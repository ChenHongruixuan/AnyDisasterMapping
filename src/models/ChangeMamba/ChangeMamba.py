import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from src.models.ChangeMamba.Mamba_backbone import Backbone_VSSM
from src.models.ChangeMamba.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from src.models.ChangeMamba.ChangeDecoder import ChangeDecoder
from src.models.ChangeMamba.SemanticDecoder import SemanticDecoder


class ChangeMamba(nn.Module):
    def __init__(self, output_building, output_damage, pretrained=None,
                 backbone: str = 'vssm_tiny_224_0229flex',
                 siamese_mode: str = 'pseudo', **kwargs):
        super(ChangeMamba, self).__init__()

        from os.path import dirname, join
        from src.models.ChangeMamba.configs.config import get_config

        self.siamese_mode = self._normalize_mode(siamese_mode)
        self.shared_encoder = self.siamese_mode == 'shared'

        # Load VSSM config from backbone name (same pattern as wildfire)
        cfg_file = join(dirname(__file__), f"configs/vssm1/{backbone}.yaml")
        config = get_config(cfg_file=cfg_file)
        cfg = config.MODEL
        vssm = cfg.VSSM
        ssm_dt_rank = vssm.SSM_DT_RANK
        if isinstance(ssm_dt_rank, str) and ssm_dt_rank.lower() != 'auto':
            ssm_dt_rank = int(ssm_dt_rank)

        encoder_kwargs = dict(
            patch_size=vssm.PATCH_SIZE,
            in_chans=vssm.IN_CHANS,
            num_classes=cfg.NUM_CLASSES,
            depths=vssm.DEPTHS,
            dims=vssm.EMBED_DIM,
            ssm_d_state=vssm.SSM_D_STATE,
            ssm_ratio=vssm.SSM_RATIO,
            ssm_rank_ratio=vssm.SSM_RANK_RATIO,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=vssm.SSM_ACT_LAYER,
            ssm_conv=vssm.SSM_CONV,
            ssm_conv_bias=vssm.SSM_CONV_BIAS,
            ssm_drop_rate=vssm.SSM_DROP_RATE,
            ssm_init=vssm.SSM_INIT,
            forward_type=vssm.SSM_FORWARDTYPE,
            mlp_ratio=vssm.MLP_RATIO,
            mlp_act_layer=vssm.MLP_ACT_LAYER,
            mlp_drop_rate=vssm.MLP_DROP_RATE,
            drop_path_rate=cfg.DROP_PATH_RATE,
            patch_norm=vssm.PATCH_NORM,
            norm_layer=vssm.NORM_LAYER,
            downsample_version=vssm.DOWNSAMPLE,
            patchembed_version=vssm.PATCHEMBED,
            gmlp=vssm.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

        # If in_chans was explicitly passed and differs from config default,
        # build with original channels first (for pretrained weight loading),
        # then adapt patch_embed to the target channels.
        target_in_chans = kwargs.pop('in_chans', None)
        post_channels = kwargs.pop('post_channels', None)

        self.encoder_1 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **encoder_kwargs)
        if self.shared_encoder:
            self.encoder_2 = self.encoder_1
        else:
            self.encoder_2 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **encoder_kwargs)

        # Adapt patch_embed if target channels differ from pretrained (3ch)
        if target_in_chans is not None and target_in_chans != encoder_kwargs.get('in_chans', 3):
            from src.models.channel_adapt import adapt_conv_channels
            self.encoder_1.patch_embed[0] = adapt_conv_channels(
                self.encoder_1.patch_embed[0], target_in_chans)
            # Post branch: use post_channels if specified, else same as pre
            post_ch = post_channels if post_channels is not None else target_in_chans
            if not self.shared_encoder:
                self.encoder_2.patch_embed[0] = adapt_conv_channels(
                    self.encoder_2.patch_embed[0], post_ch)
        elif post_channels is not None and post_channels != encoder_kwargs.get('in_chans', 3):
            from src.models.channel_adapt import adapt_conv_channels
            if not self.shared_encoder:
                self.encoder_2.patch_embed[0] = adapt_conv_channels(
                    self.encoder_2.patch_embed[0], post_channels)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder_1.channel_first

        norm_layer: nn.Module = _NORMLAYERS.get(encoder_kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(encoder_kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(encoder_kwargs['mlp_act_layer'].lower(), None)

        clean_kwargs = {k: v for k, v in encoder_kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder_1.dims,
            channel_first=self.encoder_1.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder_2.dims,
            channel_first=self.encoder_2.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
      
        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder_1(pre_data)
        post_features = self.encoder_2(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_building = self.decoder_building(pre_features)
        output_damage = self.decoder_damage(pre_features, post_features)
       
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')

        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_data.size()[-2:], mode='bilinear')
       
        return output_building, output_damage

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or 'pseudo').strip().lower()
        if normalized not in {'shared', 'pseudo'}:
            raise ValueError("siamese_mode must be either 'shared' or 'pseudo'.")
        return normalized
