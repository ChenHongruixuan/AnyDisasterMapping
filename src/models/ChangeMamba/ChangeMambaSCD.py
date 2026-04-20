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

class ChangeMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained=None,
                 backbone: str = 'vssm_tiny_224_0229flex', **kwargs):
        super(ChangeMambaSCD, self).__init__()

        from os.path import dirname, join
        from src.models.ChangeMamba.configs.config import get_config

        # Load VSSM config from backbone name (same pattern as ChangeMamba)
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

        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **encoder_kwargs)

        # Adapt patch_embed if target channels differ from pretrained (3ch)
        if target_in_chans is not None and target_in_chans != encoder_kwargs.get('in_chans', 3):
            from src.models.channel_adapt import adapt_conv_channels
            self.encoder.patch_embed[0] = adapt_conv_channels(
                self.encoder.patch_embed[0], target_in_chans)

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

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(encoder_kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(encoder_kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(encoder_kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in encoder_kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )


        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)


    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)


        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')

        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2
