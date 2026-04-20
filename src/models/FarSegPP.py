# Repository-local FarSeg++-style adaptation built on retained FarSeg and
# SegFormer components retain their original upstream license terms.

import logging
import os
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork as FPN

from .FarSeg import LightWeightDecoder
from .SegFormer import mix_transformer

LOGGER = logging.getLogger("trainer")


class MiTBackbone(nn.Module):
    """MiT-B2 feature extractor that mimics the ResNet backbone interface used by FarSeg."""

    def __init__(
        self,
        backbone: str = 'mit_b2',
        in_channels: int = 3,
        pretrained_path: Optional[str] = None,
        freeze: bool = False,
        input_adapt_mode: str = 'mean_pad',
    ) -> None:
        super().__init__()
        self.input_adapt_mode = self._normalize_input_adapt_mode(input_adapt_mode)
        builder = getattr(mix_transformer, backbone, None)
        if builder is None:
            raise ValueError(f'Unknown MiT backbone "{backbone}".')

        self.model = builder(in_chans=in_channels)
        self.embed_dims: Tuple[int, int, int, int] = tuple(getattr(self.model, 'embed_dims', (64, 128, 320, 512)))
        self.feature_strides: Tuple[int, int, int, int] = (4, 8, 16, 32)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_pretrained(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'Pretrained backbone weights not found: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ('module.', 'encoder.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value

        patch_key = 'patch_embed1.proj.weight'
        if patch_key in cleaned:
            pretrained_weight = cleaned[patch_key]
            target_weight = self.model.patch_embed1.proj.weight
            if pretrained_weight.shape != target_weight.shape:
                if self.input_adapt_mode != 'mean_pad':
                    self._log_input_adapt(self.input_adapt_mode, pretrained_weight.shape[1], target_weight.shape[1])
                cleaned[patch_key] = self._align_input_channels(
                    pretrained_weight,
                    target_weight.shape,
                    self.input_adapt_mode,
                )

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            preview = ', '.join(missing[:5])
            print(f'[MiTBackbone] Missing keys while loading pretrained weights (showing first 5): {preview}')
        if unexpected:
            preview = ', '.join(unexpected[:5])
            print(f'[MiTBackbone] Unexpected keys while loading pretrained weights (showing first 5): {preview}')

    @staticmethod
    def _normalize_input_adapt_mode(mode: str) -> str:
        normalized = (mode or 'mean_pad').lower()
        aliases = {
            'current': 'mean_pad',
            'default': 'mean_pad',
            'meanpad': 'mean_pad',
            'legacy': 'repeat_scale',
            'wildfire': 'repeat_scale',
            'repeat': 'repeat_scale',
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {'mean_pad', 'repeat_scale'}:
            raise ValueError(
                "Unsupported input_adapt_mode '{}'. Expected 'mean_pad' or 'repeat_scale'.".format(mode)
            )
        return normalized

    @staticmethod
    def _log_input_adapt(mode: str, source_in: int, target_in: int) -> None:
        LOGGER.info(
            "FarSeg++ input_adapt_mode active: mode=%s | source_in=%d | target_in=%d",
            mode,
            source_in,
            target_in,
        )

    @staticmethod
    def _align_input_channels(
        weight: torch.Tensor,
        target_shape: torch.Size,
        input_adapt_mode: str,
    ) -> torch.Tensor:
        """Adapt checkpoint conv weights when the input channel count differs."""
        target_in = target_shape[1]
        source_in = weight.shape[1]
        if target_in == source_in:
            return weight
        if target_in < source_in:
            return weight[:, :target_in, :, :]

        if input_adapt_mode == 'repeat_scale':
            repeats = target_in // source_in
            remainder = target_in % source_in
            aligned = weight.repeat(1, repeats, 1, 1)
            if remainder > 0:
                aligned = torch.cat([aligned, weight[:, :remainder, :, :]], dim=1)
            return aligned[:, :target_in, :, :] * (source_in / float(target_in))

        if input_adapt_mode == 'mean_pad':
            repeat_channels = target_in - source_in
            channel_avg = weight.mean(dim=1, keepdim=True)
            padding = channel_avg.repeat(1, repeat_channels, 1, 1)
            aligned = torch.cat([weight, padding], dim=1)
            return aligned[:, :target_in, :, :]

        raise ValueError(
            f"Unsupported input_adapt_mode '{input_adapt_mode}'. Expected 'mean_pad' or 'repeat_scale'."
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features: Sequence[torch.Tensor] = self.model(x)
        if len(features) != 4:
            raise RuntimeError('MiT backbone is expected to return four stages of features.')
        return tuple(features)  # type: ignore[return-value]


class FSRelation(nn.Module):
    """Foreground-Scene Relation module adapted from the original FarSeg++ implementation."""

    def __init__(
        self,
        scene_channels: int,
        in_channels_list: Sequence[int],
        out_channels: int,
        *,
        scale_aware_proj: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale_aware_proj = scale_aware_proj

        if self.scale_aware_proj:
            self.scene_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(scene_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ])
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout_rate),
                )
                for _ in in_channels_list
            ])
        else:
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            self.projector = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
            )

        self.content_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_channels in in_channels_list
        ])
        self.feature_reencoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_channels in in_channels_list
        ])
        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature: torch.Tensor, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        content_features = [enc(feat) for enc, feat in zip(self.content_encoders, features)]

        if self.scale_aware_proj:
            scene_features = [enc(scene_feature) for enc in self.scene_encoders]
            relations = [
                self.normalizer((scene_feat * content_feat).sum(dim=1, keepdim=True))
                for scene_feat, content_feat in zip(scene_features, content_features)
            ]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [
                self.normalizer((scene_feat * content_feat).sum(dim=1, keepdim=True))
                for content_feat in content_features
            ]

        projected = [enc(feat) for enc, feat in zip(self.feature_reencoders, features)]
        refined = [torch.cat([relation * proj, feat], dim=1) for relation, proj, feat in zip(relations, projected, features)]

        if self.scale_aware_proj:
            return [proj_module(feat) for proj_module, feat in zip(self.projectors, refined)]
        return [self.projector(feat) for feat in refined]


class FarSegPP(nn.Module):
    """FarSeg variant that swaps the ResNet backbone for MiT-B2 while keeping the original decoder design."""

    def __init__(
        self,
        num_classes: int = 16,
        backbone: str = 'mit_b2',
        in_channels: int = 3,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
        decoder_channels: int = 256,
        relation_channels: int = 256,
        decoder_mid_channels: int = 128,
        scale_aware_relation: bool = True,
        relation_dropout: float = 0.1,
        input_adapt_mode: str = 'mean_pad',
    ) -> None:
        super().__init__()
        self.backbone = MiTBackbone(
            backbone=backbone,
            in_channels=in_channels,
            pretrained_path=pretrained_backbone,
            freeze=freeze_backbone,
            input_adapt_mode=input_adapt_mode,
        )

        self.fpn = FPN(
            in_channels_list=list(self.backbone.embed_dims),
            out_channels=decoder_channels,
        )
        self.fsr = FSRelation(
            scene_channels=self.backbone.embed_dims[-1],
            in_channels_list=[decoder_channels] * 4,
            out_channels=relation_channels,
            scale_aware_proj=scale_aware_relation,
            dropout_rate=relation_dropout,
        )
        self.decoder = LightWeightDecoder(
            in_channels=relation_channels,
            mid_channels=decoder_mid_channels,
            num_classes=num_classes,
        )

    def extract_backbone_feats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.extract_backbone_feats(x)

        fpn_inputs = OrderedDict({'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5})
        fpn_feats = self.fpn(fpn_inputs)
        feats = [fpn_feats[key] for key in ('c2', 'c3', 'c4', 'c5')]

        scene_embed = F.adaptive_avg_pool2d(c5, 1)
        feats = self.fsr(scene_embed, feats)

        logits = self.decoder(feats)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits


__all__ = ['FarSegPP']
