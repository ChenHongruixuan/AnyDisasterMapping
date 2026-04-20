from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SAM2FPN import SAM2FeatureExtractor
from src.models.ChangeMamba.ChangeDecoder import ChangeDecoder
from src.models.ChangeMamba.SemanticDecoder import SemanticDecoder
from src.models.ChangeMamba.vmamba import LayerNorm2d


_NORM_LOOKUP: Dict[str, type[nn.Module]] = {
    'ln': nn.LayerNorm,
    'ln2d': LayerNorm2d,
    'bn': nn.BatchNorm2d,
}

_ACT_LOOKUP: Dict[str, type[nn.Module]] = {
    'silu': nn.SiLU,
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
}


class SAM2Mamba(nn.Module):
    """Two-stream change-detection model using SAM2 encoder and Mamba decoders."""

    def __init__(
        self,
        *,
        config_file: str,
        checkpoint: Optional[str] = None,
        pre_in_channels: int = 3,
        post_in_channels: int = 3,
        semantic_classes: int = 2,
        damage_classes: int = 4,
        decoder_kwargs: Dict[str, object],
        hydra_overrides: Optional[Iterable[str]] = None,
        apply_postprocessing: bool = False,
        freeze_encoder: bool = False,
        device: str = 'cuda',
        mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.pre_encoder = SAM2FeatureExtractor(
            config_file=config_file,
            checkpoint=checkpoint,
            device=device,
            mode=mode,
            hydra_overrides=hydra_overrides,
            apply_postprocessing=apply_postprocessing,
            in_channels=pre_in_channels,
            freeze_encoder=freeze_encoder,
        )
        self.post_encoder = self.pre_encoder if freeze_encoder else SAM2FeatureExtractor(
            config_file=config_file,
            checkpoint=checkpoint,
            device=device,
            mode=mode,
            hydra_overrides=hydra_overrides,
            apply_postprocessing=apply_postprocessing,
            in_channels=post_in_channels,
            freeze_encoder=freeze_encoder,
        )

        if freeze_encoder:
            self.post_encoder = self.pre_encoder

        channel_first = True
        encoder_dims = tuple(reversed(self.pre_encoder.channel_list))

        norm_name = str(decoder_kwargs.get('norm_layer', 'bn')).lower()
        norm_layer = _NORM_LOOKUP.get(norm_name)
        if norm_layer is None:
            raise ValueError(f'Unsupported norm_layer "{norm_name}" for SAM2Mamba.')

        ssm_act = str(decoder_kwargs.get('ssm_act_layer', 'silu')).lower()
        mlp_act = str(decoder_kwargs.get('mlp_act_layer', 'gelu')).lower()
        ssm_act_layer = _ACT_LOOKUP.get(ssm_act)
        mlp_act_layer = _ACT_LOOKUP.get(mlp_act)
        if ssm_act_layer is None or mlp_act_layer is None:
            raise ValueError('Invalid activation specified in decoder_kwargs.')

        clean_kwargs = dict(decoder_kwargs)
        clean_kwargs.pop('norm_layer', None)
        clean_kwargs.pop('ssm_act_layer', None)
        clean_kwargs.pop('mlp_act_layer', None)

        self.semantic_decoder = SemanticDecoder(
            encoder_dims=encoder_dims,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )

        self.change_decoder = ChangeDecoder(
            encoder_dims=encoder_dims,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )

        self.semantic_head = nn.Conv2d(128, semantic_classes, kernel_size=1)
        self.damage_head = nn.Conv2d(128, damage_classes, kernel_size=1)

    @staticmethod
    def _reorder_features(features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return list(reversed(features))

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_features = self._reorder_features(self.pre_encoder(pre_image))
        post_features = self._reorder_features(self.post_encoder(post_image))

        semantic_features = self.semantic_decoder(pre_features)
        damage_features = self.change_decoder(pre_features, post_features)

        semantic_logits = self.semantic_head(semantic_features)
        damage_logits = self.damage_head(damage_features)

        semantic_logits = F.interpolate(semantic_logits, size=pre_image.shape[-2:], mode='bilinear', align_corners=False)
        damage_logits = F.interpolate(damage_logits, size=post_image.shape[-2:], mode='bilinear', align_corners=False)

        return semantic_logits, damage_logits


__all__ = ['SAM2Mamba']
