from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.SwinUperNet import ConvBNReLU, UPerNetDecoder

try:
    from torchvision import models as tv_models
    from torchvision.models import get_model_weights
except ImportError:  # pragma: no cover - optional dependency
    tv_models = None
    get_model_weights = None


__all__ = ['ConvNeXtUPerNet']


class ConvNeXtFeatureExtractor(nn.Module):
    """Helper that exposes intermediate ConvNeXt stage features."""

    def __init__(self, backbone: nn.Module, out_indices: Sequence[int]) -> None:
        super().__init__()
        if not hasattr(backbone, 'features'):
            raise ValueError('ConvNeXt backbone is expected to expose a "features" attribute.')

        self.backbone = backbone
        self.stage_indices = self._find_stage_indices(backbone.features)
        if not self.stage_indices:
            raise ValueError('Unable to identify ConvNeXt stages for feature extraction.')

        unique_indices: List[int] = []
        for idx in out_indices:
            if idx < 0 or idx >= len(self.stage_indices):
                raise ValueError(f'Invalid stage index {idx}; available stages: {len(self.stage_indices)}.')
            if idx not in unique_indices:
                unique_indices.append(idx)
        self.requested = tuple(unique_indices)

    @staticmethod
    def _find_stage_indices(features: nn.Module) -> List[int]:
        indices: List[int] = []
        for layer_idx, layer in enumerate(features):
            if isinstance(layer, nn.Sequential) and len(layer) > 0:
                first = layer[0]
                if first.__class__.__name__ == 'CNBlock':
                    indices.append(layer_idx)
        return indices

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        collected: List[torch.Tensor] = []
        stage_ptr = 0

        for layer_idx, layer in enumerate(self.backbone.features):
            x = layer(x)
            if stage_ptr < len(self.stage_indices) and layer_idx == self.stage_indices[stage_ptr]:
                outputs.append(x)
                stage_ptr += 1

        for idx in self.requested:
            feature = outputs[idx]
            if feature.dim() != 4:
                raise ValueError(f'Expected 4D ConvNeXt feature map, got shape {tuple(feature.shape)}.')
            collected.append(feature.contiguous())
        return collected


class ConvNeXtUPerNet(nn.Module):
    """ConvNeXt encoder paired with a UPerNet decoder for semantic segmentation."""

    _DEFAULT_SAMPLE_SIZE = 224

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 5,
        backbone: str = 'convnext_tiny',
        pretrained_backbone: bool = True,
        backbone_kwargs: Optional[Dict[str, object]] = None,
        decoder_channels: int = 256,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        freeze_backbone: bool = False,
        sample_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if tv_models is None:
            raise ImportError('ConvNeXtUPerNet requires torchvision with ConvNeXt support.')

        self.backbone_name = backbone
        backbone_kwargs = dict(backbone_kwargs or {})
        self.backbone_model = self._build_backbone(
            backbone,
            in_channels,
            pretrained_backbone=pretrained_backbone,
            kwargs=backbone_kwargs,
        )

        self.feature_extractor = ConvNeXtFeatureExtractor(self.backbone_model, out_indices)

        probe_size = int(sample_size) if sample_size is not None else self._DEFAULT_SAMPLE_SIZE
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, probe_size, probe_size)
            was_training = self.feature_extractor.training
            self.feature_extractor.eval()
            feature_maps = self.feature_extractor(sample)
            if was_training:
                self.feature_extractor.train()

        if len(feature_maps) != len(self.feature_extractor.requested):
            raise ValueError('Mismatch between requested ConvNeXt stages and extracted feature maps.')

        in_channels_list = [feat.shape[1] for feat in feature_maps]
        self.decoder = UPerNetDecoder(in_channels_list, fpn_channels=decoder_channels, pool_scales=pool_scales)
        self.classifier = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[2:]
        features = self.forward_features(x)
        fused = self.decoder(features)
        logits = self.classifier(fused)
        if logits.shape[-2:] != spatial_size:
            logits = F.interpolate(logits, size=spatial_size, mode='bilinear', align_corners=False)
        return logits

    def _build_backbone(
        self,
        backbone_name: str,
        in_channels: int,
        *,
        pretrained_backbone: bool,
        kwargs: Dict[str, object],
    ) -> nn.Module:
        canonical = self._resolve_backbone_name(backbone_name)
        constructor = getattr(tv_models, canonical, None)
        if constructor is None:
            raise ValueError(f'Unsupported ConvNeXt backbone "{backbone_name}".')

        kwargs = dict(kwargs)
        if 'pretrained' in kwargs:
            kwargs.pop('pretrained')

        if 'weights' not in kwargs:
            kwargs['weights'] = self._default_weights(canonical) if pretrained_backbone else None

        backbone = constructor(**kwargs)
        if not hasattr(backbone, 'features'):
            raise ValueError('Expected ConvNeXt backbone to expose a "features" attribute.')

        if in_channels != 3:
            self._reset_stem_conv(backbone, in_channels)

        return backbone

    @staticmethod
    def _resolve_backbone_name(name: str) -> str:
        normalized = name.replace('-', '_').lower()
        alias_map = {
            'tiny': 'convnext_tiny',
            'small': 'convnext_small',
            'base': 'convnext_base',
            'large': 'convnext_large',
        }
        if normalized in alias_map:
            return alias_map[normalized]
        if normalized.startswith('convnext_'):
            return normalized
        raise ValueError(f'Unknown ConvNeXt variant "{name}".')

    def _reset_stem_conv(self, backbone: nn.Module, in_channels: int) -> None:
        stem = backbone.features[0]
        if not isinstance(stem, nn.Sequential) or len(stem) == 0:
            raise ValueError('Unexpected ConvNeXt stem structure; unable to modify input channels.')

        conv = None
        for module in stem:
            if isinstance(module, nn.Conv2d):
                conv = module
                break

        if conv is None:
            raise ValueError('ConvNeXt stem does not contain a convolution layer.')

        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        nn.init.trunc_normal_(new_conv.weight, std=0.02)
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
        stem[0] = new_conv

    @staticmethod
    def _default_weights(backbone_name: str) -> Optional[object]:
        if get_model_weights is None:
            return None
        try:
            weights_enum = get_model_weights(backbone_name)
        except (AttributeError, ValueError, RuntimeError):
            return None
        return getattr(weights_enum, 'DEFAULT', None)
