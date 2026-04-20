from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_sam import build_sam2
from .decoders import FPNDecoder, UPerNetDecoder

LOGGER = logging.getLogger("trainer")


class SAM2FeatureExtractor(nn.Module):
    """Wraps SAM2 image encoder and exposes raw multi-scale feature maps."""

    def __init__(
        self,
        config_file: str,
        checkpoint: Optional[str] = None,
        device: str = 'cuda',
        mode: Optional[str] = None,
        hydra_overrides: Optional[Iterable[str]] = None,
        apply_postprocessing: bool = False,
        in_channels: int = 3,
        patch_embed_init: str = "repeat_scale",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        overrides = list(hydra_overrides) if hydra_overrides is not None else []

        build_mode = mode or ('eval' if freeze_encoder else 'train')
        sam_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint,
            device=device,
            mode=build_mode,
            hydra_overrides_extra=overrides,
            apply_postprocessing=apply_postprocessing,
        )

        self.image_encoder = sam_model.image_encoder
        self.image_size = getattr(sam_model, 'image_size', None)
        self.backbone_stride = getattr(sam_model, 'backbone_stride', None)
        self.in_channels = int(in_channels)

        self._adjust_patch_embed(self.in_channels, patch_embed_init)

        channel_list: Optional[Sequence[int]] = None
        trunk = getattr(self.image_encoder, 'trunk', None)
        if trunk is not None and hasattr(trunk, 'channel_list'):
            channel_list = list(trunk.channel_list)  # type: ignore[attr-defined]
        elif hasattr(self.image_encoder, 'channel_list'):
            channel_list = list(self.image_encoder.channel_list)  # type: ignore[attr-defined]

        if channel_list is None:
            raise ValueError('SAM2 image encoder does not expose channel_list; unable to infer feature dimensions.')

        scalp = max(0, int(getattr(self.image_encoder, 'scalp', 0)))
        if scalp:
            channel_list = channel_list[:-scalp] if scalp < len(channel_list) else []
        if not channel_list:
            raise ValueError('SAM2 image encoder exposes no feature levels after applying scalp configuration.')

        self.channel_list = tuple(int(c) for c in channel_list)
        num_levels = len(self.channel_list)
        self.highest_level = num_levels - 1

        base_stride = self._infer_base_stride()
        self.output_strides = tuple(base_stride * (2 ** (num_levels - 1 - idx)) for idx in range(num_levels))

        if freeze_encoder:
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            self.image_encoder.train()

        del sam_model

    def _infer_base_stride(self) -> int:
        proj = getattr(self.image_encoder.trunk.patch_embed, 'proj', None)
        if proj is None or not hasattr(proj, 'stride'):
            return 4
        stride = proj.stride
        if isinstance(stride, tuple):
            return int(stride[0])
        return int(stride)

    @staticmethod
    def _normalize_patch_embed_init_mode(mode: str) -> str:
        normalized = (mode or "repeat_scale").lower()
        alias_map = {
            "current": "repeat_scale",
            "legacy": "copy_ch0",
            "legacy_copy_fill": "copy_ch0",
            "copy_first": "copy_ch0",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized not in {"repeat_scale", "copy_ch0"}:
            raise ValueError(
                "Unsupported patch_embed_init '{}'. Expected 'repeat_scale' or 'copy_ch0'.".format(mode)
            )
        return normalized

    @staticmethod
    def _repeat_scale_patch_embed_weights(source_w: torch.Tensor, in_channels: int) -> torch.Tensor:
        # Keep the default multi-channel init exactly aligned with the pre-ablation behavior.
        weight = source_w
        current_in = weight.shape[1]
        if in_channels < current_in:
            weight = weight[:, :in_channels]
        elif in_channels > current_in:
            repeats = in_channels // current_in
            remainder = in_channels % current_in
            weight = weight.repeat(1, repeats, 1, 1)
            if remainder:
                weight = torch.cat([weight, weight[:, :remainder]], dim=1)
            weight = weight * (current_in / float(in_channels))
        return weight

    @staticmethod
    def _copy_ch0_patch_embed_weights(source_w: torch.Tensor, in_channels: int) -> torch.Tensor:
        current_in = source_w.shape[1]
        if in_channels <= current_in:
            return source_w[:, :in_channels]

        weight = source_w.new_empty(source_w.shape[0], in_channels, source_w.shape[2], source_w.shape[3])
        weight[:, :current_in].copy_(source_w)
        extra = in_channels - current_in
        weight[:, current_in:].copy_(source_w[:, :1].repeat(1, extra, 1, 1))
        return weight

    def _adjust_patch_embed(self, in_channels: int, patch_embed_init: str = "repeat_scale") -> None:
        patch_embed = getattr(self.image_encoder.trunk.patch_embed, 'proj', None)
        if patch_embed is None:
            raise ValueError('SAM2 image encoder does not expose a convolutional patch embedding.')
        if patch_embed.in_channels == in_channels:
            return
        patch_embed_init = self._normalize_patch_embed_init_mode(patch_embed_init)

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=patch_embed.out_channels,
            kernel_size=patch_embed.kernel_size,
            stride=patch_embed.stride,
            padding=patch_embed.padding,
            bias=patch_embed.bias is not None,
            device=patch_embed.weight.device,
            dtype=patch_embed.weight.dtype,
        )

        with torch.no_grad():
            source_w = patch_embed.weight
            if patch_embed_init == "copy_ch0":
                LOGGER.info(
                    "SAM2 patch_embed ablation active: patch_embed_init=copy_ch0 | pretrained_in=%d | target_in=%d",
                    source_w.shape[1],
                    in_channels,
                )
                weight = self._copy_ch0_patch_embed_weights(source_w, in_channels)
            else:
                weight = self._repeat_scale_patch_embed_weights(source_w, in_channels)
            new_conv.weight.copy_(weight)
            if patch_embed.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(patch_embed.bias)

        self.image_encoder.trunk.patch_embed.proj = new_conv

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() != 4:
            raise ValueError('Expected input tensor with shape (B, C, H, W).')
        if x.shape[1] != self.in_channels:
            raise ValueError(f'Expected {self.in_channels} input channels, received {x.shape[1]}.')

        outputs = self.image_encoder(x)
        if not isinstance(outputs, (list, tuple)):
            raise ValueError('SAM2 image encoder did not return a sequence of feature maps.')
        if not outputs:
            raise RuntimeError('SAM2 image encoder returned no feature maps.')

        expected = len(self.channel_list)
        if len(outputs) < expected:
            raise RuntimeError(
                f'SAM2 image encoder returned {len(outputs)} feature maps but {expected} were expected.'
            )

        selected = list(outputs[-expected:])

        ordered: List[torch.Tensor] = []
        remaining = list(selected)
        for expected_channels in self.channel_list:
            match_idx = next((idx for idx, feat in enumerate(remaining) if feat.shape[1] == expected_channels), None)
            if match_idx is None:
                available = [feat.shape[1] for feat in remaining]
                raise RuntimeError(
                    f'Unable to match expected channel dimension {expected_channels} among available features {available}.'
                )
            ordered.append(remaining.pop(match_idx))

        return ordered


class FPNDecoder(nn.Module):
    """Simple FPN decoder that fuses multi-scale features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        use_bn: bool = True,
        upsample_mode: str = 'bilinear',
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError('FPNDecoder requires at least one input channel description.')

        self.upsample_mode = upsample_mode
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for channels in in_channels:
            self.lateral_convs.append(nn.Conv2d(channels, out_channels, kernel_size=1, bias=False))

            layers: List[nn.Module] = [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.output_convs.append(nn.Sequential(*layers))

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != len(self.lateral_convs):
            raise ValueError(
                f'FPNDecoder expected {len(self.lateral_convs)} feature maps, received {len(features)}.'
            )

        results: List[torch.Tensor] = [torch.empty(0)] * len(features)
        prev_feature: Optional[torch.Tensor] = None

        for idx in reversed(range(len(features))):
            lateral = self.lateral_convs[idx](features[idx])
            if prev_feature is not None:
                prev_feature = F.interpolate(
                    prev_feature,
                    size=lateral.shape[-2:],
                    mode=self.upsample_mode,
                    align_corners=False if self.upsample_mode == 'bilinear' else None,
                )
                lateral = lateral + prev_feature
            fused = self.output_convs[idx](lateral)
            results[idx] = fused
            prev_feature = fused

        return results


class SAM2FPN(nn.Module):
    """SAM2 backbone followed by an FPN decoder and classification head."""

    def __init__(
        self,
        config_file: str,
        checkpoint: Optional[str] = None,
        *,
        in_channels: int = 3,
        num_classes: int = 4,
        decoder_type: str = 'fpn',
        decoder_channels: Optional[int] = None,
        freeze_encoder: bool = False,
        use_decoder_bn: bool = True,
        hydra_overrides: Optional[Iterable[str]] = None,
        apply_postprocessing: bool = False,
        device: str = 'cuda',
        mode: Optional[str] = None,
        decoder_pool_scales: Iterable[int] = (1, 2, 3, 6),
        patch_embed_init: str = "repeat_scale",
    ) -> None:
        super().__init__()

        decoder_type = (decoder_type or 'fpn').lower()
        if decoder_type not in {'fpn', 'upernet'}:
            raise ValueError(f"Unsupported decoder_type '{decoder_type}'. Expected 'fpn' or 'upernet'.")
        self.decoder_type = decoder_type

        self.encoder = SAM2FeatureExtractor(
            config_file=config_file,
            checkpoint=checkpoint,
            device=device,
            mode=mode,
            hydra_overrides=hydra_overrides,
            apply_postprocessing=apply_postprocessing,
            in_channels=in_channels,
            patch_embed_init=patch_embed_init,
            freeze_encoder=freeze_encoder,
        )

        if self.decoder_type == 'fpn':
            self.decoder_channels = int(
                decoder_channels if decoder_channels is not None else self.encoder.channel_list[-1]
            )
            self.decoder = FPNDecoder(
                in_channels=self.encoder.channel_list,
                out_channels=self.decoder_channels,
                use_bn=use_decoder_bn,
            )
        else:
            default_channels = self.encoder.channel_list[-1]
            self.decoder_channels = int(decoder_channels if decoder_channels is not None else default_channels)
            self.decoder = UPerNetDecoder(
                in_channels=self.encoder.channel_list,
                fpn_channels=self.decoder_channels,
                pool_scales=tuple(decoder_pool_scales),
            )

        self.output_stride = int(self.encoder.output_strides[self.encoder.highest_level])

        self.classifier = nn.Conv2d(self.decoder_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        decoded = self.decoder(features)
        if isinstance(decoded, list):
            if not decoded:
                raise RuntimeError('Decoder returned no feature maps.')
            selected = decoded[self.encoder.highest_level]
        else:
            selected = decoded

        logits = self.classifier(selected)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits


__all__ = ['SAM2FPN']
