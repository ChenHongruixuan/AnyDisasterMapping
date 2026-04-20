import math
import warnings
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None

from src.models.DinoV2DPT import (
    DINOv2FeatureExtractor,
    DPTDecoder,
    DPTHead,
    _default_stage_channels,
)


class LoRALinear(nn.Module):
    """LoRA adapter that wraps a linear layer while keeping the base weights frozen."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError('LoRA rank must be non-negative.')

        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # freeze the base weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, linear.in_features))
            self.lora_B = nn.Parameter(torch.zeros(linear.out_features, self.rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.rank == 0:
            return result

        dropped = self.dropout(x)
        original_shape = dropped.shape
        if dropped.dim() > 2:
            dropped = dropped.reshape(-1, dropped.shape[-1])
            lora = F.linear(dropped, self.lora_A)
            lora = F.linear(lora, self.lora_B)
            lora = lora.reshape(*original_shape[:-1], -1)
        else:
            lora = F.linear(dropped, self.lora_A)
            lora = F.linear(lora, self.lora_B)
        return result + self.scaling * lora

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.rank == 0:
            return []
        return [self.lora_A, self.lora_B]


class DINOv2FeatureExtractorLoRA(DINOv2FeatureExtractor):
    """DINOv2 feature extractor that injects LoRA adapters into attention blocks."""

    def __init__(
        self,
        model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        pretrained: bool = True,
        in_channels: int = 3,
        hook_indices: Sequence[int] = (5, 11, 17, 23),
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Sequence[str] = ('qkv', 'proj'),
    ) -> None:
        if timm is None:
            raise ImportError('DINOv2 backbones require the timm package. Please install timm to continue.')
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            hook_indices=hook_indices,
            freeze_backbone=False,
        )
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = tuple(target_modules)

        self._enable_flexible_patch_embed()
        self._inject_lora()
        self._lock_except_lora()

    def _enable_flexible_patch_embed(self) -> None:
        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is None:
            return
        if hasattr(patch_embed, 'strict_img_size'):
            patch_embed.strict_img_size = False
        if hasattr(patch_embed, 'img_size'):
            patch_embed.img_size = None
        if hasattr(patch_embed, 'dynamic_img_pad'):
            patch_embed.dynamic_img_pad = True

    def _inject_lora(self) -> None:
        if self.lora_rank == 0:
            return
        for block in self.model.blocks:
            if 'qkv' in self.target_modules and hasattr(block.attn, 'qkv'):
                block.attn.qkv = LoRALinear(block.attn.qkv, self.lora_rank, self.lora_alpha, self.lora_dropout)
            if 'proj' in self.target_modules and hasattr(block.attn, 'proj'):
                block.attn.proj = LoRALinear(block.attn.proj, self.lora_rank, self.lora_alpha, self.lora_dropout)

    def _lock_except_lora(self) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith('patch_embed')
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and module.rank > 0:
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and module.rank > 0:
                yield from module.lora_parameters()


class DinoV2DPTLoRA(nn.Module):
    """DINOv2-DPT model that fine-tunes only LoRA adapters on a frozen DINOv2-L backbone."""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        backbone: str = 'vit_large_patch14_dinov2.lvd142m',
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Sequence[int] = (5, 11, 17, 23),
        output_strides: Sequence[int] = (32, 16, 8, 4),
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
        pretrained_backbone: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Sequence[str] = ('qkv', 'proj'),
    ) -> None:
        super().__init__()
        if len(hook_indices) != len(output_strides):
            raise ValueError('hook_indices and output_strides must have the same length.')

        self.encoder = DINOv2FeatureExtractorLoRA(
            model_name=backbone,
            pretrained=pretrained_backbone,
            in_channels=in_channels,
            hook_indices=hook_indices,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        stage_channels = tuple(int(c) for c in (
            decoder_stage_channels if decoder_stage_channels is not None else _default_stage_channels(self.encoder.embed_dim)
        ))
        self.decoder_channels = int(decoder_channels) if decoder_channels is not None else stage_channels[0]
        self.head_channels = int(head_channels) if head_channels is not None else self.decoder_channels
        self.decoder = DPTDecoder(
            embed_dim=self.encoder.embed_dim,
            features=self.decoder_channels,
            out_channels=stage_channels,
            use_bn=use_bn,
            use_cls_token=True,
        )
        self.head = DPTHead(self.decoder_channels, self.head_channels, use_bn=use_bn)
        self.classifier = nn.Conv2d(self.head_channels, num_classes, kernel_size=1)

        self.hook_indices = self.encoder.hook_indices
        self.output_strides = tuple(output_strides)
        if self.output_strides != (32, 16, 8, 4):
            warnings.warn('output_strides parameter is kept for backward compatibility but not used by the decoder.')
        self.num_features = len(self.hook_indices)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features, patch_shape = self.encoder(x)
        return self.decoder(features, patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        input_size = x.shape[-2:]
        decoded = self.forward_features(x)
        refined = self.head(decoded)
        logits = self.classifier(refined)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.lora_parameters()
