
import math
import os
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels
from src.models.dinov3.models import vision_transformer as dv3_vit

__all__ = ['DinoV3FeatureExtractor', 'DinoV3DPT']


_DINOV3_ALIASES = {
    'dinov3_vits': 'dinov3_vits16',
    'dinov3_vits16': 'dinov3_vits16',
    'dinov3_vits16plus': 'dinov3_vits16plus',
    'dinov3_vitb': 'dinov3_vitb16',
    'dinov3_vitb16': 'dinov3_vitb16',
    'dinov3_vitl': 'dinov3_vitl16',
    'dinov3_vitl16': 'dinov3_vitl16',
    'dinov3_vith': 'dinov3_vith16',
    'dinov3_vith16': 'dinov3_vith16',
    'dinov3_vith16plus': 'dinov3_vith16plus',
    'dinov3_vit7b': 'dinov3_vit7b16',
    'dinov3_vit7b16': 'dinov3_vit7b16',
}


_DINOV3_BUILDERS = {
    'dinov3_vits16': lambda **kwargs: dv3_vit.vit_small(patch_size=16, **kwargs),
    'dinov3_vits16plus': lambda **kwargs: dv3_vit.vit_small(patch_size=16, **kwargs),
    'dinov3_vitb16': lambda **kwargs: dv3_vit.vit_base(patch_size=16, **kwargs),
    'dinov3_vitl16': lambda **kwargs: dv3_vit.vit_large(patch_size=16, **kwargs),
    'dinov3_vith16': lambda **kwargs: dv3_vit.vit_huge2(patch_size=16, **kwargs),
    'dinov3_vith16plus': lambda **kwargs: dv3_vit.vit_huge2(patch_size=16, **kwargs),
    'dinov3_vit7b16': lambda **kwargs: dv3_vit.vit_7b(patch_size=16, **kwargs),
}


class DinoV3FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = 'dinov3_vitl16',
        pretrained: bool = True,
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        hub_repo: Optional[str] = None,
        hub_source: Optional[str] = None,
        hub_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

        self.model = torch.hub.load(
            repo_or_dir=hub_repo,
            model=model_name,
            source="local",
            weights=hub_weights
        )

        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is None:
            raise ValueError('Backbone does not expose a patch embedding module.')

        self.embed_dim = getattr(self.model, 'embed_dim', None)
        if self.embed_dim is None:
            raise ValueError('Unable to determine embedding dimension for backbone.')

        patch_size = getattr(patch_embed, 'patch_size', (16, 16))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))

        original_in = getattr(patch_embed, 'in_chans', in_channels)
        if in_channels != original_in:
            self._adapt_input_channels(in_channels)
        else:
            patch_embed.in_chans = in_channels

        depth = len(getattr(self.model, 'blocks', []))
        if depth == 0:
            raise ValueError('Backbone exposes no transformer blocks.')

        if hook_indices is None:
            if num_features <= 0:
                raise ValueError('num_features must be positive when hook_indices is not provided.')
            step = depth / float(num_features)
            indices: List[int] = []
            for i in range(1, num_features + 1):
                idx = int(round(i * step) - 1)
                idx = max(0, min(depth - 1, idx))
                indices.append(idx)
            hook_indices = tuple(sorted(set(indices)))
        else:
            filtered = [int(idx) for idx in hook_indices if 0 <= int(idx) < depth]
            if not filtered:
                raise ValueError('Provided hook_indices are invalid for the selected backbone.')
            hook_indices = tuple(sorted(set(filtered)))

        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        if getattr(self.model, 'chunked_blocks', False):
            warnings.warn('DINOv3 backbone uses chunked blocks; custom adapters may require adjustments.')

    def _adapt_input_channels(self, in_channels: int) -> None:
        patch_embed = self.model.patch_embed
        proj = getattr(patch_embed, 'proj', None)
        if proj is None or not isinstance(proj, nn.Conv2d):
            raise ValueError('Patch embedding projection is not a convolution; cannot adapt input channels.')
        if proj.in_channels == in_channels:
            patch_embed.in_chans = in_channels
            return

        new_proj = nn.Conv2d(
            in_channels,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=proj.bias is not None,
        ).to(device=proj.weight.device, dtype=proj.weight.dtype)

        with torch.no_grad():
            weight = proj.weight
            current_in = weight.shape[1]
            if in_channels < current_in:
                weight = weight[:, :in_channels, :, :]
            elif in_channels > current_in:
                repeats = in_channels // current_in
                remainder = in_channels % current_in
                weight = weight.repeat(1, repeats, 1, 1)
                if remainder:
                    weight = torch.cat([weight, weight[:, :remainder, :, :]], dim=1)
                weight = weight * (current_in / float(in_channels))
            new_proj.weight.copy_(weight)
            if proj.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(proj.bias)

        patch_embed.proj = new_proj
        patch_embed.in_chans = in_channels

    def _compute_patch_shape(self, num_tokens: int, height: int, width: int) -> Tuple[int, int]:
        patch_h_size, patch_w_size = self.patch_size
        patch_h = max(1, int(round(height / float(patch_h_size))))
        patch_w = max(1, int(round(width / float(patch_w_size))))

        if patch_h * patch_w != num_tokens:
            patch_w = max(1, num_tokens // patch_h)
        if patch_h * patch_w != num_tokens and patch_w > 0:
            patch_h = max(1, num_tokens // patch_w)
        if patch_h * patch_w != num_tokens:
            patch_h = max(1, int(round(math.sqrt(num_tokens))))
            patch_w = max(1, num_tokens // patch_h)
        if patch_h * patch_w != num_tokens:
            raise RuntimeError('Failed to infer patch grid size from token count.')
        return patch_h, patch_w

    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        height, width = x.shape[-2:]

        outputs = self.model.get_intermediate_layers(
            x,
            n=self.hook_indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        features: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for entry in outputs:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                raise RuntimeError('Unexpected output format from DINOv3 backbone.')
            features.append((entry[0], entry[1]))

        num_tokens = features[0][0].shape[1]
        patch_shape = self._compute_patch_shape(num_tokens, height, width)
        return features, patch_shape


class DinoV3DPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 11,
        backbone: str = 'dinov3_vitl16',
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
        torchhub_repo: Optional[str] = None,
        torchhub_source: Optional[str] = None,
        torchhub_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = DinoV3FeatureExtractor(
            model_name=backbone,
            pretrained=pretrained_backbone,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
            hub_repo=torchhub_repo,
            hub_source=torchhub_source,
            hub_weights=torchhub_weights,
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
        self.num_features = len(self.hook_indices)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features, patch_shape = self.encoder(x)
        return self.decoder(features, patch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        decoded = self.forward_features(x)
        refined = self.head(decoded)
        logits = self.classifier(refined)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits
