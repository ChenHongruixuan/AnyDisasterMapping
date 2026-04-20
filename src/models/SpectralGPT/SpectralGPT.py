import argparse
import os
import pickle
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .models_vit_tensor import vit_base_patch8_128
from src.models.sam2.decoders.upernet import UPerNetDecoder


def _default_stage_channels(embed_dim: int) -> Tuple[int, int, int, int]:
    if embed_dim >= 1024:
        return 256, 512, 1024, 1024
    if embed_dim >= 768:
        return 256, 512, 768, 768
    if embed_dim >= 512:
        return 192, 384, 768, 768
    return 128, 256, 512, 512


def _make_scratch(in_shape: Sequence[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()

    out1 = out_shape
    out2 = out_shape
    out3 = out_shape
    out4 = out_shape

    if expand:
        out1 = out_shape
        out2 = out_shape * 2
        out3 = out_shape * 4
        out4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, use_bn: bool) -> None:
        super().__init__()
        self.use_bn = use_bn
        self.activation = activation

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        use_bn: bool,
        align_corners: bool = True,
        expand: bool = False,
        default_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        self.default_size = default_size

        self.out_conv = nn.Conv2d(
            features,
            features // 2 if expand else features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.residual_1 = ResidualConvUnit(features, activation, use_bn)
        self.residual_2 = ResidualConvUnit(features, activation, use_bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        output = x
        if skip is not None:
            output = self.skip_add.add(output, self.residual_1(skip))

        output = self.residual_2(output)

        if size is not None:
            resize_kwargs = {'size': size}
        elif self.default_size is not None:
            resize_kwargs = {'size': self.default_size}
        else:
            resize_kwargs = {'scale_factor': 2.0}

        output = F.interpolate(output, mode='bilinear', align_corners=self.align_corners, **resize_kwargs)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features: int, use_bn: bool, size: Optional[Tuple[int, int]] = None) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=False),
        use_bn=use_bn,
        align_corners=True,
        expand=False,
        default_size=size,
    )


class SpectralDPTDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        features: int,
        out_channels: Sequence[int],
        use_bn: bool = False,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError('SpectralDPTDecoder expects four encoder stages.')

        self.use_cls_token = use_cls_token
        self.features = int(features)
        self.projection_channels = tuple(int(c) for c in out_channels)

        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, c, kernel_size=1, stride=1, padding=0, bias=True)
            for c in self.projection_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(self.projection_channels[0], self.projection_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(self.projection_channels[1], self.projection_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(self.projection_channels[3], self.projection_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        if use_cls_token:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.GELU(),
                )
                for _ in range(4)
            ])
        else:
            self.readout_projects = nn.ModuleList([nn.Identity() for _ in range(4)])

        self.scratch = _make_scratch(self.projection_channels, self.features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(self.features, use_bn)

    @staticmethod
    def _reshape_tokens(tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.permute(0, 2, 1).reshape(batch, channels, patch_h, patch_w)

    def forward(
        self,
        features: Sequence[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
        patch_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f'SpectralDPTDecoder expects four feature tensors, received {len(features)}.')

        patch_h, patch_w = patch_shape
        processed: List[torch.Tensor] = []

        for idx, feat in enumerate(features):
            if isinstance(feat, (tuple, list)):
                patch_tokens = feat[0]
                cls_token = feat[1] if len(feat) > 1 else None
            else:
                patch_tokens = feat
                cls_token = None

            if self.use_cls_token and cls_token is not None:
                readout = cls_token.unsqueeze(1).expand_as(patch_tokens)
                patch_tokens = torch.cat((patch_tokens, readout), dim=-1)
                patch_tokens = self.readout_projects[idx](patch_tokens)
            else:
                patch_tokens = self.readout_projects[idx](patch_tokens)

            spatial = self._reshape_tokens(patch_tokens, patch_h, patch_w)
            spatial = self.projects[idx](spatial)
            spatial = self.resize_layers[idx](spatial)
            processed.append(spatial)

        layer1, layer2, layer3, layer4 = processed

        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[2:])
        path3 = self.scratch.refinenet3(path4, layer3_rn, size=layer2_rn.shape[2:])
        path2 = self.scratch.refinenet2(path3, layer2_rn, size=layer1_rn.shape[2:])
        path1 = self.scratch.refinenet1(path2, layer1_rn)

        return path1


class DPTHead(nn.Module):
    def __init__(self, in_channels: int, head_channels: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(head_channels))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralGPTFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_frames: int = 12,
        img_size: int = 128,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.model = vit_base_patch8_128()
        self.model.head = nn.Identity()
        self.model.dropout = nn.Identity()

        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is None:
            raise ValueError('SpectralGPT backbone does not expose a patch embedding module.')

        self.embed_dim = getattr(self.model, 'pos_embed', None).shape[-1]
        self.num_frames = int(num_frames)
        self.img_size = int(img_size)
        self.time_bins = patch_embed.input_size[0]
        self.spatial_height = patch_embed.input_size[1]
        self.spatial_width = patch_embed.input_size[2]
        self.spatial_bins = self.spatial_height * self.spatial_width

        self._base_time_bins = self.time_bins
        self._base_spatial_height = self.spatial_height
        self._base_spatial_width = self.spatial_width
        self._base_spatial_bins = self.spatial_bins

        depth = len(getattr(self.model, 'blocks', []))
        if depth == 0:
            raise ValueError('SpectralGPT backbone exposes no transformer blocks.')

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
                raise ValueError('Provided hook_indices are invalid for the SpectralGPT backbone.')
            hook_indices = tuple(sorted(set(filtered)))

        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_pretrained(self, path: str) -> None:
        resolved = os.path.expanduser(path)
        if not os.path.isfile(resolved):
            warnings.warn(f'Pretrained backbone path {resolved} not found. Proceeding without loading weights.', stacklevel=2)
            return

        def _torch_load(weights_only: Optional[bool] = True):
            load_kwargs = dict(map_location='cpu')
            if weights_only is not None:
                load_kwargs['weights_only'] = weights_only
            try:
                return torch.load(resolved, **load_kwargs)
            except TypeError:
                load_kwargs.pop('weights_only', None)
                return torch.load(resolved, **load_kwargs)

        try:
            checkpoint = _torch_load()
        except pickle.UnpicklingError as exc:
            checkpoint = None
            message = str(exc)
            safe_globals = getattr(torch.serialization, 'add_safe_globals', None)
            if safe_globals and 'Namespace' in message:
                try:
                    safe_globals([argparse.Namespace])
                    checkpoint = _torch_load()
                except pickle.UnpicklingError:
                    checkpoint = None
            if checkpoint is None:
                warnings.warn(
                    'Falling back to torch.load(weights_only=False); ensure the SpectralGPT checkpoint is trusted.',
                    stacklevel=2,
                )
                checkpoint = _torch_load(weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ('model', 'state_dict', 'module'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        if isinstance(checkpoint, dict):
            checkpoint = {k.replace('module.', '', 1): v for k, v in checkpoint.items()}

        model_state = self.model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in checkpoint.items():
            target = model_state.get(key)
            if target is None:
                skipped.append((key, 'missing in model'))
                continue
            if target.shape != value.shape:
                skipped.append((key, f'shape mismatch {tuple(value.shape)} -> {tuple(target.shape)}'))
                continue
            filtered_state[key] = value

        loaded_keys = sorted(filtered_state.keys())
        print(f'[SpectralGPT] Loading pretrained weights from {resolved}')
        print(f'[SpectralGPT] Matched parameters ({len(loaded_keys)}):')
        for name in loaded_keys:
            tensor = filtered_state[name]
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else 'non-tensor'
            print(f'  - {name}: {shape}')

        incompatible = self.model.load_state_dict(filtered_state, strict=False)
        missing = getattr(incompatible, 'missing_keys', ())
        unexpected = getattr(incompatible, 'unexpected_keys', ())
        if missing or unexpected:
            warnings.warn(
                f'Loaded SpectralGPT backbone with missing keys: {missing} and unexpected keys: {unexpected}.',
                stacklevel=2,
            )
        if skipped:
            print('[SpectralGPT] Skipped incompatible weights:')
            for name, reason in skipped:
                print(f'  - {name}: {reason}')

    def _resize_backbone(self, frames: int, height: int, width: int) -> None:
        patch = self.model.patch_embed

        if frames % patch.t_patch_size != 0:
            raise ValueError(
                f'Input frames ({frames}) must be divisible by temporal patch size ({patch.t_patch_size}).'
            )
        if height % patch.patch_size[0] != 0 or width % patch.patch_size[1] != 0:
            raise ValueError(
                f'Input spatial size ({height}, {width}) must be divisible by patch size {patch.patch_size}.'
            )

        time_bins = frames // patch.t_patch_size
        spatial_h = height // patch.patch_size[0]
        spatial_w = width // patch.patch_size[1]

        patch.img_size = (height, width)
        patch.frames = frames
        patch.input_size = (time_bins, spatial_h, spatial_w)
        patch.num_patches = time_bins * spatial_h * spatial_w
        patch.grid_size = spatial_h
        patch.t_grid_size = time_bins

        for block in self.model.blocks:
            attn = getattr(block, 'attn', None)
            if attn is not None and hasattr(attn, 'input_size'):
                attn.input_size = (time_bins, spatial_h, spatial_w)

        self.num_frames = frames
        self.time_bins = time_bins
        self.spatial_height = spatial_h
        self.spatial_width = spatial_w
        self.spatial_bins = spatial_h * spatial_w

    def _interpolate_positional_embeddings(
        self,
        time_bins: int,
        spatial_h: int,
        spatial_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if getattr(self.model, 'sep_pos_embed', False):
            pos_spatial = self.model.pos_embed_spatial
            base_h = self._base_spatial_height
            base_w = self._base_spatial_width

            spatial = pos_spatial.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)
            spatial = torch.nn.functional.interpolate(
                spatial.to(device=device, dtype=dtype),
                size=(spatial_h, spatial_w),
                mode='bicubic',
                align_corners=False,
            )
            spatial = spatial.permute(0, 2, 3, 1).reshape(1, spatial_h * spatial_w, -1)

            temporal = self.model.pos_embed_temporal.to(device=device, dtype=dtype)
            base_time = self._base_time_bins
            if time_bins != base_time:
                temporal = temporal.transpose(1, 2).unsqueeze(-1)
                temporal = torch.nn.functional.interpolate(
                    temporal,
                    size=time_bins,
                    mode='linear',
                    align_corners=False,
                )
                temporal = temporal.squeeze(-1).transpose(1, 2)

            temporal = temporal.repeat_interleave(spatial_h * spatial_w, dim=1)
            pos_embed = spatial.repeat(1, time_bins, 1) + temporal

            if getattr(self.model, 'cls_embed', False):
                pos_class = self.model.pos_embed_class.to(device=device, dtype=dtype)
                pos_embed = torch.cat([pos_class.expand(pos_embed.shape[0], -1, -1), pos_embed], dim=1)
            return pos_embed

        pos_embed = self.model.pos_embed
        base_time = self._base_time_bins
        base_h = self._base_spatial_height
        base_w = self._base_spatial_width

        cls_token = None
        tokens = pos_embed
        if getattr(self.model, 'cls_embed', False):
            cls_token = tokens[:, :1, :].to(device=device, dtype=dtype)
            tokens = tokens[:, 1:, :]

        embed_dim = tokens.shape[-1]
        tokens = tokens.reshape(1, base_time, base_h, base_w, embed_dim)

        # Spatial interpolation (per time-bin)
        tokens = tokens.permute(0, 1, 4, 2, 3).reshape(base_time, embed_dim, base_h, base_w)
        tokens = torch.nn.functional.interpolate(
            tokens.to(device=device, dtype=dtype),
            size=(spatial_h, spatial_w),
            mode='bicubic',
            align_corners=False,
        )
        # tokens: (base_time, embed_dim, spatial_h, spatial_w)

        # Temporal interpolation (only when input requires MORE time bins than
        # the pretrained backbone provides; when fewer are needed, the caller
        # slices via pos_embed[:, :N, :] which preserves existing behaviour
        # and keeps backward-compatibility with already-trained checkpoints).
        if time_bins > base_time:
            tokens = tokens.reshape(1, base_time, embed_dim, spatial_h * spatial_w)
            tokens = tokens.permute(0, 2, 3, 1)  # (1, embed_dim, spatial_bins, base_time)
            tokens = tokens.reshape(embed_dim * spatial_h * spatial_w, 1, base_time)
            tokens = torch.nn.functional.interpolate(
                tokens,
                size=time_bins,
                mode='linear',
                align_corners=False,
            )
            tokens = tokens.reshape(1, embed_dim, spatial_h * spatial_w, time_bins)
            tokens = tokens.permute(0, 3, 2, 1)  # (1, time_bins, spatial_bins, embed_dim)
            tokens = tokens.reshape(1, time_bins * spatial_h * spatial_w, embed_dim)
        else:
            tokens = tokens.reshape(base_time, embed_dim, spatial_h * spatial_w).permute(0, 2, 1)
            tokens = tokens.reshape(1, base_time * spatial_h * spatial_w, embed_dim)

        if cls_token is not None:
            tokens = torch.cat([cls_token, tokens], dim=1)
        return tokens

    def _reduce_temporal(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, channels = tokens.shape
        if num_tokens != self.time_bins * self.spatial_bins:
            raise RuntimeError('Token count does not match expected temporal-spatial configuration.')
        tokens = tokens.view(batch, self.time_bins, self.spatial_bins, channels)
        tokens = tokens.mean(dim=1)
        return tokens

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, T, H, W).')
        batch, frames, height, width = x.shape

        self._resize_backbone(frames, height, width)

        tokens = self.model.patch_embed(x.unsqueeze(1))
        batch_size, time_bins, spatial_bins, embed_dim = tokens.shape
        tokens = tokens.view(batch_size, time_bins * spatial_bins, embed_dim)

        pos_embed = self._interpolate_positional_embeddings(
            self.time_bins,
            self.spatial_height,
            self.spatial_width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + pos_embed[:, : tokens.shape[1], :]

        features: List[torch.Tensor] = []

        for idx, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if idx in self.hook_indices:
                normalized = self.model.norm(tokens)
                reduced = self._reduce_temporal(normalized)
                features.append(reduced)

        if not features:
            normalized = self.model.norm(tokens)
            reduced = self._reduce_temporal(normalized)
            features.append(reduced)

        if len(features) < self.num_features:
            normalized = self.model.norm(tokens)
            reduced = self._reduce_temporal(normalized)
            while len(features) < self.num_features:
                features.append(reduced)

        patch_shape = (self.spatial_height, self.spatial_width)
        return features[: self.num_features], patch_shape


class SpectralGPT(nn.Module):
    def __init__(
        self,
        num_frames: int = 12,
        num_classes: int = 4,
        img_size: int = 128,
        decoder_channels: int = 256,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        pretrained_backbone_path: Optional[str] = None,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()

        if pretrained_backbone_path is None:
            project_root = Path(__file__).resolve().parents[3]
            pretrained_backbone_path = str(project_root / 'pretrained_weight' / 'SpectralGPT+.pth')

        self.encoder = SpectralGPTFeatureExtractor(
            num_frames=num_frames,
            img_size=img_size,
            hook_indices=hook_indices,
            num_features=num_features,
            pretrained=pretrained_backbone,
            pretrained_path=pretrained_backbone_path,
            freeze_backbone=freeze_backbone,
        )

        self.decoder_channels = int(decoder_channels)
        self.classifier = nn.Conv2d(self.decoder_channels, num_classes, kernel_size=1)

        self.proj_4 = self._make_projection(self.encoder.embed_dim, self.decoder_channels)
        self.proj_8 = self._make_projection(self.encoder.embed_dim, self.decoder_channels)
        self.proj_16 = self._make_projection(self.encoder.embed_dim, self.decoder_channels)
        self.proj_32 = self._make_projection(self.encoder.embed_dim, self.decoder_channels)

        self.decoder = UPerNetDecoder(
            in_channels=[self.decoder_channels] * 4,
            fpn_channels=self.decoder_channels,
            pool_scales=pool_scales,
        )

    @staticmethod
    def _make_projection(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, T, H, W).')

        batch, frames, height, width = x.shape

        # Pad temporal dimension to align with t_patch_size (matching flood legacy)
        patch_embed = getattr(self.encoder.model, 'patch_embed', None)
        if patch_embed is not None:
            t_patch = getattr(patch_embed, 't_patch_size', 1)
            if t_patch > 0 and frames % t_patch != 0:
                import warnings
                pad = t_patch - (frames % t_patch)
                pad_frame = x[:, -1:].expand(-1, pad, -1, -1)
                x = torch.cat([x, pad_frame], dim=1)
                frames = x.shape[1]
                warnings.warn(
                    f'Padded temporal dimension with {pad} frame(s) '
                    f'to align with t_patch_size={t_patch}.',
                    stacklevel=2,
                )

        self.encoder._resize_backbone(frames, height, width)

        tokens = self.encoder.model.patch_embed(x.unsqueeze(1))
        batch_size, time_bins, spatial_bins, embed_dim = tokens.shape
        tokens = tokens.view(batch_size, time_bins * spatial_bins, embed_dim)

        pos_embed = self.encoder._interpolate_positional_embeddings(
            self.encoder.time_bins,
            self.encoder.spatial_height,
            self.encoder.spatial_width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + pos_embed[:, : tokens.shape[1], :]

        for block in self.encoder.model.blocks:
            tokens = block(tokens)

        normalized = self.encoder.model.norm(tokens)
        reduced = self.encoder._reduce_temporal(normalized)
        base_feat = reduced.transpose(1, 2).reshape(
            batch_size,
            embed_dim,
            self.encoder.spatial_height,
            self.encoder.spatial_width,
        )
        return base_feat, (height, width)

    def _build_pyramid(self, base_feat: torch.Tensor, height: int, width: int) -> List[torch.Tensor]:
        def _resize(feature: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
            if feature.shape[-2:] == target:
                return feature
            return F.interpolate(feature, size=target, mode='bilinear', align_corners=False)

        size_4 = (max(1, height // 4), max(1, width // 4))
        size_8 = (max(1, height // 8), max(1, width // 8))
        size_16 = (max(1, height // 16), max(1, width // 16))
        size_32 = (max(1, height // 32), max(1, width // 32))

        feat_4 = self.proj_4(_resize(base_feat, size_4))
        feat_8 = self.proj_8(_resize(base_feat, size_8))
        feat_16 = self.proj_16(_resize(base_feat, size_16))
        feat_32 = self.proj_32(_resize(base_feat, size_32))

        return [feat_4, feat_8, feat_16, feat_32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_feat, input_size = self.forward_features(x)
        height, width = input_size

        pyramid = self._build_pyramid(base_feat, height, width)
        fused = self.decoder(pyramid)
        logits = self.classifier(fused)
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        return logits
