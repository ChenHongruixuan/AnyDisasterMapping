import warnings
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.models.SwinUperNet import ConvBNReLU, UPerNetDecoder
from .swin_transformer_v2 import SwinTransformerV2


_SizeType = Union[int, Tuple[int, int]]


class SkySenseEncoder(nn.Module):
    """Swin Transformer V2 encoder that exposes intermediate stage features."""

    def __init__(
        self,
        *,
        img_size: _SizeType,
        patch_size: _SizeType,
        in_channels: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained_window_sizes: Sequence[int] = (0, 0, 0, 0),
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        img_size = self._to_2tuple(img_size)
        patch_size = self._to_2tuple(patch_size)

        backbone_kwargs: Dict[str, object] = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes,
        )
        self.backbone = SwinTransformerV2(**backbone_kwargs)
        self.backbone.head = nn.Identity()
        self.out_indices = tuple(sorted(set(out_indices)))
        self.out_channels = [int(embed_dim * 2 ** idx) for idx in self.out_indices]

    @staticmethod
    def _to_2tuple(value: _SizeType) -> Tuple[int, int]:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            seq = tuple(int(v) for v in value)
            if len(seq) != 2:
                raise ValueError(f'Expected tuple of length 2, got {value}')
            return seq
        return int(value), int(value)

    def _update_resolution_if_needed(self, x: torch.Tensor) -> None:
        patch_embed = self.backbone.patch_embed
        current_size = tuple(int(v) for v in patch_embed.img_size)
        spatial_size = (int(x.shape[2]), int(x.shape[3]))
        if spatial_size == current_size:
            return

        patch_size = tuple(int(v) for v in patch_embed.patch_size)
        if spatial_size[0] % patch_size[0] != 0 or spatial_size[1] % patch_size[1] != 0:
            raise ValueError(
                f'Input size {spatial_size} is not divisible by patch size {patch_size}.'
            )

        if self.backbone.ape:
            raise ValueError('Absolute positional embeddings are enabled; dynamic resizing is not supported.')

        patches_resolution = (
            spatial_size[0] // patch_size[0],
            spatial_size[1] // patch_size[1],
        )
        patch_embed.img_size = spatial_size
        patch_embed.patches_resolution = patches_resolution
        patch_embed.num_patches = patches_resolution[0] * patches_resolution[1]
        self.backbone.patches_resolution = patches_resolution

        for stage_idx, layer in enumerate(self.backbone.layers):
            stage_resolution = (
                patches_resolution[0] // (2 ** stage_idx),
                patches_resolution[1] // (2 ** stage_idx),
            )
            layer.input_resolution = stage_resolution
            if layer.downsample is not None:
                layer.downsample.input_resolution = stage_resolution
    @staticmethod
    def _tokens_to_feature(tokens: torch.Tensor, resolution: Tuple[int, int]) -> torch.Tensor:
        h, w = resolution
        b, n, c = tokens.shape
        if n != h * w:
            raise ValueError(f'Number of tokens ({n}) does not match resolution {resolution}.')
        return tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _forward_layer(layer: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in layer.blocks:
            if layer.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        stage_tokens = x
        if layer.downsample is not None:
            x = layer.downsample(x)
        return stage_tokens, x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._update_resolution_if_needed(x)

        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        collected: Dict[int, torch.Tensor] = {}
        for stage_idx, layer in enumerate(self.backbone.layers):
            stage_tokens, x = self._forward_layer(layer, x)
            if stage_idx in self.out_indices:
                collected[stage_idx] = self._tokens_to_feature(stage_tokens, layer.input_resolution)

        return [collected[idx] for idx in self.out_indices]


class SkySenseUPerNet(nn.Module):
    """SkySense segmentation model with Swin Transformer V2 encoder and UPerNet decoder."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 5,
        img_size: _SizeType = 512,
        decoder_channels: int = 256,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        pretrained_backbone: bool = True,
        pretrained_backbone_path: Optional[Union[str, Path]] = None,
        freeze_backbone: bool = False,
        encoder_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()

        defaults: Dict[str, object] = dict(
            img_size=img_size,
            patch_size=4,
            embed_dim=352,
            depths=(2, 2, 18, 2),
            num_heads=(8, 16, 32, 64),
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=(0, 0, 0, 0),
            out_indices=out_indices,
        )
        if encoder_kwargs:
            defaults.update(encoder_kwargs)

        self.encoder = SkySenseEncoder(in_channels=in_channels, **defaults)
        self.decoder = UPerNetDecoder(self.encoder.out_channels, fpn_channels=decoder_channels, pool_scales=pool_scales)
        self.classifier = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if pretrained_backbone:
            weights_path = self._resolve_pretrained_path(pretrained_backbone_path)
            if weights_path.is_file():
                self._load_pretrained_backbone(weights_path)
            else:
                warnings.warn(
                    f'Pretrained backbone requested but checkpoint not found at {weights_path}. '
                    'Training will proceed without loading weights.',
                    stacklevel=2,
                )

    @staticmethod
    def _resolve_pretrained_path(path: Optional[Union[str, Path]]) -> Path:
        if path is not None:
            return Path(path).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[3]
        return project_root / 'pretrained_weight' / 'skysense_model_backbone_hr.pth'

    def _load_pretrained_backbone(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

        if isinstance(checkpoint, dict):
            top_level_keys = list(checkpoint.keys())
            print(f'[SkySense] Loaded checkpoint keys from {checkpoint_path.name}: {top_level_keys}')
            for key in ('model', 'state_dict', 'backbone'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

        state_dict = dict(checkpoint)
        state_dict = self._remap_checkpoint_keys(state_dict)
        if state_dict:
            sample_keys = list(state_dict.keys())[:10]
            print(f'[SkySense] state_dict contains {len(state_dict)} entries. Sample keys: {sample_keys}')

        self._prepare_checkpoint_for_in_channels(state_dict)
        self._drop_incompatible_relative_pos(state_dict)

        incompatible = self.encoder.backbone.load_state_dict(state_dict, strict=False)
        missing = getattr(incompatible, 'missing_keys', ())
        unexpected = getattr(incompatible, 'unexpected_keys', ())
        if missing or unexpected:
            warnings.warn(
                f'Loaded backbone with missing keys: {missing} and unexpected keys: {unexpected}.',
                stacklevel=2,
            )

    def _prepare_checkpoint_for_in_channels(self, state_dict: Dict[str, torch.Tensor]) -> None:
        patch_embed = getattr(self.encoder.backbone, 'patch_embed', None)
        proj = getattr(patch_embed, 'proj', None)
        if proj is None or not isinstance(proj, nn.Conv2d):
            return

        target_in_channels = proj.in_channels
        if target_in_channels <= 0:
            return

        keys = [key for key in state_dict.keys() if key.endswith('patch_embed.proj.weight')]
        for key in keys:
            weight = state_dict.get(key)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 4:
                continue
            if weight.shape[1] == target_in_channels:
                continue
            state_dict[key] = self._expand_input_channels(weight, target_in_channels)

    @staticmethod
    def _expand_input_channels(weight: torch.Tensor, new_in_channels: int) -> torch.Tensor:
        current_in_channels = weight.shape[1]
        if new_in_channels == current_in_channels:
            return weight.clone()

        if new_in_channels < current_in_channels:
            return weight[:, :new_in_channels, :, :].clone()

        repeats = new_in_channels // current_in_channels
        remainder = new_in_channels % current_in_channels
        expanded = weight.repeat(1, repeats, 1, 1)
        if remainder:
            expanded = torch.cat([expanded, weight[:, :remainder, :, :]], dim=1)
        scale = current_in_channels / float(new_in_channels)
        return expanded * scale

    def _drop_incompatible_relative_pos(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Remove relative position params whose shapes mismatch current window size.

        When window_size differs between checkpoint and model (e.g. pretrained
        with window_size=8 but running with window_size=7), the relative_coords_table
        and relative_position_index buffers have different shapes. Drop them so they
        are re-initialized from the model's current window configuration.
        No-op when window sizes match.
        """
        model_state = self.encoder.backbone.state_dict()
        drop_keys = []
        for k, v in state_dict.items():
            if "relative_coords_table" in k or "relative_position_index" in k:
                target = model_state.get(k)
                if target is not None and target.shape != v.shape:
                    drop_keys.append(k)
        if drop_keys:
            for k in drop_keys:
                state_dict.pop(k, None)
            print(f"[SkySense] Dropped {len(drop_keys)} relative position params "
                  f"due to window_size mismatch; first keys: {drop_keys[:4]}")

    @staticmethod
    def _remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        remapped: Dict[str, torch.Tensor] = {}
        downsample_pattern = re.compile(r'^stages\.(\d+)\.downsample\.(.+)$')
        stage_prefix_pattern = re.compile(r'^stages\.(\d+)\.(.+)$')

        for key, value in state_dict.items():
            if key == 'mask_token':
                continue

            new_key = key
            downsample_match = downsample_pattern.match(new_key)
            if downsample_match:
                stage_idx = int(downsample_match.group(1))
                if stage_idx == 0:
                    continue
                new_key = f'stages.{stage_idx - 1}.downsample.{downsample_match.group(2)}'
            else:
                stage_match = stage_prefix_pattern.match(new_key)
                if stage_match:
                    stage_idx = int(stage_match.group(1))
                    remainder = stage_match.group(2)
                    new_key = f'layers.{stage_idx}.{remainder}'

            if new_key.startswith('patch_embed.projection.'):
                new_key = new_key.replace('patch_embed.projection.', 'patch_embed.proj.')

            if 'stages.' in new_key:
                new_key = new_key.replace('stages.', 'layers.')

            if 'attn.w_msa.' in new_key:
                new_key = new_key.replace('attn.w_msa.', 'attn.')

            if 'ffn.layers.0.0.' in new_key:
                new_key = new_key.replace('ffn.layers.0.0.', 'mlp.fc1.')

            if 'ffn.layers.1.' in new_key:
                new_key = new_key.replace('ffn.layers.1.', 'mlp.fc2.')

            if '.norm3.' in new_key:
                new_key = new_key.replace('.norm3.', '.norm.')

            if new_key.startswith('norm3.'):
                new_key = new_key.replace('norm3.', 'norm.', 1)

            remapped[new_key] = value

        return remapped

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[2:]
        features = self.forward_features(x)
        decoded = self.decoder(features)
        logits = self.classifier(decoded)
        return F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
