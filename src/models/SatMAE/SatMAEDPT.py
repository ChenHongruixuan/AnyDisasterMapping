import argparse
import math
import pickle
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import resample_abs_pos_embed

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels
from src.models.SatMAE.models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14


_MODEL_BUILDERS = {
    'satmae-vit-base-patch16': (vit_base_patch16, dict(img_size=224, patch_size=16)),
    'satmae-vit-large-patch16': (vit_large_patch16, dict(img_size=224, patch_size=16)),
    'satmae-vit-huge-patch14': (vit_huge_patch14, dict(img_size=224, patch_size=14)),
}

_DEFAULT_CHECKPOINTS = {
    'satmae-vit-base-patch16': 'pretrained_weight/pretrain-vit-base-e199.pth',
    'satmae-vit-large-patch16': 'pretrained_weight/pretrain-vit-large-e199.pth',
    'satmae-vit-huge-patch14': 'pretrained_weight/pretrain-vit-huge-e199.pth',
}


def _select_builder(model_name: str):
    key = model_name.lower()
    if key not in _MODEL_BUILDERS:
        raise ValueError(f'Unsupported SatMAE backbone "{model_name}".')
    return _MODEL_BUILDERS[key], key


class SatMAEFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = 'satmae-vit-large-patch16',
        checkpoint_path: Optional[str] = None,
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        (builder, base_kwargs), resolved_name = _select_builder(model_name)
        builder_kwargs = dict(base_kwargs)
        self.model = builder(**builder_kwargs)

        self.embed_dim = getattr(self.model, 'embed_dim', None)
        if self.embed_dim is None:
            raise ValueError('Failed to infer embedding dimension for SatMAE backbone.')

        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is None:
            raise ValueError('SatMAE backbone does not expose a patch embedding module.')

        patch_size = getattr(patch_embed, 'patch_size', 16)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)

        if in_channels != getattr(patch_embed.proj, 'in_channels', in_channels):
            self._adapt_input_channels(in_channels)

        depth = len(getattr(self.model, 'blocks', []))
        if depth == 0:
            raise ValueError('SatMAE backbone exposes no transformer blocks.')

        if hook_indices is None:
            if num_features <= 0:
                raise ValueError('num_features must be positive when hook_indices is not provided.')
            step = depth / float(num_features)
            derived: List[int] = []
            for i in range(1, num_features + 1):
                idx = int(round(i * step) - 1)
                idx = max(0, min(depth - 1, idx))
                derived.append(idx)
            hook_indices = tuple(sorted(set(derived)))
        else:
            filtered = [int(idx) for idx in hook_indices if 0 <= int(idx) < depth]
            if not filtered:
                raise ValueError('Provided hook_indices are invalid for the selected backbone.')
            hook_indices = tuple(sorted(set(filtered)))

        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        self._num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 1 if self.model.cls_token is not None else 0)
        base_grid = getattr(self.model.patch_embed, 'grid_size', None)
        if base_grid is None:
            tokens_without_prefix = self.model.pos_embed.shape[1] - self._num_prefix_tokens
            side = max(1, int(round(math.sqrt(tokens_without_prefix))))
            other = max(1, tokens_without_prefix // side)
            base_grid = (side, other)
        elif isinstance(base_grid, int):
            base_grid = (base_grid, base_grid)
        else:
            base_grid = tuple(int(g) for g in base_grid)
        self._pos_grid_size = base_grid

        checkpoint = checkpoint_path
        if checkpoint is None:
            default_path = _DEFAULT_CHECKPOINTS.get(resolved_name)
            if default_path is not None:
                checkpoint = default_path

        if checkpoint is not None:
            if os.path.isfile(checkpoint):
                self._load_checkpoint(checkpoint)
            elif checkpoint_path is None:
                print(f'[SatMAE] Default checkpoint not found at {checkpoint}; using random initialisation.')
            else:
                raise FileNotFoundError(f'Specified SatMAE checkpoint not found: {checkpoint}')

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _adapt_input_channels(self, in_channels: int) -> None:
        proj = getattr(self.model.patch_embed, 'proj', None)
        if proj is None or not isinstance(proj, nn.Conv2d):
            raise ValueError('Patch embedding projection is not a convolution; cannot adapt input channels.')
        if proj.in_channels == in_channels:
            return

        new_proj = nn.Conv2d(
            in_channels,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=proj.bias is not None,
        )

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

        self.model.patch_embed.proj = new_proj

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'SatMAE checkpoint not found: {checkpoint_path}')

        def _torch_load(weights_only: Optional[bool] = True):
            load_kwargs = dict(map_location='cpu')
            if weights_only is not None:
                load_kwargs['weights_only'] = weights_only
            try:
                return torch.load(checkpoint_path, **load_kwargs)
            except TypeError:
                load_kwargs.pop('weights_only', None)
                return torch.load(checkpoint_path, **load_kwargs)

        try:
            state = _torch_load()
        except pickle.UnpicklingError as exc:
            state = None
            message = str(exc)
            safe_globals = getattr(torch.serialization, 'add_safe_globals', None)
            if safe_globals and 'Namespace' in message:
                try:
                    safe_globals([argparse.Namespace])
                    state = _torch_load()
                except pickle.UnpicklingError:
                    state = None
            if state is None:
                print(
                    '[SatMAE] Warning: falling back to torch.load(weights_only=False); '
                    'ensure the checkpoint source is trusted.'
                )
                state = _torch_load(weights_only=False)

        if isinstance(state, dict):
            if 'model' in state:
                state = state['model']
            elif 'state_dict' in state:
                state = state['state_dict']

        model_state = self.model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in state.items():
            target = model_state.get(key)
            if target is None:
                skipped.append((key, 'missing in model'))
                continue
            if target.shape != value.shape:
                skipped.append((key, f'shape mismatch ({value.shape} -> {target.shape})'))
                continue
            filtered_state[key] = value

        load_result = self.model.load_state_dict(filtered_state, strict=False)
        missing = getattr(load_result, 'missing_keys', None)
        unexpected = getattr(load_result, 'unexpected_keys', None)

        if missing is None and unexpected is None and isinstance(load_result, tuple):
            if len(load_result) == 2:
                missing, unexpected = load_result
            else:
                missing = list(load_result)
                unexpected = []

        missing = list(missing or [])
        unexpected = list(unexpected or [])

        print(f'[SatMAE] Missing keys: {missing}')
        print(f'[SatMAE] Unexpected keys: {unexpected}')
        if skipped:
            preview = [f'{k} ({reason})' for k, reason in skipped]
            print('[SatMAE] Skipped incompatible weights:', preview[:10])

        tokens_without_prefix = self.model.pos_embed.shape[1] - self._num_prefix_tokens
        side = max(1, int(round(math.sqrt(tokens_without_prefix))))
        other = max(1, tokens_without_prefix // side)
        self._pos_grid_size = (side, other)

    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, ...]], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        _, _, height, width = x.shape

        patch_h = max(1, int(math.ceil(height / float(self.patch_size))))
        patch_w = max(1, int(math.ceil(width / float(self.patch_size))))
        desired_grid = (patch_h, patch_w)

        prefix = self._num_prefix_tokens

        patch_embed = self.model.patch_embed
        if hasattr(patch_embed, 'img_size'):
            patch_embed.img_size = (height, width)
        if hasattr(patch_embed, 'grid_size'):
            patch_embed.grid_size = desired_grid
        if hasattr(patch_embed, 'num_patches'):
            patch_embed.num_patches = patch_h * patch_w

        tokens = patch_embed(x)
        B, num_tokens, _ = tokens.shape

        if patch_h * patch_w != num_tokens:
            patch_h = max(1, int(round(math.sqrt(num_tokens))))
            patch_w = max(1, max(1, num_tokens // patch_h))
            desired_grid = (patch_h, patch_w)
            if hasattr(patch_embed, 'grid_size'):
                patch_embed.grid_size = desired_grid
            if hasattr(patch_embed, 'num_patches'):
                patch_embed.num_patches = num_tokens

        if self.training and desired_grid != self._pos_grid_size:
            resampled = resample_abs_pos_embed(
                self.model.pos_embed.detach(),
                new_size=desired_grid,
                old_size=self._pos_grid_size,
                num_prefix_tokens=prefix,
            ).to(self.model.pos_embed.device)
            self.model.pos_embed = nn.Parameter(resampled)
            self._pos_grid_size = desired_grid
            pos_embed = self.model.pos_embed
        elif desired_grid == self._pos_grid_size:
            pos_embed = self.model.pos_embed
        else:
            pos_embed = resample_abs_pos_embed(
                self.model.pos_embed.detach(),
                new_size=desired_grid,
                old_size=self._pos_grid_size,
                num_prefix_tokens=prefix,
            ).to(self.model.pos_embed.device)

        pos_embed = pos_embed.to(tokens.device, tokens.dtype)

        if prefix > 0:
            cls_tokens = self.model.cls_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        if tokens.shape[1] != pos_embed.shape[1]:
            raise RuntimeError(
                f'Positional embedding mismatch: tokens={tokens.shape[1]} vs pos_embed={pos_embed.shape[1]}'
            )

        tokens = tokens + pos_embed
        tokens = self.model.pos_drop(tokens)
        tokens = self.model.patch_drop(tokens)
        tokens = self.model.norm_pre(tokens)

        collected: List[Tuple[torch.Tensor, ...]] = []
        hook_set = set(self.hook_indices)
        for idx, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if idx in hook_set:
                normalized = self.model.norm(tokens)
                prefix = getattr(self.model, 'num_prefix_tokens', 1 if self.model.cls_token is not None else 0)
                if prefix > 0:
                    cls_slice = normalized[:, :prefix, :]
                    cls_token = cls_slice[:, 0, :]
                    patch_tokens = normalized[:, prefix:, :]
                else:
                    cls_token = None
                    patch_tokens = normalized
                if cls_token is None:
                    collected.append((patch_tokens,))
                else:
                    collected.append((patch_tokens, cls_token))

        if len(collected) != self.num_features:
            raise RuntimeError(f'Collected {len(collected)} feature maps, expected {self.num_features}.')

        sample_tokens = collected[0][0]
        num_tokens = sample_tokens.shape[1]
        patch_h = max(1, int(round(height / float(self.patch_size))))
        patch_w = max(1, num_tokens // patch_h)
        if patch_h * patch_w != num_tokens:
            patch_w = max(1, int(round(width / float(self.patch_size))))
            patch_h = max(1, num_tokens // patch_w)
        if patch_h * patch_w != num_tokens:
            patch_h = max(1, int(round(math.sqrt(num_tokens))))
            patch_w = max(1, num_tokens // patch_h)
        if patch_h * patch_w != num_tokens:
            raise RuntimeError('Failed to infer patch grid size from token count.')

        return collected, (patch_h, patch_w)


class SatMAEDPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        backbone: str = 'satmae-vit-large-patch16',
        checkpoint_path: Optional[str] = None,
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SatMAEFeatureExtractor(
            model_name=backbone,
            checkpoint_path=checkpoint_path,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
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
