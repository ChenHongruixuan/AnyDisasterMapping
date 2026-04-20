
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel
except ImportError as exc:  # pragma: no cover - dependencies handled at runtime
    raise ImportError('CLIPDPT requires the transformers package. Install it via `pip install transformers`.') from exc

from src.models.DinoV2DPT import DPTDecoder, DPTHead, _default_stage_channels


def _auto_select_hook_indices(depth: int, num_features: int) -> Tuple[int, ...]:
    if num_features <= 1:
        return (depth - 1,)
    indices: List[int] = []
    step = (depth - 1) / float(num_features - 1)
    for idx in range(num_features):
        candidate = int(round(idx * step))
        candidate = max(0, min(depth - 1, candidate))
        if indices and candidate <= indices[-1]:
            candidate = min(depth - 1, indices[-1] + 1)
        indices.append(candidate)
    while len(indices) < num_features and len(indices) < depth:
        for candidate in range(depth - 1, -1, -1):
            if candidate not in indices:
                indices.append(candidate)
                break
    indices = sorted(indices[-num_features:])
    if indices[-1] != depth - 1:
        indices[-1] = depth - 1
    return tuple(indices)


class CLIPFeatureExtractor(nn.Module):
    """Feature extractor built on top of a pretrained CLIP vision transformer."""

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)

        print(self.model)
        self.model.config.output_hidden_states = True
        self.embed_dim = int(self.model.config.hidden_size)

        patch_embedding = self.model.vision_model.embeddings.patch_embedding
        if isinstance(patch_embedding, nn.Conv2d):
            projection = patch_embedding
            self._patch_embed_setter = lambda new_proj: setattr(
                self.model.vision_model.embeddings, 'patch_embedding', new_proj
            )
        else:
            projection = getattr(patch_embedding, 'projection', None)
            if not isinstance(projection, nn.Conv2d):
                raise ValueError('Unexpected patch embedding projection type; expected Conv2d.')
            self._patch_embed_setter = lambda new_proj: setattr(patch_embedding, 'projection', new_proj)
        self.patch_size = int(projection.kernel_size[0])

        if projection.in_channels != in_channels:
            self._adapt_input_channels(in_channels)

        depth = int(self.model.config.num_hidden_layers)
        if depth <= 0:
            raise ValueError('CLIP backbone exposes no transformer blocks.')

        if hook_indices is None:
            hook_indices = _auto_select_hook_indices(depth, num_features)
        else:
            filtered = [int(idx) for idx in hook_indices if 0 <= int(idx) < depth]
            if not filtered:
                raise ValueError('Provided hook_indices are invalid for the selected CLIP backbone.')
            hook_indices = tuple(sorted(filtered))

        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _adapt_input_channels(self, in_channels: int) -> None:
        patch_embedding = self.model.vision_model.embeddings.patch_embedding
        if isinstance(patch_embedding, nn.Conv2d):
            projection = patch_embedding
        else:
            projection = patch_embedding.projection

        new_proj = nn.Conv2d(
            in_channels,
            projection.out_channels,
            kernel_size=projection.kernel_size,
            stride=projection.stride,
            padding=projection.padding,
            bias=projection.bias is not None,
        )
        with torch.no_grad():
            weight = projection.weight
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
            if projection.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(projection.bias)
        self._patch_embed_setter(new_proj)

    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        batch, _, height, width = x.shape

        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError('CLIP backbone did not return hidden states.')

        sample_tokens = hidden_states[self.hook_indices[0] + 1]
        if sample_tokens.shape[1] < 2:
            raise RuntimeError('Expected sequence with class token and patches.')
        patch_token_count = sample_tokens.shape[1] - 1

        patch_h = max(1, int(round(height / float(self.patch_size))))
        patch_w = max(1, patch_token_count // patch_h)
        if patch_h * patch_w != patch_token_count:
            patch_w = max(1, int(round(width / float(self.patch_size))))
            patch_h = max(1, patch_token_count // patch_w)
        if patch_h * patch_w != patch_token_count:
            patch_h = max(1, int(round(math.sqrt(patch_token_count))))
            patch_w = max(1, patch_token_count // patch_h)
        if patch_h * patch_w != patch_token_count:
            raise RuntimeError('Unable to infer patch grid from token sequence length.')

        features: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx in self.hook_indices:
            tokens = hidden_states[idx + 1]
            if tokens.shape[1] < 2:
                raise RuntimeError('Expected sequence with class token and patches.')
            cls_token = tokens[:, 0, :]
            patch_tokens = tokens[:, 1:, :]
            if patch_tokens.shape[0] != batch:
                raise RuntimeError('Batch size mismatch in CLIP feature extraction.')
            features.append((patch_tokens, cls_token))

        return features, (patch_h, patch_w)


class CLIPDPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        clip_model: str = 'openai/clip-vit-base-patch32',
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = CLIPFeatureExtractor(
            model_name=clip_model,
            in_channels=in_channels,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_backbone=freeze_backbone,
        )
        if self.encoder.num_features != 4:
            raise ValueError('CLIPDPT expects exactly four encoder features to align with the DPT decoder.')

        embed_dim = self.encoder.embed_dim
        stage_channels = decoder_stage_channels or _default_stage_channels(embed_dim)

        self.decoder_channels = int(decoder_channels) if decoder_channels is not None else stage_channels[0]
        self.head_channels = int(head_channels) if head_channels is not None else self.decoder_channels

        self.decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=self.decoder_channels,
            out_channels=stage_channels,
            use_bn=use_bn,
            use_cls_token=True,
        )
        self.head = DPTHead(self.decoder_channels, self.head_channels, use_bn=use_bn)
        self.classifier = nn.Conv2d(self.head_channels, num_classes, kernel_size=1)
        self.hook_indices = self.encoder.hook_indices
        self.num_features = len(self.hook_indices)
        self.output_strides = (32, 16, 8, 4)

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


__all__ = ['CLIPDPT']
