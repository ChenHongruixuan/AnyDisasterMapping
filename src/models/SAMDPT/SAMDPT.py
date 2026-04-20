import math
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.SAMDPT.image_encoder import ImageEncoderViT


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
        out2 = out_shape * 2
        out3 = out_shape * 4
        out4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, use_bn: bool, activation: nn.Module) -> None:
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
        use_bn: bool,
        activation: nn.Module,
        align_corners: bool = True,
        expand: bool = False,
        default_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.default_size = default_size

        self.residual_1 = ResidualConvUnit(features, use_bn, activation)
        self.residual_2 = ResidualConvUnit(features, use_bn, activation)
        self.expand = expand

        out_channels = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        output = x
        if skip is not None:
            if skip.shape[-2:] != output.shape[-2:]:
                skip = F.interpolate(skip, size=output.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            output = self.skip_add.add(output, self.residual_1(skip))

        output = self.residual_2(output)

        if size is not None:
            target = {"size": size}
        elif self.default_size is not None:
            target = {"size": self.default_size}
        else:
            target = {"scale_factor": 2.0}

        output = F.interpolate(output, mode='bilinear', align_corners=self.align_corners, **target)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features: int, use_bn: bool, size: Optional[Tuple[int, int]] = None) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features=features,
        use_bn=use_bn,
        activation=nn.ReLU(inplace=False),
        align_corners=True,
        expand=False,
        default_size=size,
    )


class DPTDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        features: int,
        out_channels: Sequence[int],
        use_bn: bool = True,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError('DPTDecoder expects four encoder stages.')

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
        batch, num_tokens, channels = tokens.shape
        if patch_h * patch_w != num_tokens:
            raise ValueError('Token sequence length does not match provided patch spatial dimensions.')
        return tokens.permute(0, 2, 1).reshape(batch, channels, patch_h, patch_w)

    def forward(
        self,
        features: Sequence[Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]],
        patch_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f'SAMDPTDecoder expects four feature tensors, received {len(features)}.')

        patch_h, patch_w = patch_shape
        processed: List[torch.Tensor] = []

        for idx, feat in enumerate(features):
            if isinstance(feat, (tuple, list)):
                patch_tokens = feat[0]
                cls_token = feat[1] if len(feat) > 1 else None
            else:
                patch_tokens = feat
                cls_token = None

            if patch_tokens.dim() != 3:
                raise ValueError('Patch tokens are expected in (B, N, C) format.')

            if self.use_cls_token and cls_token is not None:
                readout = cls_token.unsqueeze(1).expand_as(patch_tokens)
                patch_tokens = torch.cat([patch_tokens, readout], dim=-1)
                patch_tokens = self.readout_projects[idx](patch_tokens)
            else:
                patch_tokens = self.readout_projects[idx](patch_tokens)

            patch_map = self._reshape_tokens(patch_tokens, patch_h, patch_w)
            proj = self.projects[idx](patch_map)
            resized = self.resize_layers[idx](proj)
            processed.append(resized)

        layer_1 = self.scratch.layer1_rn(processed[0])
        layer_2 = self.scratch.layer2_rn(processed[1])
        layer_3 = self.scratch.layer3_rn(processed[2])
        layer_4 = self.scratch.layer4_rn(processed[3])

        path_4 = self.scratch.refinenet4(layer_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1)
        return path_1


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


class SAMFeatureExtractor(nn.Module):
    _SAM_PRESETS = {
        'vit_b': dict(embed_dim=768, depth=12, num_heads=12, global_attn_indexes=(2, 5, 8, 11)),
        'vit_l': dict(embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=(5, 11, 17, 23)),
        'vit_h': dict(embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=(7, 15, 23, 31)),
    }

    def __init__(
        self,
        model_type: str = 'vit_h',
        checkpoint: Optional[str] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_encoder: bool = False,
        normalize_input: bool = True,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        if model_type not in self._SAM_PRESETS:
            available = ', '.join(sorted(self._SAM_PRESETS))
            raise ValueError(f'Unsupported SAM backbone "{model_type}". Available options: {available}.')
        cfg = self._SAM_PRESETS[model_type]

        self.image_size = 1024
        self.patch_size = 16
        self.patch_stride = (self.patch_size, self.patch_size)
        self.embed_dim = cfg['embed_dim']
        self.in_chans = int(in_chans)

        self.image_encoder = ImageEncoderViT(
            img_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=cfg['embed_dim'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            mlp_ratio=4.0,
            out_chans=256,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=tuple(cfg['global_attn_indexes']),
        )

        if checkpoint:
            self._load_checkpoint(checkpoint)
       
        depth = len(self.image_encoder.blocks)
        if depth == 0:
            raise ValueError('SAM image encoder exposes no transformer blocks.')
        if num_features < 1:
            raise ValueError('num_features must be at least 1.')

        if hook_indices is None:
            hook_indices = self._auto_select_hook_indices(depth, num_features)
        else:
            filtered = [int(idx) for idx in hook_indices if 0 <= int(idx) < depth]
            if not filtered:
                raise ValueError('Provided hook_indices are invalid for the selected SAM encoder.')
            hook_indices = tuple(sorted(set(filtered)))
        if (depth - 1) not in hook_indices:
            hook_indices = tuple(sorted((*hook_indices, depth - 1)))
        self.hook_indices = hook_indices
        self.num_features = len(self.hook_indices)

        if freeze_encoder:
            self.image_encoder.eval()
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _auto_select_hook_indices(depth: int, num_features: int) -> Tuple[int, ...]:
        if num_features <= 1:
            return (depth - 1,)

        indices: List[int] = []
        for idx in range(num_features):
            candidate = math.floor(((idx + 1) * depth) / float(num_features)) - 1
            candidate = max(0, min(depth - 1, candidate))
            if indices and candidate <= indices[-1]:
                candidate = min(depth - 1, indices[-1] + 1)
            indices.append(candidate)

        if indices[-1] != depth - 1:
            indices[-1] = depth - 1
        return tuple(indices)

    def _load_checkpoint(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location='cpu')
        if isinstance(state, dict):
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
        if not isinstance(state, dict):
            raise ValueError('Unsupported SAM checkpoint format.')

        image_state = {k.replace('image_encoder.', '', 1): v for k, v in state.items() if k.startswith('image_encoder.')}
        if not image_state:
            raise ValueError('SAM checkpoint does not contain image encoder weights.')

        patch_key = 'patch_embed.proj.weight'
        if patch_key in image_state:
            image_state[patch_key] = self._adapt_patch_embed_weights(image_state[patch_key])
        missing = self.image_encoder.load_state_dict(image_state, strict=False)
        if missing.missing_keys:
            raise ValueError(f'Missing keys when loading SAM checkpoint: {missing.missing_keys}')

    def _adapt_patch_embed_weights(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 4:
            return weight
        out_channels, in_channels, k_h, k_w = weight.shape
        if in_channels == self.in_chans:
            return weight
        if self.in_chans < in_channels:
            return weight[:, :self.in_chans]
        repeats = self.in_chans // in_channels
        remainder = self.in_chans % in_channels
        new_weight = weight.repeat(1, repeats, 1, 1)
        if remainder:
            new_weight = torch.cat([new_weight, weight[:, :remainder]], dim=1)
        new_weight = new_weight * (in_channels / float(self.in_chans))
        return new_weight



    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        if x.dim() != 4:
            raise ValueError('Expected input tensor with shape (B, C, H, W).')
        b, _, h, w = x.shape
        stride_h, stride_w = self.patch_stride
        grid_h = max(1, math.ceil(h / stride_h))
        grid_w = max(1, math.ceil(w / stride_w))

        tokens = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            pos = self.image_encoder.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
            if pos.shape[1] != tokens.shape[1] or pos.shape[2] != tokens.shape[2]:
                pos = F.interpolate(
                    pos.permute(0, 3, 1, 2),
                    size=tokens.shape[1:3],
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            tokens = tokens + pos

        tokens_2d = tokens[:, :grid_h, :grid_w, :]

        features: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        hook_iter = iter(self.hook_indices)
        next_hook = next(hook_iter, None)
        for idx, block in enumerate(self.image_encoder.blocks):
            tokens_2d = block(tokens_2d)
            while next_hook is not None and idx == next_hook:
                cropped = tokens_2d[:, :grid_h, :grid_w, :]
                flat = cropped.reshape(b, -1, cropped.shape[-1]).contiguous()
                features.append((flat, None))
                next_hook = next(hook_iter, None)

        if not features:
            cropped = tokens_2d[:, :grid_h, :grid_w, :]
            flat = cropped.reshape(b, -1, cropped.shape[-1]).contiguous()
            features.append((flat, None))

        while len(features) < self.num_features:
            last_tokens, _ = features[-1]
            features.append((last_tokens.clone(), None))
        if len(features) > self.num_features:
            features = features[-self.num_features:]

        if len(features) != self.num_features:
            raise RuntimeError('Unexpected number of features collected from SAM encoder.')
        for feat, _ in features:
            if feat.shape[0] != b:
                raise RuntimeError('Inconsistent batch size in SAM feature extraction.')
        return features, (grid_h, grid_w)


class SAMDPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        sam_type: str = 'vit_h',
        sam_checkpoint: Optional[str] = None,
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        output_strides: Optional[Sequence[int]] = None,
        freeze_encoder: bool = False,
        normalize_input: bool = True,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if output_strides is not None:
            raise ValueError('SAMDPT no longer accepts custom output_strides; DPT decoder uses fixed strides (32,16,8,4).')

        self.encoder = SAMFeatureExtractor(
            model_type=sam_type,
            checkpoint=sam_checkpoint,
            hook_indices=hook_indices,
            num_features=num_features,
            freeze_encoder=freeze_encoder,
            normalize_input=normalize_input,
            in_chans=in_channels,
        )
        if self.encoder.num_features != 4:
            raise ValueError('SAMDPT expects exactly four encoder features to match the DPT decoder.')

        embed_dim = self.encoder.embed_dim
        stage_channels = _default_stage_channels(embed_dim)

        self.decoder_channels = int(decoder_channels) if decoder_channels is not None else stage_channels[0]
        self.head_channels = int(head_channels) if head_channels is not None else self.decoder_channels
        self.hook_indices = tuple(self.encoder.hook_indices)
        self.output_strides = (32, 16, 8, 4)

        self.decoder = DPTDecoder(
            embed_dim=embed_dim,
            features=self.decoder_channels,
            out_channels=stage_channels,
            use_bn=use_bn,
            use_cls_token=False,
        )
        self.head = DPTHead(self.decoder_channels, self.head_channels, use_bn=use_bn)
        self.classifier = nn.Conv2d(self.head_channels, num_classes, kernel_size=1)

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


__all__ = ['SAMDPT']
