import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm  # noqa: F401  # Optional dependency, kept for backward compatibility
except ImportError:  # pragma: no cover - optional dependency
    timm = None


_BACKBONE_ALIASES = {
    'vit_small_patch14_dinov2.lvd142m': 'dinov2_vits14',
    'vit_base_patch14_dinov2.lvd142m': 'dinov2_vitb14',
    'vit_large_patch14_dinov2.lvd142m': 'dinov2_vitl14',
    'vit_giant_patch14_dinov2.lvd142m': 'dinov2_vitg14',
}


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
        activation=nn.ReLU(inplace=False),
        use_bn=use_bn,
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
        use_bn: bool = False,
        use_cls_token: bool = True,
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

    def _reshape_tokens(self, tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.permute(0, 2, 1).reshape(batch, channels, patch_h, patch_w)

    def forward(
        self,
        features: Sequence[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
        patch_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f'DPTDecoder expects four feature tensors, received {len(features)}.')

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


class DINOv2FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch14_dinov2.lvd142m',
        pretrained: bool = True,
        in_channels: int = 3,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        resolved = _BACKBONE_ALIASES.get(model_name, model_name)
        if not resolved.startswith('dinov2_'):
            raise ValueError(
                f'Unsupported backbone "{model_name}". Provide a DINOv2 identifier or a known alias.'
            )

        try:
            self.model = torch.hub.load('facebookresearch/dinov2', resolved, pretrained=pretrained)
        except Exception as exc:  # pragma: no cover - hub loading failures are environment specific
            raise RuntimeError(
                'Failed to load DINOv2 weights via torch.hub. Ensure torch>=1.13 and internet/connectivity '
                'or pre-download the weights with torch.hub.load_state_dict_from_url.'
            ) from exc

        if getattr(self.model, 'chunked_blocks', False):
            warnings.warn('DINOv2 backbone is chunked; LoRA or fine-grained modifications may need adaptation.')

        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is None:
            raise ValueError('Backbone does not expose a patch embedding module.')
        if hasattr(patch_embed, 'strict_img_size'):
            patch_embed.strict_img_size = False
        if hasattr(patch_embed, 'img_size'):
            patch_embed.img_size = None

        self.embed_dim = getattr(self.model, 'embed_dim', None)
        if self.embed_dim is None:
            raise ValueError('Unable to determine embedding dimension for backbone.')

        patch_size = getattr(patch_embed, 'patch_size', 14)
        if isinstance(patch_size, tuple):
            self.patch_size = int(patch_size[0])
        else:
            self.patch_size = int(patch_size)

        if in_channels != getattr(patch_embed.proj, 'in_channels', in_channels):
            self._adapt_input_channels(in_channels)

        depth = len(getattr(self.model, 'blocks', []))
        if depth == 0:
            raise ValueError('Backbone exposes no transformer blocks.')
        if hook_indices is None:
            if num_features <= 0:
                raise ValueError('num_features must be positive when hook_indices is not provided.')
            step = depth / float(num_features)
            indices = []
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

        sample = outputs[0][0] if isinstance(outputs[0], (tuple, list)) else outputs[0]
        num_tokens = sample.shape[1]

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

        return list(outputs), (patch_h, patch_w)


class DinoV2DPT(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        backbone: str = 'vit_base_patch14_dinov2.lvd142m',
        decoder_channels: Optional[int] = None,
        head_channels: Optional[int] = None,
        hook_indices: Optional[Sequence[int]] = None,
        num_features: int = 4,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        decoder_stage_channels: Optional[Sequence[int]] = None,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = DINOv2FeatureExtractor(
            model_name=backbone,
            pretrained=pretrained_backbone,
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
