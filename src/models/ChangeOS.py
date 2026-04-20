from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import get_model_weights
except ImportError:  # pragma: no cover - torchvision handles this at runtime
    torchvision = None
    get_model_weights = None


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: Optional[int] = None) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class FuseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SqueezeExcitation(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        out = self.se(out) + out
        return self.act(out)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class FuseMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x))


class SimpleFPN(nn.Module):
    def __init__(self, in_channels_list: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for in_channels in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, kernel_size=3)
            for _ in in_channels_list
        ])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral_convs):
            raise ValueError('Number of feature maps does not match FPN configuration.')
        results: List[torch.Tensor] = [None] * len(features)
        last_inner: Optional[torch.Tensor] = None
        for idx in reversed(range(len(features))):
            inner = self.lateral_convs[idx](features[idx])
            if last_inner is not None:
                inner = inner + F.interpolate(last_inner, size=inner.shape[-2:], mode='bilinear', align_corners=False)
            last_inner = inner
            results[idx] = self.output_convs[idx](inner)
        return results[0]


class DecoderRefiner(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        blocks = [ConvBNReLU(channels, channels) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ChangeOSDecoder(nn.Module):
    def __init__(self, in_channels_list: Sequence[int], out_channels: int, fusion_type: str = 'mlp') -> None:
        super().__init__()
        self.loc_fpn = SimpleFPN(in_channels_list, out_channels)
        self.loc_refine = DecoderRefiner(out_channels)

        self.dam_fpn = SimpleFPN(in_channels_list, out_channels)
        self.dam_refine = DecoderRefiner(out_channels)

        if fusion_type == 'residual_se':
            self.fuse = FuseConv(out_channels * 2, out_channels)
        elif fusion_type in {'mlp', '2mlps'}:
            self.fuse = FuseMLP(out_channels * 2, out_channels)
        else:
            raise ValueError(f'Unsupported fusion_type "{fusion_type}".')

    def forward(
        self,
        pre_features: Sequence[torch.Tensor],
        post_features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_context = self.loc_refine(self.loc_fpn(pre_features))
        post_context = self.dam_refine(self.dam_fpn(post_features))
        fused = self.fuse(torch.cat([pre_context, post_context], dim=1))
        return pre_context, fused


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for idx in range(num_blocks):
            blocks.append(ConvBNReLU(in_channels if idx == 0 else hidden_channels, hidden_channels))
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.classifier = nn.Conv2d(hidden_channels if blocks else in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        x = self.blocks(x)
        x = self.classifier(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True) -> None:
        super().__init__()
        if torchvision is None:
            raise ImportError('ChangeOS requires torchvision to provide ResNet backbones.')

        constructor = getattr(torchvision.models, backbone, None)
        if constructor is None:
            raise ValueError(f'Unsupported ResNet backbone "{backbone}".')

        kwargs = {}
        if pretrained and get_model_weights is not None:
            try:
                weights_enum = get_model_weights(backbone)
                kwargs['weights'] = weights_enum.DEFAULT
            except (ValueError, AttributeError):
                kwargs['pretrained'] = True
        else:
            kwargs['weights'] = None

        model = constructor(**kwargs)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.stage_channels = self._infer_stage_channels(model)

    @staticmethod
    def _infer_stage_channels(model: nn.Module) -> List[int]:
        stage_channels = []
        base_planes = [64, 128, 256, 512]
        for idx, planes in enumerate(base_planes, 1):
            block = getattr(model, f'layer{idx}')[0]
            expansion = getattr(block, 'expansion', 1)
            stage_channels.append(planes * expansion)
        return stage_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.maxpool(x)
        features = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            features.append(x)
        return features


class ChangeOS(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        loc_classes: Optional[int] = 2,
        num_classes: int = 4,
        backbone: str = 'resnet18',
        pretrained_backbone: bool = True,
        decoder_channels: int = 256,
        head_channels: int = 128,
        head_blocks: int = 1,
        fusion_type: str = 'mlp',
        freeze_backbone: bool = False,
        siamese_mode: str = 'shared',
    ) -> None:
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError('ChangeOS expects concatenated pre/post images with an even number of channels.')

        self.backbone_name = backbone
        self.siamese_mode = self._normalize_mode(siamese_mode)

        self.backbone_pre = self._build_backbone(backbone, pretrained_backbone)
        self.backbone = self.backbone_pre  # backward compatibility for existing checkpoints/utilities
        if self.siamese_mode == 'siamese':
            self.backbone_post = self._build_backbone(backbone, pretrained_backbone)
        else:
            self.backbone_post = self.backbone_pre

        per_branch_ch = in_channels // 2
        if per_branch_ch != 3:
            from src.models.channel_adapt import adapt_conv_channels
            self.backbone_pre.stem[0] = adapt_conv_channels(self.backbone_pre.stem[0], per_branch_ch)
            if self.siamese_mode == 'siamese':
                self.backbone_post.stem[0] = adapt_conv_channels(self.backbone_post.stem[0], per_branch_ch)

        if freeze_backbone:
            for module in {self.backbone_pre, self.backbone_post}:
                for param in module.parameters():
                    param.requires_grad = False

        self.decoder = ChangeOSDecoder(
            in_channels_list=self.backbone_pre.stage_channels,
            out_channels=decoder_channels,
            fusion_type=fusion_type,
        )

        self.has_loc_head = loc_classes is not None and loc_classes > 0
        if self.has_loc_head:
            self.loc_head = ClassificationHead(decoder_channels, head_channels, head_blocks, loc_classes)
        self.cls_head = ClassificationHead(decoder_channels, head_channels, head_blocks, num_classes)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError('Input tensor must have shape (B, C, H, W).')
        b, c, h, w = x.shape
        half = c // 2
        pre = x[:, :half]
        post = x[:, half:]

        pre_features = self.backbone_pre(pre)
        post_features = self.backbone_post(post)

        pre_context, fused_context = self.decoder(pre_features, post_features)

        cls_logits = self.cls_head(fused_context, size=(h, w))
        if self.has_loc_head:
            loc_logits = self.loc_head(pre_context, size=(h, w))
            return loc_logits, cls_logits
        return cls_logits

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or 'shared').strip().lower()
        if normalized not in {'shared', 'siamese'}:
            raise ValueError('siamese_mode must be either "shared" or "siamese".')
        return normalized

    @staticmethod
    def _build_backbone(backbone: str, pretrained: bool) -> ResNetFeatureExtractor:
        return ResNetFeatureExtractor(backbone=backbone, pretrained=pretrained)


__all__ = ['ChangeOS']
