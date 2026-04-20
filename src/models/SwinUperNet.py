import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    from torchvision import models as tv_models
    from torchvision.models import get_model_weights
    from torchvision.models.swin_transformer import SwinTransformerBlock
except ImportError:  # pragma: no cover - dependency handled at runtime
    tv_models = None
    get_model_weights = None
    SwinTransformerBlock = None

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        concat_channels = in_channels + len(pool_scales) * out_channels
        self.bottleneck = ConvBNReLU(concat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w = x.shape[2:]
        pooled = [x]
        for stage in self.stages:
            pooled.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        fused = torch.cat(pooled, dim=1)
        return self.bottleneck(fused)


class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels_list, fpn_channels=256, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        if len(in_channels_list) < 2:
            raise ValueError('UPerNetDecoder requires at least two feature maps from the backbone')

        self.ppm = PyramidPoolingModule(in_channels_list[-1], fpn_channels, pool_scales)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:
            self.lateral_convs.append(ConvBNReLU(in_channels, fpn_channels, kernel_size=1))
            self.fpn_convs.append(ConvBNReLU(fpn_channels, fpn_channels, kernel_size=3, padding=1))

        fusion_in_channels = fpn_channels * len(in_channels_list)
        self.fusion = ConvBNReLU(fusion_in_channels, fpn_channels, kernel_size=3, padding=1)

    def forward(self, features):
        if len(features) != len(self.lateral_convs) + 1:
            raise ValueError('Number of features provided does not match decoder configuration')

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features[:-1])]
        top = self.ppm(features[-1])

        fpn_results = [top]
        prev = top
        for lateral, fpn_conv in zip(reversed(laterals), reversed(self.fpn_convs)):
            prev = F.interpolate(prev, size=lateral.shape[2:], mode='bilinear', align_corners=False)
            merged = lateral + prev
            prev = fpn_conv(merged)
            fpn_results.insert(0, prev)

        target_size = fpn_results[0].shape[2:]
        resized_features = [fpn_results[0]]
        for feat in fpn_results[1:]:
            resized_features.append(F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False))

        fused = torch.cat(resized_features, dim=1)
        return self.fusion(fused)


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, swin_model: nn.Module):
        super().__init__()
        self.swin_model = swin_model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []

        for layer in self.swin_model.features:
            x = layer(x)
            if self._is_swin_stage(layer):
                features.append(self._stage_to_nchw(x, layer))

        return features

    @staticmethod
    def _is_swin_stage(module: nn.Module) -> bool:
        if SwinTransformerBlock is None:
            return False
        if isinstance(module, nn.Sequential):
            return any(isinstance(block, SwinTransformerBlock) for block in module)
        blocks = getattr(module, 'blocks', None)
        if isinstance(blocks, nn.Sequential):
            return any(isinstance(block, SwinTransformerBlock) for block in blocks)
        if blocks is not None:
            try:
                iterator = iter(blocks)
            except TypeError:
                return False
            return any(isinstance(block, SwinTransformerBlock) for block in iterator)
        return False

    @staticmethod
    def _stage_to_nchw(tensor: torch.Tensor, stage: nn.Module) -> torch.Tensor:
        if tensor.dim() == 3:
            batch, tokens, channels = tensor.shape
            resolution = getattr(stage, 'output_resolution', None)
            if resolution is None or any(dim is None for dim in resolution):
                side = int(round(tokens ** 0.5))
                resolution = (side, max(1, tokens // max(1, side)))
            height, width = resolution
            tensor = tensor.view(batch, height, width, channels)
        elif tensor.dim() != 4:
            raise ValueError('Unexpected tensor shape for Swin stage output')

        return tensor.permute(0, 3, 1, 2).contiguous()


class SwinUperNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        backbone='swin_tiny',
        fpn_channels=256,
        pool_scales=(1, 2, 3, 6),
        freeze_backbone=False,
        img_size=None,
    ):
        super().__init__()
        if tv_models is None or SwinTransformerBlock is None:
            raise ImportError('SwinUperNet requires torchvision with Swin Transformer support. Please install torchvision>=0.13.')

        self.backbone_name = backbone
        self.backbone_model = self._build_backbone(backbone, in_channels, img_size)
        self.feature_extractor = SwinTransformerFeatureExtractor(self.backbone_model)

        sample_size = img_size if img_size is not None else 224
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, sample_size, sample_size)
            was_training = self.feature_extractor.training
            self.feature_extractor.eval()
            feature_outputs = self.feature_extractor(sample)
            if was_training:
                self.feature_extractor.train()

        if len(feature_outputs) < 2:
            raise ValueError('Swin backbone did not produce enough feature maps for decoding')

        backbone_channels = [feat.shape[1] for feat in feature_outputs]
        self.decoder = UPerNetDecoder(backbone_channels, fpn_channels=fpn_channels, pool_scales=pool_scales)

        self.classifier = nn.Sequential(
            ConvBNReLU(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

    def forward_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        size = x.shape[2:]
        features = self.forward_features(x)
        decoded = self.decoder(features)
        logits = self.classifier(decoded)
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return logits

    def _build_backbone(self, backbone_name: str, in_channels: int, img_size: int = None) -> nn.Module:
        normalized_name = backbone_name.replace('-', '_')
        base_name = normalized_name.split('.')[0]
        alias_map = {
            'swin_tiny': 'swin_t',
            'swin_small': 'swin_s',
            'swin_base': 'swin_b',
            'swin_large': 'swin_l',
            'swin_t': 'swin_t',
            'swin_s': 'swin_s',
            'swin_b': 'swin_b',
            'swin_l': 'swin_l',

        }
        canonical = alias_map.get(base_name, base_name)

        constructor = getattr(tv_models, canonical, None)
        if constructor is None:
            raise ValueError(f'Unsupported Swin backbone "{backbone_name}" for torchvision models')

        kwargs = {}
        if get_model_weights is not None:
            try:
                weights_enum = get_model_weights(canonical).DEFAULT
            except (AttributeError, ValueError, RuntimeError):
                weights_enum = None
            kwargs['weights'] = weights_enum
        else:
            kwargs['weights'] = None

        backbone = constructor(**kwargs)

        if in_channels != 3:
            self._adapt_patch_embed_channels(backbone, in_channels)

        if img_size is not None and hasattr(backbone, 'features'):
            patch_embed = getattr(backbone.features, 'patch_embed', None)
            if patch_embed is not None and hasattr(patch_embed, 'img_size'):
                patch_embed.img_size = (img_size, img_size)

        return backbone

    def _adapt_patch_embed_channels(self, backbone: nn.Module, in_channels: int) -> None:
        patch_embed = getattr(backbone.features, 'patch_embed', None)
        if patch_embed is None and hasattr(backbone.features, '0'):
            patch_embed = backbone.features[0]

        if patch_embed is None:
            raise ValueError('Unable to locate patch embedding module for Swin backbone')

        conv_module = None
        conv_parent = None
        conv_name = None

        if hasattr(patch_embed, 'proj') and isinstance(patch_embed.proj, nn.Conv2d):
            conv_module = patch_embed.proj
            conv_parent = patch_embed
            conv_name = 'proj'
        else:
            for name, module in patch_embed.named_children():
                if isinstance(module, nn.Conv2d):
                    conv_module = module
                    conv_parent = patch_embed
                    conv_name = name
                    break

        if conv_module is None:
            raise ValueError('Unable to adapt patch embedding convolution for Swin backbone')

        if conv_module.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels,
            conv_module.out_channels,
            kernel_size=conv_module.kernel_size,
            stride=conv_module.stride,
            padding=conv_module.padding,
            bias=conv_module.bias is not None,
        )

        with torch.no_grad():
            new_weight = self._expand_input_channels(conv_module.weight.data, in_channels)
            new_conv.weight.copy_(new_weight)
            if conv_module.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv_module.bias.data)

        if conv_parent is not None and conv_name is not None:
            conv_parent._modules[conv_name] = new_conv
        else:
            raise ValueError('Unable to assign adapted patch embedding convolution')

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
