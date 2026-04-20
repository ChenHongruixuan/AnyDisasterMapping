# farseg_pytorch.py
# MIT License
# Implementation: FarSeg (CVPR 2020) without a simplecv dependency.
# Requires: torch >= 1.10, torchvision >= 0.11.

import math
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN

import logging

LOGGER = logging.getLogger("trainer")


class FSRelation(nn.Module):
    """
    Foreground-Scene Relation module.

    - scene_feature: [B, C_s, 1, 1] scene embedding from GAP on the coarsest feature map
    - features: List[[B, C_i, Hi, Wi]] multi-scale features from the FPN
    Uses the scene embedding to reweight each feature scale.
    """

    def __init__(self, scene_channels: int, in_channels_list: List[int], out_channels: int):
        super().__init__()
        # Encode the scene feature to the same width as the content features.
        self.scene_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(scene_channels, out_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            for _ in in_channels_list
        ])

        # Encode content features for per-pixel scene correlation.
        self.content_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for c_in in in_channels_list
        ])

        # Re-encode the original features before applying the relation weights.
        self.feature_reencoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_in, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for c_in in in_channels_list
        ])

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # content_feats: features used for scene correlation.
        content_feats = [enc(feat) for enc, feat in zip(self.content_encoders, features)]
        # scene_feats: one scene-guidance feature per scale.
        scene_feats = [enc(scene_feature) for enc in self.scene_encoders]
        # relations: [B, 1, Hi, Wi] per-pixel correlation weights.
        relations = [
            self.normalizer((sf * cf).sum(dim=1, keepdim=True))
            for sf, cf in zip(scene_feats, content_feats)
        ]
        # p_feats: features after relation-weight re-encoding.
        p_feats = [enc(feat) for enc, feat in zip(self.feature_reencoders, features)]
        refined = [w * p for w, p in zip(relations, p_feats)]
        return refined


class LightWeightDecoder(nn.Module):
    """
    Lightweight decoder.

    Upsamples each scale to a common output stride (default 1/4), averages
    them per pixel, then predicts with a convolution head.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_classes: int,
        in_feature_output_strides: List[int] = (4, 8, 16, 32),
        out_feature_output_stride: int = 4,
    ):
        super().__init__()
        blocks = []
        for os in in_feature_output_strides:
            # Number of 2x upsampling stages needed to reach the target stride.
            num_upsample = int(math.log2(os) - math.log2(out_feature_output_stride))
            num_layers = max(1, num_upsample)
            layers = []
            for li in range(num_layers):
                layers += [
                    nn.Conv2d(in_channels if li == 0 else mid_channels, mid_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                ]
                # Apply bilinear 2x upsampling when this scale needs it.
                if num_upsample > 0:
                    layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(
            nn.Conv2d(mid_channels, num_classes, kernel_size=3, padding=1, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=4),  # Upsample from 1/4 stride to input resolution.
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        aligned: List[torch.Tensor] = []
        target_size = None
        for blk, feat in zip(self.blocks, features):
            out = blk(feat)
            if target_size is None:
                target_size = out.shape[-2:]
            elif out.shape[-2:] != target_size:
                # Bilinear upsampling stacks can drift by 1px; re-align to the reference size.
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(out)
        fused = sum(aligned) / len(aligned)
        logits = self.classifier(fused)
        return logits


class FarSeg(nn.Module):
    """
    FarSeg backbone stack (ResNet + FPN + FSRelation + LightWeightDecoder).

    Args:
        backbone: ["resnet18","resnet34","resnet50","resnet101"]
        num_classes: number of semantic classes
        pretrained: whether to load torchvision ImageNet weights
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 16,
        pretrained: bool = True,
        in_channels: int = 3,
        scale_stem_weight: bool = False,
    ):
        super().__init__()
        if backbone not in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Load the torchvision ResNet backbone.
        if pretrained:
            weights_attr = f"ResNet{backbone[6:]}_Weights"
            weights = getattr(resnet, weights_attr).DEFAULT
            self.backbone = getattr(resnet, backbone)(weights=weights)
        else:
            self.backbone = getattr(resnet, backbone)(weights=None)

        if in_channels != 3:
            self._replace_stem_conv(in_channels, scale_stem_weight=scale_stem_weight)

        # Select the C5 channel width from the ResNet variant.
        max_channels = 512 if backbone in ["resnet18", "resnet34"] else 2048

        # FPN input stages: C2 through C5.
        self.fpn = FPN(
            in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)],  # [C2, C3, C4, C5] channels
            out_channels=256,
        )

        # FS-Relation uses the C5 channel width as the scene width.
        self.fsr = FSRelation(scene_channels=max_channels, in_channels_list=[256, 256, 256, 256], out_channels=256)

        # Lightweight decoder head.
        self.decoder = LightWeightDecoder(in_channels=256, mid_channels=128, num_classes=num_classes)

    def _replace_stem_conv(self, in_channels: int, *, scale_stem_weight: bool = False) -> None:
        """Replace the first convolution to accommodate arbitrary input channels."""
        old_conv: nn.Conv2d = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            old_weight = old_conv.weight
            if in_channels <= old_weight.shape[1]:
                new_conv.weight[:, :in_channels] = old_weight[:, :in_channels]
            else:
                repeat = int(math.ceil(in_channels / old_weight.shape[1]))
                expanded = old_weight.repeat(1, repeat, 1, 1)
                if scale_stem_weight:
                    LOGGER.info(
                        "FarSeg scale_stem_weight active: repeat %d->%d channels with scale %.6f",
                        old_weight.shape[1],
                        in_channels,
                        old_weight.shape[1] / float(in_channels),
                    )
                    expanded = expanded * (old_weight.shape[1] / float(in_channels))
                new_conv.weight.copy_(expanded[:, :in_channels])
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.conv1 = new_conv

    def extract_backbone_feats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the ResNet C2, C3, C4, and C5 feature maps.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return c2, c3, c4, c5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        c2, c3, c4, c5 = self.extract_backbone_feats(x)

        # Scene embedding from GAP on the coarsest feature map -> [B, C5, 1, 1].
        scene_embed = F.adaptive_avg_pool2d(c5, 1)

        # FPN output order: {"c2": ..., "c3": ..., "c4": ..., "c5": ...}.
        fpn_feats = self.fpn(OrderedDict({"c2": c2, "c3": c3, "c4": c4, "c5": c5}))
        feats = [fpn_feats[k] for k in ["c2", "c3", "c4", "c5"]]

        # Reweight features with FS-Relation.
        feats = self.fsr(scene_embed, feats)

        # Decode back to the input image size.
        logits = self.decoder(feats)
        # Safeguard alignment to the input resolution.
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# Optional simple Focal Loss implementation without simplecv.
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss suitable as an alternative FarSeg loss.

    gamma defaults to 2.0.
    target: [B, H, W] int64 in the range [0, C-1]
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = 255, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        # pt = exp(-ce)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss[target != self.ignore_index].mean()
        elif self.reduction == "sum":
            return loss[target != self.ignore_index].sum()
        else:
            return loss


if __name__ == "__main__":
    # quick smoke test
    model = FarSeg(backbone="resnet50", num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("logits:", y.shape)  # -> [2, 6, 512, 512]
