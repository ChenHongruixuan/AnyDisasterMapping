from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: Iterable[int] = (1, 2, 3, 6)):
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

        concat_channels = in_channels + len(list(pool_scales)) * out_channels
        self.bottleneck = ConvBNReLU(concat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        pooled = [x]
        for stage in self.stages:
            pooled.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        fused = torch.cat(pooled, dim=1)
        return self.bottleneck(fused)


class UPerNetDecoder(nn.Module):
    """UPerNet-style decoder used by several segmentation backbones."""

    def __init__(
        self,
        in_channels: Sequence[int],
        fpn_channels: int = 256,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError('UPerNetDecoder requires at least two feature maps.')

        self.ppm = PyramidPoolingModule(in_channels[-1], fpn_channels, pool_scales=pool_scales)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for channels in in_channels[:-1]:
            self.lateral_convs.append(ConvBNReLU(channels, fpn_channels, kernel_size=1))
            self.fpn_convs.append(ConvBNReLU(fpn_channels, fpn_channels, kernel_size=3, padding=1))

        fusion_in_channels = fpn_channels * len(in_channels)
        self.fusion = ConvBNReLU(fusion_in_channels, fpn_channels, kernel_size=3, padding=1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral_convs) + 1:
            raise ValueError('Number of features provided does not match decoder configuration.')

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features[:-1])]
        top = self.ppm(features[-1])

        fpn_results: List[torch.Tensor] = [top]
        prev = top
        for lateral, fpn_conv in zip(reversed(laterals), reversed(self.fpn_convs)):
            prev = F.interpolate(prev, size=lateral.shape[2:], mode='bilinear', align_corners=False)
            merged = lateral + prev
            prev = fpn_conv(merged)
            fpn_results.insert(0, prev)

        target_size = fpn_results[0].shape[2:]
        resized = [fpn_results[0]]
        for feat in fpn_results[1:]:
            resized.append(F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False))

        fused = torch.cat(resized, dim=1)
        return self.fusion(fused)
