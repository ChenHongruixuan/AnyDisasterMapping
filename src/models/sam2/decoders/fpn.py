from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    """Lightweight Feature Pyramid Network decoder."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        use_bn: bool = True,
        upsample_mode: str = 'bilinear',
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError('FPNDecoder requires at least one input channel description.')

        self.upsample_mode = upsample_mode
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for channels in in_channels:
            lateral = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
            self.lateral_convs.append(lateral)

            layers: List[nn.Module] = [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.output_convs.append(nn.Sequential(*layers))

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != len(self.lateral_convs):
            raise ValueError(
                f'FPNDecoder expected {len(self.lateral_convs)} feature maps, received {len(features)}.'
            )

        results: List[torch.Tensor] = [torch.empty(0)] * len(features)
        prev_feature: torch.Tensor | None = None

        for idx in reversed(range(len(features))):
            lateral = self.lateral_convs[idx](features[idx])
            if prev_feature is not None:
                prev_feature = F.interpolate(
                    prev_feature,
                    size=lateral.shape[-2:],
                    mode=self.upsample_mode,
                    align_corners=False if self.upsample_mode == 'bilinear' else None,
                )
                lateral = lateral + prev_feature
            fused = self.output_convs[idx](lateral)
            results[idx] = fused
            prev_feature = fused

        return results
