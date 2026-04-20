
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.post = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.post(x)


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 5,
        use_bn: bool = True,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError('UNetPlusPlus depth must be at least 2.')
        self.depth = depth
        self.deep_supervision = deep_supervision

        filters = [base_channels * (2 ** i) for i in range(depth)]

        self.down_blocks = nn.ModuleList()
        for idx in range(depth):
            in_ch = in_channels if idx == 0 else filters[idx - 1]
            self.down_blocks.append(ConvBlock(in_ch, filters[idx], use_bn=use_bn))

        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(depth - 1)])

        self.upsamplers: nn.ModuleDict = nn.ModuleDict()
        self.conv_blocks: nn.ModuleDict = nn.ModuleDict()
        for i in range(depth - 1):
            for j in range(1, depth - i):
                key = self._key(i, j)
                self.upsamplers[key] = UpsampleBlock(filters[i + 1], filters[i], use_bn=use_bn)
                in_ch = (j + 1) * filters[i]
                self.conv_blocks[key] = ConvBlock(in_ch, filters[i], use_bn=use_bn)

        if deep_supervision:
            self.output_convs = nn.ModuleList(
                [nn.Conv2d(filters[0], num_classes, kernel_size=1) for _ in range(depth - 1)]
            )
        else:
            self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    @staticmethod
    def _key(i: int, j: int) -> str:
        return f"{i}_{j}"

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        input_size = x.shape[-2:]
        nodes: Dict[Tuple[int, int], torch.Tensor] = {}

        current = x
        for i in range(self.depth):
            if i == 0:
                nodes[(i, 0)] = self.down_blocks[i](current)
            else:
                current = self.pools[i - 1](nodes[(i - 1, 0)])
                nodes[(i, 0)] = self.down_blocks[i](current)

        for j in range(1, self.depth):
            for i in range(self.depth - j):
                up_key = self._key(i, j)
                upsampled = self.upsamplers[up_key](nodes[(i + 1, j - 1)], nodes[(i, 0)].shape[-2:])
                concat_tensors = [nodes[(i, k)] for k in range(j)] + [upsampled]
                merged = torch.cat(concat_tensors, dim=1)
                nodes[(i, j)] = self.conv_blocks[up_key](merged)

        if self.deep_supervision:
            outputs = []
            for idx, conv in enumerate(self.output_convs, start=1):
                logits = conv(nodes[(0, idx)])
                if logits.shape[-2:] != input_size:
                    logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
                outputs.append(logits)
            return tuple(outputs)

        logits = self.final_conv(nodes[(0, self.depth - 1)])
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits


__all__ = ['UNetPlusPlus']
