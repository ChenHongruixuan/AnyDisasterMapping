"""
FCN-8s: Fully Convolutional Networks for Semantic Segmentation

Faithful reproduction based on Long, Shelhamer & Darrell (2015), arXiv:1411.4038.
VGG-16 backbone with pool3/pool4/conv7 multi-scale skip-connection fusion.

Adapted from https://github.com/wkentaro/pytorch-fcn
Upstream project is MIT licensed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for transposed-convolution init."""
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):
    """FCN-8s with VGG-16 backbone and 3-scale skip fusion (pool3 + pool4 + conv7)."""

    encoder_param_prefixes = (
        "features1", "features2", "features3", "features4", "features5",
        "classifier",
    )

    def __init__(self, in_channels=3, num_classes=21, pretrained_backbone=True,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels

        # ----- VGG-16 backbone split into 5 stages -----
        # Stage 1 -> 1/2
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # Stage 2 -> 1/4
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # Stage 3 -> 1/8 (pool3)
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # Stage 4 -> 1/16 (pool4)
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # Stage 5 -> 1/32 (pool5)
        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # ----- Convolutionalized FC layers (fc6 / fc7) -----
        # padding=3 keeps spatial dims (replaces the original padding=100 hack)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        # ----- 1x1 scoring layers -----
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        # ----- Learned 2x transposed convolutions -----
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2, padding=1, bias=False)

        self._initialize_weights()

        if pretrained_backbone:
            self._load_pretrained_vgg16()

    # ------------------------------------------------------------------
    def _initialize_weights(self):
        # Scoring layers: zero-init (paper Section 4.2)
        for m in [self.score_fr, self.score_pool4, self.score_pool3]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # Transposed convolutions: bilinear init
        for m in [self.upscore2, self.upscore_pool4]:
            m.weight.data.copy_(
                _get_upsampling_weight(m.in_channels, m.out_channels,
                                       m.kernel_size[0]))

    # ------------------------------------------------------------------
    def _load_pretrained_vgg16(self):
        import torchvision
        vgg16 = torchvision.models.vgg16(weights="IMAGENET1K_V1")

        # VGG-16 .features index ranges for each stage
        vgg_feat = list(vgg16.features.children())
        ranges = [(0, 5), (5, 10), (10, 17), (17, 24), (24, 31)]
        stages = [self.features1, self.features2, self.features3,
                  self.features4, self.features5]

        for stage, (lo, hi) in zip(stages, ranges):
            src_convs = [vgg_feat[i] for i in range(lo, hi)
                         if isinstance(vgg_feat[i], nn.Conv2d)]
            dst_convs = [m for m in stage.modules()
                         if isinstance(m, nn.Conv2d)]
            for sc, dc in zip(src_convs, dst_convs):
                if sc.weight.shape == dc.weight.shape:
                    dc.weight.data.copy_(sc.weight.data)
                    dc.bias.data.copy_(sc.bias.data)
                elif sc.weight.shape[1] != dc.weight.shape[1]:
                    # in_channels mismatch (conv1 when in_channels != 3)
                    # → adapt via repeat+scale using codebase utility
                    from src.models.channel_adapt import adapt_conv_channels
                    adapted = adapt_conv_channels(sc, dc.in_channels)
                    dc.weight.data.copy_(adapted.weight.data)
                    if adapted.bias is not None and dc.bias is not None:
                        dc.bias.data.copy_(adapted.bias.data)

        # Convolutionalized fc6 / fc7
        cls_convs = [m for m in self.classifier.modules()
                     if isinstance(m, nn.Conv2d)]
        # fc6: Linear(512*7*7, 4096) → Conv2d(512, 4096, 7)
        fc6_w = vgg16.classifier[0].weight.data
        cls_convs[0].weight.data.copy_(fc6_w.view(4096, 512, 7, 7))
        cls_convs[0].bias.data.copy_(vgg16.classifier[0].bias.data)
        # fc7: Linear(4096, 4096) → Conv2d(4096, 4096, 1)
        fc7_w = vgg16.classifier[3].weight.data
        cls_convs[1].weight.data.copy_(fc7_w.view(4096, 4096, 1, 1))
        cls_convs[1].bias.data.copy_(vgg16.classifier[3].bias.data)

    # ------------------------------------------------------------------
    def forward(self, x):
        input_size = x.shape[2:]

        # ---- Encoder ----
        h = self.features1(x)           # 1/2
        h = self.features2(h)           # 1/4
        pool3 = self.features3(h)       # 1/8
        pool4 = self.features4(pool3)   # 1/16
        h = self.features5(pool4)       # 1/32

        # ---- Convolutionalized FC ----
        h = self.classifier(h)

        # ---- Score from conv7 → 2× upsample → fuse with pool4 ----
        h = self.score_fr(h)
        h = self.upscore2(h)
        score_pool4 = self.score_pool4(pool4)
        if h.shape[2:] != score_pool4.shape[2:]:
            h = F.interpolate(h, size=score_pool4.shape[2:],
                              mode="bilinear", align_corners=False)
        h = h + score_pool4             # 1/16

        # ---- 2× upsample → fuse with pool3 ----
        h = self.upscore_pool4(h)
        score_pool3 = self.score_pool3(pool3)
        if h.shape[2:] != score_pool3.shape[2:]:
            h = F.interpolate(h, size=score_pool3.shape[2:],
                              mode="bilinear", align_corners=False)
        h = h + score_pool3             # 1/8

        # ---- 8× upsample to input resolution ----
        h = F.interpolate(h, size=input_size,
                          mode="bilinear", align_corners=False)
        return h
