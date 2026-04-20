import torch
import torch.nn as nn
from src.models.DSIFN.VGG import vgg16
import torch.nn.functional as F
from typing import List


class vgg16_base(nn.Module):
    def __init__(self, in_channels, pretrained):
        super(vgg16_base, self).__init__()
        features = list(vgg16(in_channels, pretrained=pretrained).features)[:30]
        self.features = nn.ModuleList(features)
        for param in self.features.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22, 29}:
                results.append(x)
        return results

    def train(self, mode: bool = True):
        super().train(False)
        return self


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True, out=None)[0]

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )


class DSIFN(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        post_channels: int = None,
        num_classes: int = 2,
        siamese_mode: str = 'shared',
        vgg_pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.siamese_mode = self._normalize_mode(siamese_mode)
        self.shared_backbone = self.siamese_mode == 'shared'

        post_dim = post_channels if post_channels is not None else in_dim
        self.t1_base = vgg16_base(in_channels=in_dim, pretrained=vgg_pretrained)
        if self.shared_backbone:
            self.t2_base = self.t1_base
        else:
            self.t2_base = vgg16_base(in_channels=post_dim, pretrained=vgg_pretrained)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.sigmoid = nn.Sigmoid()

        # branch1
        self.ca1 = ChannelAttention(in_channels=1024)
        self.bn_ca1 = nn.BatchNorm2d(1024)
        self.o1_conv1 = conv2d_bn(1024, 512)
        self.o1_conv2 = conv2d_bn(512, 512)
        self.bn_sa1 = nn.BatchNorm2d(512)
        self.o1_conv3 = nn.Conv2d(512, self.num_classes, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=1536)
        self.bn_ca2 = nn.BatchNorm2d(1536)
        self.o2_conv1 = conv2d_bn(1536, 512)
        self.o2_conv2 = conv2d_bn(512, 256)
        self.o2_conv3 = conv2d_bn(256, 256)
        self.bn_sa2 = nn.BatchNorm2d(256)
        self.o2_conv4 = nn.Conv2d(256, self.num_classes, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=768)
        self.o3_conv1 = conv2d_bn(768, 256)
        self.o3_conv2 = conv2d_bn(256, 128)
        self.o3_conv3 = conv2d_bn(128, 128)
        self.bn_sa3 = nn.BatchNorm2d(128)
        self.o3_conv4 = nn.Conv2d(128, self.num_classes, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=384)
        self.o4_conv1 = conv2d_bn(384, 128)
        self.o4_conv2 = conv2d_bn(128, 64)
        self.o4_conv3 = conv2d_bn(64, 64)
        self.bn_sa4 = nn.BatchNorm2d(64)
        self.o4_conv4 = nn.Conv2d(64, self.num_classes, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=192)
        self.o5_conv1 = conv2d_bn(192, 64)
        self.o5_conv2 = conv2d_bn(64, 32)
        self.o5_conv3 = conv2d_bn(32, 16)
        self.bn_sa5 = nn.BatchNorm2d(16)
        self.o5_conv4 = nn.Conv2d(16, self.num_classes, 1)

    def forward(self, t1_input, t2_input):
        t1_list = self.t1_base(t1_input)
        t2_list = self.t2_base(t2_input)

        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_list[0], t1_list[1], t1_list[2], t1_list[3], t1_list[4]
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29 = t2_list[0], t2_list[1], t2_list[2], t2_list[3], t2_list[4]

        x = torch.cat((t1_f_l29, t2_f_l29), dim=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        branch_1_out = self.o1_conv3(x)

        x = self.trans_conv1(x)
        x = F.interpolate(x, size=t1_f_l22.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, t1_f_l22, t2_f_l22), dim=1)
        x = self.ca2(x) * x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) * x
        x = self.bn_sa2(x)

        branch_2_out = self.o2_conv4(x)

        x = self.trans_conv2(x)
        x = F.interpolate(x, size=t1_f_l15.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, t1_f_l15, t2_f_l15), dim=1)
        x = self.ca3(x) * x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) * x
        x = self.bn_sa3(x)

        branch_3_out = self.o3_conv4(x)

        x = self.trans_conv3(x)
        x = F.interpolate(x, size=t1_f_l8.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, t1_f_l8, t2_f_l8), dim=1)
        x = self.ca4(x) * x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) * x
        x = self.bn_sa4(x)

        branch_4_out = self.o4_conv4(x)

        x = self.trans_conv4(x)
        x = F.interpolate(x, size=t1_f_l3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, t1_f_l3, t2_f_l3), dim=1)
        x = self.ca5(x) * x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) * x
        x = self.bn_sa5(x)

        final_logits = self.o5_conv4(x)

        target_size = t1_input.shape[-2:]
        outputs: List[torch.Tensor] = [
            F.interpolate(final_logits, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(branch_4_out, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(branch_3_out, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(branch_2_out, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(branch_1_out, size=target_size, mode='bilinear', align_corners=False),
        ]

        return outputs

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or 'shared').strip().lower()
        if normalized not in {'shared', 'pseudo'}:
            raise ValueError("siamese_mode must be either 'shared' or 'pseudo'.")
        return normalized
