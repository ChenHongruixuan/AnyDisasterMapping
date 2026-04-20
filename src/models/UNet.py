'''
Ronneberger, O., Fischer, P., and Brox, T.: U-Net: Convolutional Networks for Biomedical Image Segmentation, 
in: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pp. 234–241, 2015
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = self.bn(x)
        x = self.relu(x)
        return torch.cat((x, skip), dim=1)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        
        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 1024)
        
        self.decoder4 = ConvBlock(512 * 2, 512)
        self.decoder3 = ConvBlock(256 * 2, 256)
        self.decoder2 = ConvBlock(128 * 2, 128)
        self.decoder1 = ConvBlock(64 * 2, 64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up4 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))
        
        # Decoder path
        dec4 = self.up4(enc5, enc4)
        dec4 = self.decoder4(dec4)

        dec3 = self.up3(dec4, enc3)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3, enc2)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2, enc1)
        dec1 = self.decoder1(dec1)
        
        out = self.final_conv(dec1)
        
        return out
    
