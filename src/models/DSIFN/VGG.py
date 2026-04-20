import warnings

import torch
import torch.nn as nn

__model_file = {
    16: 'pretrained_weight/vgg-16-pytorch.pth',
}

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, in_channels, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], in_channels, batch_norm=batch_norm), **kwargs)
    if pretrained:
        import os
        # Try local path first, then torchvision auto-download
        local_path = __model_file.get(16)
        if local_path and os.path.isfile(local_path):
            print(f'[DSIFN/VGG] Loading pretrained weights from local path: {local_path}')
            pretrain_dict = torch.load(local_path, map_location='cpu')
            weight_source = local_path
        else:
            from torchvision.models import vgg16, VGG16_Weights
            warnings.warn(
                f'[DSIFN/VGG] Local pretrained weights not found at {local_path}; '
                'falling back to torchvision VGG16 ImageNet weights.',
                stacklevel=2,
            )
            pretrain_dict = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).state_dict()
            weight_source = 'torchvision::VGG16_Weights.IMAGENET1K_V1'
        # Adapt first conv weight if input channels differ from pretrained (3)
        first_key = 'features.0.weight'
        if first_key in pretrain_dict and pretrain_dict[first_key].shape[1] != in_channels:
            old_w = pretrain_dict[first_key]
            old_in = old_w.shape[1]
            if in_channels < old_in:
                pretrain_dict[first_key] = old_w[:, :in_channels]
            else:
                repeats = in_channels // old_in
                remainder = in_channels % old_in
                new_w = old_w.repeat(1, repeats, 1, 1)
                if remainder:
                    new_w = torch.cat([new_w, old_w[:, :remainder]], dim=1)
                new_w = new_w * (old_in / in_channels)
                pretrain_dict[first_key] = new_w
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        print(f'[DSIFN/VGG] Pretrained weights loaded successfully from {weight_source}')

    return model


def vgg16(in_channels, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', in_channels, False, pretrained, progress, **kwargs)


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
