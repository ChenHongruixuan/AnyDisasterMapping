import torch
import torch.nn as nn
import torch.nn.functional as F 
from .segformer_head import SegFormerHead
from . import mix_transformer

class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=4, embedding_dim=256, pretrained_weight=None, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.input_channels = in_channels

        self.encoder = getattr(mix_transformer, backbone)(in_chans=in_channels)
        self.in_channels = self.encoder.embed_dims

        if pretrained_weight:
            self._load_pretrained_weights(pretrained_weight)

        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
        )

        self.classifier = nn.Conv2d(
            in_channels=self.in_channels[-1],
            out_channels=self.num_classes,
            kernel_size=1,
            bias=False,
        )

    def _load_pretrained_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = state_dict.copy()

        # Handle common wrappers around encoder parameters
        cleaned_state = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            if new_key.startswith('encoder.'):
                new_key = new_key[len('encoder.'):]
            cleaned_state[new_key] = value

        # Remove classification head weights if present
        cleaned_state.pop('head.weight', None)
        cleaned_state.pop('head.bias', None)

        patch_key = 'patch_embed1.proj.weight'
        if patch_key in cleaned_state:
            weight = cleaned_state[patch_key]
            if weight.shape[1] != self.input_channels:
                cleaned_state[patch_key] = self._resize_patch_embed_weight(weight, self.input_channels)

        encoder_state = self.encoder.state_dict()
        filtered_state = {k: v for k, v in cleaned_state.items() if k in encoder_state}

        missing, unexpected = self.encoder.load_state_dict(filtered_state, strict=False)
        if missing:
            preview = missing[:5]
            print(f'[SegFormer] Missing keys when loading pretrained weights (showing first 5 of {len(missing)}): {preview}')
        if unexpected:
            preview = unexpected[:5]
            print(f'[SegFormer] Unexpected keys when loading pretrained weights (showing first 5 of {len(unexpected)}): {preview}')
        print('pretrained weight loaded successfully!')

    @staticmethod
    def _resize_patch_embed_weight(weight: torch.Tensor, new_channels: int) -> torch.Tensor:
        old_channels = weight.shape[1]
        if new_channels == old_channels:
            return weight
        if new_channels < old_channels:
            return weight[:, :new_channels, :, :]
        extra = new_channels - old_channels
        channel_mean = weight.mean(dim=1, keepdim=True)
        repeated = channel_mean.repeat(1, extra, 1, 1)
        return torch.cat([weight, repeated], dim=1)

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, input_data):

        _x = self.encoder(input_data)
        # _x1, _x2, _x3, _x4 = _x
        # cls = self.classifier(_x4)
        x = self.decoder(_x)
        x = F.interpolate(x, size=input_data.size()[2:], mode='bilinear', align_corners=False)
        return x