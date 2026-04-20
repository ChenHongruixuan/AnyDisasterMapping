"""Shared utility for adapting pretrained model input channels."""
import torch
import torch.nn as nn


def adapt_conv_channels(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Replace a Conv2d with one accepting different input channels.

    Weights are adapted by repeating+scaling (expansion) or slicing (reduction).
    """
    if old_conv.in_channels == in_channels:
        return old_conv
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        w = old_conv.weight  # (out, old_in, kH, kW)
        old_in = w.shape[1]
        if in_channels < old_in:
            new_w = w[:, :in_channels]
        else:
            repeats = in_channels // old_in
            remainder = in_channels % old_in
            new_w = w.repeat(1, repeats, 1, 1)
            if remainder:
                new_w = torch.cat([new_w, w[:, :remainder]], dim=1)
            new_w = new_w * (old_in / in_channels)
        new_conv.weight.copy_(new_w)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv
