"""Decoder modules for SAM2-based models."""

from .fpn import FPNDecoder  # noqa: F401
from .upernet import UPerNetDecoder  # noqa: F401

__all__ = [
    'FPNDecoder',
    'UPerNetDecoder',
]
