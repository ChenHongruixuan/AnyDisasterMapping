"""HRGLDD segmentation dataset adapter.

Pre-computed NPY-based multi-spectral landslide segmentation dataset.
The entire dataset (images + labels) is loaded into memory from .npy files
at construction time for fast random access.

Input format::

    data_npy_path  : .npy file of shape (N, H, W, C), float/uint8
    label_npy_path : .npy file of shape (N, H, W) or (N, H, W, 1), float/uint8

Returns
-------
3-tuple : (image, label, sample_id)
    - image    : float32 (C, H, W)  -- typically 4 channels
    - label    : int64   (H, W)     -- 0 / 1 binary
    - sample_id: str (index as string)

Normalization
-------------
None.  The raw float32 values from the NPY arrays are used directly.
"""

from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class HRGLDDDataset(Dataset):
    """HRGLDD NPY-based segmentation dataset.

    Parameters
    ----------
    data_npy_path : str
        Path to the .npy file containing image data, shape (N, H, W, C).
    label_npy_path : str
        Path to the .npy file containing labels, shape (N, H, W) or (N, H, W, 1).
    crop_size : int, optional
        Kept for interface compatibility; unused directly.
    split : str
        Split name (train / val / test).
    transforms : callable, optional
        Albumentations transform pipeline built by the trainer.
    """

    def __init__(
        self,
        data_npy_path: str,
        label_npy_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
    ) -> None:
        self.transforms = transforms
        self.crop_size = crop_size
        self.split = split
        self.data = np.load(data_npy_path)
        self.labels = np.load(label_npy_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data[index].astype(np.float32)         # (H, W, C)
        mask = self.labels[index].astype(np.float32).squeeze()  # (H, W)

        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob["image"]
            mask = blob["mask"]

        image = np.ascontiguousarray(image.transpose(2, 0, 1))  # (C, H, W)
        image = image.astype(np.float32, copy=False)
        mask = mask.astype(np.int64, copy=False)
        return image, mask, str(index)
