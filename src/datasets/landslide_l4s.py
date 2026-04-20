"""Landslide4Sense segmentation dataset adapter.

Multi-spectral (14-channel) landslide segmentation from H5 files.
Per-channel normalization is applied internally (NOT via augmentation),
similar to KuroSiwo.

File layout under *dataset_path*::

    {dataset_path}/TrainData/img/image_*.h5   (key: 'img',  shape HWC)
    {dataset_path}/TrainData/mask/mask_*.h5    (key: 'mask', shape HW)
    {dataset_path}/TestData/img/image_*.h5
    {dataset_path}/TestData/mask/mask_*.h5

Split files list relative paths from dataset_path, e.g.::

    TrainData/img/image_1.h5
    TrainData/img/image_2.h5

Returns
-------
3-tuple : (image, label, sample_id)
    - image    : float32 (C, H, W)  -- 14ch (or 3ch if rgb_only)
    - label    : int64   (H, W)     -- 0=non-landslide, 1=landslide
    - sample_id: str

Normalization
-------------
Internal per-channel normalization applied AFTER augmentation transforms.
Augmentation config should contain geometric-only transforms (no Normalize).
"""

import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None

# Per-channel statistics computed from training set
L4S_MEAN = np.array([
    -0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644,
     0.0802,  0.3000,  0.4082, 0.0823, 0.0516, 0.3338, 0.7819,
], dtype=np.float32)

L4S_STD = np.array([
    0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354,
    0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913,
], dtype=np.float32)


class Landslide4SenseDataset(Dataset):
    """Landslide4Sense multi-spectral segmentation dataset.

    Parameters
    ----------
    dataset_path : str
        Root directory containing H5 image/mask files.
    data_list_path : str
        Text file listing relative paths to image H5 files from *dataset_path*.
    crop_size : int, optional
        Kept for interface compatibility; unused directly.
    split : str
        Split name (train / val / test).
    transforms : callable, optional
        Albumentations transform pipeline (geometric-only, no Normalize).
    rgb_only : bool
        If True, extract RGB channels [R=3, G=2, B=1] from 14-band input
        *after* per-channel normalization.  Output is (3, H, W).
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        rgb_only: bool = False,
    ) -> None:
        if h5py is None:
            raise ImportError("h5py is required to read Landslide4Sense H5 files")

        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size
        self.rgb_only = rgb_only

        raw_list = self._read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if self._sample_exists(d)]
        if len(self.data_list) < len(raw_list):
            print(
                f"[{self.__class__.__name__}] Filtered "
                f"{len(raw_list) - len(self.data_list)} missing samples "
                f"(kept {len(self.data_list)}/{len(raw_list)})"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _read_data_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _sample_exists(self, name: str) -> bool:
        img_path = os.path.join(self.dataset_path, name)
        lbl_name = name.replace("img", "mask").replace("image", "mask")
        lbl_path = os.path.join(self.dataset_path, lbl_name)
        return os.path.isfile(img_path) and os.path.isfile(lbl_path)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        name = self.data_list[index]
        img_path = os.path.join(self.dataset_path, name)
        lbl_name = name.replace("img", "mask").replace("image", "mask")
        lbl_path = os.path.join(self.dataset_path, lbl_name)

        with h5py.File(img_path, "r") as hf:
            image = hf["img"][:].astype(np.float32)   # (H, W, C)
        with h5py.File(lbl_path, "r") as hf:
            mask = hf["mask"][:].astype(np.float32)    # (H, W)

        # Apply augmentation transforms (geometric only, no Normalize)
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob["image"]
            mask = blob["mask"]

        # Transpose to CHW
        image = image.transpose(2, 0, 1)   # (C, H, W)

        # Per-channel normalization (legacy: applied in CHW after transforms)
        for i in range(len(L4S_MEAN)):
            image[i] -= L4S_MEAN[i]
            image[i] /= L4S_STD[i]

        # RGB extraction (after normalization, matching legacy order)
        if self.rgb_only:
            image = np.stack([image[3], image[2], image[1]], axis=0)

        image = np.ascontiguousarray(image).astype(np.float32, copy=False)
        mask = mask.astype(np.int64, copy=False)
        return image, mask, name
