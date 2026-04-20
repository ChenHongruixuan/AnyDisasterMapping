"""FLOGA wildfire change detection dataset adapter.

Supports two data types:
  - ``"rgb"``  : 3ch per temporal (BGR→RGB reorder from 9-band) → (H, W, 3) [0-1]
  - ``"sen2"`` : 9ch per temporal → clamp_and_scale → (H, W, 9) [0-1]

Directory layout::

    {dataset_path}/patch{patch_size}/T1/{id}.npy
    {dataset_path}/patch{patch_size}/T2/{id}.npy
    {dataset_path}/patch{patch_size}/GT/{id}.npy
    {dataset_path}/patch{patch_size}/{split}.txt

Returns (CD 4-tuple)::

    (pre_CHW, post_CHW, label_HW, sample_id)
"""

import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset


def _clamp_and_scale(arr, a_max=10000):
    return np.clip(arr, a_min=0, a_max=a_max).astype(np.float32) / a_max


class FLOGADataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        data_type: str = "sen2",
        patch_size: int = 256,
    ) -> None:
        self.data_dir = os.path.join(dataset_path, f"patch{patch_size}")
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size
        self.data_type = data_type
        assert data_type in ("rgb", "sen2"), data_type

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
            return [l.strip() for l in f if l.strip()]

    def _sample_exists(self, name: str) -> bool:
        return (
            os.path.isfile(os.path.join(self.data_dir, "T1", name + ".npy"))
            and os.path.isfile(os.path.join(self.data_dir, "T2", name + ".npy"))
            and os.path.isfile(os.path.join(self.data_dir, "GT", name + ".npy"))
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        name = self.data_list[index]

        # Load Sentinel-2 data: (C, H, W) raw → clamp_and_scale → (H, W, C)
        t1_raw = np.load(os.path.join(self.data_dir, "T1", name + ".npy")).astype(np.float32)
        t2_raw = np.load(os.path.join(self.data_dir, "T2", name + ".npy")).astype(np.float32)
        t1 = _clamp_and_scale(t1_raw).transpose(1, 2, 0)   # (H, W, 9)
        t2 = _clamp_and_scale(t2_raw).transpose(1, 2, 0)

        if self.data_type == "rgb":
            # BGR → RGB: select channels [2, 1, 0]
            t1 = t1[:, :, [2, 1, 0]]
            t2 = t2[:, :, [2, 1, 0]]

        # Label
        mask = np.load(os.path.join(self.data_dir, "GT", name + ".npy")).astype(np.uint8)
        mask[mask == 2] = 255   # unknown → ignore

        # NaN handling
        nan_mask = np.any(np.isnan(np.concatenate([t1, t2], axis=-1)), axis=-1)
        t1[nan_mask] = 0.0
        t2[nan_mask] = 0.0
        mask[nan_mask] = 255

        # Apply transforms (CD dual-image)
        if self.transforms is not None:
            blob = self.transforms(image=t1, image_post=t2, mask=mask)
            t1 = blob["image"]
            t2 = blob["image_post"]
            mask = blob["mask"]

        # HWC → CHW
        t1 = np.ascontiguousarray(t1.transpose(2, 0, 1)).astype(np.float32)
        t2 = np.ascontiguousarray(t2.transpose(2, 0, 1)).astype(np.float32)
        mask = mask.astype(np.int64)
        return t1, t2, mask, name
