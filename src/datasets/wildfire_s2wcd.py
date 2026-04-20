"""S2WCD (Sentinel-2 Wildfire Change Detection) dataset adapter.

Supports two data types:
  - ``"rgb"``  : 3ch per temporal → (H, W, 3) float32 [0-1]
  - ``"sen2"`` : 13ch per temporal → clamp_and_scale → (H, W, 13) float32 [0-1]

Directory layout::

    {dataset_path}/T1/{id}.npy        (rgb mode)
    {dataset_path}/T2/{id}.npy
    {dataset_path}/T1_sen2/{id}.npy   (sen2 mode)
    {dataset_path}/T2_sen2/{id}.npy
    {dataset_path}/GT/{id}.png
    {dataset_path}/{split}.txt

Returns (CD 4-tuple)::

    (pre_CHW, post_CHW, label_HW, sample_id)
"""

import os
from typing import List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _clamp_and_scale(arr, a_max=10000):
    return np.clip(arr, a_min=0, a_max=a_max).astype(np.float32) / a_max


class S2WCDDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        data_type: str = "rgb",
    ) -> None:
        self.dataset_path = dataset_path
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
        t1_dir = "T1" if self.data_type == "rgb" else "T1_sen2"
        t2_dir = "T2" if self.data_type == "rgb" else "T2_sen2"
        return (
            os.path.isfile(os.path.join(self.dataset_path, t1_dir, name + ".npy"))
            and os.path.isfile(os.path.join(self.dataset_path, t2_dir, name + ".npy"))
            and os.path.isfile(os.path.join(self.dataset_path, "GT", name + ".png"))
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        name = self.data_list[index]

        if self.data_type == "rgb":
            t1 = np.load(os.path.join(self.dataset_path, "T1", name + ".npy")).astype(np.float32)
            t2 = np.load(os.path.join(self.dataset_path, "T2", name + ".npy")).astype(np.float32)
            # Already (H, W, 3) in [0, 1]
        else:
            t1_raw = np.load(os.path.join(self.dataset_path, "T1_sen2", name + ".npy")).astype(np.float32)
            t2_raw = np.load(os.path.join(self.dataset_path, "T2_sen2", name + ".npy")).astype(np.float32)
            # (13, H, W) → clamp_and_scale → (H, W, 13)
            t1 = _clamp_and_scale(t1_raw).transpose(1, 2, 0)
            t2 = _clamp_and_scale(t2_raw).transpose(1, 2, 0)

        # Label: grayscale PNG → binary, invalid → 255
        mask = np.array(Image.open(
            os.path.join(self.dataset_path, "GT", name + ".png")
        ).convert("L"), dtype=np.uint8)
        mask[mask > 1] = 255

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
