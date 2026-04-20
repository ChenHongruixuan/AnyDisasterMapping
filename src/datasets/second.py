"""SECOND (Semantic Change Detection) dataset.

Loads dual-temporal optical imagery and three label maps (binary change,
T1 semantic, T2 semantic) from the SECOND dataset. Adapted from the
ChangeMamba semantic change detection loader and restyled to match the
albumentations-based interface used by ``xBDDataset``.

Directory layout expected under *dataset_path*::

    {dataset_path}/T1/{id}.png       -- pre-change RGB image
    {dataset_path}/T2/{id}.png       -- post-change RGB image
    {dataset_path}/GT_CD/{id}.png    -- binary change label {0, 255}
    {dataset_path}/GT_T1/{id}.png    -- T1 semantic label   {0..6}  (0=no-change, 1..6=semantic)
    {dataset_path}/GT_T2/{id}.png    -- T2 semantic label   {0..6}  (0=no-change, 1..6=semantic)

Note: SECOND uses 6 foreground semantic classes indexed from 1, with 0
reserved for no-change/background. This is why downstream consumers
(model head output_clf, scd_metrics) use num_classes=7, not 6 — the
(t1-1)*6+t2 encoding in scd_metrics._encode_scd requires indices 1..6.

Split files (one ID per line, with ``.png`` extension):
    data/SECOND/train.txt
    data/SECOND/test.txt

Returns
-------
6-tuple : (pre_img, post_img, cd_label, t1_label, t2_label, data_idx)
"""

import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
from torch.utils.data import Dataset


def _img_loader(path: str) -> np.ndarray:
    """Load an image as float32 via imageio."""
    return np.asarray(imageio.imread(path), dtype=np.float32)


class SECONDDataset(Dataset):
    """SECOND semantic change detection dataset.

    Returns
    -------
    6-tuple : (pre_img, post_img, cd_label, t1_label, t2_label, data_idx)
        - pre_img  : float32 (C, H, W)
        - post_img : float32 (C, H, W)
        - cd_label : int64   (H, W)  -- binary change {0, 1}
        - t1_label : int64   (H, W)  -- 7-valued {0..6}: 0=no-change, 1..6=semantic
        - t2_label : int64   (H, W)  -- 7-valued {0..6}: 0=no-change, 1..6=semantic
        - data_idx : str             -- sample identifier
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size  # kept for compat; unused directly

        # Read data list from file, filter out samples with any missing file
        raw_list = self._read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if self._sample_exists(d)]
        if len(self.data_list) < len(raw_list):
            print(
                f"[{self.__class__.__name__}] Filtered "
                f"{len(raw_list) - len(self.data_list)} missing samples "
                f"(kept {len(self.data_list)}/{len(raw_list)})"
            )

    @staticmethod
    def _read_data_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _sample_exists(self, data_id: str) -> bool:
        """Check that all 5 files (T1, T2, GT_CD, GT_T1, GT_T2) exist."""
        name = data_id
        for subdir in ("T1", "T2", "GT_CD", "GT_T1", "GT_T2"):
            if not os.path.isfile(os.path.join(self.dataset_path, subdir, name)):
                return False
        return True

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        data_id = self.data_list[index]
        name = data_id  # e.g. "02203.png"

        pre_path = os.path.join(self.dataset_path, "T1", name)
        post_path = os.path.join(self.dataset_path, "T2", name)
        cd_label_path = os.path.join(self.dataset_path, "GT_CD", name)
        t1_label_path = os.path.join(self.dataset_path, "GT_T1", name)
        t2_label_path = os.path.join(self.dataset_path, "GT_T2", name)

        # Load images as float32 (H, W, 3)
        pre_img = _img_loader(pre_path)
        post_img = _img_loader(post_path)

        # Load labels as float32 then convert
        cd_label = _img_loader(cd_label_path)
        t1_label = _img_loader(t1_label_path)
        t2_label = _img_loader(t2_label_path)

        # CD label: {0, 255} -> {0, 1}
        cd_label = cd_label / 255.0

        # Apply albumentations transforms if provided
        # additional_targets: image_post=image, mask_t1=mask, mask_cd=mask
        # where mask=t2_label (primary), mask_t1=t1_label, mask_cd=cd_label
        if self.transforms is not None:
            blob = self.transforms(
                image=pre_img,
                image_post=post_img,
                mask=t2_label,
                mask_t1=t1_label,
                mask_cd=cd_label,
            )
            pre_img = blob["image"]
            post_img = blob["image_post"]
            t2_label = blob["mask"]
            t1_label = blob["mask_t1"]
            cd_label = blob["mask_cd"]

        # Transpose to (C, H, W)
        pre_img = np.ascontiguousarray(pre_img.transpose(2, 0, 1))
        post_img = np.ascontiguousarray(post_img.transpose(2, 0, 1))

        cd_label = np.asarray(cd_label, dtype=np.int64)
        t1_label = np.asarray(t1_label, dtype=np.int64)
        t2_label = np.asarray(t2_label, dtype=np.int64)

        return pre_img, post_img, cd_label, t1_label, t2_label, data_id
