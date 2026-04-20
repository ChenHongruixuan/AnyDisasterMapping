"""GVLM change detection dataset adapter.

Loads pre/post temporal RGB images and binary change masks from the GVLM
dataset.  Supports both segmentation (stacked 6ch) and change detection
(separate 3ch pre/post) modes via the ``task`` kwarg.

Directory layout expected under *dataset_path*::

    {dataset_path}/t1/{id}.jpg   -- time-1 RGB image
    {dataset_path}/t2/{id}.jpg   -- time-2 RGB image
    {dataset_path}/label/{id}.png -- binary change mask

Split files (one sample ID per line, e.g. ``image_001.jpg``):
    data/landslide/gvlm/train.txt
    data/landslide/gvlm/test.txt

Returns
-------
Seg mode -- 3-tuple : (image, label, sample_id)
    - image    : float32 (6, H, W) -- t1+t2 channels stacked
    - label    : int64   (H, W)    -- 0=no-change, 1=change
    - sample_id: str

CD mode -- 4-tuple : (pre_img, post_img, label, sample_id)
    - pre_img  : float32 (3, H, W)
    - post_img : float32 (3, H, W)
    - label    : int64   (H, W)    -- 0=no-change, 1=change
    - sample_id: str

Normalization
-------------
Handled via albumentations ``Normalize`` in the augmentation config
(ImageNet mean/std, max_pixel_value=1.0).  The adapter returns raw
float32 images in [0, 255] range before transforms are applied.
"""

import os
from typing import List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class GVLMDataset(Dataset):
    """GVLM binary change detection dataset.

    Parameters
    ----------
    dataset_path : str
        Root directory containing t1/, t2/, label/ sub-folders.
    data_list_path : str
        Path to a text file listing sample IDs (one per line, e.g. ``image_001.jpg``).
    crop_size : int, optional
        Kept for interface compatibility; unused directly.
    split : str
        Split name (train / val / test).
    transforms : callable, optional
        Albumentations transform pipeline built by the trainer.
    task : str
        ``"seg"`` for segmentation (stacked 6ch) or ``"cd"`` for change detection.
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        task: str = "cd",
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size
        self.task = task.lower()

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

    def _sample_exists(self, sample_id: str) -> bool:
        t1 = os.path.join(self.dataset_path, "t1", sample_id)
        t2 = os.path.join(self.dataset_path, "t2", sample_id)
        lbl = os.path.join(
            self.dataset_path, "label",
            sample_id.replace(".jpg", ".png"),
        )
        return os.path.isfile(t1) and os.path.isfile(t2) and os.path.isfile(lbl)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]

        t1_path = os.path.join(self.dataset_path, "t1", sample_id)
        t2_path = os.path.join(self.dataset_path, "t2", sample_id)
        lbl_path = os.path.join(
            self.dataset_path, "label",
            sample_id.replace(".jpg", ".png"),
        )

        # Load images -- PIL gives uint8 HWC; cast to float32 [0, 255]
        t1 = np.asarray(Image.open(t1_path), dtype=np.float32)
        t2 = np.asarray(Image.open(t2_path), dtype=np.float32)
        mask = np.asarray(Image.open(lbl_path))

        if self.task == "seg":
            # Stack t1 + t2 along channel axis → HW6
            image = np.concatenate([t1, t2], axis=-1)

            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob["image"]
                mask = blob["mask"]

            # Binarize label AFTER transforms (matching legacy order)
            mask = mask.astype(np.int64)
            mask[mask > 128] = 1

            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            image = image.astype(np.float32, copy=False)
            return image, mask, sample_id

        # CD mode -- return separate pre / post
        if self.transforms is not None:
            blob = self.transforms(image=t1, image_post=t2, mask=mask)
            t1 = blob["image"]
            t2 = blob["image_post"]
            mask = blob["mask"]

        # Binarize label AFTER transforms (matching legacy order)
        mask = mask.astype(np.int64)
        mask[mask > 128] = 1

        t1 = np.ascontiguousarray(t1.transpose(2, 0, 1)).astype(np.float32, copy=False)
        t2 = np.ascontiguousarray(t2.transpose(2, 0, 1)).astype(np.float32, copy=False)
        return t1, t2, mask, sample_id
