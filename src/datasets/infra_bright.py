"""BRIGHT dataset for building damage change detection.

Loads pre/post disaster satellite imagery (.tif) and multi-class damage labels
from the BRIGHT dataset.  All inline augmentation has been removed; an optional
albumentations ``transforms`` pipeline is applied instead.
"""

import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
from torch.utils.data import Dataset


def _img_loader(path: str) -> np.ndarray:
    """Load an image as float32 via imageio."""
    return np.asarray(imageio.imread(path), dtype=np.float32)


class BRIGHTDataset(Dataset):
    """BRIGHT change-detection dataset.

    Returns
    -------
    5-tuple : (pre_img, post_img, loc_label, clf_label, data_idx)
        - pre_img  : float32 (C, H, W)
        - post_img : float32 (C, H, W)
        - loc_label: int64   (H, W)  -- binary localisation mask
        - clf_label: int64   (H, W)  -- 4-class damage, 255 = ignore
        - data_idx : str             -- sample identifier
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        suffix: str = ".tif",
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.suffix = suffix
        self.crop_size = crop_size  # kept for compat; unused directly

        # Read data list from file, filter out samples with any missing file
        raw_list = self._read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if (
            os.path.isfile(os.path.join(dataset_path, "pre-event", d + "_pre_disaster" + suffix))
            and os.path.isfile(os.path.join(dataset_path, "post-event", d + "_post_disaster" + suffix))
            and os.path.isfile(os.path.join(dataset_path, "target", d + "_building_damage" + suffix))
        )]
        if len(self.data_list) < len(raw_list):
            print(f"[{self.__class__.__name__}] Filtered {len(raw_list) - len(self.data_list)} "
                  f"missing samples (kept {len(self.data_list)}/{len(raw_list)})")

    @staticmethod
    def _read_data_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        data_id = self.data_list[index]

        pre_path = os.path.join(
            self.dataset_path, "pre-event", data_id + "_pre_disaster" + self.suffix
        )
        post_path = os.path.join(
            self.dataset_path, "post-event", data_id + "_post_disaster" + self.suffix
        )
        label_path = os.path.join(
            self.dataset_path, "target", data_id + "_building_damage" + self.suffix
        )

        # Load images as float32, values in [0, 255]
        pre_img = _img_loader(pre_path)[:, :, 0:3]   # take first 3 channels only
        post_img = _img_loader(post_path)             # single channel

        # Stack single-channel post to 3ch
        if post_img.ndim == 2:
            post_img = np.stack((post_img,) * 3, axis=-1)
        elif post_img.ndim == 3 and post_img.shape[2] == 1:
            post_img = np.stack((post_img[:, :, 0],) * 3, axis=-1)

        # Load label
        clf_label = _img_loader(label_path)
        if clf_label.ndim == 3:
            clf_label = clf_label[:, :, 0]

        # Localisation label: classes 2,3 -> 1 (building present)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        # Apply albumentations transforms if provided
        if self.transforms is not None:
            blob = self.transforms(image=pre_img, image_post=post_img, mask=clf_label)
            pre_img = blob["image"]
            post_img = blob["image_post"]
            clf_label = blob["mask"]
            # Recompute loc_label after augmentation
            loc_label = clf_label.copy()
            loc_label[loc_label == 2] = 1
            loc_label[loc_label == 3] = 1

        # Transpose to (C, H, W) -- always after transforms
        pre_img = np.ascontiguousarray(pre_img.transpose(2, 0, 1))
        post_img = np.ascontiguousarray(post_img.transpose(2, 0, 1))

        clf_label = np.asarray(clf_label, dtype=np.int64)
        loc_label = np.asarray(loc_label, dtype=np.int64)

        return pre_img, post_img, loc_label, clf_label, data_id
