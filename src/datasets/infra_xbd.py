"""xBD dataset for building damage change detection.

Loads pre/post disaster imagery and multi-class damage labels from the xBD
dataset.  All inline augmentation has been removed; an optional albumentations
``transforms`` pipeline is applied instead.
"""

import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
from torch.utils.data import Dataset


def _img_loader(path: str) -> np.ndarray:
    """Load an image as float32 via imageio."""
    return np.asarray(imageio.imread(path), dtype=np.float32)


class xBDDataset(Dataset):
    """xBD change-detection dataset.

    Returns
    -------
    5-tuple : (pre_img, post_img, loc_label, clf_label, data_idx)
        - pre_img  : float32 (C, H, W)
        - post_img : float32 (C, H, W)
        - loc_label: int64   (H, W)  -- binary localisation mask
        - clf_label: int64   (H, W)  -- 5-class damage, 255 = ignore
        - data_idx : str             -- sample identifier
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        suffix: str = ".png",
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.suffix = suffix
        self.crop_size = crop_size  # kept for compat; unused directly

        # Read data list from file, filter out missing samples
        raw_list = self._read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if self._sample_exists(d)]
        if len(self.data_list) < len(raw_list):
            print(f"[{self.__class__.__name__}] Filtered {len(raw_list) - len(self.data_list)} "
                  f"missing samples (kept {len(self.data_list)}/{len(raw_list)})")

    @staticmethod
    def _read_data_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _sample_exists(self, data_id: str) -> bool:
        name = data_id
        if name.endswith(self.suffix):
            name = name[: -len(self.suffix)]
        if "_post_disaster" not in name and "_pre_disaster" not in name:
            pre_name = f"{name}_pre_disaster"
            post_name = f"{name}_post_disaster"
        else:
            pre_name = name.replace("_post_disaster", "_pre_disaster")
            post_name = name.replace("_pre_disaster", "_post_disaster")
        pre_path = os.path.join(self.dataset_path, "images", pre_name + self.suffix)
        post_path = os.path.join(self.dataset_path, "images", post_name + self.suffix)
        label_path = os.path.join(self.dataset_path, "masks", post_name + self.suffix)
        return os.path.isfile(pre_path) and os.path.isfile(post_path) and os.path.isfile(label_path)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        data_id = self.data_list[index]
        post_name = data_id
        if post_name.endswith(self.suffix):
            post_name = post_name[: -len(self.suffix)]

        pre_name = post_name.replace("_post_disaster", "_pre_disaster")
        if pre_name == post_name and "_pre_disaster" not in post_name:
            pre_name = f"{post_name}_pre_disaster"
            post_name = f"{post_name}_post_disaster"

        pre_path = os.path.join(self.dataset_path, "images", pre_name + self.suffix)
        post_path = os.path.join(self.dataset_path, "images", post_name + self.suffix)
        label_path = os.path.join(self.dataset_path, "masks", post_name + self.suffix)

        # Load images as float32 (H, W, 3), values in [0, 255]
        pre_img = _img_loader(pre_path)   # (H, W, 3)
        post_img = _img_loader(post_path) # (H, W, 3)

        # Ensure 3 channels
        if pre_img.ndim == 2:
            pre_img = np.stack((pre_img,) * 3, axis=-1)
        if post_img.ndim == 2:
            post_img = np.stack((post_img,) * 3, axis=-1)

        # Load label
        clf_label = _img_loader(label_path)
        if clf_label.ndim == 3:
            clf_label = clf_label[:, :, 0]

        # Damage class mapping:  values > 4 -> 255 (ignore)
        clf_label[clf_label > 4] = 255

        # Localisation label: binary (building vs background)
        loc_label = clf_label.copy()
        loc_label[clf_label > 1] = 1

        # Apply albumentations transforms if provided
        if self.transforms is not None:
            blob = self.transforms(image=pre_img, image_post=post_img, mask=clf_label)
            pre_img = blob["image"]
            post_img = blob["image_post"]
            clf_label = blob["mask"]
            # Recompute loc_label after crop / augmentation
            loc_label = clf_label.copy()
            loc_label[clf_label > 1] = 1

        # Transpose to (C, H, W) -- always after transforms
        pre_img = np.ascontiguousarray(pre_img.transpose(2, 0, 1))
        post_img = np.ascontiguousarray(post_img.transpose(2, 0, 1))

        clf_label = np.asarray(clf_label, dtype=np.int64)
        loc_label = np.asarray(loc_label, dtype=np.int64)

        return pre_img, post_img, loc_label, clf_label, data_id
