"""RescueNet dataset for single-temporal semantic segmentation.

Loads single RGB imagery and 11-class semantic segmentation labels from the
RescueNet dataset.  All inline augmentation has been removed; an optional
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


class RescueNetDataset(Dataset):
    """RescueNet single-temporal segmentation dataset.

    Returns
    -------
    3-tuple : (image, label, sample_id)
        - image    : float32 (C, H, W)
        - label    : int64   (H, W)  -- 11-class, 255 = ignore
        - sample_id: str             -- sample identifier
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        image_suffix: str = ".jpg",
        label_suffix: str = "_lab.png",
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size  # kept for compat; unused directly
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix

        # Read data list from file, filter out samples with any missing file
        raw_list = self._read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if (
            os.path.isfile(os.path.join(dataset_path, "img", d + image_suffix))
            and os.path.isfile(os.path.join(dataset_path, "label", d + label_suffix))
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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        sample_id = self.data_list[index]

        image_path = os.path.join(
            self.dataset_path, "img", sample_id + self.image_suffix
        )
        label_path = os.path.join(
            self.dataset_path, "label", sample_id + self.label_suffix
        )

        # Load image as float32 (H, W, 3), values in [0, 255]
        image = _img_loader(image_path)

        # Ensure 3 channels
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.stack((image[:, :, 0],) * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        # Load label
        label = _img_loader(label_path)
        if label.ndim == 3:
            label = label[:, :, 0]

        # Apply albumentations transforms if provided
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=label)
            image = blob["image"]
            label = blob["mask"]

        # Transpose to (C, H, W) -- always after transforms
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        label = np.asarray(label, dtype=np.int64)

        return image, label, sample_id
