"""RescueNet dataset wrapper with documented single-temporal augmentation pipeline."""

import os
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
import imageio

from .imulti_new import augment_images, to_tensor


def default_image_loader(path: str) -> np.ndarray:
    """Load an image from ``path`` using imageio and return it as ``float32``."""

    return np.asarray(imageio.imread(path), dtype=np.float32)


class RescueNetDataset(Dataset):
    """Dataset for RescueNet single-temporal imagery with optional augmentation.

    Parameters
    ----------
    dataset_path: str
        Root directory containing the ``img`` and ``label`` sub-folders.
    data_list: Iterable[str]
        Identifiers of samples without suffixes (e.g. ``disaster_0001``).
    crop_size: Optional[int]
        Size of the random crop used during training augmentations. Set to
        ``None`` to disable cropping.
    max_iters: Optional[int]
        When provided, the ``data_list`` is repeated to reach ``max_iters``
        samples—useful for iteration-based training loops.
    split: str
        Name of the split (``train`` / ``val`` / ``test``). Augmentations are
        only enabled when ``"train"`` appears in the name.
    loader: Callable[[str], np.ndarray]
        Function responsible for loading an image or label from disk.
    image_suffix: str
        Suffix appended to ``sample_id`` to locate the imagery file.
    label_suffix: str
        Suffix appended to ``sample_id`` to locate the label mask.
    ignore_index: int
        Value used to pad labels during random cropping.
    """

    def __init__(
        self,
        dataset_path: str,
        data_list: Iterable[str],
        crop_size: Optional[int],
        max_iters: Optional[int] = None,
        split: str = "train",
        loader: Callable[[str], np.ndarray] = default_image_loader,
        image_suffix: str = ".jpg",
        label_suffix: str = "_lab.png",
        ignore_index: int = 255,
    ) -> None:
        self.dataset_path = dataset_path
        self.data_list = list(data_list)
        self.loader = loader
        self.split = split
        self.crop_size = crop_size
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.ignore_index = ignore_index

        if max_iters is not None and self.data_list:
            repeat = int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = (self.data_list * repeat)[:max_iters]

    def __len__(self) -> int:
        """Return the number of samples in the expanded ``data_list``."""
        return len(self.data_list)

    def _build_paths(self, sample_id: str) -> Tuple[str, str]:
        """Construct absolute paths pointing to the image and label for ``sample_id``."""
        image_path = os.path.join(self.dataset_path, "img", sample_id + self.image_suffix)
        label_path = os.path.join(self.dataset_path, "label", sample_id + self.label_suffix)
        return image_path, label_path

    def _prepare_inputs(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure imagery has three channels and labels are 2-D arrays."""
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] > 3:
            image = image[..., :3]

        if label.ndim == 3:
            label = label[..., 0]

        return image, label

    def _apply_transforms(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations (when enabled) and convert the image to tensor form."""
        augment = "train" in self.split
        crop_target = self.crop_size if augment else None
        images, label = augment_images(
            [image],
            label,
            crop_size=crop_target,
            augment=augment,
            ignore_index=self.ignore_index,
        )
        tensor = to_tensor(images)[0]
        return tensor, label

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Load, transform, and return ``(image_tensor, label_mask, sample_id)``."""
        sample_id = self.data_list[index]
        image_path, label_path = self._build_paths(sample_id)

        image = self.loader(image_path)
        label = self.loader(label_path)
        image, label = self._prepare_inputs(image, label)
        tensor, label = self._apply_transforms(image, label)

        label = np.asarray(label).astype(np.int64, copy=False)
        return tensor, label, sample_id
