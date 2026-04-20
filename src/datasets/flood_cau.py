"""CAU Flood dataset for flood change detection / segmentation.

Loads pre/post disaster imagery and binary flood masks from the CAU Flood
dataset.  Supports both segmentation (stacked pre+post as multi-channel
image) and change detection (separate pre/post) modes via the ``task`` kwarg.

Directory layout expected under *dataset_path*::

    {dataset_path}/PRE/{id}_pre.tif     -- pre-event image
    {dataset_path}/POST/{id}_post.tif   -- post-event image
    {dataset_path}/GT/{id}_gt.tif       -- binary flood mask

Split files (one ID per line):
    data/cau_flood/train.txt
    data/cau_flood/test.txt

Returns
-------
Seg mode -- 3-tuple : (image, label, sample_id)
    - image    : float32 (C, H, W)  -- pre+post channels stacked
    - label    : int64   (H, W)     -- binary flood mask, 255 = ignore
    - sample_id: str

CD mode -- 4-tuple : (pre_img, post_img, label, sample_id)
    - pre_img  : float32 (C, H, W)
    - post_img : float32 (C, H, W)
    - label    : int64   (H, W)     -- binary flood mask, 255 = ignore
    - sample_id: str
"""

import os
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

try:
    import tifffile
except ImportError:
    tifffile = None


def _read_image(path: str, return_invalid: bool = False):
    """Load image from TIFF or other formats (e.g. PNG), handle NaN/invalid pixels.

    Returns (array_HWC_float32, invalid_mask_HW_or_None).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        if tifffile is None:
            raise ImportError("tifffile is required to read TIFF images")
        array = np.asarray(tifffile.imread(path), dtype=np.float32)
    else:
        import imageio
        array = np.asarray(imageio.imread(path), dtype=np.float32)
    invalid = None
    if return_invalid:
        invalid = ~np.isfinite(array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure HWC layout
    array = _move_channels_last(array)
    if return_invalid and invalid is not None:
        invalid = _move_channels_last(invalid.astype(bool))
        if invalid.ndim == 3:
            invalid = np.any(invalid, axis=-1)
    return array.astype(np.float32, copy=False), invalid


def _move_channels_last(array: np.ndarray) -> np.ndarray:
    """Move channels-first to channels-last if shape looks like (C, H, W)."""
    if (array.ndim == 3
            and array.shape[0] <= 16
            and array.shape[0] <= array.shape[1]
            and array.shape[0] <= array.shape[2]):
        return np.moveaxis(array, 0, -1)
    return array


def _ensure_3d(array: np.ndarray) -> np.ndarray:
    """Ensure array is at least 3-D (H, W, C)."""
    if array.ndim == 2:
        return array[:, :, None]
    return array


def _tile_channels(array: np.ndarray, target_channels: int) -> np.ndarray:
    """Tile last axis to reach *target_channels*, then truncate to exact size."""
    channels = array.shape[-1]
    if channels == target_channels:
        return array
    repeats = (target_channels + channels - 1) // channels
    tiled = np.tile(array, (1, 1, repeats))
    return tiled[..., :target_channels]


class CAUFloodDataset(Dataset):
    """CAU Flood dataset supporting both seg and CD modes.

    Parameters
    ----------
    dataset_path : str
        Root directory containing PRE/, POST/, GT/ sub-folders.
    data_list_path : str
        Path to a text file listing sample IDs (one per line).
    crop_size : int, optional
        Kept for interface compatibility; unused directly.
    split : str
        Split name (train / val / test).
    transforms : callable, optional
        Albumentations transform pipeline.
    task : str
        ``"seg"`` for segmentation or ``"cd"`` for change detection.
    input_cfg : dict, optional
        Channel configuration.  For seg mode, ``mode`` defaults to ``"stack"``.
        For cd mode, ``mode`` defaults to ``"siamese"``.
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        task: str = "cd",
        input_cfg: Optional[Dict] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size
        self.task = task.lower()
        self.input_cfg = input_cfg or {}

        default_mode = "stack" if self.task == "seg" else "siamese"
        self.input_mode = (self.input_cfg.get("mode") or default_mode).lower()

        # Read sample list and validate file existence
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

    def _sample_exists(self, sample_id: str) -> bool:
        """Check that pre, post, and GT files all exist for this sample."""
        name = str(sample_id)
        pre_ok = self._find_file(os.path.join(self.dataset_path, "PRE"), name)
        post_ok = self._find_file(os.path.join(self.dataset_path, "POST"), name)
        gt_ok = self._find_file(os.path.join(self.dataset_path, "GT"), name)
        return pre_ok and post_ok and gt_ok

    @staticmethod
    def _find_file(directory: str, name: str) -> bool:
        """Check if a file with any of the supported extensions exists."""
        for ext in ("", ".png", ".tif", ".tiff"):
            if os.path.isfile(os.path.join(directory, name + ext)):
                return True
        return False

    @staticmethod
    def _resolve_path(directory: str, name: str) -> str:
        """Find the actual file path, trying multiple extensions."""
        base = os.path.join(directory, name)
        if os.path.isfile(base):
            return base
        for ext in (".png", ".tif", ".tiff"):
            candidate = base + ext
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"File not found for '{name}' in {directory}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]
        name = str(sample_id)

        pre_path = self._resolve_path(
            os.path.join(self.dataset_path, "PRE"), name
        )
        post_path = self._resolve_path(
            os.path.join(self.dataset_path, "POST"), name
        )
        gt_path = self._resolve_path(
            os.path.join(self.dataset_path, "GT"), name
        )

        # Load images -- returns (H, W, C) float32
        pre, invalid_pre = _read_image(pre_path, return_invalid=True)
        post, invalid_post = _read_image(post_path, return_invalid=True)

        # Load mask -- squeeze to (H, W)
        mask = _read_image(gt_path, return_invalid=False)[0].squeeze()

        # Mark invalid (NaN) pixels as ignore_index = 255
        invalid_mask = np.zeros(pre.shape[:2], dtype=bool)
        if invalid_pre is not None:
            invalid_mask |= invalid_pre
        if invalid_post is not None:
            invalid_mask |= invalid_post
        if np.any(invalid_mask):
            mask = mask.copy()
            mask[invalid_mask] = 255

        mask = mask.astype(np.int64)

        if self.task == "seg":
            # Stack pre + post along channel axis
            image = np.concatenate(
                (_ensure_3d(pre), _ensure_3d(post)), axis=-1
            )

            # Apply albumentations transforms
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob["image"]
                mask = blob["mask"]

            # Transpose to (C, H, W)
            image = _ensure_3d(image)
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            image = image.astype(np.float32, copy=False)
            mask = mask.astype(np.int64, copy=False)

            return image, mask, sample_id

        # CD mode -- return separate pre/post
        pre = _ensure_3d(pre)
        post = _ensure_3d(post)

        # Tile channels to match when pre/post have different channel counts
        if pre.shape[-1] != post.shape[-1]:
            target = max(pre.shape[-1], post.shape[-1])
            pre = _tile_channels(pre, target)
            post = _tile_channels(post, target)

        # Apply albumentations transforms with dual-image support
        if self.transforms is not None:
            blob = self.transforms(image=pre, image_post=post, mask=mask)
            pre = blob["image"]
            post = blob["image_post"]
            mask = blob["mask"]

        # Transpose to (C, H, W)
        pre = _ensure_3d(pre)
        post = _ensure_3d(post)
        pre = np.ascontiguousarray(pre.transpose(2, 0, 1))
        post = np.ascontiguousarray(post.transpose(2, 0, 1))
        pre = pre.astype(np.float32, copy=False)
        post = post.astype(np.float32, copy=False)
        mask = mask.astype(np.int64, copy=False)

        return pre, post, mask, sample_id
