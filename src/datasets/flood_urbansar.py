"""UrbanSARFloods dataset for flood change detection / segmentation.

Loads multi-channel SAR imagery and 3-class flood masks from the
UrbanSARFloods dataset.  Data is stored as TIFF files.  Supports both
segmentation (all channels stacked) and change detection (pre/post split
via channel_router) modes.

Label classes: 0 = background (no flood), 1 = flood open area,
               2 = flood urban area, 255 = ignore.

Directory layout expected under *dataset_path*::

    {dataset_path}/SAR/{id}.tif    -- multi-channel SAR image
    {dataset_path}/GT/{id}.tif     -- 3-class flood mask

Split files (one ID per line):
    data/urbansar/train.txt
    data/urbansar/test.txt

Returns
-------
Seg mode -- 3-tuple : (image, label, sample_id)
    - image    : float32 (C, H, W)
    - label    : int64   (H, W)     -- 3-class flood, 255 = ignore
    - sample_id: str

CD mode -- 4-tuple : (pre_img, post_img, label, sample_id)
    - pre_img  : float32 (C, H, W)
    - post_img : float32 (C, H, W)
    - label    : int64   (H, W)     -- 3-class flood, 255 = ignore
    - sample_id: str
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


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
    """Tile last axis to reach target_channels (matching legacy BaseDataset)."""
    c = array.shape[-1]
    if c >= target_channels:
        return array[..., :target_channels]
    repeats = (target_channels + c - 1) // c
    return np.tile(array, (1, 1, repeats))[..., :target_channels]


def _load_array(path: str, return_invalid: bool = False):
    """Load array from .tif/.tiff file, handle NaN/invalid pixels.

    Returns (array_HWC_float32, invalid_mask_or_None).  The invalid mask
    preserves its original dimensionality (2-D or 3-D HWC) so that callers
    can compute per-branch invalidity when needed.
    """
    import tifffile
    array = tifffile.imread(path).astype(np.float32)
    array = np.asarray(array)
    invalid = None
    if return_invalid:
        invalid = ~np.isfinite(array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure HWC layout
    array = _move_channels_last(array)
    if return_invalid and invalid is not None:
        invalid = _move_channels_last(invalid.astype(bool))
    return array.astype(np.float32, copy=False), invalid


def _resolve_path(directory: str, name: str, extensions: Tuple[str, ...] = ()) -> str:
    """Find the actual file path, trying the base name then extensions."""
    base = os.path.join(directory, name)
    if os.path.isfile(base):
        return base
    for ext in extensions:
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"File not found for '{name}' in {directory}")


class UrbanSARFloodsDataset(Dataset):
    """UrbanSARFloods dataset supporting both seg and CD modes.

    Parameters
    ----------
    dataset_path : str
        Root directory containing SAR/ and GT/ sub-folders.
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
        Channel routing configuration for CD mode.  Must contain
        ``channel_router`` dict with ``pre_channels`` and ``post_channels``
        lists of integer channel indices.  Example::

            input_cfg = {
                "channel_router": {
                    "pre_channels": [0, 1, 2],
                    "post_channels": [3, 4, 5],
                }
            }
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str,
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        task: str = "seg",
        input_cfg: Optional[Dict] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size
        self.task = task.lower()
        self.input_cfg = input_cfg or {}

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
        """Check that both SAR and GT files exist for this sample."""
        name = str(sample_id)
        sar_ok = self._file_exists(
            os.path.join(self.dataset_path, "SAR"), name
        )
        gt_ok = self._file_exists(
            os.path.join(self.dataset_path, "GT"), name
        )
        return sar_ok and gt_ok

    @staticmethod
    def _file_exists(directory: str, name: str) -> bool:
        base = os.path.join(directory, name)
        if os.path.isfile(base):
            return True
        for ext in (".tif", ".tiff"):
            if os.path.isfile(base + ext):
                return True
        return False

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]
        name = str(sample_id)

        sar_path = _resolve_path(
            os.path.join(self.dataset_path, "SAR"),
            name,
            extensions=(".tif", ".tiff"),
        )
        gt_path = _resolve_path(
            os.path.join(self.dataset_path, "GT"),
            name,
            extensions=(".tif", ".tiff"),
        )

        # Load image -- (H, W, C) float32
        image, invalid = _load_array(sar_path, return_invalid=True)

        # Load mask -- squeeze to (H, W)
        mask = _load_array(gt_path, return_invalid=False)[0].squeeze()

        mask = mask.astype(np.int64)

        if self.task == "seg":
            # Collapse 3D invalid to 2D (any-channel) for whole-image masking
            invalid_2d = None
            if invalid is not None:
                invalid_2d = invalid.any(axis=-1) if invalid.ndim == 3 else invalid
            if invalid_2d is not None and np.any(invalid_2d):
                mask = mask.copy()
                mask[invalid_2d] = 255

            image = _ensure_3d(image)

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

        # CD mode -- split channels via channel_router
        router = self.input_cfg.get("channel_router", {})
        pre_idx = router.get("pre_channels")
        post_idx = router.get("post_channels")
        if pre_idx is None or post_idx is None:
            raise KeyError(
                "input_cfg must contain 'channel_router' with "
                "'pre_channels' and 'post_channels' for change detection mode"
            )

        image = _ensure_3d(image)
        pre = image[..., pre_idx]
        post = image[..., post_idx]

        # Tile channels to match if pre/post have different counts
        if pre.shape[-1] != post.shape[-1]:
            target = max(pre.shape[-1], post.shape[-1])
            pre = _tile_channels(pre, target)
            post = _tile_channels(post, target)

        # Per-branch invalid masking: compute invalidity from the raw 3D
        # invalid array using only the channels belonging to each branch,
        # matching the legacy per-branch logic.
        if invalid is not None:
            if invalid.ndim == 3:
                invalid_2d = (invalid[..., pre_idx].any(axis=-1)
                              | invalid[..., post_idx].any(axis=-1))
            else:
                invalid_2d = invalid.astype(bool)
            if np.any(invalid_2d):
                mask = mask.copy()
                mask[invalid_2d] = 255

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
