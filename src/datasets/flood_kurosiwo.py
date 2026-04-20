"""KuroSiwo dataset for flood change detection / segmentation.

Loads multi-channel SAR imagery (pre1/pre2/post VV/VH), DEM, slope, and
3-class flood labels from the KuroSiwo dataset.  Supports both segmentation
(all channels stacked) and change detection (pre/post split) modes.

Directory layout expected under *dataset_path*::

    {dataset_path}/pre1_vv/{id}.tif
    {dataset_path}/pre1_vh/{id}.tif
    {dataset_path}/pre2_vv/{id}.tif
    {dataset_path}/pre2_vh/{id}.tif
    {dataset_path}/post_vv/{id}.tif
    {dataset_path}/post_vh/{id}.tif
    {dataset_path}/DEM/{id}.tif
    {dataset_path}/SLOPE/{id}.tif
    {dataset_path}/GT/{id}.tif
    {dataset_path}/MASK_NODATA/{id}.tif

Split files (one ID per line):
    data/kurosiwo/train.txt
    data/kurosiwo/val.txt
    data/kurosiwo/test.txt

Label classes: 0=no water, 1=permanent water, 2=flood (per Bountos et al. 2022, MK0_MLU encoding).
Label value 3 = nodata (mapped to ignore_index 255).

Returns
-------
Seg mode -- 3-tuple : (image, label, sample_id)
    - image    : float32 (C, H, W)  -- per-channel normalised
    - label    : int64   (H, W)     -- 3-class, 255 = ignore
    - sample_id: str

CD mode -- 4-tuple : (pre_img, post_img, label, sample_id)
    - pre_img  : float32 (C, H, W)  -- normalised pre channels (padded if needed)
    - post_img : float32 (C, H, W)  -- normalised post channels (padded if needed)
    - label    : int64   (H, W)     -- 3-class, 255 = ignore
    - sample_id: str
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import rioxarray
except ImportError:
    rioxarray = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_image(path: str, return_invalid: bool = False):
    """Load a TIFF, handle NaN. Returns (array, invalid_mask_or_None)."""
    if tifffile is None:
        raise ImportError("tifffile is required to read TIFF images")
    array = np.asarray(tifffile.imread(path))
    invalid = None
    if return_invalid:
        invalid = ~np.isfinite(array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = _move_channels_last(array)
    if return_invalid and invalid is not None:
        invalid = _move_channels_last(invalid.astype(bool))
        if invalid.ndim == 3:
            invalid = np.any(invalid, axis=-1)
    return array, invalid


def _move_channels_last(array: np.ndarray) -> np.ndarray:
    if (array.ndim == 3
            and array.shape[0] <= 16
            and array.shape[0] <= array.shape[1]
            and array.shape[0] <= array.shape[2]):
        return np.moveaxis(array, 0, -1)
    return array


def _ensure_3d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[:, :, None]
    return array


def _tile_channels(array: np.ndarray, target_channels: int) -> np.ndarray:
    """Repeat channels along last axis until reaching *target_channels*."""
    channels = array.shape[-1]
    if channels == target_channels:
        return array
    repeats = (target_channels + channels - 1) // channels
    tiled = np.tile(array, (1, 1, repeats))
    return tiled[..., :target_channels]


def _resolve_path(directory: str, name: str) -> str:
    """Find the actual file path, trying multiple TIFF extensions."""
    base = os.path.join(directory, name)
    if os.path.isfile(base):
        return base
    for ext in (".tif", ".tiff"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"File not found for '{name}' in {directory}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KuroSiwoDataset(Dataset):
    """KuroSiwo flood dataset supporting both seg and CD modes.

    Parameters
    ----------
    dataset_path : str
        Root directory containing channel sub-folders and GT/.
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
        Channel configuration.  Supported keys:

        * ``channels`` (list[str]) -- for seg mode, which channels to use
          (default: all 8).
        * ``pre_channels`` / ``post_channels`` (list[str]) -- for CD mode,
          which channels belong to pre vs post branch (defaults:
          [pre1_vv, pre1_vh, pre2_vv, pre2_vh] / [post_vv, post_vh]).
    """

    # Channel name -> sub-folder name mapping
    CHANNEL_FOLDERS: Dict[str, str] = {
        "pre1_vv": "pre1_vv",
        "pre1_vh": "pre1_vh",
        "pre2_vv": "pre2_vv",
        "pre2_vh": "pre2_vh",
        "post_vv": "post_vv",
        "post_vh": "post_vh",
        "dem":     "DEM",
        "slope":   "SLOPE",
    }

    DEFAULT_SEG_CHANNELS = [
        "pre1_vv", "pre1_vh", "pre2_vv", "pre2_vh",
        "post_vv", "post_vh", "dem", "slope",
    ]
    DEFAULT_PRE_CHANNELS = ["pre1_vv", "pre1_vh", "pre2_vv", "pre2_vh"]
    DEFAULT_POST_CHANNELS = ["post_vv", "post_vh"]
    EXTRA_CHANNELS = {"dem", "slope"}

    # Label value 3 = nodata in the GT
    NODATA_LABEL = 3

    # Per-channel normalisation statistics
    SAR_STATS = {
        "vv": {"mean": 0.0953, "std": 0.0427},
        "vh": {"mean": 0.0264, "std": 0.0215},
    }
    DEM_STATS = {"mean": 93.4313, "std": 1410.8382}
    SLOPE_STATS = {"mean": 2.1277, "std": 67.5048}

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

        self._available_channels = set(self.CHANNEL_FOLDERS.keys())

        # ---- resolve channel lists & normalisation stats ----
        if self.task == "seg":
            channels = self.input_cfg.get("channels")
            if channels is None:
                channels = list(self.DEFAULT_SEG_CHANNELS)
            self.seg_channels = self._validate_channel_list(channels)
            self._seg_mean, self._seg_std = self._gather_stats(
                self.seg_channels
            )
        else:
            pre_channels = self.input_cfg.get("pre_channels")
            post_channels = self.input_cfg.get("post_channels")
            if pre_channels is None:
                pre_channels = list(self.DEFAULT_PRE_CHANNELS)
            if post_channels is None:
                post_channels = list(self.DEFAULT_POST_CHANNELS)
            self.pre_channels = self._validate_channel_list(pre_channels)
            self.post_channels = self._validate_channel_list(post_channels)
            self._validate_cd_channels()
            self._pre_mean, self._pre_std = self._gather_stats(
                self.pre_channels
            )
            self._post_mean, self._post_std = self._gather_stats(
                self.post_channels
            )
            # When pre/post have different channel counts, pad to the max
            self._cd_channels = max(
                len(self.pre_channels), len(self.post_channels)
            )
            self._pre_mean_full = self._expand_stats(
                self._pre_mean, self._cd_channels
            )
            self._pre_std_full = self._expand_stats(
                self._pre_std, self._cd_channels
            )
            self._post_mean_full = self._expand_stats(
                self._post_mean, self._cd_channels
            )
            self._post_std_full = self._expand_stats(
                self._post_std, self._cd_channels
            )

        # Read sample list and validate file existence
        raw_list = self._read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if self._sample_exists(d)]
        if len(self.data_list) < len(raw_list):
            print(
                f"[{self.__class__.__name__}] Filtered "
                f"{len(raw_list) - len(self.data_list)} missing samples "
                f"(kept {len(self.data_list)}/{len(raw_list)})"
            )

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_data_list(path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _sample_exists(self, sample_id: str) -> bool:
        """Check GT + all required channel files exist."""
        name = str(sample_id)
        # GT is always required
        if not self._file_exists(
            os.path.join(self.dataset_path, "GT"), name
        ):
            return False
        # Check channels used by current mode
        if self.task == "seg":
            channels = self.seg_channels
        else:
            channels = list(self.pre_channels) + list(self.post_channels)
        for ch in channels:
            folder = self.CHANNEL_FOLDERS[ch]
            if not self._file_exists(
                os.path.join(self.dataset_path, folder), name
            ):
                return False
        # MASK_NODATA is always read in __getitem__
        if not self._file_exists(
            os.path.join(self.dataset_path, "MASK_NODATA"), name
        ):
            return False
        return True

    @staticmethod
    def _file_exists(directory: str, name: str) -> bool:
        base = os.path.join(directory, name)
        if os.path.isfile(base):
            return True
        for ext in (".tif", ".tiff"):
            if os.path.isfile(base + ext):
                return True
        return False

    def _validate_channel_list(self, channels: List[str]) -> List[str]:
        normalised = []
        for ch in channels:
            key = ch.lower()
            if key not in self._available_channels:
                raise KeyError(
                    f"Unknown channel '{ch}'. "
                    f"Available: {sorted(self._available_channels)}"
                )
            if key not in normalised:
                normalised.append(key)
        return normalised

    def _validate_cd_channels(self) -> None:
        """DEM/SLOPE must be in both branches if used in either."""
        extras_pre = {
            ch for ch in self.pre_channels if ch in self.EXTRA_CHANNELS
        }
        extras_post = {
            ch for ch in self.post_channels if ch in self.EXTRA_CHANNELS
        }
        if extras_pre != extras_post:
            raise ValueError(
                "DEM/SLOPE must be included in both pre and post branches "
                "when used for change detection."
            )

    # ------------------------------------------------------------------
    # Normalisation statistics
    # ------------------------------------------------------------------

    def _gather_stats(
        self, channels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        means, stds = [], []
        for ch in channels:
            stats = self._get_channel_stats(ch)
            means.append(stats["mean"])
            stds.append(stats["std"])
        return (
            np.asarray(means, dtype=np.float32),
            np.asarray(stds, dtype=np.float32),
        )

    @classmethod
    def _is_sar_channel(cls, channel: str) -> bool:
        name = channel.lower()
        return name.endswith("_vv") or name.endswith("_vh")

    @classmethod
    def _get_channel_stats(cls, channel: str) -> Dict[str, float]:
        lower = channel.lower()
        if lower == "dem":
            return cls.DEM_STATS
        if lower == "slope":
            return cls.SLOPE_STATS
        if lower.endswith("_vv"):
            return cls.SAR_STATS["vv"]
        if lower.endswith("_vh"):
            return cls.SAR_STATS["vh"]
        raise KeyError(
            f"Unknown statistics mapping for channel '{channel}'."
        )

    @classmethod
    def _clamp_sar_channels(
        cls, array: np.ndarray, channels: List[str]
    ) -> np.ndarray:
        """Clip SAR channels to [0, 0.15] range."""
        if array.ndim != 3:
            return array
        for idx, ch in enumerate(channels):
            if cls._is_sar_channel(ch):
                np.clip(array[..., idx], 0.0, 0.15, out=array[..., idx])
        return array

    @staticmethod
    def _expand_stats(values: np.ndarray, target: int) -> np.ndarray:
        if values.size == target:
            return values
        repeats = (target + values.size - 1) // values.size
        tiled = np.tile(values, repeats)
        return tiled[:target]

    @staticmethod
    def _normalize(
        array: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        std_safe = np.where(std == 0, 1.0, std).reshape((1, 1, -1))
        mean = mean.reshape((1, 1, -1))
        return (array - mean) / std_safe

    # ------------------------------------------------------------------
    # Channel loading
    # ------------------------------------------------------------------

    def _load_channel(
        self, sample_id: str, channel: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load a single channel and its invalid mask."""
        folder = self.CHANNEL_FOLDERS[channel]
        path = _resolve_path(
            os.path.join(self.dataset_path, folder), sample_id
        )
        if channel in {"dem", "slope"}:
            return self._load_dem_slope_channel(path)
        array, invalid = _read_image(path, return_invalid=True)
        array = np.asarray(array).squeeze()
        if invalid is not None:
            invalid = np.asarray(invalid, dtype=bool)
            if invalid.ndim == 3:
                invalid = np.any(invalid, axis=-1)
            invalid = invalid.squeeze()
        return array, invalid

    def _load_dem_slope_channel(
        self, path: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load DEM/SLOPE with optional rioxarray interpolation."""
        if rioxarray is not None:
            data = rioxarray.open_rasterio(path)
            array = data.squeeze().values
            invalid = ~np.isfinite(array)
            filled = data.rio.interpolate_na()
            array = filled.squeeze().values
        else:
            array, invalid_arr = _read_image(path, return_invalid=True)
            array = np.asarray(array).squeeze()
            invalid = (
                invalid_arr.squeeze()
                if invalid_arr is not None
                else ~np.isfinite(array)
            )

        array = np.asarray(array)
        invalid = np.asarray(invalid, dtype=bool) if invalid is not None else None
        if invalid is not None:
            remaining_invalid = ~np.isfinite(array)
            if remaining_invalid.any():
                invalid = invalid | remaining_invalid
        array = np.where(np.isfinite(array), array, 0.0)
        return array, invalid

    def _stack_channels(
        self, sample_id: str, channels: List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load and stack multiple channels into (H, W, C)."""
        arrays = []
        invalid = None
        for ch in channels:
            arr, invalid_arr = self._load_channel(sample_id, ch)
            arrays.append(arr)
            if invalid_arr is not None:
                invalid = (
                    invalid_arr if invalid is None else (invalid | invalid_arr)
                )
        stacked = np.stack(arrays, axis=-1)
        stacked = self._clamp_sar_channels(stacked, channels)
        return stacked, invalid

    def _load_valid_mask(self, sample_id: str) -> np.ndarray:
        """Load the MASK_NODATA validity mask (True = valid)."""
        path = _resolve_path(
            os.path.join(self.dataset_path, "MASK_NODATA"), sample_id
        )
        mask = _read_image(path)[0].squeeze()
        return mask > 0

    def _load_gt(self, sample_id: str) -> np.ndarray:
        """Load ground truth label map."""
        path = _resolve_path(
            os.path.join(self.dataset_path, "GT"), sample_id
        )
        return _read_image(path)[0].squeeze()

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]

        # --- Load ground truth and invalidity masks ---
        gt = self._load_gt(sample_id)
        valid_mask = self._load_valid_mask(sample_id)
        invalid_from_mask = ~valid_mask

        # NODATA_LABEL (3) in GT -> remap to class 0, mark as invalid
        nodata_from_gt = gt == self.NODATA_LABEL
        if np.any(nodata_from_gt):
            gt = gt.copy()
            gt[nodata_from_gt] = 0

        # --- Seg mode ---
        if self.task == "seg":
            image, invalid_data = self._stack_channels(
                sample_id, self.seg_channels
            )
            # Combine all invalid sources
            invalid_mask = invalid_from_mask | nodata_from_gt
            if invalid_data is not None:
                invalid_mask = invalid_mask | invalid_data

            # Apply ignore_index = 255
            mask = gt.copy()
            mask[invalid_mask] = 255
            mask = mask.astype(np.int64)

            image = _ensure_3d(image)

            # Apply albumentations transforms
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob["image"]
                mask = blob["mask"]

            # Per-channel normalisation
            image = image.astype(np.float32, copy=False)
            image = self._normalize(image, self._seg_mean, self._seg_std)

            # Transpose to (C, H, W)
            image = _ensure_3d(image)
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            mask = mask.astype(np.int64, copy=False)

            return image, mask, sample_id

        # --- CD mode ---
        pre, invalid_pre = self._stack_channels(
            sample_id, self.pre_channels
        )
        post, invalid_post = self._stack_channels(
            sample_id, self.post_channels
        )

        # Combine all invalid sources
        invalid_mask = invalid_from_mask | nodata_from_gt
        if invalid_pre is not None:
            invalid_mask = invalid_mask | invalid_pre
        if invalid_post is not None:
            invalid_mask = invalid_mask | invalid_post

        # Apply ignore_index = 255
        mask = gt.copy()
        mask[invalid_mask] = 255
        mask = mask.astype(np.int64)

        pre = _ensure_3d(pre)
        post = _ensure_3d(post)

        # Pad to equal channel count if pre/post differ
        if pre.shape[-1] != post.shape[-1]:
            target_channels = max(pre.shape[-1], post.shape[-1])
            pre = _tile_channels(pre, target_channels)
            post = _tile_channels(post, target_channels)
        else:
            target_channels = pre.shape[-1]

        # Apply albumentations transforms with dual-image support
        if self.transforms is not None:
            blob = self.transforms(image=pre, image_post=post, mask=mask)
            pre = blob["image"]
            post = blob["image_post"]
            mask = blob["mask"]

        # Per-channel normalisation with stats expanded to match channel count
        pre = pre.astype(np.float32, copy=False)
        post = post.astype(np.float32, copy=False)
        if hasattr(self, "_pre_mean_full"):
            if target_channels != self._pre_mean_full.size:
                pre_mean = self._expand_stats(self._pre_mean, target_channels)
                pre_std = self._expand_stats(self._pre_std, target_channels)
                post_mean = self._expand_stats(self._post_mean, target_channels)
                post_std = self._expand_stats(self._post_std, target_channels)
            else:
                pre_mean = self._pre_mean_full
                pre_std = self._pre_std_full
                post_mean = self._post_mean_full
                post_std = self._post_std_full
        else:
            pre_mean = self._pre_mean
            pre_std = self._pre_std
            post_mean = self._post_mean
            post_std = self._post_std

        pre = self._normalize(pre, pre_mean, pre_std)
        post = self._normalize(post, post_mean, post_std)

        # Transpose to (C, H, W)
        pre = _ensure_3d(pre)
        post = _ensure_3d(post)
        pre = np.ascontiguousarray(pre.transpose(2, 0, 1))
        post = np.ascontiguousarray(post.transpose(2, 0, 1))
        mask = mask.astype(np.int64, copy=False)

        return pre, post, mask, sample_id
