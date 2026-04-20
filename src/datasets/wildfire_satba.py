"""Satellite Burned Area wildfire change detection dataset adapter.

Supports four data types:
  - ``"sen2"``  : pre-S2 (12ch) + post-S2 (12ch) = 24ch concatenated
  - ``"rgb"``   : pre-S2 RGB (3ch) + post-S2 RGB (3ch) = 6ch concatenated
  - ``"s2_s1"`` : pre-S2 (12ch) + post-S1 VV/VH (2ch) = 14ch concatenated
  - ``"s1"``    : pre-S1 VV/VH (2ch) + post-S1 VV/VH (2ch) = 4ch concatenated

Data is discovered from CSV + folder structure (no data_list_path).

Directory layout::

    {dataset_path}/satellite_data.csv
    {dataset_path}/{event_folder}/sentinel2_YYYY-MM-DD.tiff
    {dataset_path}/{event_folder}/sentinel1_YYYY-MM-DD.tiff
    {dataset_path}/{event_folder}/{event_folder}_mask.tiff

Split assignment via fold colours in CSV:
  - train: ['purple', 'pink', 'grey', 'lime', 'magenta']
  - val:   ['coral']
  - test:  ['cyan']

Returns (CD 4-tuple)::

    (pre_CHW, post_CHW, label_HW, sample_id)
"""

import os
import re
import math
from os.path import join, basename
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize, rescale


def _clamp_and_scale(sen2_value, a_max=10000):
    scaled_sample = np.clip(sen2_value, a_max=a_max, a_min=0)
    scaled_sample = scaled_sample / a_max
    return scaled_sample


split_libs = dict(
    train=['purple', 'pink', 'grey', 'lime', 'magenta'],
    val=['coral'],
    test=['cyan'],
)


class SatelliteBurnedAreaDataset(Dataset):
    """Unified adapter for the Satellite Burned Area wildfire dataset.

    Tiles satellite imagery into fixed-size patches at init time,
    then returns (pre, post, label, id) 4-tuples as numpy arrays.
    """

    def __init__(
        self,
        dataset_path: str,
        data_list_path: str = None,   # unused — splits from CSV fold assignment
        crop_size: Optional[int] = None,
        split: str = "train",
        transforms=None,
        data_type: str = "sen2",
        mask_intervals: Optional[List[Tuple[int, int]]] = None,
        height: int = 512,
        width: int = 512,
        filter_validity_mask: bool = True,
        only_burnt: bool = True,
    ):
        self.data_dir = dataset_path
        self.data_type = data_type
        self.crop_size = crop_size
        self.split = split
        self.transforms = transforms

        # Channel config -- identical to legacy
        if data_type == "sen2":
            self.channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            self.pre_channels = 12
        elif data_type == "rgb":
            self.channel_indices = [3, 2, 1]
            self.pre_channels = 3
        elif data_type == "s2_s1":
            # Pre-S2 (12ch) + Post-S1 (2ch: VV, VH)
            self.s2_channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            self.s1_channel_indices = [0, 1]  # VV, VH only
            self.pre_channels = 12
        elif data_type == "s1":
            # Pre-S1 (2ch: VV, VH) + Post-S1 (2ch: VV, VH)
            self.s1_channel_indices = [0, 1]  # VV, VH only
            self.pre_channels = 2
        else:
            raise ValueError(self.data_type)

        split_fold_list = split_libs[split]

        csv_path = join(dataset_path, "satellite_data.csv")
        df = pd.read_csv(csv_path)

        self.activation_dates = {}
        for _, row in df.iterrows():
            folder_val = row.get('folder', None)
            activation_val = row.get('activation_date', None)
            if folder_val and pd.notna(activation_val) and str(activation_val).strip():
                self.activation_dates[str(folder_val)] = str(activation_val).strip()

        self.folder_list = []
        for fold, folder in zip(df['fold'], df['folder']):
            if fold in split_fold_list:
                self.folder_list.append(folder)

        if mask_intervals is None:
            mask_intervals = [(0, 36), (37, 255)]
        self.mask_intervals = mask_intervals
        self.height = height
        self.width = width

        self.filter_validity_mask = filter_validity_mask
        self.only_burnt = only_burnt

        # Load and process all data
        self.images = []
        self.masks = []
        self._load_all_data()

    # ------------------------------------------------------------------
    # Folder scanning helpers
    # ------------------------------------------------------------------

    def _extract_date(self, filename: str) -> Optional[str]:
        """
        Extract date from filename in format: product_YYYY-MM-DD.tiff
        Replicates scanner.py:42-57
        """
        pattern = r'.+_([0-9]{4}-[0-9]{2}-[0-9]{2})\.tiff'
        match = re.search(pattern, filename, re.IGNORECASE)
        return match.group(1) if match else None

    def _scan_folder(self, folder_path: str) -> Dict[str, str]:
        """
        Scan folder to find pre-fire and post-fire sentinel2 files.
        Replicates scanner.py:73-133

        Returns:
            dict with 'pre' and 'post' keys containing dates
        """
        sentinel2_files = []

        for filename in os.listdir(folder_path):
            if filename.startswith('sentinel2') and filename.endswith('.tiff'):
                date = self._extract_date(filename)
                if date:
                    sentinel2_files.append(date)

        result = {'pre': '', 'post': ''}

        if len(sentinel2_files) > 1:
            # Multiple files: min=pre, max=post (scanner.py:113-115)
            result['pre'] = min(sentinel2_files)
            result['post'] = max(sentinel2_files)
        elif len(sentinel2_files) == 1:
            # Single file: use activation_date if available (scanner.py:116-127)
            date = sentinel2_files[0]
            folder_name = basename(folder_path)

            if folder_name in self.activation_dates:
                activation_date = self._convert_date(self.activation_dates[folder_name])
                if date > activation_date:
                    result['post'] = date
                else:
                    result['pre'] = date
            else:
                # No activation date: assume it's post-fire
                result['post'] = date

        return result

    def _scan_folder_s1(self, folder_path: str) -> Dict[str, str]:
        """
        Scan folder to find pre-fire and post-fire sentinel1 files.
        Logic mirrors _scan_folder but for sentinel1 files.

        Returns:
            dict with 'pre' and 'post' keys containing dates (str)
        """
        sentinel1_files = []

        for filename in os.listdir(folder_path):
            if filename.startswith('sentinel1') and filename.endswith('.tiff'):
                date = self._extract_date(filename)
                if date:
                    sentinel1_files.append(date)

        result = {'pre': '', 'post': ''}

        if len(sentinel1_files) > 1:
            result['pre'] = min(sentinel1_files)
            result['post'] = max(sentinel1_files)
        elif len(sentinel1_files) == 1:
            date = sentinel1_files[0]
            folder_name = basename(folder_path)

            if folder_name in self.activation_dates:
                activation_date = self._convert_date(self.activation_dates[folder_name])
                if date > activation_date:
                    result['post'] = date
                else:
                    result['pre'] = date
            else:
                result['post'] = date

        return result

    @staticmethod
    def _convert_date(date_str: str) -> str:
        """
        Convert date from dd/mm/yyyy to yyyy-mm-dd format.
        Replicates scanner.py:60-71
        """
        if '/' in date_str:
            parts = date_str.split('/')
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return date_str

    # ------------------------------------------------------------------
    # Loading helpers -- identical to legacy
    # ------------------------------------------------------------------

    def _load_sentinel2(self, folder_path: str) -> Optional[np.ndarray]:
        """
        Load sentinel2 image for specified mode (pre or post).
        Replicates scanner.py:159-197

        Returns:
            Image array with shape (H, W, 13) or None if not found
        """
        dates = self._scan_folder(folder_path)
        pre_date = dates["pre"]
        post_date = dates["post"]

        pre_filename = f"sentinel2_{pre_date}.tiff"
        pre_filepath = join(folder_path, pre_filename)
        pre_img = io.imread(str(pre_filepath))
        if len(pre_img.shape) == 2:
            pre_img = pre_img[..., np.newaxis]

        # # Step 4: Apply validity filter (dataset.py:99-109)
        # # Note: Modifies image in-place, zeros out invalid pixels in channels 0-11
        pre_img = self._apply_validity_filter(pre_img)
        # # Step 5: Select channels (dataset.py:114-115)
        # # Select channels [0-11], drop channel 12 (validity mask)
        pre_img = self._select_channels(pre_img)

        post_filename = f"sentinel2_{post_date}.tiff"
        post_filepath = join(folder_path, post_filename)
        post_img = io.imread(str(post_filepath))
        if len(post_img.shape) == 2:
            post_img = post_img[..., np.newaxis]
        post_img = self._apply_validity_filter(post_img)
        post_img = self._select_channels(post_img)

        img = np.concatenate((pre_img, post_img), axis=-1)
        return img.astype(np.float32)

    def _load_sentinel1(self, folder_path: str, mode: str = "both") -> Optional[np.ndarray]:
        """
        Load sentinel1 SAR image (VV + VH channels only).

        Args:
            folder_path: Path to event folder
            mode: "both" for pre+post concatenated (H,W,4),
                  "pre" or "post" for single temporal (H,W,2)

        Returns:
            Image array or None if required file not found
        """
        dates = self._scan_folder_s1(folder_path)

        parts = []
        for temporal in (["pre", "post"] if mode == "both" else [mode]):
            date = dates[temporal]
            if not date:
                return None

            filename = f"sentinel1_{date}.tiff"
            filepath = join(folder_path, filename)
            try:
                img = io.imread(str(filepath))
            except Exception as e:
                import warnings
                warnings.warn(f"S1 read error ({filepath}): {e}")
                return None

            if len(img.shape) == 2:
                img = img[..., np.newaxis]

            # Apply validity filter (last channel is validity mask)
            img = self._apply_validity_filter(img)
            # Select VV and VH channels only (indices 0, 1)
            img = img[:, :, self.s1_channel_indices]
            parts.append(img)

        if not parts:
            return None

        img = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        return img.astype(np.float32)

    def _load_s2_s1(self, folder_path: str) -> Optional[np.ndarray]:
        """
        Load pre-fire S2 (12ch) + post-fire S1 (2ch: VV, VH) = 14ch.

        Returns:
            Image array with shape (H, W, 14) or None
        """
        # Load pre-fire S2
        dates_s2 = self._scan_folder(folder_path)
        pre_date_s2 = dates_s2["pre"]
        if not pre_date_s2:
            return None

        pre_filename = f"sentinel2_{pre_date_s2}.tiff"
        pre_filepath = join(folder_path, pre_filename)
        pre_img = io.imread(str(pre_filepath))
        if len(pre_img.shape) == 2:
            pre_img = pre_img[..., np.newaxis]
        pre_img = self._apply_validity_filter(pre_img)
        pre_img = pre_img[:, :, self.s2_channel_indices]  # 12ch

        # Load post-fire S1
        dates_s1 = self._scan_folder_s1(folder_path)
        post_date_s1 = dates_s1["post"]
        if not post_date_s1:
            return None

        post_filename = f"sentinel1_{post_date_s1}.tiff"
        post_filepath = join(folder_path, post_filename)
        try:
            post_img = io.imread(str(post_filepath))
        except Exception as e:
            import warnings
            warnings.warn(f"S1 read error ({post_filepath}): {e}")
            return None

        if len(post_img.shape) == 2:
            post_img = post_img[..., np.newaxis]
        post_img = self._apply_validity_filter(post_img)
        post_img = post_img[:, :, self.s1_channel_indices]  # 2ch (VV, VH)

        # Shape check before concatenate
        if pre_img.shape[:2] != post_img.shape[:2]:
            import warnings
            warnings.warn(
                f"Shape mismatch S2/S1: {pre_img.shape} vs {post_img.shape} in {folder_path}"
            )
            return None

        img = np.concatenate((pre_img, post_img), axis=-1)  # 14ch
        return img.astype(np.float32)

    def _load_mask(self, folder_path: str) -> np.ndarray:
        """
        Load and discretize mask file.
        Replicates scanner.py:199-236

        Returns:
            Discretized mask with shape (H, W, 1), values in {0, 1, ...}
        """
        folder_name = basename(folder_path)
        mask_filename = f"{folder_name}_mask.tiff"
        mask_path = join(folder_path, mask_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Read raw mask (scanner.py:220)
        raw_mask = io.imread(str(mask_path))

        # Apply mask discretization (scanner.py:221-228)
        result = np.zeros_like(raw_mask, dtype=np.float32)

        for idx, (lower, upper) in enumerate(self.mask_intervals):
            # Closed interval matching [lower, upper]
            bin_mask = (raw_mask >= lower) & (raw_mask <= upper)
            result[bin_mask] = idx

        # Ensure 3D shape (H, W, 1)
        if len(result.shape) == 2:
            result = result[..., np.newaxis]

        return result

    def _apply_validity_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply validity mask filtering: zero out pixels where validity==0.
        Replicates dataset.py:99-109

        The last channel (index -1) is treated as validity mask.
        Pixels where validity==0 have their other channel values set to 0.

        Args:
            image: Shape (H, W, C) where C includes validity as last channel

        Returns:
            Filtered image (modified in-place, but returned for clarity)
        """
        if not self.filter_validity_mask:
            return image

        # Extract validity mask (last channel)
        validity_mask = image[:, :, -1]

        # Find invalid positions (validity == 0)
        invalid_positions = (validity_mask == 0)

        # Zero out all channels except validity for invalid positions
        # dataset.py:108-109: img[:, :, :-1][bool_mask] = 0
        image[:, :, :-1][invalid_positions] = 0

        return image

    def _select_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Select specified channels from image.
        Replicates image_processor.py:249-258

        For otsu_baseline.py config: selects channels [0-11], drops channel 12 (validity)

        Args:
            image: Shape (H, W, 13)

        Returns:
            Selected channels, shape (H, W, len(channel_indices))
        """
        return image[:, :, self.channel_indices]

    def _upscale_if_needed(self, image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Upscale image if it's smaller than target dimensions.
        Replicates image_processor.py:323-346

        Args:
            image: Input image
            target_h: Minimum height
            target_w: Minimum width

        Returns:
            Upscaled image if needed, otherwise original
        """
        h, w = image.shape[:2]

        # Compute scale factors (image_processor.py:337-338)
        rescale_height = target_h / h
        rescale_width = target_w / w

        # Check if upscaling is needed (image_processor.py:340-341)
        if rescale_height <= 1 and rescale_width <= 1:
            return image

        # Upscale with max scale factor (image_processor.py:343)
        scale = max(rescale_height, rescale_width)

        # Use skimage rescale (original uses rescale, not resize!)
        # Note: rescale takes scale factor, not target dimensions
        result = rescale(image, scale, preserve_range=True, channel_axis=2).astype(image.dtype)
        return result

    def _cut_into_tiles(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cut image and mask into 512x512 tiles with overlap handling.
        Replicates image_processor.py:348-396

        If image dimensions are not multiples of 512, last tiles overlap with previous ones.

        Args:
            image: Shape (H, W, C)
            mask: Shape (H, W, 1)

        Returns:
            (tiles_image, tiles_mask) with shapes (N, 512, 512, C) and (N, 512, 512, 1)
        """
        # Ensure minimum size (upscale if needed)
        image = self._upscale_if_needed(image, self.height, self.width)
        mask = self._upscale_if_needed(mask, self.height, self.width)

        # Round mask values after potential upscaling
        mask = np.rint(mask).astype(np.float32)

        h, w = image.shape[:2]

        # Calculate number of tiles (image_processor.py:364-365)
        max_i = math.ceil(h / self.height)
        max_j = math.ceil(w / self.width)

        image_tiles = []
        mask_tiles = []

        # Cut tiles (image_processor.py:368-376)
        for i in range(max_i):
            for j in range(max_j):
                # Handle overlap at boundaries (image_processor.py:370-371)
                # Note: Original uses slice(min(h*i, H-h), min(h*(i+1), H))
                # This ensures v_end - v_start always equals self.height
                v_start = min(self.height * i, h - self.height)
                v_end = v_start + self.height  # Changed: always self.height pixels
                h_start = min(self.width * j, w - self.width)
                h_end = h_start + self.width   # Changed: always self.width pixels

                img_tile = image[v_start:v_end, h_start:h_end, :]
                mask_tile = mask[v_start:v_end, h_start:h_end, :]

                # Verify tile dimensions
                assert img_tile.shape[0] == self.height and img_tile.shape[1] == self.width
                assert mask_tile.shape[0] == self.height and mask_tile.shape[1] == self.width

                image_tiles.append(img_tile)
                mask_tiles.append(mask_tile)

        return np.array(image_tiles), np.array(mask_tiles)

    def _filter_tiles_by_burnt(
        self,
        image_tiles: np.ndarray,
        mask_tiles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out tiles that don't contain any burned pixels.
        Replicates dataset.py:137-144

        Args:
            image_tiles: Shape (N, H, W, C)
            mask_tiles: Shape (N, H, W, 1)

        Returns:
            Filtered (image_tiles, mask_tiles) with only tiles containing burned pixels
        """
        if not self.only_burnt:
            return image_tiles, mask_tiles

        # Find tiles with at least one burned pixel (mask > 0)
        # dataset.py:141: valid_cut = (tmp_mask > 0).any(axis=(1, 2))
        has_burnt = (mask_tiles > 0).any(axis=(1, 2, 3))

        # Get indices of valid tiles
        valid_indices = np.where(has_burnt)[0]

        if len(valid_indices) == 0:
            return np.array([]), np.array([])

        # Filter tiles (dataset.py:143-144)
        return image_tiles[valid_indices], mask_tiles[valid_indices]

    def _load_all_data(self):
        """
        Load and process all data from folder list.
        Replicates dataset.py:86-155

        Processing pipeline:
        1. For each event folder:
           a. Load post-fire sentinel2 image (13 channels)
           b. Load and discretize mask
           c. [Image-level filtering] Skip if no burned pixels (only_burnt)
           d. Apply validity mask filtering
           e. Select channels (0-11, drop validity)
           f. Cut into 512x512 tiles
           g. [Tile-level filtering] Remove tiles without burned pixels (only_burnt)
        2. Concatenate all tiles from all events
        """
        all_images = []
        all_masks = []

        for folder_name in self.folder_list:
            folder_path = join(self.data_dir, folder_name)

            if not os.path.isdir(folder_path) or not os.path.exists(folder_path):
                continue

            print(f"Processing folder: {folder_name}")

            # Step 1: Load image based on data_type
            if self.data_type in ("sen2", "rgb"):
                image = self._load_sentinel2(folder_path)
            elif self.data_type == "s2_s1":
                image = self._load_s2_s1(folder_path)
            elif self.data_type == "s1":
                image = self._load_sentinel1(folder_path, mode="both")
            else:
                raise ValueError(f"Unknown data_type: {self.data_type}")

            if image is None:
                print(f"  Skipping {folder_name}: No image for data_type={self.data_type}")
                continue

            # Step 2: Load mask (dataset.py:88)
            try:
                mask = self._load_mask(folder_path)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            # Verify dimensions match
            if image.shape[:2] != mask.shape[:2]:
                print(f"  Skipping: Image and mask dimensions don't match")
                continue

            # Step 3: Image-level filtering (dataset.py:92-97)
            if self.only_burnt:
                if not (mask >= 1).any():
                    print(f"  Skipping: No burned pixels in entire image")
                    continue

            # Now image has shape (H, W, 12x2) and mask has shape (H, W, 1)

            # Step 6: Cut into tiles (dataset.py:131-134)
            image_tiles, mask_tiles = self._cut_into_tiles(image, mask)

            if len(image_tiles) == 0:
                print(f"  Skipping: No tiles generated")
                continue

            # Step 7: Tile-level filtering (dataset.py:137-144)
            image_tiles, mask_tiles = self._filter_tiles_by_burnt(image_tiles, mask_tiles)

            if len(image_tiles) == 0:
                print(f"  Skipping: No tiles with burned pixels")
                continue

            print(f"  Added {len(image_tiles)} tiles")

            # Step 8: Accumulate tiles (dataset.py:146-147)
            all_images.append(image_tiles)
            all_masks.append(mask_tiles)

        # Step 9: Concatenate all tiles (dataset.py:151-152)
        if len(all_images) > 0:
            self.images = np.concatenate(all_images, axis=0)
            self.masks = np.concatenate(all_masks, axis=0)
            print(f"\nTotal tiles loaded: {len(self.images)}")
        else:
            self.images = np.array([])
            self.masks = np.array([])
            print("\nNo valid tiles found!")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of tiles in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int):
        """Return a single tile as (pre_CHW, post_CHW, label_HW, sample_id).

        The concatenated HWC image is split into pre/post based on
        ``self.pre_channels``, transforms are applied with dual-image
        support, then both images are transposed to CHW numpy float32.
        """
        pixel_values = self.images[idx]  # (H, W, total_channels)
        labels = self.masks[idx]         # (H, W, 1)

        # Squeeze mask to (H, W)
        if len(labels.shape) == 3 and labels.shape[2] == 1:
            labels = labels[..., 0]
        elif len(labels.shape) != 2:
            raise RuntimeError("")

        # Split concatenated channels into pre and post
        pre = pixel_values[:, :, :self.pre_channels]   # (H, W, pre_ch)
        post = pixel_values[:, :, self.pre_channels:]   # (H, W, post_ch)

        mask = labels.astype(np.int64)

        # Apply transforms (CD dual-image)
        if self.transforms is not None:
            blob = self.transforms(image=pre, image_post=post, mask=mask)
            pre = blob["image"]
            post = blob["image_post"]
            mask = blob["mask"]

        # HWC -> CHW, ensure float32
        pre = np.ascontiguousarray(pre.transpose(2, 0, 1)).astype(np.float32)
        post = np.ascontiguousarray(post.transpose(2, 0, 1)).astype(np.float32)
        mask = mask.astype(np.int64)

        sample_id = f"satellite_burned_area_{idx}"
        return pre, post, mask, sample_id
