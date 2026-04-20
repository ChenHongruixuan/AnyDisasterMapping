"""Augmentation utilities for handling single or multiple images with a shared label."""

import random
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _ensure_numpy(arr):
    """Return the input as a NumPy array without copying when possible."""
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def normalize_img(
    img: np.ndarray,
    mean: Sequence[float] = (123.675, 116.28, 103.53),
    std: Sequence[float] = (58.395, 57.12, 57.375),
) -> np.ndarray:
    """Apply channel-wise normalization using the provided mean and std statistics."""
    img_array = _ensure_numpy(img).astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    channels = img_array.shape[-1]
    normalized = np.empty_like(img_array, dtype=np.float32)
    for channel in range(channels):
        mean_value = mean[channel] if channel < len(mean) else mean[-1]
        std_value = std[channel] if channel < len(std) else std[-1]
        normalized[..., channel] = (img_array[..., channel] - mean_value) / std_value
    return normalized


def _resize_array(arr: np.ndarray, output_size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """Resize a 2-D or 3-D array using OpenCV while preserving the data type range."""
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    h, w = arr.shape[:2]
    new_h, new_w = output_size
    if (h, w) == (new_h, new_w):
        return arr.copy()

    target_size = (new_w, new_h)
    if arr.ndim == 2:
        resized = cv2.resize(arr, target_size, interpolation=interpolation)
    else:
        resized_channels = [
            cv2.resize(arr[..., c], target_size, interpolation=interpolation)
            for c in range(arr.shape[2])
        ]
        resized = np.stack(resized_channels, axis=-1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return np.clip(resized, info.min, info.max).astype(arr.dtype)
    return resized.astype(arr.dtype, copy=False)


def random_scale(
    label: np.ndarray,
    images: Iterable[np.ndarray],
    scales: Sequence[float] = (0.75, 1.0, 1.25),
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Randomly scale all images and the label using one of the provided factors."""
    if not scales:
        raise ValueError("`scales` must contain at least one value")

    scale = random.choice(scales)
    if np.isclose(scale, 1.0):
        return label, [img.copy() for img in images]

    label = _ensure_numpy(label)
    h, w = label.shape[:2]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    output_size = (new_h, new_w)

    scaled_images = [
        _resize_array(_ensure_numpy(img), output_size, interpolation=cv2.INTER_LINEAR)
        for img in images
    ]
    scaled_label = _resize_array(label, output_size, interpolation=cv2.INTER_NEAREST)
    return scaled_label, scaled_images


def random_fliplr(label: np.ndarray, images: Iterable[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Flip all inputs horizontally with a probability of 0.5."""
    if random.random() <= 0.5:
        return label, [img.copy() for img in images]

    flipped_label = np.fliplr(label)
    flipped_images = [np.fliplr(_ensure_numpy(img)) for img in images]
    return flipped_label, flipped_images


def random_flipud(label: np.ndarray, images: Iterable[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Flip all inputs vertically with a probability of 0.5."""
    if random.random() <= 0.5:
        return label, [img.copy() for img in images]

    flipped_label = np.flipud(label)
    flipped_images = [np.flipud(_ensure_numpy(img)) for img in images]
    return flipped_label, flipped_images


def random_rot90(label: np.ndarray, images: Iterable[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Rotate all inputs by a random multiple of 90 degrees."""
    k = random.randrange(4)
    if k == 0:
        return label, [img.copy() for img in images]

    rotated_label = np.rot90(label, k).copy()
    rotated_images = [np.rot90(_ensure_numpy(img), k).copy() for img in images]
    return rotated_label, rotated_images


def random_crop(
    label: np.ndarray,
    images: Iterable[np.ndarray],
    crop_size: int,
    mean_rgb: Sequence[float] = (0.0, 0.0, 0.0),
    ignore_index: int = 255,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Crop all inputs to the requested size, padding beforehand to avoid boundary issues."""
    label = _ensure_numpy(label)
    h, w = label.shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    padded_label = np.ones((H, W), dtype=label.dtype) * ignore_index
    padded_label[:h, :w] = label

    padded_images: List[np.ndarray] = []
    for img in images:
        img = _ensure_numpy(img)
        if img.ndim == 2:
            padded = np.zeros((H, W), dtype=img.dtype)
            padded[:h, :w] = img
        elif img.ndim == 3:
            channels = img.shape[2]
            padded = np.zeros((H, W, channels), dtype=img.dtype)
            for c in range(min(3, channels)):
                padded[..., c] = mean_rgb[c]
            padded[:h, :w, :] = img
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")
        padded_images.append(padded)

    def _sample_window():
        """Sample a crop whose label distribution is not overly dominated by one class."""
        for _ in range(10):
            h_start = random.randrange(0, H - crop_size + 1)
            w_start = random.randrange(0, W - crop_size + 1)
            window = padded_label[h_start:h_start + crop_size, w_start:w_start + crop_size]
            unique, counts = np.unique(window, return_counts=True)
            valid_counts = counts[unique != ignore_index]
            if len(valid_counts) > 0 and np.max(valid_counts) / np.sum(valid_counts) < 0.75:
                return h_start, w_start
        return 0, 0

    h_start, w_start = _sample_window()
    h_end = h_start + crop_size
    w_end = w_start + crop_size

    cropped_label = padded_label[h_start:h_end, w_start:w_end]
    cropped_images = []
    for padded in padded_images:
        if padded.ndim == 2:
            cropped_images.append(padded[h_start:h_end, w_start:w_end])
        else:
            cropped_images.append(padded[h_start:h_end, w_start:w_end, :])

    return cropped_label, cropped_images


def augment_images(
    images: Iterable[np.ndarray],
    label: np.ndarray,
    crop_size: Optional[int],
    augment: bool,
    scales: Sequence[float] = (0.75, 1.0, 1.25),
    mean_rgb: Sequence[float] = (0.0, 0.0, 0.0),
    ignore_index: int = 255,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Apply the full augmentation pipeline and return transformed images plus label."""
    images_list = [img.copy() for img in images]
    label_array = _ensure_numpy(label)

    if augment:
        if crop_size is None:
            raise ValueError('`crop_size` must be provided when augmenting images.')
        # label_array, images_list = random_scale(label_array, images_list, scales=scales)
        label_array, images_list = random_crop(label_array, images_list, crop_size, mean_rgb=mean_rgb, ignore_index=ignore_index)
        label_array, images_list = random_fliplr(label_array, images_list)
        label_array, images_list = random_flipud(label_array, images_list)
        label_array, images_list = random_rot90(label_array, images_list)

    return images_list, label_array


def to_tensor(
    images: Iterable[np.ndarray],
    mean: Sequence[float] = (123.675, 116.28, 103.53),
    std: Sequence[float] = (58.395, 57.12, 57.375),
) -> List[np.ndarray]:
    """Normalize images and convert them to channel-first tensors."""
    tensors: List[np.ndarray] = []
    for img in images:
        normalized = normalize_img(img, mean=mean, std=std)
        if normalized.ndim == 2:
            normalized = np.expand_dims(normalized, axis=-1)
        tensors.append(np.transpose(normalized, (2, 0, 1)))
    return tensors
