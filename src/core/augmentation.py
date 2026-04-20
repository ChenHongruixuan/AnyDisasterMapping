# src/core/augmentation.py

import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class SmartCrop(DualTransform):
    """
    Smart crop with rejection sampling to avoid all-background or
    class-imbalanced patches.

    Ported from infra_damage imutils.random_crop(), adapted for the
    albumentations Compose framework.  Supports additional_targets
    (pre/post synchronised cropping for change-detection).

    Parameters
    ----------
    crop_size : int
        Output crop height and width.
    cat_max_ratio : float
        Maximum allowed ratio of any single class among valid pixels.
        Crops exceeding this are rejected.
    max_retry : int
        Maximum number of rejection-sampling attempts.
    ignore_index : int
        Value in the label that marks invalid / padding pixels.
        These pixels are excluded from class-ratio computation.
    pad_fill_value : float
        Fill value for image padding regions.
    """

    def __init__(
        self,
        crop_size: int,
        cat_max_ratio: float = 0.75,
        max_retry: int = 50,
        ignore_index: int = 255,
        pad_fill_value: float = 0.0,
        crop_mask_key: str = "mask",
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.max_retry = max_retry
        self.ignore_index = ignore_index
        self.pad_fill_value = pad_fill_value
        self.crop_mask_key = crop_mask_key

    # -- albumentations interface --

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("crop_size", "cat_max_ratio", "max_retry", "ignore_index",
                "pad_fill_value", "crop_mask_key")

    @property
    def targets_as_params(self):
        # Only request the mask key actually used for crop sampling.
        # Requesting non-existent keys (e.g. mask_t1 for CD/Seg) causes
        # albumentations to raise ValueError.
        if self.crop_mask_key == "mask":
            return ["mask"]
        return ["mask", self.crop_mask_key]

    def get_params_dependent_on_data(self, params, data):
        mask = data.get(self.crop_mask_key)
        if mask is None:
            mask = data.get("mask")
        if mask is None:
            return {"h_start": 0, "w_start": 0, "pad_h": 0, "pad_w": 0}

        h, w = mask.shape[:2]
        H = max(self.crop_size, h)
        W = max(self.crop_size, w)

        # random offset for placing original image on the padded canvas
        pad_h = np.random.randint(0, H - h + 1) if H > h else 0
        pad_w = np.random.randint(0, W - w + 1) if W > w else 0

        # build padded mask for sampling decisions
        padded_mask = np.full((H, W), self.ignore_index, dtype=mask.dtype)
        padded_mask[pad_h:pad_h + h, pad_w:pad_w + w] = mask

        # smart sampling
        crop_h, crop_w = self._sample_crop_box(padded_mask, H, W)

        return {"h_start": crop_h, "w_start": crop_w, "pad_h": pad_h, "pad_w": pad_w,
                "canvas_H": H, "canvas_W": W, "orig_h": h, "orig_w": w}

    def _sample_crop_box(self, padded_mask, H, W):
        """Rejection sampling: up to max_retry attempts for a balanced crop."""
        fallback = (0, 0)
        cs = self.crop_size

        for _ in range(self.max_retry):
            h_s = np.random.randint(0, H - cs + 1)
            w_s = np.random.randint(0, W - cs + 1)

            crop = padded_mask[h_s:h_s + cs, w_s:w_s + cs]
            valid = crop[crop != self.ignore_index]

            # check 1: must have valid pixels
            if len(valid) == 0:
                continue
            # check 2: must not be all class 0
            if np.all(valid == 0):
                continue

            fallback = (h_s, w_s)

            # check 3: class balance
            _, counts = np.unique(valid, return_counts=True)
            if counts.max() / counts.sum() < self.cat_max_ratio:
                return h_s, w_s

        return fallback

    # -- actual crop operations --

    def apply(self, img, h_start=0, w_start=0, pad_h=0, pad_w=0,
              canvas_H=0, canvas_W=0, orig_h=0, orig_w=0, **params):
        """Crop image. Pad to canvas size first, then crop."""
        h, w = img.shape[:2]
        H, W = canvas_H, canvas_W
        cs = self.crop_size

        if img.ndim == 2:
            padded = np.full((H, W), self.pad_fill_value, dtype=img.dtype)
        else:
            padded = np.full((H, W, img.shape[2]), self.pad_fill_value, dtype=img.dtype)

        padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
        return padded[h_start:h_start + cs, w_start:w_start + cs]

    def apply_to_mask(self, mask, h_start=0, w_start=0, pad_h=0, pad_w=0,
                      canvas_H=0, canvas_W=0, orig_h=0, orig_w=0, **params):
        """Crop mask. Padding regions are filled with ignore_index."""
        h, w = mask.shape[:2]
        H, W = canvas_H, canvas_W
        cs = self.crop_size

        padded = np.full((H, W), self.ignore_index, dtype=mask.dtype)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = mask
        return padded[h_start:h_start + cs, w_start:w_start + cs]


class _PerBranchNormalize:
    """Wraps an albumentations Compose to apply different normalization per CD branch."""

    def __init__(self, compose, pre_cfg, post_cfg):
        self.compose = compose
        self.pre_norm = A.Normalize(**pre_cfg)
        self.post_norm = A.Normalize(**post_cfg)

    def __call__(self, **kwargs):
        result = self.compose(**kwargs)
        # Apply pre normalize to image
        img = result["image"]
        result["image"] = self.pre_norm(image=img)["image"]
        # Apply post normalize to image_post
        if "image_post" in result:
            post = result["image_post"]
            result["image_post"] = self.post_norm(image=post)["image"]
        return result


def build_transforms(config_dict: dict, task: str = "seg"):
    """
    Build an albumentations pipeline from a YAML config dict.

    config_dict example:
        SmartCrop: {crop_size: 512, cat_max_ratio: 0.75}
        HorizontalFlip: {p: 0.5}
        Normalize: {mean: [...], std: [...]}
    """
    # custom transform registry
    CUSTOM_TRANSFORMS = {
        "SmartCrop": SmartCrop,
    }

    transforms = []
    normalize_cfg = None

    for key, params in config_dict.items():
        # Normalize is handled last
        if key == "Normalize":
            normalize_cfg = params
            continue

        # look up custom transforms first
        if key in CUSTOM_TRANSFORMS:
            transforms.append(CUSTOM_TRANSFORMS[key](**params))
        elif hasattr(A, key):
            transforms.append(getattr(A, key)(**params))
        else:
            raise ValueError(f"Unknown transform: {key}")

    # No ToTensorV2: return numpy, let DataLoader default collate convert to tensor

    from src.tasks import get_task_handler
    targets = get_task_handler(task).augmentation_targets()

    if normalize_cfg:
        if "pre" in normalize_cfg and "post" in normalize_cfg:
            # Per-branch normalization for CD
            compose = A.Compose(transforms, additional_targets=targets) if targets else A.Compose(transforms)
            return _PerBranchNormalize(compose, normalize_cfg["pre"], normalize_cfg["post"])
        else:
            transforms.append(A.Normalize(**normalize_cfg))

    if targets:
        return A.Compose(transforms, additional_targets=targets)
    return A.Compose(transforms)


def apply_cd_transforms(transform, pre_img, post_img, label):
    """
    Apply transform to change-detection data.
    pre_img, post_img: (H, W, C) numpy
    label: (H, W) numpy
    Returns: pre (C, H, W), post (C, H, W), label (H, W) -- all numpy
    """
    result = transform(image=pre_img, image_post=post_img, mask=label)
    pre_out = result["image"].transpose(2, 0, 1)        # (H,W,C) -> (C,H,W)
    post_out = result["image_post"].transpose(2, 0, 1)
    label_out = result["mask"]
    return pre_out, post_out, label_out


def apply_seg_transforms(transform, image, label):
    """
    Apply transform to segmentation data.
    image: (H, W, C) numpy
    label: (H, W) numpy
    Returns: image (C, H, W), label (H, W)
    """
    result = transform(image=image, mask=label)
    image_out = result["image"].transpose(2, 0, 1)
    label_out = result["mask"]
    return image_out, label_out
