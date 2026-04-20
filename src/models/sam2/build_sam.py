# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
# if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
#     # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
#     # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
#     # This typically happens because the user is running Python from the parent directory
#     # that contains the sam2 repo they cloned.
#     raise RuntimeError(
#         "You're likely running Python from the parent directory of the sam2 repository "
#         "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
#         "This is not supported since the `sam2` Python package could be shadowed by the "
#         "repository name (the repository is also named `sam2` and contains the Python package "
#         "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
#         "rather than its parent dir, or from your home directory) after installing SAM 2."
#     )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def _resolve_config_path(config_file: str) -> Path:
    """Locate the requested Hydra config relative to this module if needed."""
    raw_path = Path(config_file)
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)

    module_root = Path(__file__).resolve().parent
    config_root = module_root / 'configs'
    if not raw_path.is_absolute():
        relative_variants: List[Path] = [raw_path]
        if raw_path.parts and raw_path.parts[0].lower() == 'configs':
            relative_variants.append(Path(*raw_path.parts[1:]))
        for variant in relative_variants:
            candidates.append((module_root / variant).resolve())
            candidates.append((config_root / variant).resolve())

    # Also try matching by filename only inside the canonical configs directory.
    candidates.append((module_root / raw_path.name).resolve())
    candidates.append((config_root / raw_path.name).resolve())

    checked: List[Path] = []
    for candidate in candidates:
        if candidate not in checked and candidate.is_file():
            return candidate
        checked.append(candidate)

    search_paths = '\n  - '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Unable to locate SAM2 config file '{config_file}'. Tried:\n  - {search_paths}"
    )


def _compose_config(config_file: str, overrides: Optional[Sequence[str]] = None) -> DictConfig:
    """Compose a Hydra config from an explicit file path and optional overrides."""
    resolved_path = _resolve_config_path(config_file)
    config_dir = str(resolved_path.parent)
    config_name = resolved_path.stem
    overrides = list(overrides) if overrides is not None else []

    with initialize_config_dir(config_dir=config_dir, job_name="sam2_build", version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def build_sam2(config_file, ckpt_path=None, device="cuda", mode="eval", **kwargs):
    hydra_overrides_extra: Iterable[str] = kwargs.pop('hydra_overrides_extra', []) if kwargs else []

    overrides = list(hydra_overrides_extra)

    if kwargs:
        logging.debug('Unused kwargs in build_sam2: %s', kwargs)
    
    cfg = _compose_config(config_file, overrides=overrides)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = _compose_config(config_file, hydra_overrides)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
