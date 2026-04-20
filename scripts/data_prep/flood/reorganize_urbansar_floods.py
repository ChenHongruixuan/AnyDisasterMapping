#!/usr/bin/env python3
"""Reorganise Urban SAR flood datasets into a standard structure.

This utility merges the training/validation subsets, normalises file names,
updates split definition files, and optionally tiles the provided testing
scenes into 256x256 patches while dropping tile windows dominated by NaNs.

Example usage (from the repository root):
    python scripts/data_prep/flood/reorganize_urbansar_floods.py

Use --help for the complete list of options.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


TRAINING_SUFFIX = "_GT"
SAR_SUFFIX = "_SAR"
TIF_SUFFIX = ".tif"
DEFAULT_TEST_SPLITS = {
    "20210727_Weihui": "test_weihui.txt",
    "20230609_NovaKakhovka": "test_nova.txt",
    "20231201_Jubba_1": "test_jubba.txt",
    "20231201_Jubba_2": "test_jubba.txt",
}
DEFAULT_RAW_ROOT = Path("data/flood/UrbanSARFloods/raw")
DEFAULT_SOURCE_ROOT = DEFAULT_RAW_ROOT / "urban_sar_floods"
DEFAULT_TEST_ROOT = DEFAULT_RAW_ROOT / "testing_case_orig"
DEFAULT_DEST_ROOT = Path("data/flood/UrbanSARFloods")


@dataclass(frozen=True)
class Args:
    source_root: Path
    test_root: Path
    dest_root: Path
    patch_size: int
    stride: int
    nan_threshold: float
    move_files: bool
    overwrite: bool
    skip_train: bool
    skip_test: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Organise Urban SAR Floods data into a unified layout.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Path to the original training/validation dataset root.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help="Path to the original testing dataset root.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help="Destination root directory for the reorganised dataset.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Tile width/height for test scenes.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride in pixels used when tiling test scenes.",
    )
    parser.add_argument(
        "--nan-threshold",
        type=float,
        default=0.5,
        help="Discard patches whose all-channel-NaN pixel ratio exceeds this value.",
    )
    parser.add_argument(
        "--move",
        dest="move_files",
        action="store_true",
        help="Move files instead of copying them when merging train/val data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination directory.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip merging the training/validation datasets.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip tiling the testing datasets.",
    )

    parsed = parser.parse_args()
    return Args(
        source_root=parsed.source_root.resolve(),
        test_root=parsed.test_root.resolve(),
        dest_root=parsed.dest_root.resolve(),
        patch_size=parsed.patch_size,
        stride=parsed.stride,
        nan_threshold=parsed.nan_threshold,
        move_files=parsed.move_files,
        overwrite=parsed.overwrite,
        skip_train=parsed.skip_train,
        skip_test=parsed.skip_test,
    )


def strip_suffix(text: str, suffix: str) -> str:
    if not text.endswith(suffix):
        raise ValueError(f"Expected '{text}' to end with '{suffix}'.")
    return text[: -len(suffix)]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, move: bool, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            if dst.is_dir():
                raise IsADirectoryError(f"Destination {dst} is a directory.")
            dst.unlink()
        else:
            raise FileExistsError(f"Destination file already exists: {dst}")
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def find_data_subfolders(source_root: Path) -> List[Path]:
    folders = [
        p
        for p in source_root.iterdir()
        if p.is_dir() and (p / "GT").is_dir() and (p / "SAR").is_dir()
    ]
    if not folders:
        raise FileNotFoundError(
            f"No subfolders with GT/SAR found under {source_root}."
        )
    return sorted(folders)


def reorganise_train_validation(args: Args) -> Sequence[str]:
    dest_gt = args.dest_root / "GT"
    dest_sar = args.dest_root / "SAR"
    ensure_directory(dest_gt)
    ensure_directory(dest_sar)

    seen_samples = []
    for subset_dir in find_data_subfolders(args.source_root):
        logging.info("Processing subset %s", subset_dir.name)
        gt_files = sorted((subset_dir / "GT").glob(f"*{TRAINING_SUFFIX}{TIF_SUFFIX}"))
        for gt_path in gt_files:
            sample_stem = strip_suffix(gt_path.stem, TRAINING_SUFFIX)
            sar_path = (subset_dir / "SAR" / f"{sample_stem}{SAR_SUFFIX}{TIF_SUFFIX}")
            if not sar_path.exists():
                raise FileNotFoundError(
                    f"Missing SAR counterpart for {gt_path.name} in {sar_path.parent}."
                )
            dst_name = f"{sample_stem}{TIF_SUFFIX}"
            copy_file(gt_path, dest_gt / dst_name, move=args.move_files, overwrite=args.overwrite)
            copy_file(sar_path, dest_sar / dst_name, move=args.move_files, overwrite=args.overwrite)
            seen_samples.append(sample_stem)
    return list(dict.fromkeys(seen_samples))


def load_split_file(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            stem = Path(line).stem
            try:
                normalised = strip_suffix(stem, TRAINING_SUFFIX)
            except ValueError:
                try:
                    normalised = strip_suffix(stem, SAR_SUFFIX)
                except ValueError:
                    normalised = stem
            names.append(normalised)
    return names


def write_split_file(path: Path, names: Iterable[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Split file already exists: {path}")
    ordered_unique = list(dict.fromkeys(names))
    with path.open("w", encoding="utf-8") as handle:
        if ordered_unique:
            handle.write("\n".join(ordered_unique) + "\n")
        else:
            handle.write("")


def update_train_validation_splits(args: Args, available: Sequence[str]) -> None:
    src_train = args.source_root / "Train_dataset.txt"
    src_valid = args.source_root / "Valid_dataset.txt"

    train_names = load_split_file(src_train)
    valid_names = load_split_file(src_valid)

    missing_train = sorted(set(train_names) - set(available))
    missing_valid = sorted(set(valid_names) - set(available))
    if missing_train:
        logging.warning("%d training samples were not found in merged data: %s", len(missing_train), ", ".join(missing_train[:5]))
    if missing_valid:
        logging.warning("%d validation samples were not found in merged data: %s", len(missing_valid), ", ".join(missing_valid[:5]))

    write_split_file(args.dest_root / "train.txt", train_names, overwrite=args.overwrite)
    write_split_file(args.dest_root / "validation.txt", valid_names, overwrite=args.overwrite)


def find_unique_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}.")
    if len(matches) > 1:
        raise FileExistsError(
            f"Expected a single file matching {pattern} in {directory}, found {len(matches)}."
        )
    return matches[0]


def tile_test_regions(args: Args) -> None:
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Tiling test regions requires 'rasterio' and 'numpy' to be installed."
        ) from exc

    dest_gt = args.dest_root / "GT"
    dest_sar = args.dest_root / "SAR"
    ensure_directory(dest_gt)
    ensure_directory(dest_sar)

    split_entries = defaultdict(list)
    for region_dir in sorted(p for p in args.test_root.iterdir() if p.is_dir()):
        if region_dir.name not in DEFAULT_TEST_SPLITS:
            logging.info("Skipping unrecognised test region %s", region_dir.name)
            continue
        split_file = DEFAULT_TEST_SPLITS[region_dir.name]
        region_prefix = region_dir.name
        logging.info("Tiling test region %s", region_prefix)

        gt_path = find_unique_file(region_dir, f"*{TRAINING_SUFFIX}{TIF_SUFFIX}")
        sar_path = find_unique_file(region_dir, f"*{SAR_SUFFIX}{TIF_SUFFIX}")

        with rasterio.open(gt_path) as gt_ds, rasterio.open(sar_path) as sar_ds:
            if gt_ds.width != sar_ds.width or gt_ds.height != sar_ds.height:
                raise ValueError(
                    f"Mismatched raster dimensions for region {region_prefix}: "
                    f"GT {gt_ds.width}x{gt_ds.height} vs SAR {sar_ds.width}x{sar_ds.height}."
                )
            height, width = sar_ds.height, sar_ds.width
            generated = 0
            for row in range(0, height - args.patch_size + 1, args.stride):
                for col in range(0, width - args.patch_size + 1, args.stride):
                    if row + args.patch_size > height or col + args.patch_size > width:
                        continue
                    window = Window(col_off=col, row_off=row, width=args.patch_size, height=args.patch_size)
                    sar_patch = sar_ds.read(window=window)

                    if np.issubdtype(sar_patch.dtype, np.floating):
                        nan_mask = np.all(np.isnan(sar_patch), axis=0)
                        nan_ratio = float(nan_mask.sum()) / nan_mask.size
                        if nan_ratio > args.nan_threshold:
                            continue
                    row_idx = row // args.stride
                    col_idx = col // args.stride
                    patch_name = f"{region_prefix}_ID_{row_idx}_{col_idx}"
                    patch_filename = f"{patch_name}{TIF_SUFFIX}"

                    dst_sar_path = dest_sar / patch_filename
                    dst_gt_path = dest_gt / patch_filename

                    if dst_sar_path.exists() or dst_gt_path.exists():
                        if args.overwrite:
                            if dst_sar_path.exists():
                                dst_sar_path.unlink()
                            if dst_gt_path.exists():
                                dst_gt_path.unlink()
                        else:
                            raise FileExistsError(
                                f"Destination patch already exists: {dst_sar_path}"
                            )

                    sar_profile = sar_ds.profile.copy()
                    sar_profile.update({
                        "height": args.patch_size,
                        "width": args.patch_size,
                        "transform": sar_ds.window_transform(window),
                    })
                    with rasterio.open(dst_sar_path, "w", **sar_profile) as dst:
                        dst.write(sar_patch)

                    gt_patch = gt_ds.read(window=window)
                    gt_profile = gt_ds.profile.copy()
                    gt_profile.update({
                        "height": args.patch_size,
                        "width": args.patch_size,
                        "transform": gt_ds.window_transform(window),
                    })
                    with rasterio.open(dst_gt_path, "w", **gt_profile) as dst:
                        dst.write(gt_patch)

                    split_entries[split_file].append(patch_name)
                    generated += 1
            logging.info("Generated %d patches for %s", generated, region_prefix)

    for split_name in sorted(set(DEFAULT_TEST_SPLITS.values())):
        items = split_entries.get(split_name, [])
        write_split_file(args.dest_root / split_name, items, overwrite=args.overwrite)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ensure_directory(args.dest_root)
    ensure_directory(args.dest_root / "GT")
    ensure_directory(args.dest_root / "SAR")

    merged_samples: Sequence[str] = []
    if not args.skip_train:
        merged_samples = reorganise_train_validation(args)
        update_train_validation_splits(args, merged_samples)

    if not args.skip_test:
        tile_test_regions(args)

    logging.info("Dataset reorganisation completed. Output saved to %s", args.dest_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface any unexpected failure
        logging.error("%s", exc)
        sys.exit(1)
