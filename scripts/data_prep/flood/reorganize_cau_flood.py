#!/usr/bin/env python3
"""
Reorganize the CAU_Flood dataset into a flat directory with modality folders
and generate train/validation/test splits.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MODALITY_MAP = {
    "flood_vv": "GT",
    "opt": "PRE",
    "vv": "POST",
}
DEFAULT_INPUT_ROOT = Path("data/flood/CAU_Flood/raw")
DEFAULT_OUTPUT_DIR = Path("data/flood/CAU_Flood")


class DatasetError(RuntimeError):
    """Raised when the dataset validation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize CAU_Flood dataset and create split lists."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Path to the directory that contains train/ and test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the reorganized dataset.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train",
        help="Name of the train split directory located under input-root.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="test",
        help="Name of the test split directory located under input-root.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio to split from the original train set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split.",
    )
    return parser.parse_args()


def ensure_png_only(files: Iterable[Path], *, context: str) -> None:
    for path in files:
        if path.suffix.lower() != ".png":
            raise DatasetError(f"{context}: found non-PNG file {path}")


def list_png_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise DatasetError(f"Expected directory missing: {directory}")
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    png_files = [p for p in files if p.suffix.lower() == ".png"]
    skipped = [p for p in files if p.suffix.lower() != ".png"]
    if skipped:
        print(
            f"  Skipping {len(skipped)} non-PNG files in {directory}", file=sys.stderr
        )
    if not png_files:
        raise DatasetError(f"No PNG files found under {directory}.")
    return png_files


def collect_samples(split_root: Path) -> Dict[str, List[Path]]:
    modality_files: Dict[str, List[Path]] = {}
    name_sets = []

    for modality, target_folder in MODALITY_MAP.items():
        modality_dir = split_root / modality
        files = list_png_files(modality_dir)
        modality_files[modality] = files
        name_sets.append({path.name for path in files})

    first_set = name_sets[0] if name_sets else set()
    for idx, names in enumerate(name_sets[1:], start=1):
        if names != first_set:
            raise DatasetError(
                f"Inconsistent file names across modalities under {split_root}. "
                f"Mismatch at modality #{idx + 1}."
            )

    return modality_files


def split_train_validation(
    sample_names: List[str], val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("Validation ratio must satisfy 0 ≤ ratio < 1.")

    names = sample_names[:]
    rng = random.Random(seed)
    rng.shuffle(names)

    total = len(names)
    if total <= 1:
        return names, []

    tentative_val = max(1, int(round(total * val_ratio)))
    val_count = min(tentative_val, total - 1)

    val_names = sorted(names[:val_count])
    train_names = sorted(names[val_count:])
    return train_names, val_names


def prepare_output_dirs(output_root: Path) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    modality_dirs: Dict[str, Path] = {}
    for modality, target in MODALITY_MAP.items():
        dest_dir = output_root / target
        dest_dir.mkdir(parents=True, exist_ok=True)
        if any(dest_dir.iterdir()):
            raise DatasetError(f"Output directory is not empty: {dest_dir}")
        modality_dirs[modality] = dest_dir
    return modality_dirs


def copy_samples(
    all_samples: Dict[str, Dict[str, List[Path]]],
    dest_dirs: Dict[str, Path],
) -> None:
    for modality, files_per_split in all_samples.items():
        dest_dir = dest_dirs[modality]
        for files in files_per_split.values():
            for src_path in files:
                dest_path = dest_dir / src_path.name
                if dest_path.exists():
                    raise DatasetError(
                        f"Duplicate sample detected for {dest_path.name}; "
                        "files from different splits share the same name."
                    )
                shutil.copy2(src_path, dest_path)


def verify_output_consistency(dest_dirs: Dict[str, Path]) -> None:
    all_name_sets = []
    expected_count = None

    for modality, dest_dir in dest_dirs.items():
        files = sorted(p for p in dest_dir.iterdir() if p.is_file())
        ensure_png_only(files, context=str(dest_dir))
        names = {p.name for p in files}
        if expected_count is None:
            expected_count = len(names)
        elif len(names) != expected_count:
            raise DatasetError(
                f"Output directories have different file counts; mismatch at {dest_dir}."
            )
        all_name_sets.append(names)

    reference = all_name_sets[0] if all_name_sets else set()
    for idx, names in enumerate(all_name_sets[1:], start=1):
        if names != reference:
            raise DatasetError(
                f"Output modality directories do not contain identical file sets; "
                f"mismatch at index {idx}."
            )


def write_split_file(path: Path, names: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{name}\n")


def main() -> int:
    args = parse_args()

    input_root: Path = args.input_root.resolve()
    output_root: Path = args.output_dir.resolve()

    splits = {
        "train": input_root / args.train_dir,
        "test": input_root / args.test_dir,
    }

    for split_name, split_path in splits.items():
        if not split_path.is_dir():
            raise DatasetError(f"Missing split directory: {split_path}")

    print(f"Input root: {input_root}")
    print(f"Output directory: {output_root}")

    # Gather and validate samples for each split.
    split_samples: Dict[str, Dict[str, List[Path]]] = {}
    for split_name, split_path in splits.items():
        print(f"Validating split: {split_name}")
        split_samples[split_name] = collect_samples(split_path)
        sample_count = len(next(iter(split_samples[split_name].values()), []))
        print(f"  Found {sample_count} samples.")

    # Prepare destination directories.
    print("Creating output directories...")
    dest_dirs = prepare_output_dirs(output_root)

    # Copy all samples.
    print("Copying samples...")
    modality_to_files: Dict[str, Dict[str, List[Path]]] = {
        modality: {
            split: split_samples[split][modality]
            for split in split_samples
        }
        for modality in MODALITY_MAP
    }
    copy_samples(modality_to_files, dest_dirs)

    # Verify copied files.
    print("Verifying output consistency...")
    verify_output_consistency(dest_dirs)

    # Generate split name lists.
    train_names = sorted(
        path.stem for path in split_samples["train"]["flood_vv"]
    )
    test_names = sorted(
        path.stem for path in split_samples["test"]["flood_vv"]
    )
    train_list, val_list = split_train_validation(
        train_names, args.val_ratio, args.seed
    )

    print(f"Writing split files to {output_root}...")
    write_split_file(output_root / "train.txt", train_list)
    write_split_file(output_root / "validation.txt", val_list)
    write_split_file(output_root / "test.txt", test_names)

    print("All done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DatasetError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
