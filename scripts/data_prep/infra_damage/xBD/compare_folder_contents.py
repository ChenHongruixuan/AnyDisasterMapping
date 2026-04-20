#!/usr/bin/env python3
"""Compare PNG files across two folders and flag pixel-level differences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


@dataclass
class ComparisonResult:
    name: str
    first_path: Path
    second_path: Path
    reason: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Find PNG files that share a name across two directories but differ in pixel values.'
    )
    parser.add_argument('--first_dir', type=Path, required=True, help='First directory to inspect.')
    parser.add_argument('--second_dir', type=Path, required=True, help='Second directory to inspect.')
    parser.add_argument(
        '--pattern',
        default='*.png',
        help='Glob pattern for files to include (default: *.png).'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search directories recursively. Otherwise only the top level is compared.'
    )
    parser.add_argument(
        '--convert-mode',
        default='RGBA',
        help=(
            "Optional Pillow mode to convert images before comparison (default: RGBA). "
            "Use 'none' to keep the original mode."
        ),
    )
    return parser.parse_args(argv)


def collect_files(directory: Path, pattern: str, recursive: bool) -> Dict[str, Path]:
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f'Directory not found or not a directory: {directory}')

    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)

    files: Dict[str, Path] = {}
    for path in iterator:
        if not path.is_file():
            continue
        files.setdefault(path.name, path)
    return files


def compare_directories(
    first_dir: Path,
    second_dir: Path,
    pattern: str,
    recursive: bool,
    convert_mode: Optional[str],
) -> List[ComparisonResult]:
    first_files = collect_files(first_dir, pattern, recursive)
    second_files = collect_files(second_dir, pattern, recursive)

    print(f'Found {len(first_files)} files in {first_dir} and {len(second_files)} in {second_dir}.')

    mismatches: List[ComparisonResult] = []
    for name, first_path in first_files.items():
        second_path = second_files.get(name)
        if second_path is None:
            continue

        reason = compare_png(first_path, second_path, convert_mode)
        if reason is not None:
            mismatches.append(ComparisonResult(name, first_path, second_path, reason))

    return mismatches


def compare_png(first_path: Path, second_path: Path, convert_mode: Optional[str]) -> Optional[str]:
    img1 = load_image(first_path, convert_mode)
    img2 = load_image(second_path, convert_mode)

    if img1.size != img2.size:
        return f'size mismatch {img1.size} vs {img2.size}'

    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)

    if arr1.shape != arr2.shape:
        return f'array shape mismatch {arr1.shape} vs {arr2.shape}'

    if arr1.ndim == 3:
        diff_mask = np.any(arr1 != arr2, axis=-1)
    else:
        diff_mask = arr1 != arr2

    diff_pixels = int(np.count_nonzero(diff_mask))
    if diff_pixels == 0:
        return None

    total_pixels = arr1.shape[0] * arr1.shape[1]
    return f'pixel mismatch ({diff_pixels} of {total_pixels} pixels differ)'


def load_image(path: Path, convert_mode: Optional[str]) -> Image.Image:
    with Image.open(path) as image:
        if convert_mode:
            image = image.convert(convert_mode)
        else:
            image = image.copy()
    return image


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    convert_mode = None if args.convert_mode.lower() == 'none' else args.convert_mode

    mismatches = compare_directories(
        first_dir=args.first_dir.resolve(),
        second_dir=args.second_dir.resolve(),
        pattern=args.pattern,
        recursive=args.recursive,
        convert_mode=convert_mode,
    )

    if not mismatches:
        print('No mismatched files found.')
        return 0

    print('Found files with identical names but different pixel values:')
    for result in mismatches:
        print(f'- {result.name}\n  {result.first_path}\n  {result.second_path}\n  Reason: {result.reason}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
