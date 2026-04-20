#!/usr/bin/env python3
"""Inspect RescueNet crop folders and normalize all images to the dominant size.

The script scans the requested RescueNet subdirectories (defaults to test_crop and
val_crop), reports the distribution of image sizes, identifies the dominant size,
and resizes the remaining images in-place to match the dominant size.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
Size = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Harmonize RescueNet crop image sizes by resizing to the dominant size.'
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('./data/infra_damage/rescuenet'),
        help='Root directory that contains RescueNet crop folders (default: ./data/infra_damage/rescuenet).',
    )
    parser.add_argument(
        '--subdirs',
        nargs='+',
        default=['test_crop', 'val_crop'],
        help='One or more subdirectories under the root to process (default: test_crop val_crop).',
    )
    parser.add_argument(
        '--target-size',
        type=str,
        default=None,
        help='Force a specific target size given as WIDTHxHEIGHT. If omitted, use the dominant size.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report detected sizes without modifying any files.',
    )
    return parser.parse_args()


def iter_images(directory: Path) -> Iterator[Path]:
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_size_arg(size_arg: str) -> Size:
    if 'x' not in size_arg:
        raise ValueError('Target size must be formatted as WIDTHxHEIGHT (e.g., 1024x1024).')
    width_str, height_str = size_arg.lower().split('x', 1)
    width = int(width_str)
    height = int(height_str)
    if width <= 0 or height <= 0:
        raise ValueError('Target size dimensions must be positive integers.')
    return (width, height)


def collect_sizes(image_paths: Iterable[Path]) -> Tuple[Counter, dict]:
    sizes = Counter()
    records = {}
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                size = image.size  # (width, height)
        except (OSError, ValueError) as exc:
            print(f'[warn] Skipping {path}: failed to read image ({exc}).')
            continue
        sizes[size] += 1
        records[path] = size
    return sizes, records


def choose_dominant_size(size_counters: Iterable[Counter]) -> Optional[Size]:
    aggregate = Counter()
    for counter in size_counters:
        aggregate.update(counter)
    if not aggregate:
        return None
    return aggregate.most_common(1)[0][0]


def preferred_resample(image_mode: str) -> int:
    if image_mode.upper() in {'RGB', 'RGBA', 'CMYK', 'YCbCr'}:
        return Image.BILINEAR
    return Image.NEAREST


def resize_image(path: Path, target_size: Size) -> bool:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.size == target_size:
            return False
        resample = preferred_resample(image.mode)
        resized = image.resize(target_size, resample=resample)
        resized.save(path)
    return True


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    size_counters = []
    image_records = {}

    for subdir in args.subdirs:
        directory = root / subdir
        if not directory.is_dir():
            print(f'[warn] Skipping {directory}: directory does not exist.')
            continue

        images = list(iter_images(directory))
        counter, records = collect_sizes(images)
        size_counters.append(counter)
        image_records[subdir] = records

        total = sum(counter.values())
        print(f'[{subdir}] Found {total} images in {directory}')
        if not counter:
            print('  No readable images discovered.')
            continue
        for (width, height), count in counter.most_common():
            print(f'  {width}x{height}: {count}')

    if not image_records:
        print('No images were processed; aborting.')
        return

    target_size = parse_size_arg(args.target_size) if args.target_size else choose_dominant_size(size_counters)
    if target_size is None:
        print('Unable to determine a dominant size; aborting.')
        return

    print(f'Dominant size selected: {target_size[0]}x{target_size[1]}')

    if args.dry_run:
        print('Dry run requested; no files will be modified.')
        return

    total_resized = 0
    for subdir, records in image_records.items():
        resized_here = 0
        for path, size in records.items():
            if size == target_size:
                continue
            if resize_image(path, target_size):
                resized_here += 1
        total_resized += resized_here
        print(f'[{subdir}] Resized {resized_here} images to {target_size[0]}x{target_size[1]}')

    if total_resized == 0:
        print('All images already matched the dominant size; no files modified.')
    else:
        print(f'Resized {total_resized} images across all directories.')


if __name__ == '__main__':
    main()
