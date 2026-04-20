#!/usr/bin/env python3

"""Create a txt file listing cropped xBD tiles with non-empty masks."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Scan cropped xBD tiles, exclude tiles whose masks are entirely zeros, '
            'and write the remaining image names to a txt file.'
        )
    )
    parser.add_argument(
        '--image-dir',
        type=Path,
        default=Path('./data/infra_damage/xBD/train_crop_512/cropped_images'),
        help='Directory containing cropped images. Default: ./data/infra_damage/xBD/train_crop_512/cropped_images'
    )
    parser.add_argument(
        '--mask-dir',
        type=Path,
        default=Path('./data/infra_damage/xBD/train_crop_512/cropped_masks'),
        help='Directory containing the masks corresponding to the cropped images. '
             'Default: ./data/infra_damage/xBD/train_crop_512/cropped_masks'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./data/infra_damage/xBD/train_valid_tiles.txt'),
        help='Destination txt file (default: ./data/infra_damage/xBD/train_valid_tiles.txt)'
    )
    parser.add_argument(
        '--suffix',
        default='.png',
        help='Image file suffix to consider (case-insensitive). Default: .png'
    )
    parser.add_argument(
        '--keep-extension',
        action='store_true',
        help='Write the full filename (including extension). Default writes the stem only.'
    )
    parser.add_argument(
        '--keyword',
        default='post_disaster',
        help='Only keep image names containing this substring (default: post_disaster). Use an empty string to disable.'
    )
    return parser.parse_args(argv)


def is_mask_nonzero(mask_path: Path) -> bool:
    with Image.open(mask_path) as image:
        array = np.asarray(image)
    return np.any(array)


def collect_valid_images(image_dir: Path, mask_dir: Path, suffix: str, keyword: Optional[str]) -> List[Path]:
    suffix = suffix.lower()
    files = []
    for path in sorted(image_dir.glob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() != suffix:
            continue
        if keyword and keyword not in path.name:
            continue
        mask_path = mask_dir / path.name
        if not mask_path.is_file():
            print(f'Warning: mask not found for {path.name}. Skipping.', file=sys.stderr)
            continue
        try:
            if not is_mask_nonzero(mask_path):
                continue
        except Exception as error:
            print(f'Warning: failed to read mask {mask_path}: {error}. Skipping.', file=sys.stderr)
            continue
        files.append(path)
    return files


def write_names(paths: List[Path], output_path: Path, keep_extension: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [path.name if keep_extension else path.stem for path in paths]
    with output_path.open('w', encoding='utf-8') as handle:
        if names:
            handle.write('\n'.join(names))
            handle.write('\n')


def main(argv=None) -> int:
    args = parse_args(argv)

    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    output_path = args.output.resolve()

    if not image_dir.exists():
        print(f'Image directory not found: {image_dir}', file=sys.stderr)
        return 1
    if not image_dir.is_dir():
        print(f'Image directory is not a directory: {image_dir}', file=sys.stderr)
        return 1
    if not mask_dir.exists():
        print(f'Mask directory not found: {mask_dir}', file=sys.stderr)
        return 1
    if not mask_dir.is_dir():
        print(f'Mask directory is not a directory: {mask_dir}', file=sys.stderr)
        return 1

    keyword = args.keyword.strip()
    keyword_filter = keyword if keyword else None

    valid_paths = collect_valid_images(image_dir, mask_dir, args.suffix, keyword_filter)
    if not valid_paths:
        print('No valid tiles found. Output will not be created.', file=sys.stderr)
        return 1

    write_names(valid_paths, output_path, args.keep_extension)
    print(f'Wrote {len(valid_paths)} entries to {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
