#!/usr/bin/env python3
"""Inspect RescueNet crop outputs for missing or malformed tiles."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio

DEFAULT_IMAGE_SUFFIX = '.jpg'
DEFAULT_LABEL_SUFFIX = '_lab.png'


Index = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check RescueNet cropped tiles against the originals.')
    parser.add_argument('--original-dir', type=Path, required=True,
                        help='Original dataset root containing img/ and label/ folders.')
    parser.add_argument('--cropped-dir', type=Path, required=True,
                        help='Directory containing cropped_img/ and cropped_label/.')
    parser.add_argument('--crop-height', type=int, default=750, help='Crop height used during tiling (default: 750).')
    parser.add_argument('--crop-width', type=int, default=1000, help='Crop width used during tiling (default: 1000).')
    parser.add_argument('--image-suffix', type=str, default=DEFAULT_IMAGE_SUFFIX,
                        help='Suffix of original image files (default: .jpg).')
    parser.add_argument('--label-suffix', type=str, default=DEFAULT_LABEL_SUFFIX,
                        help='Suffix of original label files (default: _lab.png).')
    parser.add_argument('--max-items', type=int, default=None,
                        help='Limit the number of samples to inspect (useful for quick checks).')
    parser.add_argument('--check-labels', action='store_true',
                        help='Also validate cropped labels alongside images.')
    return parser.parse_args()


def compute_start_positions(length: int, tile: int) -> List[int]:
    if tile <= 0:
        raise ValueError('Tile dimension must be positive.')
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def load_shape(path: Path) -> Tuple[int, int]:
    array = imageio.imread(path)
    if array.ndim < 2:
        raise ValueError(f'{path} is not an image (ndim < 2).')
    return array.shape[:2]


def collect_originals(image_dir: Path, suffix: str) -> Dict[str, Path]:
    originals: Dict[str, Path] = {}
    for path in sorted(image_dir.glob(f'*{suffix}')):
        if path.is_file() and path.suffix.lower() == suffix.lower():
            originals[path.stem] = path
    return originals


def collect_crops(crop_dir: Path, base: str, suffix: str) -> Dict[Index, Path]:
    pattern = re.compile(rf'^{re.escape(base)}_r(\d+)_c(\d+){re.escape(suffix)}$')
    found: Dict[Index, Path] = {}
    for path in crop_dir.glob(f'{base}_r*_c*{suffix}'):
        match = pattern.match(path.name)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        found[key] = path
    return found


def compare_shapes(path: Path, expected_h: int, expected_w: int) -> bool:
    h, w = load_shape(path)
    return h == expected_h and w == expected_w


def main() -> None:
    args = parse_args()

    image_dir = args.original_dir / 'img'
    label_dir = args.original_dir / 'label'

    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if args.check_labels and not label_dir.is_dir():
        raise FileNotFoundError(f'Label directory not found: {label_dir}')

    cropped_img_dir = args.cropped_dir / 'cropped_img'
    cropped_label_dir = args.cropped_dir / 'cropped_label'

    if not cropped_img_dir.is_dir():
        raise FileNotFoundError(f'Cropped image directory not found: {cropped_img_dir}')
    if args.check_labels and not cropped_label_dir.is_dir():
        raise FileNotFoundError(f'Cropped label directory not found: {cropped_label_dir}')

    originals = collect_originals(image_dir, args.image_suffix)
    if args.max_items is not None:
        originals = dict(list(originals.items())[: args.max_items])

    summary = Counter()
    issues: Dict[str, Dict[str, Iterable]] = defaultdict(dict)

    for base, image_path in originals.items():
        height, width = load_shape(image_path)
        y_starts = compute_start_positions(height, args.crop_height)
        x_starts = compute_start_positions(width, args.crop_width)
        expected_indices = {(ri, ci) for ri in range(len(y_starts)) for ci in range(len(x_starts))}
        expected_count = len(expected_indices)
        summary['original_tiles_expected'] += expected_count

        crops = collect_crops(cropped_img_dir, base, args.image_suffix)
        actual_count = len(crops)
        summary['cropped_tiles_found'] += actual_count

        missing = sorted(expected_indices - set(crops))
        unexpected = sorted(set(crops) - expected_indices)

        wrong_shape: List[Tuple[Index, Tuple[int, int]]] = []
        for index, crop_path in crops.items():
            expected_h = min(args.crop_height, height - y_starts[index[0]])
            expected_w = min(args.crop_width, width - x_starts[index[1]])
            h, w = load_shape(crop_path)
            if (h, w) != (expected_h, expected_w):
                wrong_shape.append((index, (h, w)))

        summary['samples_checked'] += 1
        if missing:
            summary['samples_with_missing'] += 1
            issues[base]['missing_tiles'] = missing
        if unexpected:
            summary['samples_with_unexpected'] += 1
            issues[base]['unexpected_tiles'] = unexpected
        if wrong_shape:
            summary['samples_with_wrong_shape'] += 1
            issues[base]['image_size_mismatch'] = wrong_shape

        if args.check_labels:
            label_path = label_dir / f'{base}{args.label_suffix}'
            if label_path.exists():
                label_crops = collect_crops(cropped_label_dir, base, args.label_suffix)
                missing_labels = sorted(expected_indices - set(label_crops))
                unexpected_labels = sorted(set(label_crops) - expected_indices)
                wrong_label_shape: List[Tuple[Index, Tuple[int, int]]] = []
                for index, crop_path in label_crops.items():
                    expected_h = min(args.crop_height, height - y_starts[index[0]])
                    expected_w = min(args.crop_width, width - x_starts[index[1]])
                    h, w = load_shape(crop_path)
                    if (h, w) != (expected_h, expected_w):
                        wrong_label_shape.append((index, (h, w)))
                if missing_labels:
                    summary['label_samples_with_missing'] += 1
                    issues[base]['missing_label_tiles'] = missing_labels
                if unexpected_labels:
                    summary['label_samples_with_unexpected'] += 1
                    issues[base]['unexpected_label_tiles'] = unexpected_labels
                if wrong_label_shape:
                    summary['label_samples_with_wrong_shape'] += 1
                    issues[base]['label_size_mismatch'] = wrong_label_shape
            else:
                summary['samples_without_label'] += 1
                issues[base]['label_missing'] = str(label_path)

    print('=== Summary ===')
    for key, value in summary.items():
        print(f'{key}: {value}')

    if issues:
        print('=== Detailed issues ===')
        for base, problem in sorted(issues.items()):
            print(f'[{base}]')
            for name, data in problem.items():
                print(f'  - {name}: {data}')
    else:
        print('No inconsistencies detected.')


if __name__ == '__main__':
    main()
