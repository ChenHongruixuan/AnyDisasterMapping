#!/usr/bin/env python3
"""Crop RescueNet imagery/labels while merging trailing remainders into the preceding tile."""

import argparse
from pathlib import Path
from typing import Iterator, List, Tuple

import imageio.v2 as imageio
import numpy as np

DEFAULT_IMAGE_SUFFIX = '.jpg'
DEFAULT_LABEL_SUFFIX = '_lab.png'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Crop RescueNet imagery/labels into tiles (default 750x1000). '
            'If the final strip on an axis is smaller than the tile size, it is merged into the previous tile.'
        )
    )
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Root directory containing the original img/ and label/ folders.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Destination directory where cropped_img/ and cropped_label/ will be created.')
    parser.add_argument('--crop-height', type=int, default=750,
                        help='Tile height in pixels (default: 750).')
    parser.add_argument('--crop-width', type=int, default=1000,
                        help='Tile width in pixels (default: 1000).')
    parser.add_argument('--image-suffix', type=str, default=DEFAULT_IMAGE_SUFFIX,
                        help=f'Suffix of the original image files (default: {DEFAULT_IMAGE_SUFFIX}).')
    parser.add_argument('--label-suffix', type=str, default=DEFAULT_LABEL_SUFFIX,
                        help=f'Suffix of the original label files (default: {DEFAULT_LABEL_SUFFIX}).')
    parser.add_argument('--image-dirname', type=str, default='img',
                        help='Name of the folder that stores images inside the dataset root.')
    parser.add_argument('--label-dirname', type=str, default='label',
                        help='Name of the folder that stores labels inside the dataset root.')
    return parser.parse_args()


def compute_start_positions(length: int, tile: int) -> List[int]:
    """Return tile start indices while merging a short remainder into the previous tile."""
    if tile <= 0:
        raise ValueError('Tile dimension must be positive.')
    if length <= 0:
        raise ValueError('Image dimension must be positive.')

    if length <= tile:
        return [0]

    starts: List[int] = []
    current = 0
    while True:
        starts.append(current)
        next_start = current + tile
        if next_start >= length:
            break
        if length - next_start < tile:
            # Remaining span is shorter than one tile; merge into the current tile
            break
        current = next_start
    return starts


def load_array(path: Path) -> np.ndarray:
    return np.asarray(imageio.imread(path))


def ensure_valid_dimensions(array: np.ndarray, min_height: int, min_width: int, name: str) -> None:
    h, w = array.shape[:2]
    if h < min_height or w < min_width:
        raise ValueError(
            f'{name}: size {h}x{w} is smaller than the requested minimum {min_height}x{min_width}.'
        )


def build_output_name(base: str, row_idx: int, col_idx: int, row_pad: int, col_pad: int, suffix: str) -> str:
    return f"{base}_r{row_idx:0{row_pad}d}_c{col_idx:0{col_pad}d}{suffix}"


def crop_tiles(image: np.ndarray,
               label: np.ndarray,
               crop_height: int,
               crop_width: int) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    h, w = image.shape[:2]
    y_starts = compute_start_positions(h, crop_height)
    x_starts = compute_start_positions(w, crop_width)

    for r_idx, top in enumerate(y_starts):
        bottom = h if r_idx == len(y_starts) - 1 else top + crop_height
        for c_idx, left in enumerate(x_starts):
            right = w if c_idx == len(x_starts) - 1 else left + crop_width
            yield (
                r_idx,
                c_idx,
                image[top:bottom, left:right].copy(),
                label[top:bottom, left:right].copy(),
            )


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    image_dir = input_dir / args.image_dirname
    label_dir = input_dir / args.label_dirname

    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if not label_dir.is_dir():
        raise FileNotFoundError(f'Label directory not found: {label_dir}')

    output_img_dir = output_dir / 'cropped_img'
    output_label_dir = output_dir / 'cropped_label'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob(f'*{args.image_suffix}'))
    if not image_paths:
        raise ValueError(f'No image files ending with {args.image_suffix} found under {image_dir}')

    total_tiles = 0
    for image_path in image_paths:
        base_name = image_path.name[:-len(args.image_suffix)]
        label_path = label_dir / f'{base_name}{args.label_suffix}'
        if not label_path.exists():
            print(f'[WARN] Missing label for {image_path.name}; expected {label_path.name}. Skipping.')
            continue

        image = load_array(image_path)
        label = load_array(label_path)
        ensure_valid_dimensions(image, min_height=1, min_width=1, name=image_path.name)
        ensure_valid_dimensions(label, min_height=1, min_width=1, name=label_path.name)

        y_starts = compute_start_positions(image.shape[0], args.crop_height)
        x_starts = compute_start_positions(image.shape[1], args.crop_width)
        row_pad = max(2, len(str(len(y_starts) - 1)))
        col_pad = max(2, len(str(len(x_starts) - 1)))

        for r_idx, c_idx, image_tile, label_tile in crop_tiles(
            image,
            label,
            args.crop_height,
            args.crop_width,
        ):
            image_out = output_img_dir / build_output_name(base_name, r_idx, c_idx, row_pad, col_pad, args.image_suffix)
            label_out = output_label_dir / build_output_name(base_name, r_idx, c_idx, row_pad, col_pad, args.label_suffix)
            imageio.imwrite(image_out, image_tile)
            imageio.imwrite(label_out, label_tile)
            total_tiles += 1

    print(f'Finished. Wrote {total_tiles} tiles to {output_img_dir} and {output_label_dir}.')


if __name__ == '__main__':
    main()
