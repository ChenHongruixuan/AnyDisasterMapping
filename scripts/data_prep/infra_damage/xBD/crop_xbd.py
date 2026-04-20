#!/usr/bin/env python3
"""Crop xBD dataset imagery and masks into fixed-size tiles."""

import argparse
from pathlib import Path
from typing import Iterator, Tuple

import imageio.v2 as imageio
import numpy as np

DEFAULT_SUFFIX = '.png'
PRE_SUFFIX = '_pre_disaster'
POST_SUFFIX = '_post_disaster'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Crop xBD pre/post imagery and masks (default 512x512 tiles).'
    )
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='xBD root directory containing images/ and masks/ folders.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Destination directory for cropped outputs.')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size for height and width (default: 512). Must divide 1024.')
    parser.add_argument('--suffix', type=str, default=DEFAULT_SUFFIX,
                        help=f'File suffix for imagery and masks (default: {DEFAULT_SUFFIX}).')
    parser.add_argument('--image-dirname', type=str, default='images',
                        help='Folder name containing xBD images (default: images).')
    parser.add_argument('--mask-dirname', type=str, default='masks',
                        help='Folder name containing xBD masks (default: masks).')
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    return np.asarray(imageio.imread(path))


def build_output_name(base: str, row: int, col: int, suffix: str) -> str:
    return f"{base}_r{row:02d}_c{col:02d}{suffix}"


def crop_tiles(array: np.ndarray, tile: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    h, w = array.shape[:2]
    if h % tile != 0 or w % tile != 0:
        raise ValueError(f'Image shape {h}x{w} is not divisible by tile size {tile}.')
    rows = h // tile
    cols = w // tile
    for r in range(rows):
        top = r * tile
        bottom = top + tile
        for c in range(cols):
            left = c * tile
            right = left + tile
            yield r, c, array[top:bottom, left:right]


def ensure_dirs(output_root: Path) -> Tuple[Path, Path]:
    cropped_img = output_root / 'cropped_images'
    cropped_mask = output_root / 'cropped_masks'
    for directory in (cropped_img, cropped_mask):
        directory.mkdir(parents=True, exist_ok=True)
    return cropped_img, cropped_mask


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    image_dir = input_dir / args.image_dirname
    mask_dir = input_dir / args.mask_dirname

    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if not mask_dir.is_dir():
        raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

    cropped_img_dir, cropped_mask_dir = ensure_dirs(output_dir)

    tile = args.tile_size
    suffix = args.suffix

    pre_images = sorted(image_dir.glob(f'*{PRE_SUFFIX}{suffix}'))
    if not pre_images:
        raise ValueError(f'No images ending with {PRE_SUFFIX}{suffix} found in {image_dir}')

    total_tiles = 0
    for pre_path in pre_images:
        base_id = pre_path.name[:-len(PRE_SUFFIX + suffix)]
        post_path = image_dir / f'{base_id}{POST_SUFFIX}{suffix}'
        pre_mask_path = mask_dir / f'{base_id}{PRE_SUFFIX}{suffix}'
        post_mask_path = mask_dir / f'{base_id}{POST_SUFFIX}{suffix}'

        if not post_path.exists():
            print(f'[WARN] Missing post image for {base_id}; expected {post_path.name}. Skipping.')
            continue
        if not pre_mask_path.exists():
            print(f'[WARN] Missing pre mask for {base_id}; expected {pre_mask_path.name}. Skipping.')
            continue
        if not post_mask_path.exists():
            print(f'[WARN] Missing post mask for {base_id}; expected {post_mask_path.name}. Skipping.')
            continue

        pre_img = load_image(pre_path)
        post_img = load_image(post_path)
        pre_mask = load_image(pre_mask_path)
        post_mask = load_image(post_mask_path)

        for (r, c, crop_pre), (_, _, crop_post), (_, _, crop_pre_mask), (_, _, crop_post_mask) in zip(
            crop_tiles(pre_img, tile),
            crop_tiles(post_img, tile),
            crop_tiles(pre_mask, tile),
            crop_tiles(post_mask, tile),
        ):
            pre_out = cropped_img_dir / build_output_name(f'{base_id}{PRE_SUFFIX}', r, c, suffix)
            post_out = cropped_img_dir / build_output_name(f'{base_id}{POST_SUFFIX}', r, c, suffix)
            pre_mask_out = cropped_mask_dir / build_output_name(f'{base_id}{PRE_SUFFIX}', r, c, suffix)
            post_mask_out = cropped_mask_dir / build_output_name(f'{base_id}{POST_SUFFIX}', r, c, suffix)
            imageio.imwrite(pre_out, crop_pre)
            imageio.imwrite(post_out, crop_post)
            imageio.imwrite(pre_mask_out, crop_pre_mask)
            imageio.imwrite(post_mask_out, crop_post_mask)
            total_tiles += 1

    print(f'Finished. Wrote {total_tiles} tiles (per modality) to {output_dir}.')


if __name__ == '__main__':
    main()
