"""Utilities to visualize xBD dataset masks with fixed color palettes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

POST_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (0x46, 0xB5, 0x79),
    2: (0x81, 0xD0, 0xC4),
    3: (0xEF, 0x86, 0xAC),
    4: (0xB6, 0x46, 0x45),
    5: (0xB1, 0x7D, 0xBA),
}

PRE_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (0x46, 0xB5, 0x79),
}

DEFAULT_COLOR = (0, 0, 0)


def build_palette(mapping: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """Create a lookup table for fast colorization."""
    if not mapping:
        raise ValueError('Color mapping must not be empty.')
    max_index = max(mapping.keys())
    palette = np.zeros((max_index + 1, 3), dtype=np.uint8)
    palette[:] = DEFAULT_COLOR
    for label_value, rgb in mapping.items():
        palette[label_value] = rgb
    return palette


POST_PALETTE = build_palette(POST_COLOR_MAP)
PRE_PALETTE = build_palette(PRE_COLOR_MAP)


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Apply the palette to the mask values, clipping out-of-range indices."""
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, palette.shape[0] - 1)
    return palette[mask]


def determine_palette(path: Path) -> np.ndarray:
    name = path.name.lower()
    if 'post_disaster' in name:
        return POST_PALETTE
    if 'pre_disaster' in name:
        return PRE_PALETTE
    raise ValueError(f'File name does not contain "pre_disaster" or "post_disaster": {path}')


def collect_masks(directory: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from sorted(p for p in directory.rglob(pattern) if p.is_file())
    else:
        yield from sorted(p for p in directory.glob(pattern) if p.is_file())


def visualize_masks(input_dir: Path, output_dir: Path, pattern: str = '*.png', recursive: bool = False, overwrite: bool = False) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist or is not a directory: {input_dir}')

    masks = list(collect_masks(input_dir, pattern, recursive))
    if not masks:
        raise FileNotFoundError(f'No files matching pattern {pattern!r} found in {input_dir}')

    for mask_path in masks:
        try:
            palette = determine_palette(mask_path)
        except ValueError as error:
            print(f'Skipping {mask_path}: {error}', file=sys.stderr)
            continue

        relative_path = mask_path.relative_to(input_dir)
        output_path = (output_dir / relative_path).with_suffix('.png')

        if output_path.exists() and not overwrite:
            print(f'Skipping existing file: {output_path}', file=sys.stderr)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(mask_path) as mask_image:
            mask_array = np.asarray(mask_image)

        colorized = colorize_mask(mask_array, palette)
        Image.fromarray(colorized).save(output_path)
        print(f'Saved {output_path}')


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize xBD mask labels using fixed color palettes for pre/post disaster tiles.'
    )
    parser.add_argument('--input_dir', type=Path, help='Directory containing mask images.')
    parser.add_argument('-o', '--output-dir', type=Path, required=True, help='Directory where visualizations will be saved.')
    parser.add_argument('--pattern', default='*.png', help='Glob pattern for mask filenames (default: *.png).')
    parser.add_argument('--recursive', action='store_true', help='Search for masks recursively.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing visualization files.')
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        visualize_masks(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
