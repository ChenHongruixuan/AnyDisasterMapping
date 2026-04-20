#!/usr/bin/env python3
"""Clip raw GVLM site folders into flat 256x256 t1/t2/label tiles.

This is a cleaned version of the original GVLM helper.
The clipping logic is preserved; only the path handling is converted to argparse and repo-relative usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

DEFAULT_SITES = [
    'Los Lagos_Chile', 'Tbilisi_Georgia', 'Shimen_China', 'Askja_Iceland', 'Kodagu_India',
    'Asakura_Japan', 'Osh_Kyrgyzstan', 'Tenejapa_Mexico', 'Taitung_China', 'A Luoi_Vietnam',
    'Santa Catarina_Brazil', 'Jiuzhaigou_China', 'Chimanimani_Zimbabwe',
    'Big Sur_United States', 'Kupang_Indonesia', 'Kurucasile_Turkey', 'Kaikoura_New Zealand',
]


def start_points(size: int, split_size: int, overlap: float = 0.0) -> list[int]:
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        points.append(pt)
        counter += 1
    return points


def clear_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clip raw GVLM sites into flat t1/t2/label tile folders.')
    parser.add_argument('--input-root', type=Path, default=Path('./data/landslides/GVLM_CD/raw/GVLM_CD'), help='Root folder containing per-site raw folders with im1.png, im2.png, and ref.png.')
    parser.add_argument('--output-root', type=Path, default=Path('./data/landslides/GVLM_CD'), help='Output root where t1/, t2/, and label/ will be created.')
    parser.add_argument('--tile-width', type=int, default=256, help='Tile width in pixels (default: 256).')
    parser.add_argument('--tile-height', type=int, default=256, help='Tile height in pixels (default: 256).')
    parser.add_argument('--overlap', type=float, default=0.0, help='Tile overlap ratio in [0, 1) (default: 0.0).')
    parser.add_argument('--clear-output', action='store_true', help='Delete existing files under t1/, t2/, and label/ before writing new tiles.')
    parser.add_argument('--sites', nargs='*', default=DEFAULT_SITES, help='Site folder names to process. Defaults to the original site list used by the upstream helper.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    output_t1 = output_root / 't1'
    output_t2 = output_root / 't2'
    output_label = output_root / 'label'

    for path in (output_t1, output_t2, output_label):
        path.mkdir(parents=True, exist_ok=True)
        if args.clear_output:
            clear_folder(path)

    count = 0
    for site in args.sites:
        site_root = input_root / site
        img1_path = site_root / 'im1.png'
        img2_path = site_root / 'im2.png'
        label_path = site_root / 'ref.png'
        if not img1_path.is_file() or not img2_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f'Missing expected GVLM files under {site_root}')

        img1 = np.asarray(Image.open(img1_path))
        img2 = np.asarray(Image.open(img2_path))
        label = np.asarray(Image.open(label_path))
        img_h, img_w, _ = img1.shape
        x_points = start_points(img_w, args.tile_width, args.overlap)
        y_points = start_points(img_h, args.tile_height, args.overlap)

        for y in y_points:
            for x in x_points:
                split1 = img1[y:y + args.tile_height, x:x + args.tile_width, :]
                split2 = img2[y:y + args.tile_height, x:x + args.tile_width, :]
                split3 = label[y:y + args.tile_height, x:x + args.tile_width]
                Image.fromarray(split1).save(output_t1 / f'{count}.jpg')
                Image.fromarray(split2).save(output_t2 / f'{count}.jpg')
                Image.fromarray(split3).save(output_label / f'{count}.png')
                count += 1

    print(f'Wrote {count} GVLM tiles to {output_root}')


if __name__ == '__main__':
    main()
