#!/usr/bin/env python3
"""Split flat GVLM tiles into train/val/test txt files.

This is a cleaned version of the original GVLM helper.
The 60/20/20 random split logic is preserved; only path handling is converted to argparse.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate GVLM train/val/test txt files from a flat t1/ tile folder.')
    parser.add_argument('--source-root', type=Path, default=Path('./data/landslides/GVLM_CD'), help='Root containing t1/, t2/, and label/.')
    parser.add_argument('--val-percentage', type=float, default=0.2, help='Validation split ratio (default: 0.2).')
    parser.add_argument('--test-percentage', type=float, default=0.2, help='Test split ratio (default: 0.2).')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed. Leave unset to preserve upstream nondeterministic behavior.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    t1_dir = source_root / 't1'
    if not t1_dir.is_dir():
        raise FileNotFoundError(f'Missing t1 directory: {t1_dir}')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    train_txt_path = source_root / 'train.txt'
    val_txt_path = source_root / 'val.txt'
    test_txt_path = source_root / 'test.txt'

    image_list = sorted(os.listdir(t1_dir))
    image_num = len(image_list)
    rand_list = range(0, image_num - 1)

    indexes_for_test = random.sample(rand_list, np.round(image_num * args.test_percentage).astype(np.int16))
    indexes_for_val = random.sample([i for i in range(0, image_num) if i not in indexes_for_test], np.round(image_num * args.val_percentage).astype(np.int16))

    for file_path in (train_txt_path, val_txt_path, test_txt_path):
        if file_path.exists():
            file_path.unlink()

    with train_txt_path.open('w', encoding='utf-8') as f_train:
        with val_txt_path.open('w', encoding='utf-8') as f_val:
            with test_txt_path.open('w', encoding='utf-8') as f_test:
                for i in range(image_num):
                    if i in indexes_for_test:
                        print(image_list[i], file=f_test)
                    elif i in indexes_for_val:
                        print(image_list[i], file=f_val)
                    else:
                        print(image_list[i], file=f_train)

    print(f'Wrote {train_txt_path}, {val_txt_path}, and {test_txt_path}')


if __name__ == '__main__':
    main()
