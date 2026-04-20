#!/usr/bin/env python3

import argparse
from pathlib import Path
import re

def collect_stems(image_root: Path, recursive: bool = True):
    pattern = re.compile(r"^(?P<stem>.+)_(?:pre|post)_disaster\.png$")
    stems = set()

    walk_iter = image_root.rglob('*.png') if recursive else image_root.glob('*.png')
    for path in walk_iter:
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        stems.add(match.group('stem'))

    return sorted(stems)

def write_stems(stems, destination: Path):
    if not stems:
        raise ValueError('No matching PNG files were found; verify the input directory.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as handle:
        handle.write('\n'.join(stems) + '\n')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Generate xBD file list from pre/post disaster image pairs.')
    parser.add_argument('--image_dir', type=Path, help='Root directory that contains xBD PNG files.')
    parser.add_argument('--output_txt', type=Path, help='Path to the output txt file to write stems into.')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive search when gathering PNG files.')
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    image_dir: Path = args.image_dir.expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f'Input directory {image_dir} does not exist.')

    recursive = not args.no_recursive
    stems = collect_stems(image_dir, recursive=recursive)
    write_stems(stems, args.output_txt.expanduser())
    print(f'Wrote {len(stems)} entries to {args.output_txt}')

if __name__ == '__main__':
    main()
