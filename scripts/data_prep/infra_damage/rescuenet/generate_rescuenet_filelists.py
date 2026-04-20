#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, List, Set


def collect_names(root: Path, suffix: str = '.png', recursive: bool = True) -> List[str]:
    suffix = suffix.lower()
    if recursive:
        iterator = root.rglob(f'*{suffix}')
    else:
        iterator = root.glob(f'*{suffix}')

    names: Set[str] = set()
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() != suffix:
            continue
        names.add(path.stem)

    return sorted(names)


def write_names(names: Iterable[str], destination: Path) -> None:
    names = list(names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as handle:
        handle.write('\n'.join(names))
        handle.write('\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Collect PNG base names (without suffix) from one or more RescueNet directories.'
    )
    parser.add_argument(
        '--directories',
        type=Path,
        nargs='+',
        required=True,
        help='One or more directories containing RescueNet imagery or labels.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        required=True,
        help='Directory where generated txt files will be stored.'
    )
    parser.add_argument(
        '--suffix',
        default='.png',
        help='File suffix to filter on (default: .png).'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Disable recursive search when collecting file names.'
    )
    parser.add_argument(
        '--combined',
        type=str,
        help='Optional filename for a combined list containing entries from all input directories.'
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recursive = not args.no_recursive
    all_names: Set[str] = set()

    directories = []
    for directory in args.directories:
        directory = directory.expanduser().resolve()
        directories.append(directory)

    for directory in directories:
        if not directory.exists():
            raise FileNotFoundError(f'Directory {directory} does not exist.')
        if not directory.is_dir():
            raise NotADirectoryError(f'{directory} is not a directory.')

        names = collect_names(directory, suffix=args.suffix, recursive=recursive)
        if not names:
            raise ValueError(f'No files with suffix {args.suffix} found under {directory}.')

        all_names.update(names)
        output_path = output_dir / f'{directory.name}_filelist.txt'
        write_names(names, output_path)
        print(f'Wrote {len(names)} entries to {output_path}')

    if args.combined:
        combined_path = output_dir / args.combined
        write_names(sorted(all_names), combined_path)
        print(f'Wrote {len(all_names)} unique entries to {combined_path}')


if __name__ == '__main__':
    main()
