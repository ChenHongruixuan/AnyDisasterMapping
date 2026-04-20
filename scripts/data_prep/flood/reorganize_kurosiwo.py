#!/usr/bin/env python3
"""Reorganize KuroSiwo data into split-specific folders with consistent naming."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple, Union

GridDict = Dict[str, Dict[str, object]]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = Path("data/flood/kurosiwo/raw")
DEFAULT_CONFIG_PATH = DEFAULT_RAW_ROOT / "configs/train/data_config.json"
DEFAULT_SOURCE_ROOT = DEFAULT_RAW_ROOT / "data"
DEFAULT_OUTPUT_ROOT = Path("data/flood/kurosiwo")
SPLIT_FILENAMES = {"train": "train.txt", "val": "validation.txt", "test": "test.txt"}

FILE_LAYOUT: "OrderedDict[str, str]" = OrderedDict(
    [
        ("MS1_IVV", "post_vv"),
        ("MS1_IVH", "post_vh"),
        ("SL1_IVV", "pre1_vv"),
        ("SL1_IVH", "pre1_vh"),
        ("SL2_IVV", "pre2_vv"),
        ("SL2_IVH", "pre2_vh"),
        ("MK0_MNA", "MASK_NODATA"),
        ("MK0_MLU", "GT"),
        ("MK0_SLOPE", "SLOPE"),
        ("MK0_DEM", "DEM"),
    ]
)

SUBDIRS = list(dict.fromkeys(FILE_LAYOUT.values()))


def remove_json_comments(text: str) -> str:
    cleaned: List[str] = []
    for line in text.splitlines():
        if "//" in line:
            line = line.split("//", 1)[0]
        cleaned.append(line)
    return "\n".join(cleaned)


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    return json.loads(remove_json_comments(raw))


def load_pickle(path: Path) -> GridDict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def normalise_targets(
    targets: Sequence[Union[int, str]], track: str
) -> set[Union[int, str]]:
    if track == "Climatic":
        return {str(target) for target in targets}
    return set(targets)


def activation_key(info: MutableMapping[str, object], track: str) -> Union[int, str]:
    if track == "Climatic":
        act_id = info["actid"]
        aoi_id = int(info["aoiid"])
        return f"{act_id}_{aoi_id:02d}"
    return info["actid"]


def collect_records(
    grid_dict: GridDict,
    targets: set[Union[int, str]],
    track: str,
) -> Dict[str, Dict[str, object]]:
    selected: Dict[str, Dict[str, object]] = {}
    for grid_id, payload in grid_dict.items():
        if activation_key(payload["info"], track) in targets:
            selected[grid_id] = payload
    return selected


def resolve_source_root(project_root: Path, config: Dict[str, object], override: Path | None) -> Path:
    if override is not None:
        root = override
        if not root.is_absolute():
            root = project_root / root
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Source root '{root}' does not exist")
        return root

    candidates: List[Path] = []
    config_root = config.get("root_path")
    if isinstance(config_root, str) and config_root:
        cfg_path = Path(config_root)
        candidates.append(cfg_path / "data")
        candidates.append(cfg_path)
        candidates.append(Path("data") / cfg_path)
    candidates.extend(
        [
            DEFAULT_SOURCE_ROOT,
            DEFAULT_SOURCE_ROOT / "KuroSiwo",
            Path(".data"),
            Path("data"),
            Path("data") / "KuroSiwo",
        ]
    )

    for cand in candidates:
        candidate = cand if cand.is_absolute() else (project_root / cand)
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to locate source data root. Provide --source-root explicitly."
    )


def ensure_output_dirs(root: Path) -> None:
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)


def find_source_file(scene_dir: Path, prefix: str) -> Path | None:
    for entry in scene_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".tif":
            continue
        if entry.name.startswith(prefix):
            return entry
    return None


def scene_identifier(payload: Dict[str, object]) -> Tuple[str, Path]:
    relative = Path(str(payload["path"]))
    return "_".join(relative.parts), relative


def copy_scene(
    scene_name: str,
    relative: Path,
    grid_id: str,
    source_root: Path,
    output_root: Path,
    skip_existing: bool,
) -> Tuple[str, List[str]]:
    scene_dir = source_root / relative
    if not scene_dir.exists():
        raise FileNotFoundError(f"Missing source directory for grid {grid_id}: {scene_dir}")

    missing: List[str] = []

    for prefix, folder in FILE_LAYOUT.items():
        source_file = find_source_file(scene_dir, prefix)
        if source_file is None:
            missing.append(prefix)
            continue
        dest_dir = output_root / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{scene_name}.tif"
        if skip_existing and dest_file.exists():
            continue
        shutil.copy2(source_file, dest_file)

    return scene_name, missing


def write_split_list(path: Path, names: Iterable[str]) -> None:
    unique_names = sorted(dict.fromkeys(names))
    with path.open("w", encoding="utf-8") as handle:
        for name in unique_names:
            handle.write(f"{name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize the dataset into split-specific folders."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration file with split definitions.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Base directory used to resolve relative paths.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory containing the original disaster scene folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the reorganized dataset will be written.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copying if the destination file already exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads used for copying (default: cpu_count).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress after this many scenes per split (set 0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)

    source_root = resolve_source_root(project_root, config, args.source_root)
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs(output_root)
    config_root = config_path.parents[2] if len(config_path.parents) > 2 else project_root
    default_pickle_dir = config_root / "pickle"

    track = config.get("track", "RandomEvents")

    train_targets = normalise_targets(config["train_acts"], track)
    val_targets = normalise_targets(config["val_acts"], track)
    test_targets = normalise_targets(config["test_acts"], track)

    train_pickle_path = project_root / config["train_pickle"]
    test_pickle_path = project_root / config["test_pickle"]
    if not train_pickle_path.exists():
        train_pickle_path = default_pickle_dir / Path(config["train_pickle"]).name
    if not train_pickle_path.exists():
        train_pickle_path = project_root / "pickle" / Path(config["train_pickle"]).name
    if not test_pickle_path.exists():
        test_pickle_path = default_pickle_dir / Path(config["test_pickle"]).name
    if not test_pickle_path.exists():
        test_pickle_path = project_root / "pickle" / Path(config["test_pickle"]).name

    train_dict = load_pickle(train_pickle_path)
    test_dict = load_pickle(test_pickle_path)

    splits = {
        "train": collect_records(train_dict, train_targets, track),
        "val": collect_records(test_dict, val_targets, track),
        "test": collect_records(test_dict, test_targets, track),
    }

    scene_registry: Dict[str, str] = {}
    missing_summary: Dict[str, List[str]] = {}

    worker_count = args.workers or max(1, os.cpu_count() or 1)

    for split, records in splits.items():
        ordered_entries = OrderedDict()
        for grid_id, payload in sorted(
            records.items(),
            key=lambda item: (
                item[1]["info"].get("actid"),
                item[1]["info"].get("aoiid"),
                item[0],
            ),
        ):
            scene_name, relative = scene_identifier(payload)
            ordered_entries.setdefault(scene_name, (grid_id, relative))

        total = len(ordered_entries)
        names: List[str] = []
        missing_details: List[str] = []

        if total == 0:
            write_split_list(output_root / SPLIT_FILENAMES[split], names)
            missing_summary[split] = missing_details
            print(f"[{split}] Processed 0 scenes.")
            continue

        print(
            f"[{split}] Dispatching {total} scenes with {worker_count} workers...",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_scene = {
                executor.submit(
                    copy_scene,
                    scene_name,
                    relative,
                    grid_id,
                    source_root,
                    output_root,
                    args.skip_existing,
                ): scene_name
                for scene_name, (grid_id, relative) in ordered_entries.items()
            }

            completed = 0
            for future in as_completed(future_to_scene):
                scene_name, missing = future.result()
                completed += 1
                names.append(scene_name)
                if missing:
                    missing_details.append(f"{scene_name}: missing {', '.join(missing)}")
                previous_split = scene_registry.get(scene_name)
                if previous_split and previous_split != split:
                    raise RuntimeError(
                        f"Scene {scene_name} appears in both {previous_split} and {split}."
                    )
                scene_registry[scene_name] = split

                if (
                    args.progress_interval > 0
                    and (completed % args.progress_interval == 0 or completed == total)
                ):
                    print(
                        f"[{split}] {completed}/{total} scenes copied",
                        flush=True,
                    )

        write_split_list(output_root / SPLIT_FILENAMES[split], names)
        missing_summary[split] = missing_details
        print(f"[{split}] Processed {len(names)} scenes.")

    overlap = set()
    splits_sets = {
        split: set(Path(output_root / SPLIT_FILENAMES[split]).read_text().split())
        for split in splits
    }
    overlap.update(splits_sets["train"] & splits_sets["val"])
    overlap.update(splits_sets["train"] & splits_sets["test"])
    overlap.update(splits_sets["val"] & splits_sets["test"])
    if overlap:
        raise RuntimeError(f"Overlapping scene names detected: {sorted(overlap)}")

    for split, missing in missing_summary.items():
        if missing:
            print(f"[{split}] Scenes with missing files ({len(missing)}):")
            for entry in missing:
                print(f"  - {entry}")


if __name__ == "__main__":
    main()
