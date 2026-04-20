#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

from huggingface_hub import snapshot_download
try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # huggingface_hub<0.20 exposes the exception from utils.
    from huggingface_hub.utils import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UrbanSARFloods assets from HuggingFace."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/flood/UrbanSARFloods/raw"),
        help="Snapshot target directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def non_empty_dir(path: Path) -> bool:
        return path.is_dir() and any(path.iterdir())

    allow_patterns: list[str] = []
    if non_empty_dir(output_dir / "testing_case_orig"):
        print(f"Skip existing directory: {output_dir / 'testing_case_orig'}")
    else:
        allow_patterns.append("testing_case_orig/**")

    if non_empty_dir(output_dir / "testing_case_256"):
        print(f"Skip existing directory: {output_dir / 'testing_case_256'}")
    else:
        allow_patterns.append("testing_case_256/**")

    if (output_dir / "urban_sar_floods.tar.gz").is_file():
        print(f"Skip existing file: {output_dir / 'urban_sar_floods.tar.gz'}")
    else:
        allow_patterns.append("urban_sar_floods.tar.gz")

    ignore_patterns: list[str] = []

    if not allow_patterns:
        print("All expected UrbanSARFloods download targets already exist. Nothing to do.")
        return

    backoff_base = 5
    max_attempts = 8

    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(
                repo_id="S1Floodbenchmark/UrbanSARFloods_v1",
                repo_type="dataset",
                local_dir=str(output_dir),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                resume_download=True,
                max_workers=4,
                token=os.environ.get("HF_TOKEN", None),
            )
            print("Done.")
            return
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429:
                wait = min(300, backoff_base * (2 ** (attempt - 1))) + random.uniform(0, 3)
                print(f"Rate limited (attempt {attempt}), waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue
            raise


if __name__ == "__main__":
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    main()
