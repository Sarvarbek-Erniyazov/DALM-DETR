"""Download a Roboflow Universe dataset in COCO JSON format.

Used for CityPersons and WiderPerson cross-dataset OOD evaluation, since both
require lengthy manual approval (CityPersons/Cityscapes) or Google/Baidu Drive
downloads (WiderPerson) from their official sources. Roboflow mirrors are
instant to access but are NOT guaranteed to match the official train/val/test
split or to preserve "ignore" regions -- see README's Limitations section.
This is a pragmatic choice for cross-dataset generalization testing, not for
reproducing published CityPersons/WiderPerson leaderboard numbers.

IMPORTANT: the target directory (--out) must NOT already exist. The Roboflow
SDK checks os.path.exists(location) and silently skips the download (zero
files, no error) if the directory is already there. If a previous attempt
failed, delete the directory first.

Requires:
    pip install roboflow
    export ROBOFLOW_API_KEY=xxxx   # free account, instant, no approval wait
                                    # get it at https://app.roboflow.com/settings/api

Usage:
    python tools/download_roboflow_dataset.py \
        --url https://universe.roboflow.com/citypersons-conversion/citypersons-woqjq/dataset/9 \
        --out datasets/citypersons
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset in COCO format.")
    parser.add_argument("--url", required=True, help="Roboflow Universe dataset URL")
    parser.add_argument("--out", required=True, help="output directory (must not already exist)")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key is None:
        raise SystemExit(
            "ROBOFLOW_API_KEY not set. Get a free key at "
            "https://app.roboflow.com/settings/api and run:\n"
            "  export ROBOFLOW_API_KEY=xxxx"
        )

    if os.path.exists(args.out):
        raise SystemExit(
            f"ERROR: {args.out} already exists. The Roboflow SDK silently skips "
            f"the download if the target directory exists. Delete it first:\n"
            f"  rm -rf {args.out}"
        )

    from roboflow import download_dataset

    print(f"Downloading {args.url} -> {args.out} (COCO format) ...")
    dataset = download_dataset(args.url, "coco", location=args.out)
    print("Dataset object location:", dataset.location)

    n_files = sum(len(files) for _, _, files in os.walk(args.out))
    print(f"Done. {n_files} files written under {args.out}")
    if n_files == 0:
        raise SystemExit(
            "ERROR: 0 files downloaded. Check the ROBOFLOW_API_KEY and dataset URL."
        )


if __name__ == "__main__":
    main()
