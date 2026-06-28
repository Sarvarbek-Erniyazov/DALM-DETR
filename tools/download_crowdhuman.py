"""Download CrowdHuman from the Hugging Face Hub.

Source: https://huggingface.co/datasets/sshao0516/CrowdHuman

The dataset is gated: you must (1) accept the terms on the dataset page while
logged in, and (2) provide a read token. Pass the token via the HF_TOKEN
environment variable (never hard-code it):

    export HF_TOKEN=hf_xxx           # in git bash
    python tools/download_crowdhuman.py --split val
    python tools/download_crowdhuman.py --split train

Downloaded zips are extracted under datasets/crowdhuman/, producing:

    datasets/crowdhuman/
        annotation_train.odgt
        annotation_val.odgt
        Images/                 # all .jpg images (train + val share this dir)
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "sshao0516/CrowdHuman"
REPO_TYPE = "dataset"

# Which files belong to each split.
SPLIT_FILES = {
    "val": ["annotation_val.odgt", "CrowdHuman_val.zip"],
    "train": [
        "annotation_train.odgt",
        "CrowdHuman_train01.zip",
        "CrowdHuman_train02.zip",
        "CrowdHuman_train03.zip",
    ],
}


def _download_file(filename: str, cache_dir: str, token: str | None) -> str:
    print(f"  downloading {filename} ...")
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
    )


def _extract_zip(zip_path: str, out_dir: Path) -> None:
    print(f"  extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CrowdHuman from HF Hub.")
    parser.add_argument(
        "--split", choices=["train", "val"], required=True,
        help="which split to download",
    )
    parser.add_argument(
        "--out", default="datasets/crowdhuman",
        help="output directory (default: datasets/crowdhuman)",
    )
    parser.add_argument(
        "--cache", default="datasets/.hf_cache",
        help="HF download cache directory",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("WARNING: HF_TOKEN not set. If the download fails with a 401/403, "
              "set it with:  export HF_TOKEN=hf_xxx")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in SPLIT_FILES[args.split]:
        local = _download_file(filename, args.cache, token)
        if filename.endswith(".zip"):
            _extract_zip(local, out_dir)
        else:
            # Copy the .odgt next to the images.
            dst = out_dir / filename
            if not dst.exists():
                dst.write_bytes(Path(local).read_bytes())

    print(f"\nDone. '{args.split}' split is ready under: {out_dir}")


if __name__ == "__main__":
    main()
