"""Perceptual-hash leakage audit between CrowdHuman (train) and an OOD dataset.

Our cross-dataset OOD claim ("the model has never seen these images") is only
as strong as the guarantee that CityPersons / WiderPerson do not silently
share images with CrowdHuman's training set. Public "in the wild" pedestrian
datasets are all scraped from broadly overlapping web sources, so exact or
near-duplicate images across datasets is a real, previously documented risk
(see the FreqFD leakage audit this project follows the spirit of).

Method: perceptual hash (pHash, 16x16) every CrowdHuman train image and every
OOD image, then flag OOD images whose nearest CrowdHuman-train hash is within
a small Hamming distance (near-duplicate threshold). This is a similarity
check, not exact-match only, so it also catches re-encoded / re-scaled copies.

Usage:
    python tools/audit_ood_leakage.py \
        --crowdhuman_dir datasets/crowdhuman/Images \
        --crowdhuman_odgt datasets/crowdhuman/annotation_train.odgt \
        --ood_dir datasets/citypersons/valid \
        --ood_ann datasets/citypersons/valid/_annotations.coco.json \
        --ood_name citypersons \
        --threshold 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imagehash
from PIL import Image


def _crowdhuman_filenames(odgt_path: str, image_dir: str) -> list[str]:
    """Resolve CrowdHuman .odgt image IDs to actual file paths on disk."""
    paths = []
    with open(odgt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            img_id = rec["ID"]
            for ext in (".jpg", ".png", ""):
                p = os.path.join(image_dir, img_id + ext)
                if os.path.exists(p):
                    paths.append(p)
                    break
    return paths


def _ood_filenames(ann_path: str, image_dir: str) -> list[str]:
    """Resolve a COCO-format annotation file's image list to file paths."""
    with open(ann_path, "r") as f:
        coco = json.load(f)
    paths = []
    for img in coco["images"]:
        p = os.path.join(image_dir, img["file_name"])
        if os.path.exists(p):
            paths.append(p)
    return paths


def _phash(path: str, hash_size: int = 16):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"), hash_size=hash_size)


def audit(crowdhuman_paths: list[str], ood_paths: list[str],
          hash_size: int = 16, threshold: int = 8,
          limit_crowdhuman: int = 0) -> dict:
    """Return a leakage report comparing ood_paths against crowdhuman_paths."""
    if limit_crowdhuman > 0:
        crowdhuman_paths = crowdhuman_paths[:limit_crowdhuman]

    print(f"Hashing {len(crowdhuman_paths)} CrowdHuman train images ...")
    ch_hashes = {}
    for i, p in enumerate(crowdhuman_paths):
        try:
            ch_hashes[p] = _phash(p, hash_size)
        except Exception as e:
            print(f"  skip (unreadable): {p} ({e})")
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(crowdhuman_paths)}")

    print(f"Hashing {len(ood_paths)} OOD images ...")
    matches = []
    for i, p in enumerate(ood_paths):
        try:
            h = _phash(p, hash_size)
        except Exception as e:
            print(f"  skip (unreadable): {p} ({e})")
            continue
        best_dist, best_path = None, None
        for ch_path, ch_hash in ch_hashes.items():
            d = h - ch_hash  # Hamming distance
            if best_dist is None or d < best_dist:
                best_dist, best_path = d, ch_path
        if best_dist is not None and best_dist <= threshold:
            matches.append({"ood_image": p, "crowdhuman_match": best_path,
                            "hamming_distance": int(best_dist)})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ood_paths)}")

    return {
        "num_crowdhuman_hashed": len(ch_hashes),
        "num_ood_images": len(ood_paths),
        "num_near_duplicates": len(matches),
        "leakage_rate": len(matches) / max(len(ood_paths), 1),
        "threshold": threshold,
        "matches": matches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="pHash leakage audit: CrowdHuman train vs an OOD dataset.")
    p.add_argument("--crowdhuman_dir", required=True)
    p.add_argument("--crowdhuman_odgt", required=True)
    p.add_argument("--ood_dir", required=True)
    p.add_argument("--ood_ann", required=True, help="COCO-format json for the OOD dataset")
    p.add_argument("--ood_name", required=True)
    p.add_argument("--hash_size", type=int, default=16)
    p.add_argument("--threshold", type=int, default=8, help="max Hamming distance = near-duplicate")
    p.add_argument("--limit_crowdhuman", type=int, default=0,
                   help="if >0, only hash this many CrowdHuman train images (for a quick pass)")
    p.add_argument("--out_dir", default="outputs/logs")
    args = p.parse_args()

    ch_paths = _crowdhuman_filenames(args.crowdhuman_odgt, args.crowdhuman_dir)
    ood_paths = _ood_filenames(args.ood_ann, args.ood_dir)
    print(f"CrowdHuman train images found: {len(ch_paths)}")
    print(f"OOD ({args.ood_name}) images found: {len(ood_paths)}")

    report = audit(ch_paths, ood_paths, hash_size=args.hash_size,
                   threshold=args.threshold, limit_crowdhuman=args.limit_crowdhuman)

    print(f"\n=== Leakage audit: CrowdHuman(train) vs {args.ood_name} ===")
    print(f"near-duplicates: {report['num_near_duplicates']} / {report['num_ood_images']} "
          f"({report['leakage_rate']*100:.2f}%) at Hamming <= {args.threshold}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leakage_audit_{args.ood_name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
