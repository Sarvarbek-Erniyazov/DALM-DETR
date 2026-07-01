"""Cross-dataset (out-of-distribution) evaluation.

Loads a checkpoint trained on CrowdHuman and evaluates it -- with NO further
training -- on a dataset it has never seen. This is the core evidence for
whether the location-aware matching term (offset cost) generalizes beyond the
dataset it was trained on, not just a CrowdHuman-specific quirk.

Works with any dataset that follows the (images, targets) contract used by
CrowdHumanDataset / CocoPersonDataset: targets are dicts with "labels" and
"boxes" (normalized cx,cy,w,h).

Example:
    python scripts/eval_ood.py \
        --checkpoint outputs/checkpoints/offsetiou_baseline_best.pth \
        --dataset_type coco \
        --image_dir datasets/citypersons/images \
        --ann_path datasets/citypersons/annotations.json \
        --dataset_name citypersons \
        --tag baseline_on_citypersons
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from offsetiou_det.models.detector import OffsetIoUDet
from offsetiou_det.engine.evaluator import evaluate_model
from offsetiou_det.data.coco_person import CocoPersonDataset, collate_fn as coco_collate
from offsetiou_det.data.crowdhuman import CrowdHumanDataset, collate_fn as ch_collate


def parse_args():
    p = argparse.ArgumentParser(description="Cross-dataset OOD evaluation of a trained checkpoint.")
    p.add_argument("--checkpoint", required=True, help="path to a .pth saved by scripts/train.py")
    p.add_argument("--dataset_type", choices=["coco", "crowdhuman"], required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--ann_path", required=True, help="COCO json (coco) or .odgt file (crowdhuman)")
    p.add_argument("--dataset_name", default="ood", help="short label used in the COCO loader / logs")
    p.add_argument("--image_size", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="if >0, evaluate on only this many images")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--tag", default="ood_eval", help="name for the results file")
    return p.parse_args()


def build_dataset(args):
    if args.dataset_type == "coco":
        ds = CocoPersonDataset(
            image_dir=args.image_dir, ann_path=args.ann_path,
            image_size=args.image_size, dataset_name=args.dataset_name,
        )
        collate = coco_collate
    else:
        ds = CrowdHumanDataset(
            image_dir=args.image_dir, odgt_path=args.ann_path, image_size=args.image_size,
        )
        collate = ch_collate

    if args.limit > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    return ds, collate


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | checkpoint={args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    w_offset = ckpt.get("w_offset", None)
    print(f"checkpoint w_offset={w_offset} | trained epoch={ckpt.get('epoch')}")

    model = OffsetIoUDet(num_classes=1, num_queries=args.num_queries,
                         pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds, collate = build_dataset(args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers)
    print(f"OOD eval images: {len(ds)}")

    metrics = evaluate_model(model, loader, device=device, image_size=args.image_size)
    print(f"OOD metrics on {args.dataset_name} "
          f"(w_offset={w_offset}): MR^-2={metrics['mr']:.4f} "
          f"AP={metrics['ap']:.4f} JI={metrics['ji']:.4f}")

    out_dir = Path(args.out_dir) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": args.checkpoint,
        "w_offset": w_offset,
        "dataset_name": args.dataset_name,
        "num_images": len(ds),
        **metrics,
    }
    out_path = out_dir / f"ood_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
