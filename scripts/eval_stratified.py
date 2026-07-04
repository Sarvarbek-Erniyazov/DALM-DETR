"""Density-stratified evaluation of trained checkpoints.

Runs one or more checkpoints over an evaluation set, splits images into
density strata (sparse / medium / dense by GT person count), and reports
MR^-2 / AP / JI per stratum for each model side by side.

This directly tests the project central claim: the location-aware matching
term should help most where crowding is worst, and stay neutral where it
is not needed.

Example:
    python scripts/eval_stratified.py \\
        --checkpoints outputs/checkpoints/offsetiou_baseline_v3_best.pth \\
                      outputs/checkpoints/offsetiou_ours_const_v3_best.pth \\
                      outputs/checkpoints/offsetiou_ours_adaptive_v3_best.pth \\
        --names baseline ours_const ours_adaptive \\
        --dataset_type crowdhuman \\
        --image_dir datasets/crowdhuman/Images \\
        --ann_path datasets/crowdhuman/annotation_val.odgt \\
        --image_size 640 --tag crowdhuman_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from offsetiou_det.models.detector import OffsetIoUDet
from offsetiou_det.losses.box_ops import box_cxcywh_to_xyxy
from offsetiou_det.evaluation.density_analysis import (
    evaluate_stratified, format_stratified_table)
from offsetiou_det.data.coco_person import CocoPersonDataset, collate_fn as coco_collate
from offsetiou_det.data.crowdhuman import CrowdHumanDataset, collate_fn as ch_collate


def parse_args():
    p = argparse.ArgumentParser(description="Density-stratified evaluation.")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True,
                   help="display name per checkpoint (same order)")
    p.add_argument("--dataset_type", choices=["coco", "crowdhuman"], required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--ann_path", required=True)
    p.add_argument("--dataset_name", default="eval")
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--score_thresh", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--tag", default="stratified")
    return p.parse_args()


def build_dataset(args):
    if args.dataset_type == "coco":
        ds = CocoPersonDataset(args.image_dir, args.ann_path,
                               image_size=args.image_size,
                               dataset_name=args.dataset_name)
        return ds, coco_collate
    ds = CrowdHumanDataset(args.image_dir, args.ann_path, image_size=args.image_size)
    return ds, ch_collate


@torch.no_grad()
def collect_predictions(model, loader, device, image_size, score_thresh):
    """Run the model once; return (predictions, ground_truths) in pixel xyxy."""
    model.eval()
    predictions, ground_truths = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        pred_logits, pred_boxes = model(images)[:2]
        scores = pred_logits.softmax(-1)[..., 0]
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes) * image_size
        for b in range(images.shape[0]):
            s = scores[b].cpu().numpy()
            keep = s >= score_thresh
            predictions.append({"boxes": boxes_xyxy[b][keep].cpu().numpy(),
                                "scores": s[keep]})
            gt = targets[b]["boxes"].numpy()
            if len(gt):
                cx, cy, w, h = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
                gt_xyxy = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 1) * image_size
            else:
                gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            ground_truths.append(gt_xyxy)
    return predictions, ground_truths


def main():
    args = parse_args()
    assert len(args.checkpoints) == len(args.names), "--checkpoints and --names must align"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, collate = build_dataset(args)
    if args.limit > 0:
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers)
    print(f"device={device} | eval images: {len(ds)}")

    results_by_model = {}
    for ckpt_path, name in zip(args.checkpoints, args.names):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = OffsetIoUDet(num_classes=1, num_queries=args.num_queries,
                             pretrained_backbone=False).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"[{name}] loaded {ckpt_path} (epoch {ckpt.get('epoch')}, "
              f"w_offset={ckpt.get('w_offset')})")

        preds, gts = collect_predictions(model, loader, device,
                                         args.image_size, args.score_thresh)
        results_by_model[name] = evaluate_stratified(preds, gts)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    table = format_stratified_table(results_by_model)
    print()
    print(table)

    out_dir = Path(args.out_dir) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stratified_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(results_by_model, f, indent=2)
    (out_dir / f"stratified_{args.tag}.txt").write_text(table)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
