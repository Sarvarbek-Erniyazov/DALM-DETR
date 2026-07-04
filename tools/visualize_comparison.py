"""Qualitative side-by-side comparison of two detectors on crowded images.

For each selected image, draws two panels (model A vs model B) with:
    - GREEN solid boxes:  ground-truth persons the model FOUND (matched TP)
    - RED dashed boxes:   ground-truth persons the model MISSED (FN)
    - YELLOW thin boxes:  false-positive predictions (optional, off by default)

Miss = no prediction with IoU >= iou_thr and score >= score_thresh matched to
that GT (greedy matching by descending score, one GT used at most once --
consistent with crowd_metrics).

The most persuasive figures come from images where model A misses people that
model B finds; use --sort_by_gap to automatically surface those images first.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless: write files, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch.utils.data import DataLoader, Subset

from offsetiou_det.models.detector import OffsetIoUDet
from offsetiou_det.losses.box_ops import box_cxcywh_to_xyxy
from offsetiou_det.evaluation.crowd_metrics import _iou_matrix
from offsetiou_det.data.crowdhuman import CrowdHumanDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side qualitative comparison.")
    p.add_argument("--checkpoint_a", required=True)
    p.add_argument("--checkpoint_b", required=True)
    p.add_argument("--name_a", default="baseline")
    p.add_argument("--name_b", default="ours")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--ann_path", required=True)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--num_images", type=int, default=6)
    p.add_argument("--score_thresh", type=float, default=0.5,
                   help="operating point for visualization")
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=300,
                   help="scan at most this many val images to pick examples")
    p.add_argument("--sort_by_gap", action="store_true",
                   help="pick images where B finds most GT that A misses")
    p.add_argument("--show_fp", action="store_true", help="also draw false positives")
    p.add_argument("--out_dir", default="outputs/figures")
    return p.parse_args()


def load_model(path, num_queries, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = OffsetIoUDet(num_classes=1, num_queries=num_queries,
                         pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("w_offset"), ckpt.get("epoch")


@torch.no_grad()
def predict(model, images, image_size, score_thresh, device):
    """Return per-image (boxes_xyxy_pixels, scores) above threshold."""
    logits, boxes = model(images.to(device))[:2]
    scores = logits.softmax(-1)[..., 0]
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * image_size
    out = []
    for b in range(images.shape[0]):
        s = scores[b].cpu().numpy()
        keep = s >= score_thresh
        out.append((boxes_xyxy[b][keep].cpu().numpy(), s[keep]))
    return out


def match_gt(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Greedy match; return boolean mask over GT: True = found (TP)."""
    found = np.zeros(len(gt_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return found
    order = np.argsort(-pred_scores)
    ious = _iou_matrix(pred_boxes[order], gt_boxes)
    for i in range(len(order)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not found[j]:
            found[j] = True
    return found


def unnormalize(img_tensor):
    """CHW normalized tensor -> HWC float image in [0,1] for display."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    x = img_tensor.numpy() * std + mean
    return np.clip(x.transpose(1, 2, 0), 0, 1)


def draw_panel(ax, image, gt_boxes, found_mask, title, fp_boxes=None):
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    for j, (x0, y0, x1, y1) in enumerate(gt_boxes):
        if found_mask[j]:
            ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, edgecolor="lime", linewidth=1.6))
        else:
            ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, edgecolor="red", linewidth=2.0,
                         linestyle="--"))
    if fp_boxes is not None:
        for (x0, y0, x1, y1) in fp_boxes:
            ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, edgecolor="yellow", linewidth=0.8))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_a, w_a, ep_a = load_model(args.checkpoint_a, args.num_queries, device)
    model_b, w_b, ep_b = load_model(args.checkpoint_b, args.num_queries, device)
    print(f"A={args.name_a} (w_offset={w_a}, epoch {ep_a}) | "
          f"B={args.name_b} (w_offset={w_b}, epoch {ep_b}) | device={device}")

    ds = CrowdHumanDataset(args.image_dir, args.ann_path, image_size=args.image_size)
    if args.limit > 0:
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    candidates = []
    for images, targets in loader:
        gt = targets[0]["boxes"].numpy()
        if len(gt) == 0:
            continue
        cx, cy, w, h = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
        gt_xyxy = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1) * args.image_size

        (pa_boxes, pa_scores), = predict(model_a, images, args.image_size,
                                          args.score_thresh, device)
        (pb_boxes, pb_scores), = predict(model_b, images, args.image_size,
                                          args.score_thresh, device)
        found_a = match_gt(pa_boxes, pa_scores, gt_xyxy, args.iou_thr)
        found_b = match_gt(pb_boxes, pb_scores, gt_xyxy, args.iou_thr)

        gap = int(((~found_a) & found_b).sum())
        candidates.append((gap, images[0], gt_xyxy, found_a, found_b,
                           pa_boxes, pb_boxes, targets[0]["image_id"]))

    if args.sort_by_gap:
        candidates.sort(key=lambda t: -t[0])
    chosen = candidates[: args.num_images]

    for idx, (gap, img, gt_xyxy, fa, fb, fpa, fpb, image_id) in enumerate(chosen):
        image = unnormalize(img)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        draw_panel(axes[0], image, gt_xyxy, fa,
                   f"{args.name_a}: {int(fa.sum())}/{len(fa)} found, "
                   f"{int((~fa).sum())} missed",
                   fpa if args.show_fp else None)
        draw_panel(axes[1], image, gt_xyxy, fb,
                   f"{args.name_b}: {int(fb.sum())}/{len(fb)} found, "
                   f"{int((~fb).sum())} missed",
                   fpb if args.show_fp else None)
        fig.suptitle(f"GREEN = found person, RED dashed = missed person | "
                     f"B rescues {gap} persons missed by A", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"compare_{idx:02d}_gap{gap}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path} (gap={gap}, id={image_id})")

    print(f"Done. {len(chosen)} figures in {out_dir}")


if __name__ == "__main__":
    main()
