"""Evaluation loop: run the model over a dataset and compute crowd metrics.

The model outputs class logits and normalized (cx, cy, w, h) boxes. For each
image we:
    1. take the foreground probability as the detection score,
    2. convert boxes to pixel xyxy at the evaluation image size,
    3. drop very low-score detections (score_thresh),
then hand everything to ``crowd_metrics.evaluate``.

No NMS is applied: the model is query-based (DETR), so each query already
yields at most one object. This is exactly the regime where the location-aware
matcher is expected to help.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..evaluation.crowd_metrics import evaluate
from ..losses.box_ops import box_cxcywh_to_xyxy


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    score_thresh: float = 0.05,
    iou_thr: float = 0.5,
    image_size: int = 800,
) -> dict[str, float]:
    """Evaluate a detector and return {"mr", "ap", "ji"}.

    Boxes are scored/compared in the resized (square) image space, which is
    consistent between predictions and targets because the dataset normalizes
    boxes to [0, 1] of the original image and we scale both to ``image_size``.
    """
    model.eval()
    predictions = []
    ground_truths = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        pred_logits, pred_boxes = model(images)[:2]        # (B,Q,C+1), (B,Q,4)

        # Foreground score = 1 - P(no-object); class 0 is "person".
        probs = pred_logits.softmax(-1)                # (B,Q,C+1)
        scores = probs[..., 0]                         # (B,Q) person prob

        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes) * image_size  # (B,Q,4)

        for b in range(images.shape[0]):
            s = scores[b].cpu().numpy()
            keep = s >= score_thresh
            predictions.append({
                "boxes": boxes_xyxy[b][keep].cpu().numpy(),
                "scores": s[keep],
            })

            gt = targets[b]["boxes"].numpy()           # (M,4) normalized cxcywh
            if len(gt):
                gt_xyxy = _cxcywh_to_xyxy_np(gt) * image_size
            else:
                gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            ground_truths.append(gt_xyxy)

    return evaluate(predictions, ground_truths, iou_thr=iou_thr)


def _cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    """(N,4) cx,cy,w,h -> x0,y0,x1,y1 in numpy."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
