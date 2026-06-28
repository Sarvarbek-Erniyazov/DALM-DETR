"""Crowd-detection metrics: MR^-2, AP@0.5, and Jaccard Index (JI).

These are the standard CrowdHuman metrics. MR^-2 (log-average miss rate over
FPPI in [1e-2, 1e0]) is the primary indicator and is exactly the quantity our
location-aware matching aims to lower, since it penalizes missed detections.

Inputs are plain numpy arrays so the metric code has no torch dependency and is
easy to unit-test. Boxes are in pixel xyxy format. A prediction matches a GT if
IoU >= iou_thr; matching is greedy by descending score (standard for detection
evaluation), and each GT is used at most once.
"""

from __future__ import annotations

import numpy as np


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Pairwise IoU between pred (N,4) and gt (M,4), both xyxy."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.clip(union, 1e-7, None)


def _match_image(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Greedy match one image. Returns (tp_flags, scores, num_gt).

    tp_flags[i] = 1 if prediction i is a true positive, else 0, ordered by the
    descending-score order used internally.
    """
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    ious = _iou_matrix(pred_boxes, gt_boxes)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=np.float32)

    for i in range(len(pred_boxes)):
        if ious.shape[1] == 0:
            break
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not gt_used[j]:
            tp[i] = 1.0
            gt_used[j] = True
    return tp, pred_scores, len(gt_boxes)


def evaluate(predictions, ground_truths, iou_thr=0.5):
    """Compute MR^-2, AP@0.5, JI over a dataset.

    Args:
        predictions: list (per image) of dicts with
            "boxes": (N,4) xyxy, "scores": (N,).
        ground_truths: list (per image) of (M,4) xyxy arrays.
        iou_thr: IoU threshold for a match.

    Returns:
        dict with "mr", "ap", "ji".
    """
    all_tp, all_scores = [], []
    total_gt = 0
    n_images = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        pb = np.asarray(pred["boxes"], dtype=np.float32).reshape(-1, 4)
        ps = np.asarray(pred["scores"], dtype=np.float32).reshape(-1)
        gb = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        tp, scores, n_gt = _match_image(pb, ps, gb, iou_thr)
        all_tp.append(tp)
        all_scores.append(scores)
        total_gt += n_gt

    if not all_tp or total_gt == 0:
        return {"mr": 1.0, "ap": 0.0, "ji": 0.0}

    tp = np.concatenate(all_tp)
    scores = np.concatenate(all_scores)

    # Sort all predictions by descending score globally.
    order = np.argsort(-scores)
    tp = tp[order]
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / total_gt
    precision = cum_tp / np.clip(cum_tp + cum_fp, 1e-7, None)

    ap = _average_precision(recall, precision)
    mr = _log_average_miss_rate(cum_fp, recall, n_images)
    ji = _jaccard_index(cum_tp, cum_fp, total_gt)

    return {"mr": float(mr), "ap": float(ap), "ji": float(ji)}


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC-style AP: area under the precision-recall curve (all points)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing.
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _log_average_miss_rate(cum_fp, recall, n_images, num_points=9):
    """Log-average miss rate over FPPI in [1e-2, 1e0] (9 points)."""
    fppi = cum_fp / max(n_images, 1)
    miss_rate = 1.0 - recall

    ref = np.logspace(-2.0, 0.0, num_points)
    samples = []
    for r in ref:
        valid = fppi <= r
        # Lowest miss rate achievable at this FPPI budget.
        mr = miss_rate[valid].min() if valid.any() else 1.0
        samples.append(max(mr, 1e-10))
    return np.exp(np.mean(np.log(samples)))


def _jaccard_index(cum_tp, cum_fp, total_gt) -> float:
    """Best Jaccard Index = TP / (TP + FP + FN) over the score sweep."""
    fn = total_gt - cum_tp
    denom = cum_tp + cum_fp + fn
    ji = cum_tp / np.clip(denom, 1e-7, None)
    return float(ji.max()) if len(ji) else 0.0
