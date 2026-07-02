"""Location-aware (offset) matching cost.

This is the core contribution of OffsetIoU-Det. In dense and occluded scenes,
two nearby ground-truth objects can produce high IoU/GIoU with the same
prediction, making the Hungarian assignment ambiguous. We add a term that
penalizes the normalized distance between predicted and ground-truth box
centers, so that a prediction is discouraged from matching a distant object
even when their boxes happen to overlap.

For a prediction i (center c_pred_i) and a ground-truth j (center c_gt_j,
width w_j, height h_j):

    C_offset(i, j) = || c_pred_i - c_gt_j ||_2 / (sqrt(w_j * h_j) + eps)

Setting the weight of this term to 0 recovers the standard DETR matching cost,
which is exactly the baseline used in the ablation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def offset_cost(
    pred_boxes_cxcywh: Tensor,
    tgt_boxes_cxcywh: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Pairwise location-aware cost between predictions and targets.

    Args:
        pred_boxes_cxcywh: (N, 4) predicted boxes in (cx, cy, w, h), normalized.
        tgt_boxes_cxcywh:  (M, 4) target boxes in (cx, cy, w, h), normalized.
        eps: numerical stability constant for the size normalizer.

    Returns:
        cost: (N, M) non-negative cost matrix. Larger = worse match.
    """
    pred_centers = pred_boxes_cxcywh[:, :2]  # (N, 2)
    tgt_centers = tgt_boxes_cxcywh[:, :2]    # (M, 2)

    # Pairwise Euclidean distance between centers: (N, M).
    center_dist = torch.cdist(pred_centers, tgt_centers, p=2)

    # Size normalizer per target: sqrt(w * h). Shape: (M,) -> (1, M).
    tgt_wh = tgt_boxes_cxcywh[:, 2:]                      # (M, 2)
    tgt_size = torch.sqrt(
        (tgt_wh[:, 0] * tgt_wh[:, 1]).clamp(min=0)
    )                                                     # (M,)
    normalizer = (tgt_size + eps).unsqueeze(0)           # (1, M)

    return (center_dist / normalizer).clamp(max=4.0)
