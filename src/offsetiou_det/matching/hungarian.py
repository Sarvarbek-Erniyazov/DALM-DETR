"""Hungarian (bipartite) matcher with an optional location-aware term.

This module assigns each predicted query to at most one ground-truth object
by minimizing a total cost built from four parts:

    C_total = w_class * C_class
            + w_l1    * C_l1
            + w_giou  * C_giou
            + w_offset * C_offset      <-- our contribution

Setting ``w_offset = 0`` recovers the standard DETR matcher exactly, which is
the baseline used in the ablation. ``w_offset > 0`` enables the location-aware
term defined in ``offset_cost.py``.

The matcher operates on a single image (one set of predictions vs one set of
targets) and returns the matched index pairs. Batch handling is done by the
caller, which loops over images.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from ..losses.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .offset_cost import offset_cost, density_weights


class HungarianMatcher:
    """Bipartite matcher between predictions and ground-truth objects.

    Args:
        w_class:  weight of the classification cost.
        w_l1:     weight of the L1 box-coordinate cost.
        w_giou:   weight of the (negative) GIoU cost.
        w_offset: weight of the location-aware offset cost (0 = DETR baseline).
    """

    def __init__(
        self,
        w_class: float = 1.0,
        w_l1: float = 5.0,
        w_giou: float = 2.0,
        w_offset: float = 0.0,
        adaptive_offset: bool = False,
    ) -> None:
        self.w_class = w_class
        self.w_l1 = w_l1
        self.w_giou = w_giou
        self.w_offset = w_offset
        self.adaptive_offset = adaptive_offset
        assert w_class != 0 or w_l1 != 0 or w_giou != 0, "all costs cannot be zero"

    @torch.no_grad()
    def __call__(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        tgt_labels: Tensor,
        tgt_boxes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the optimal assignment for one image.

        Args:
            pred_logits: (N, num_classes) raw classification logits.
            pred_boxes:  (N, 4) predicted boxes in (cx, cy, w, h), normalized.
            tgt_labels:  (M,) ground-truth class indices.
            tgt_boxes:   (M, 4) target boxes in (cx, cy, w, h), normalized.

        Returns:
            (row_idx, col_idx): matched prediction indices and target indices,
            each of shape (min(N, M)... ) as torch.long tensors.
        """
        num_preds = pred_logits.shape[0]
        num_tgts = tgt_labels.shape[0]

        # Trivial case: nothing to match.
        if num_preds == 0 or num_tgts == 0:
            empty = torch.empty(0, dtype=torch.long, device=pred_logits.device)
            return empty, empty

        # --- Classification cost: -p(correct class) ---
        prob = pred_logits.softmax(-1)            # (N, num_classes)
        cost_class = -prob[:, tgt_labels]         # (N, M)

        # --- L1 cost between box coordinates ---
        cost_l1 = torch.cdist(pred_boxes, tgt_boxes, p=1)  # (N, M)

        # --- GIoU cost (negative GIoU; lower is better) ---
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes),
        )                                         # (N, M)

        cost = (
            self.w_class * cost_class
            + self.w_l1 * cost_l1
            + self.w_giou * cost_giou
        )

        # --- Location-aware offset cost (our term) ---
        if self.w_offset != 0:
            c_off = offset_cost(pred_boxes, tgt_boxes)
            if self.adaptive_offset:
                c_off = c_off * density_weights(tgt_boxes).unsqueeze(0)
            cost = cost + self.w_offset * c_off

        # Solve the assignment on CPU (scipy), then move indices back.
        cost_np = cost.detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_np)

        device = pred_logits.device
        return (
            torch.as_tensor(row_idx, dtype=torch.long, device=device),
            torch.as_tensor(col_idx, dtype=torch.long, device=device),
        )
