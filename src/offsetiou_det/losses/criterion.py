"""Set-prediction criterion for OffsetIoU-Det.

Given the bipartite assignment from the matcher, this computes the DETR-style
training loss:

    L = w_class * L_class   (cross-entropy over ALL queries; unmatched -> "no object")
      + w_l1    * L_l1      (L1 on matched boxes only)
      + w_giou  * L_giou    (1 - GIoU on matched boxes only)

The matcher is responsible for *which* query handles *which* object; the
criterion only scores the result. The location-aware term lives in the matcher
(it changes the assignment), so the loss formula here is identical for the
baseline and our method -- exactly what a clean ablation requires.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..matching.hungarian import HungarianMatcher


class SetCriterion(nn.Module):
    """Computes the detection loss from predictions and targets.

    Args:
        num_classes: number of foreground classes (excludes "no object").
        matcher:     a HungarianMatcher instance.
        w_class, w_l1, w_giou: loss weights.
        eos_coef:    down-weight for the "no object" class in classification.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        w_class: float = 1.0,
        w_l1: float = 5.0,
        w_giou: float = 2.0,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.w_class = w_class
        self.w_l1 = w_l1
        self.w_giou = w_giou

        # Class index `num_classes` is the "no object" / background class.
        self.no_object = num_classes
        weight = torch.ones(num_classes + 1)
        weight[self.no_object] = eos_coef
        self.register_buffer("class_weight", weight)

    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Compute losses for a batch.

        Args:
            pred_logits: (B, N, num_classes + 1) logits, incl. "no object".
            pred_boxes:  (B, N, 4) boxes in (cx, cy, w, h), normalized.
            targets:     list of length B; each dict has
                         "labels": (M_i,) and "boxes": (M_i, 4).

        Returns:
            dict with keys "loss_class", "loss_l1", "loss_giou", "loss_total".
        """
        bs, num_queries = pred_logits.shape[:2]
        device = pred_logits.device

        # Default every query to "no object"; fill matched ones below.
        target_classes = torch.full(
            (bs, num_queries), self.no_object, dtype=torch.long, device=device
        )

        matched_pred_boxes = []
        matched_tgt_boxes = []

        for b in range(bs):
            t_labels = targets[b]["labels"].to(device)
            t_boxes = targets[b]["boxes"].to(device)

            # Matcher uses only the foreground logits (drop the no-object column).
            row, col = self.matcher(
                pred_logits[b],
                pred_boxes[b],
                t_labels,
                t_boxes,
            )

            if row.numel() > 0:
                target_classes[b, row] = t_labels[col]
                matched_pred_boxes.append(pred_boxes[b, row])
                matched_tgt_boxes.append(t_boxes[col])

        # --- Classification loss over all queries ---
        loss_class = F.cross_entropy(
            pred_logits.flatten(0, 1),          # (B*N, C+1)
            target_classes.flatten(0, 1),       # (B*N,)
            weight=self.class_weight.to(device),
        )

        # --- Box losses over matched queries only ---
        if matched_pred_boxes:
            pb = torch.cat(matched_pred_boxes, dim=0)  # (K, 4)
            tb = torch.cat(matched_tgt_boxes, dim=0)   # (K, 4)
            num_boxes = max(pb.shape[0], 1)

            loss_l1 = F.l1_loss(pb, tb, reduction="sum") / num_boxes

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb)
            ).diag()
            loss_giou = (1 - giou).sum() / num_boxes
        else:
            loss_l1 = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        loss_total = (
            self.w_class * loss_class
            + self.w_l1 * loss_l1
            + self.w_giou * loss_giou
        )

        return {
            "loss_class": loss_class,
            "loss_l1": loss_l1,
            "loss_giou": loss_giou,
            "loss_total": loss_total,
        }
