"""Bounding box utilities (format conversion, IoU, GIoU).

Boxes are represented in two formats:
    - "cxcywh": (center_x, center_y, width, height), normalized to [0, 1].
    - "xyxy":   (x_min, y_min, x_max, y_max), normalized to [0, 1].

All functions operate on torch tensors with the box coordinates in the
last dimension.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) -> (x0, y0, x1, y1)."""
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack((x0, y0, x1, y1), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert (x0, y0, x1, y1) -> (cx, cy, w, h)."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0
    return torch.stack((cx, cy, w, h), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Area of boxes in xyxy format. Shape: (N,) for input (N, 4)."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU between two sets of boxes in xyxy format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        iou:   (N, M) pairwise IoU.
        union: (N, M) pairwise union area (reused by GIoU).
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)            # (N, M, 2)
    inter = wh[..., 0] * wh[..., 1]        # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise GIoU between two sets of boxes in xyxy format.

    Returns:
        giou: (N, M) in the range [-1, 1].
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)                 # (N, M, 2)
    enclosing = wh[..., 0] * wh[..., 1]         # (N, M)

    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)
