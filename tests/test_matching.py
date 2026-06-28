"""Unit tests for the Hungarian matcher (baseline and location-aware)."""

import torch

from offsetiou_det.matching.hungarian import HungarianMatcher


def _exact_setup():
    """5 preds, 3 of which exactly match 3 targets."""
    torch.manual_seed(0)
    logits = torch.randn(5, 2)
    boxes = torch.rand(5, 4) * 0.5 + 0.25
    tgt_labels = torch.tensor([1, 1, 1])
    tgt_boxes = boxes[:3].clone()
    return logits, boxes, tgt_labels, tgt_boxes


def test_matches_all_targets():
    matcher = HungarianMatcher(w_offset=0.0)
    logits, boxes, tl, tb = _exact_setup()
    row, col = matcher(logits, boxes, tl, tb)
    assert len(row) == 3
    assert len(col) == 3
    assert sorted(col.tolist()) == [0, 1, 2]


def test_empty_targets():
    matcher = HungarianMatcher(w_offset=0.0)
    logits, boxes, _, _ = _exact_setup()
    row, col = matcher(logits, boxes, torch.empty(0, dtype=torch.long), torch.empty(0, 4))
    assert len(row) == 0 and len(col) == 0


def test_empty_predictions():
    matcher = HungarianMatcher(w_offset=0.0)
    row, col = matcher(
        torch.empty(0, 2),
        torch.empty(0, 4),
        torch.tensor([1]),
        torch.rand(1, 4),
    )
    assert len(row) == 0 and len(col) == 0


def test_offset_assigns_to_closest_target():
    """In a dense pair, the offset term sends each pred to its nearest GT."""
    tgt_boxes = torch.tensor([
        [0.45, 0.5, 0.18, 0.4],
        [0.55, 0.5, 0.18, 0.4],
    ])
    tgt_labels = torch.tensor([1, 1])
    pred_boxes = torch.tensor([
        [0.45, 0.5, 0.18, 0.4],
        [0.55, 0.5, 0.18, 0.4],
    ])
    pred_logits = torch.tensor([[0.0, 0.1], [0.0, 0.1]])

    matcher = HungarianMatcher(w_offset=5.0)
    row, col = matcher(pred_logits, pred_boxes, tgt_labels, tgt_boxes)
    assert sorted(zip(row.tolist(), col.tolist())) == [(0, 0), (1, 1)]


def test_baseline_and_offset_return_valid_permutation():
    """Both configurations must return a one-to-one assignment."""
    logits, boxes, tl, tb = _exact_setup()
    for w in (0.0, 2.0):
        matcher = HungarianMatcher(w_offset=w)
        row, col = matcher(logits, boxes, tl, tb)
        assert len(set(row.tolist())) == len(row)
        assert len(set(col.tolist())) == len(col)
