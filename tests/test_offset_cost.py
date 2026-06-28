"""Unit tests for the location-aware (offset) matching cost."""

import torch

from offsetiou_det.matching.offset_cost import offset_cost


def test_exact_match_is_zero():
    """A prediction identical to a target has ~zero offset cost."""
    box = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    cost = offset_cost(box, box)
    assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-5)


def test_closer_target_has_lower_cost():
    """In a dense pair, the prediction prefers the spatially closer target."""
    gt = torch.tensor([
        [0.50, 0.50, 0.20, 0.40],   # closer
        [0.58, 0.50, 0.20, 0.40],   # farther
    ])
    pred = torch.tensor([[0.50, 0.50, 0.20, 0.40]])
    cost = offset_cost(pred, gt)
    assert cost[0, 0] < cost[0, 1]


def test_size_normalization():
    """Same center distance costs less for a larger target."""
    pred = torch.tensor([[0.50, 0.50, 0.10, 0.10]])
    small_gt = torch.tensor([[0.55, 0.50, 0.10, 0.10]])
    large_gt = torch.tensor([[0.55, 0.50, 0.40, 0.40]])
    c_small = offset_cost(pred, small_gt)
    c_large = offset_cost(pred, large_gt)
    assert c_large[0, 0] < c_small[0, 0]


def test_shape():
    """Output shape is (num_preds, num_targets)."""
    pred = torch.rand(7, 4)
    tgt = torch.rand(3, 4)
    assert offset_cost(pred, tgt).shape == (7, 3)
