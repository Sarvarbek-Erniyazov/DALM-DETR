"""Pure-PyTorch multi-scale deformable attention.

Deformable DETR replaces dense attention with attention over a small set of
sampling points per query, predicted as offsets from a reference point. This
gives 10x faster convergence than vanilla DETR while keeping the end-to-end,
NMS-free formulation.

This implementation uses ``F.grid_sample`` for the sampling step, so it runs on
any CUDA or CPU PyTorch build with no custom kernel compilation -- important for
Windows. It is slightly slower than the official CUDA op but numerically
equivalent for our purposes.

Reference: Zhu et al., "Deformable DETR" (ICLR 2021).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ms_deform_attn_core_pytorch(
    value: Tensor,                 # (B, sum(HW), n_heads, head_dim)
    value_spatial_shapes: Tensor,  # (n_levels, 2) -> (H, W) per level
    sampling_locations: Tensor,    # (B, Lq, n_heads, n_levels, n_points, 2)
    attention_weights: Tensor,     # (B, Lq, n_heads, n_levels, n_points)
) -> Tensor:
    """Core deformable-attention sampling, implemented with grid_sample."""
    B, _, n_heads, head_dim = value.shape
    _, Lq, _, n_levels, n_points, _ = sampling_locations.shape

    # Split the flattened value back into per-level feature maps.
    split_sizes = [int(H * W) for H, W in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)

    # grid_sample expects coordinates in [-1, 1].
    sampling_grids = 2 * sampling_locations - 1

    sampled_value_list = []
    for lvl, (H, W) in enumerate(value_spatial_shapes):
        H, W = int(H), int(W)
        # (B, HW, n_heads, head_dim) -> (B*n_heads, head_dim, H, W)
        val_l = (
            value_list[lvl]
            .flatten(2)                       # (B, HW, n_heads*head_dim)
            .transpose(1, 2)                  # (B, n_heads*head_dim, HW)
            .reshape(B * n_heads, head_dim, H, W)
        )
        # (B, Lq, n_heads, n_points, 2) -> (B*n_heads, Lq, n_points, 2)
        grid_l = (
            sampling_grids[:, :, :, lvl]
            .transpose(1, 2)                  # (B, n_heads, Lq, n_points, 2)
            .flatten(0, 1)                    # (B*n_heads, Lq, n_points, 2)
        )
        # (B*n_heads, head_dim, Lq, n_points)
        sampled = F.grid_sample(
            val_l, grid_l, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        sampled_value_list.append(sampled)

    # Stack levels: (B*n_heads, head_dim, Lq, n_levels, n_points)
    sampled = torch.stack(sampled_value_list, dim=-2)
    # Attention weights: (B, Lq, n_heads, n_levels, n_points)
    #   -> (B*n_heads, 1, Lq, n_levels, n_points)
    attn = (
        attention_weights.transpose(1, 2)
        .reshape(B * n_heads, 1, Lq, n_levels, n_points)
    )
    out = (sampled * attn).sum(-1).sum(-1)    # (B*n_heads, head_dim, Lq)
    out = out.view(B, n_heads * head_dim, Lq).transpose(1, 2)  # (B, Lq, C)
    return out.contiguous()


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention module.

    Args:
        d_model:  feature dimension.
        n_levels: number of feature levels.
        n_heads:  number of attention heads.
        n_points: sampling points per head per level.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets so the points form a small ring around
        # the reference (the standard Deformable DETR init).
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,                 # (B, Lq, C)
        reference_points: Tensor,      # (B, Lq, n_levels, 2) in [0, 1]
        input_flatten: Tensor,         # (B, sum(HW), C)
        input_spatial_shapes: Tensor,  # (n_levels, 2)
    ) -> Tensor:
        B, Lq, _ = query.shape
        _, Lv, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(B, Lv, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points, 2
        )
        attn = self.attention_weights(query).view(
            B, Lq, self.n_heads, self.n_levels * self.n_points
        )
        attn = attn.softmax(-1).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points
        )

        # Reference point + normalized offset -> sampling location in [0, 1].
        # Normalize offsets by the per-level spatial size.
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )  # (n_levels, 2) = (W, H)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + offsets / offset_normalizer[None, None, None, :, None, :]
        )

        out = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attn
        )
        return self.output_proj(out)
