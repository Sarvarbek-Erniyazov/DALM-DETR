"""Deformable DETR transformer (encoder-decoder).

Both encoder and decoder use multi-scale deformable attention. The encoder
refines the flattened multi-level features; the decoder attends from a set of
learned object queries to those features. Reference points (one per query, in
[0,1]) anchor the deformable sampling and are predicted from the query
embeddings.

This is the standard Deformable DETR design, kept compact for an 8 GB GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .deformable import MSDeformAttn


def _sine_pos_embed(shape, dim, device):
    """2D sine positional embedding for one feature level. Returns (H*W, dim)."""
    import math
    H, W = shape
    d = dim // 2
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pos_x = x[:, None] * div[None, :]
    pos_y = y[:, None] * div[None, :]
    pe_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=1)
    pe_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=1)
    pe = torch.zeros(H, W, dim, device=device)
    pe[:, :, :d] = pe_y[:, None, :]
    pe[:, :, d:] = pe_x[None, :, :]
    return pe.flatten(0, 1)


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model, n_levels, n_heads, n_points, d_ffn, dropout):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, pos, reference_points, spatial_shapes):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        return src


class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model, n_levels, n_heads, n_points, d_ffn, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, src, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class DeformableTransformer(nn.Module):
    """Multi-scale deformable encoder-decoder with learned object queries."""

    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=1024,
        num_queries=300,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.num_queries = num_queries

        self.encoder = nn.ModuleList([
            DeformableEncoderLayer(d_model, n_levels, n_heads, n_points, d_ffn, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DeformableDecoderLayer(d_model, n_levels, n_heads, n_points, d_ffn, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.level_embed = nn.Parameter(torch.randn(n_levels, d_model))
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)
        self.reference_points = nn.Linear(d_model, 2)

    def _get_encoder_reference_points(self, spatial_shapes, device):
        ref_list = []
        for (H, W) in spatial_shapes:
            H, W = int(H), int(W)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=device),
                torch.linspace(0.5, W - 0.5, W, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1) / H
            ref_x = ref_x.reshape(-1) / W
            ref_list.append(torch.stack((ref_x, ref_y), -1))
        ref = torch.cat(ref_list, 0)
        return ref[None, :, None].repeat(1, 1, self.n_levels, 1)

    def forward(self, features: list[Tensor]) -> Tensor:
        device = features[0].device
        spatial_shapes = torch.as_tensor(
            [(f.shape[2], f.shape[3]) for f in features], dtype=torch.long, device=device
        )

        src_flatten, pos_flatten = [], []
        for lvl, f in enumerate(features):
            B, C, H, W = f.shape
            src_flatten.append(f.flatten(2).transpose(1, 2))
            pos = _sine_pos_embed((H, W), C, device)
            pos = pos + self.level_embed[lvl]
            pos_flatten.append(pos.unsqueeze(0).repeat(B, 1, 1))
        src = torch.cat(src_flatten, 1)
        pos = torch.cat(pos_flatten, 1)

        enc_ref = self._get_encoder_reference_points(spatial_shapes, device).repeat(B, 1, 1, 1)

        for layer in self.encoder:
            src = layer(src, pos, enc_ref, spatial_shapes)

        query_pos = self.query_pos.weight.unsqueeze(0).repeat(B, 1, 1)
        tgt = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        ref = self.reference_points(query_pos).sigmoid()
        dec_ref = ref[:, :, None].repeat(1, 1, self.n_levels, 1)

        for layer in self.decoder:
            tgt = layer(tgt, query_pos, dec_ref, src, spatial_shapes)

        return tgt
