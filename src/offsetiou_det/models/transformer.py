"""DETR-style transformer encoder-decoder.

The encoder refines the flattened backbone features; the decoder attends from a
fixed set of learned object queries to those features, producing one output
embedding per query. Sine positional encodings are added to the feature
sequence so the attention is spatially aware.

Built on ``torch.nn.Transformer`` to keep the implementation compact and
dependable on an 8 GB GPU.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sine_positional_encoding(h: int, w: int, dim: int, device) -> Tensor:
    """2D sine-cosine positional encoding.

    Returns:
        pos: (h * w, dim) flattened positional embedding.
    """
    assert dim % 4 == 0, "hidden_dim must be divisible by 4 for 2D pos-encoding"
    d = dim // 2
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()

    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))

    pos_x = x[:, None] * div[None, :]   # (w, d/2)
    pos_y = y[:, None] * div[None, :]   # (h, d/2)

    pe_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=1)   # (w, d)
    pe_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=1)   # (h, d)

    pe = torch.zeros(h, w, dim, device=device)
    pe[:, :, :d] = pe_y[:, None, :]
    pe[:, :, d:] = pe_x[None, :, :]
    return pe.flatten(0, 1)   # (h*w, dim)


class DetrTransformer(nn.Module):
    """Encoder-decoder transformer with learned object queries.

    Args:
        hidden_dim:    embedding dimension (must match the backbone projection).
        nheads:        number of attention heads.
        num_encoder_layers / num_decoder_layers: depth.
        dim_feedforward: FFN width.
        num_queries:   number of object slots (max detections per image).
        dropout:       dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        num_queries: int = 300,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Args:
            features: (B, hidden_dim, H, W) from the backbone.
        Returns:
            hs: (B, num_queries, hidden_dim) decoder output embeddings.
        """
        b, c, h, w = features.shape

        # Flatten spatial dims into a sequence and add positional encoding.
        src = features.flatten(2).permute(0, 2, 1)            # (B, H*W, C)
        pos = sine_positional_encoding(h, w, c, features.device)  # (H*W, C)
        src = src + pos.unsqueeze(0)

        # Object queries, repeated across the batch.
        query = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # (B, Q, C)

        return self.transformer(src, query)   # (B, Q, C)
