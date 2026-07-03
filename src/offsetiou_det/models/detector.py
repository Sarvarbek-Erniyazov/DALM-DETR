"""OffsetIoU-Det detector: Deformable DETR with deep supervision.

Two components critical for DETR-family convergence:
    1. Auxiliary decoder losses: heads applied to EVERY decoder layer.
    2. Reference-anchored boxes: centers predicted as deltas w.r.t. the
       query reference point (inverse-sigmoid space).

The novelty of this project is the *matcher* (matching/offset_cost.py);
the detector stays standard so the ablation isolates the matching term.

Forward returns:
    pred_logits: (B, Q, num_classes + 1)  -- final decoder layer
    pred_boxes:  (B, Q, 4) in [0, 1]      -- final decoder layer
    aux_outputs: list of (logits, boxes) per intermediate layer
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbone import Backbone
from .transformer import DeformableTransformer


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    """Simple multi-layer perceptron (box regression head)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x


class OffsetIoUDet(nn.Module):
    """End-to-end Deformable DETR detector with deep supervision."""

    def __init__(
        self,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 300,
        pretrained_backbone: bool = True,
        **transformer_kwargs,
    ):
        super().__init__()
        self.backbone = Backbone(hidden_dim=hidden_dim, pretrained=pretrained_backbone)
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            n_levels=self.backbone.num_feature_levels,
            num_queries=num_queries,
            **transformer_kwargs,
        )
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def _predict(self, hs: Tensor, ref: Tensor):
        logits = self.class_head(hs)
        deltas = self.box_head(hs)
        ref_inv = inverse_sigmoid(ref)
        cxcy = (deltas[..., :2] + ref_inv).sigmoid()
        wh = deltas[..., 2:].sigmoid()
        boxes = torch.cat([cxcy, wh], dim=-1)
        return logits, boxes

    def forward(self, images: Tensor):
        features = self.backbone(images)
        hs_all, ref = self.transformer(features)

        aux_outputs = []
        for lvl in range(hs_all.shape[0] - 1):
            aux_outputs.append(self._predict(hs_all[lvl], ref))

        pred_logits, pred_boxes = self._predict(hs_all[-1], ref)
        return pred_logits, pred_boxes, aux_outputs
