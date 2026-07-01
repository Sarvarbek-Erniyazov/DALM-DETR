"""OffsetIoU-Det detector: multi-scale backbone + deformable transformer + heads.

The model is a compact Deformable DETR. The novelty of this project is NOT the
backbone or the attention; it is the *matcher* used during training
(``matching/offset_cost.py``). The detector here is a clean, standard
Deformable DETR so the ablation isolates the effect of the location-aware
matching term.

Forward returns:
    pred_logits: (B, num_queries, num_classes + 1)   -- incl. "no object"
    pred_boxes:  (B, num_queries, 4)                 -- (cx, cy, w, h) in [0, 1]
"""

from __future__ import annotations

from torch import Tensor, nn

from .backbone import Backbone
from .transformer import DeformableTransformer


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
    """End-to-end Deformable DETR detector.

    Args:
        num_classes: number of foreground classes (CrowdHuman: 1).
        hidden_dim:  embedding dimension.
        num_queries: number of object slots.
        pretrained_backbone: load ImageNet weights for the ResNet.
        **transformer_kwargs: forwarded to DeformableTransformer.
    """

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

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(images)          # list of 4 levels
        hs = self.transformer(features)           # (B, Q, C)
        pred_logits = self.class_head(hs)         # (B, Q, num_classes + 1)
        pred_boxes = self.box_head(hs).sigmoid()  # (B, Q, 4) in [0, 1]
        return pred_logits, pred_boxes
