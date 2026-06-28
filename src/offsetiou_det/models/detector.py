"""OffsetIoU-Det detector: backbone + transformer + prediction heads.

The model is a compact DETR. The novelty of this project is NOT in the model
architecture but in the *matcher* used during training (see
``matching/offset_cost.py``). The detector here is a clean, standard DETR so
that the ablation isolates the effect of the location-aware matching term.

Forward returns:
    pred_logits: (B, num_queries, num_classes + 1)  -- incl. "no object"
    pred_boxes:  (B, num_queries, 4)                -- (cx, cy, w, h) in [0, 1]
"""

from __future__ import annotations

from torch import Tensor, nn

from .backbone import Backbone
from .transformer import DetrTransformer


class MLP(nn.Module):
    """Simple multi-layer perceptron (used for the box regression head)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x


class OffsetIoUDet(nn.Module):
    """End-to-end detector.

    Args:
        num_classes: number of foreground classes (CrowdHuman: 1).
        hidden_dim:  transformer/backbone embedding dimension.
        num_queries: number of object slots.
        pretrained_backbone: load ImageNet weights for the ResNet.
        **transformer_kwargs: forwarded to DetrTransformer.
    """

    def __init__(
        self,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 300,
        pretrained_backbone: bool = True,
        **transformer_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = Backbone(hidden_dim=hidden_dim, pretrained=pretrained_backbone)
        self.transformer = DetrTransformer(
            hidden_dim=hidden_dim, num_queries=num_queries, **transformer_kwargs
        )
        # +1 output class for the "no object" slot.
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
            images: (B, 3, H, W) normalized images.
        Returns:
            pred_logits: (B, num_queries, num_classes + 1)
            pred_boxes:  (B, num_queries, 4) in (cx, cy, w, h), sigmoid-normalized.
        """
        features = self.backbone(images)        # (B, C, H/32, W/32)
        hs = self.transformer(features)         # (B, Q, C)
        pred_logits = self.class_head(hs)       # (B, Q, num_classes + 1)
        pred_boxes = self.box_head(hs).sigmoid()  # (B, Q, 4) in [0, 1]
        return pred_logits, pred_boxes
