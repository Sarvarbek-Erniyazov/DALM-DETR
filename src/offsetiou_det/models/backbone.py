"""Convolutional backbone for OffsetIoU-Det.

Wraps a torchvision ResNet and returns the final feature map (stride 32),
projected to the transformer's hidden dimension by a 1x1 convolution.
ImageNet-pretrained weights give a strong starting point and let the whole
model train on a single 8 GB GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import resnet50, ResNet50_Weights


class Backbone(nn.Module):
    """ResNet-50 feature extractor with a projection to ``hidden_dim``.

    Args:
        hidden_dim: output channel dimension fed to the transformer.
        pretrained: load ImageNet-pretrained weights.
        freeze_bn:  keep BatchNorm layers in eval mode (standard for detection
                    fine-tuning, where batch sizes are small).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)

        # Keep everything up to (and including) the last residual stage.
        self.body = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.num_channels = 2048  # ResNet-50 layer4 output channels
        self.input_proj = nn.Conv2d(self.num_channels, hidden_dim, kernel_size=1)
        self.freeze_bn = freeze_bn

    def train(self, mode: bool = True):
        """Override to keep frozen BatchNorm in eval mode."""
        super().train(mode)
        if self.freeze_bn:
            for module in self.body.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Args:
            images: (B, 3, H, W) normalized image tensor.
        Returns:
            features: (B, hidden_dim, H/32, W/32) feature map.
        """
        feats = self.body(images)
        return self.input_proj(feats)
