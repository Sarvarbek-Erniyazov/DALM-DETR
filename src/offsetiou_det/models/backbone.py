from __future__ import annotations
import torch
from torch import Tensor, nn
from torchvision.models import resnet50, ResNet50_Weights

class Backbone(nn.Module):
    def __init__(self, hidden_dim=256, pretrained=True, freeze_bn=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),
            nn.Sequential(nn.Conv2d(1024, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),
            nn.Sequential(nn.Conv2d(2048, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),
        ])
        self.extra_conv = nn.Sequential(
            nn.Conv2d(2048, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
        )
        self.num_feature_levels = 4
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [
            self.input_proj[0](c3),
            self.input_proj[1](c4),
            self.input_proj[2](c5),
            self.extra_conv(c5),
        ]
