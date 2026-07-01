"""Standalone feature extractors and classifier for Dual_D training.

Module purpose:
    Provide the visual feature extractor, infrared feature extractor, classifier,
    and classification criterion required by the standalone Dual_D training
    script. These implementations are self-contained and do not import original
    JMDA-Net files.

Public interfaces:
    - VisualFeatureExtractor
    - IRFeatureExtractor
    - Classifier
    - LabelSmoothingCrossEntropy
    - set_requires_grad
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tv_models


class VisualFeatureExtractor(nn.Module):
    """ResNet-18 visual feature extractor.

    Args:
        output_dim: Output feature dimension.
        pretrained: If true, use torchvision's ImageNet weights when available.
            Set false in offline environments if weights are not cached.
    """

    def __init__(self, output_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = tv_models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = "DEFAULT"
        resnet = tv_models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, int(output_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract flattened projected visual features."""

        features = self.features(images)
        features = torch.flatten(features, 1)
        return self.proj(features)


class IRFeatureExtractor(nn.Module):
    """Lightweight convolutional encoder for infrared images."""

    def __init__(self, input_channels: int = 3, output_dim: int = 512):
        super().__init__()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.enc1 = conv_block(input_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, int(output_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract flattened projected infrared features."""

        features = self.enc1(images)
        features = self.enc2(features)
        features = self.enc3(features)
        features = self.enc4(features)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.proj(features)


class Classifier(nn.Module):
    """MLP classifier for fused multimodal features."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.30):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(int(input_dim), 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(num_classes)),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize Linear layers with Xavier weights."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits."""

        return self.fc(features)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, eps: float = 0.10, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross entropy."""

        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.sum(dim=-1)
        if self.reduction == "mean":
            smooth_loss = smooth_loss.mean()
        elif self.reduction == "sum":
            smooth_loss = smooth_loss.sum()
        nll = F.nll_loss(log_probs, target, reduction=self.reduction)
        return smooth_loss * self.eps / num_classes + (1.0 - self.eps) * nll


def set_requires_grad(model: nn.Module, requires_grad: bool = False) -> None:
    """Set requires_grad on all parameters of a module."""

    for parameter in model.parameters():
        parameter.requires_grad = bool(requires_grad)

