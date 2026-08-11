"""Standalone feature extractors and classifier for Dual_D training.

Module purpose:
    Provide the visual feature extractor, infrared feature extractor, classifier,
    and classification criterion required by the standalone Dual_D training
    script. These implementations are self-contained and do not import original
    JMDA-Net files.

Public interfaces:
    - VisualFeatureExtractor
    - IRFeatureExtractor
    - ComplexAISFeatureExtractor
    - AISMlpFeatureExtractor
    - AISFeatureExtractor
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


class ComplexConv1d(nn.Module):
    """Complex-valued 1-D convolution implemented with real convolutions.

    For ``W=A+iB`` and ``X=I+iQ`` this layer computes
    ``real=A*I-B*Q`` and ``imag=B*I+A*Q``.  It preserves I/Q coupling instead
    of treating the two channels as unrelated real-valued attributes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ):
        super().__init__()
        self.real_conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            padding=int(padding),
            bias=False,
        )
        self.imag_conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            padding=int(padding),
            bias=False,
        )

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real and imaginary outputs of the complex convolution."""

        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        return real_out, imag_out


class ComplexAISFeatureExtractor(nn.Module):
    """Complex-valued encoder for two-channel I/Q AIS sequences."""

    def __init__(self, output_dim: int = 512, dropout: float = 0.10):
        super().__init__()
        self.conv1 = ComplexConv1d(1, 32, kernel_size=5, padding=2)
        self.real_bn1 = nn.BatchNorm1d(32)
        self.imag_bn1 = nn.BatchNorm1d(32)
        self.conv2 = ComplexConv1d(32, 64, kernel_size=3, padding=1)
        self.real_bn2 = nn.BatchNorm1d(64)
        self.imag_bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(output_dim)),
        )

    @staticmethod
    def _activate_pair(
        real: torch.Tensor,
        imag: torch.Tensor,
        real_bn: nn.Module,
        imag_bn: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real = F.relu(real_bn(real), inplace=True)
        imag = F.relu(imag_bn(imag), inplace=True)
        return real, imag

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode AIS tensors with shape ``[batch, 2, sequence_length]``."""

        if signals.dim() != 3 or signals.size(1) != 2:
            raise ValueError(
                "Complex AIS encoder expects [batch, 2, sequence_length], "
                f"got {tuple(signals.shape)}."
            )
        real = signals[:, 0:1, :]
        imag = signals[:, 1:2, :]
        real, imag = self.conv1(real, imag)
        real, imag = self._activate_pair(real, imag, self.real_bn1, self.imag_bn1)
        real, imag = self.pool(real), self.pool(imag)
        real, imag = self.conv2(real, imag)
        real, imag = self._activate_pair(real, imag, self.real_bn2, self.imag_bn2)
        real = torch.flatten(self.avgpool(real), 1)
        imag = torch.flatten(self.avgpool(imag), 1)
        return self.proj(torch.cat([real, imag], dim=1))


class AISMlpFeatureExtractor(nn.Module):
    """MLP alternative for pre-extracted or non-complex AIS feature vectors."""

    def __init__(
        self,
        sequence_length: int = 128,
        output_dim: int = 512,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.net = nn.Sequential(
            nn.Linear(2 * self.sequence_length, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(output_dim)),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Flatten a fixed-size two-channel AIS tensor and return features."""

        if signals.dim() != 3 or signals.size(1) != 2:
            raise ValueError(
                "AIS MLP encoder expects [batch, 2, sequence_length], "
                f"got {tuple(signals.shape)}."
            )
        if signals.size(2) != self.sequence_length:
            raise ValueError(
                f"Expected AIS sequence length {self.sequence_length}, "
                f"got {signals.size(2)}."
            )
        return self.net(torch.flatten(signals, 1))


def AISFeatureExtractor(
    encoder_type: str = "complex",
    sequence_length: int = 128,
    output_dim: int = 512,
    dropout: float = 0.10,
) -> nn.Module:
    """Build the configured AIS encoder."""

    encoder_type = str(encoder_type).lower()
    if encoder_type == "complex":
        return ComplexAISFeatureExtractor(output_dim=output_dim, dropout=dropout)
    if encoder_type == "mlp":
        return AISMlpFeatureExtractor(
            sequence_length=sequence_length,
            output_dim=output_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported AIS encoder type: {encoder_type}")


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
