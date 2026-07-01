"""Bidirectional feature generators for dual-domain adaptation.

Module purpose:
    Implement feature-level analogues of the TACL forward/backward mappings:
    target-weather features are translated to source/stable features, and
    source/stable features are translated back to target-weather features.

Public interfaces:
    - ResidualFeatureMapper: one residual MLP domain mapper.
    - BidirectionalFeatureTranslator: owns target_to_source and source_to_target
      mappings plus cycle/identity helpers.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualFeatureMapper(nn.Module):
    """Residual MLP mapper from one feature domain to another."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.10,
        residual_scale: float = 0.50,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.residual_scale = float(residual_scale)

        layers = []
        input_dim = self.feature_dim
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            if use_layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, self.feature_dim))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize Linear layers with Xavier weights."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to the paired domain while preserving semantics."""

        residual = torch.tanh(self.net(features))
        return features + self.residual_scale * residual


class BidirectionalFeatureTranslator(nn.Module):
    """Twin feature mapper with source-target closed-loop helpers."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.10,
        residual_scale: float = 0.50,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.target_to_source = ResidualFeatureMapper(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual_scale=residual_scale,
            use_layer_norm=use_layer_norm,
        )
        self.source_to_target = ResidualFeatureMapper(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual_scale=residual_scale,
            use_layer_norm=use_layer_norm,
        )

    def cycle_from_target(self, target_features: torch.Tensor):
        """Return target -> source-like -> reconstructed-target features."""

        source_like = self.target_to_source(target_features)
        target_reconstruction = self.source_to_target(source_like)
        return source_like, target_reconstruction

    def cycle_from_source(self, source_features: torch.Tensor):
        """Return source -> target-like -> reconstructed-source features."""

        target_like = self.source_to_target(source_features)
        source_reconstruction = self.target_to_source(target_like)
        return target_like, source_reconstruction

    def identities(self, source_features: torch.Tensor, target_features: torch.Tensor):
        """Return identity-path outputs for source and target domains."""

        source_identity = self.target_to_source(source_features)
        target_identity = self.source_to_target(target_features)
        return source_identity, target_identity

