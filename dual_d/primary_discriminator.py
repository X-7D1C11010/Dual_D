"""Primary source-domain discriminator.

Module purpose:
    Independently implement the primary discriminator used to judge whether a
    feature belongs to the stable/source feature distribution. In the current
    JMDA-Net context, this corresponds to sunny source-domain fused features
    after TAL projection.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .gradient_reversal import gradient_reverse


class PrimarySourceDiscriminator(nn.Module):
    """Binary discriminator for source/stable-domain realism."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Iterable[int] = (512, 256, 128),
        dropout: float = 0.30,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        layers = []
        input_dim = int(feature_dim)

        for hidden_dim in hidden_dims:
            linear = nn.Linear(input_dim, int(hidden_dim))
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.extend(
                [
                    linear,
                    nn.LayerNorm(int(hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            input_dim = int(hidden_dim)

        output = nn.Linear(input_dim, 2)
        if use_spectral_norm:
            output = spectral_norm(output)
        layers.append(output)

        self.discriminator = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize non-normalized Linear layers safely."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        features: torch.Tensor,
        alpha: float = 1.0,
        use_grl: bool = False,
    ) -> torch.Tensor:
        """Return binary source-domain logits."""

        if use_grl:
            features = gradient_reverse(features, alpha)
        return self.discriminator(features)

