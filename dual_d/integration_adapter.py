"""Integration adapter between JMDA-Net features and Dual_D modules.

Module purpose:
    Provide a thin, stable interface that can be called from a new training
    script without changing any original JMDA-Net file. The adapter expects the
    caller to supply TAL-aligned fused source and target features.

Public interface:
    - DualDTrainingAdapter(config)
    - forward_features(feat_src, feat_tgt, labels)
    - compute_discriminator_loss(outputs)
    - compute_generator_loss(outputs, classifier, criterion_cls, ...)
    - inference_features(target_features, mode)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .collaborative_training import DualDForwardOutput, DualDiscriminatorCoordinator
from .config import DualDConfig


class DualDTrainingAdapter(nn.Module):
    """Adapter that exposes Dual_D losses in training-loop friendly methods."""

    def __init__(self, config: DualDConfig):
        super().__init__()
        self.config = config
        self.coordinator = DualDiscriminatorCoordinator(config)

    def forward_features(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> DualDForwardOutput:
        """Create all generated features required by Dual_D losses.

        Args:
            source_features: TAL-aligned source/sunny fused features.
            target_features: TAL-aligned target/weather fused features.
            labels: Optional class labels. Labels are not consumed in forward,
                but the argument keeps the adapter signature stable for callers.
        """

        del labels
        return self.coordinator(source_features, target_features)

    def compute_discriminator_loss(
        self,
        outputs: DualDForwardOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return loss/logs for updating only the two discriminators."""

        return self.coordinator.compute_discriminator_loss(outputs)

    def compute_generator_loss(
        self,
        outputs: DualDForwardOutput,
        labels: Optional[torch.Tensor] = None,
        classifier: Optional[nn.Module] = None,
        criterion_cls: Optional[nn.Module] = None,
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        adversarial_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return loss/logs for updating translators and optional base modules."""

        return self.coordinator.compute_generator_loss(
            outputs=outputs,
            labels=labels,
            classifier=classifier,
            criterion_cls=criterion_cls,
            source_labels=source_labels,
            target_labels=target_labels,
            adversarial_scale=adversarial_scale,
        )

    def discriminator_parameters(self):
        """Parameters for the discriminator optimizer."""

        return self.coordinator.discriminator_parameters()

    def generator_parameters(self):
        """Parameters for the generator-side optimizer."""

        return self.coordinator.generator_parameters()

    def set_discriminators_trainable(self, trainable: bool) -> None:
        """Enable or disable discriminator gradients."""

        self.coordinator.set_discriminators_trainable(trainable)

    @torch.no_grad()
    def inference_features(
        self,
        target_features: torch.Tensor,
        mode: str = "source_like",
    ) -> torch.Tensor:
        """Return inference-time features for target-domain samples.

        Args:
            target_features: TAL-aligned target-domain fused features.
            mode: ``source_like`` returns ``G_F(target_features)``. ``residual``
                returns the average of original and source-like features.
        """

        source_like = self.coordinator.source_like_features(target_features)
        if mode == "source_like":
            return source_like
        if mode == "residual":
            return 0.5 * (target_features + source_like)
        raise ValueError(f"Unsupported inference mode: {mode}")
