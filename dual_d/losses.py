"""Loss functions for cooperative dual-discriminator adaptation.

Module purpose:
    Centralize reusable loss functions for discriminator updates, generator
    updates, cycle consistency, identity preservation, and paired contrastive
    learning.

Public interfaces:
    - discriminator_real_fake_loss(real_logits, fake_logits)
    - generator_fooling_loss(fake_logits)
    - cycle_consistency_loss(...)
    - identity_preservation_loss(...)
    - paired_contrastive_loss(...)
    - safe_item(tensor)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _label_tensor(logits: torch.Tensor, value: int) -> torch.Tensor:
    """Create a long label tensor matching the batch size of logits."""

    return torch.full(
        (logits.size(0),),
        int(value),
        dtype=torch.long,
        device=logits.device,
    )


def discriminator_real_fake_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Binary discriminator loss.

    Label convention:
        - class 1: real feature from the discriminator's target domain.
        - class 0: fake/generated feature.
    """

    real_labels = _label_tensor(real_logits, 1)
    fake_labels = _label_tensor(fake_logits, 0)
    real_loss = F.cross_entropy(real_logits, real_labels)
    fake_loss = F.cross_entropy(fake_logits, fake_labels)
    return 0.5 * (real_loss + fake_loss)


def generator_fooling_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator-side adversarial loss that makes fake features look real."""

    target_labels = _label_tensor(fake_logits, 1)
    return F.cross_entropy(fake_logits, target_labels)


def cycle_consistency_loss(
    source_reconstruction: torch.Tensor,
    source_features: torch.Tensor,
    target_reconstruction: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """L1 cycle consistency for both source and target closed loops."""

    source_loss = F.l1_loss(source_reconstruction, source_features)
    target_loss = F.l1_loss(target_reconstruction, target_features)
    return source_loss + target_loss


def identity_preservation_loss(
    source_identity: torch.Tensor,
    source_features: torch.Tensor,
    target_identity: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """L1 identity loss that discourages unnecessary domain rewriting."""

    source_loss = F.l1_loss(source_identity, source_features)
    target_loss = F.l1_loss(target_identity, target_features)
    return source_loss + target_loss


def paired_contrastive_loss(
    anchor_features: torch.Tensor,
    positive_features: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Class-aware paired contrastive loss.

    Args:
        anchor_features: Generated features to be pulled toward positives.
        positive_features: Real features used as positive candidates.
        labels: Optional class labels for supervised positives. If omitted, the
            diagonal pair in the batch is treated as the positive pair.
        temperature: Softmax temperature.

    Returns:
        Scalar contrastive loss.
    """

    if anchor_features.size(0) == 0:
        return anchor_features.new_tensor(0.0)

    anchor_norm = F.normalize(anchor_features, p=2, dim=1)
    positive_norm = F.normalize(positive_features, p=2, dim=1)
    logits = torch.matmul(anchor_norm, positive_norm.t()) / float(temperature)
    log_probs = F.log_softmax(logits, dim=1)

    if labels is None:
        target = torch.arange(anchor_features.size(0), device=anchor_features.device)
        return F.nll_loss(log_probs, target)

    labels = labels.view(-1)
    mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)).to(log_probs.dtype)
    mask_sum = mask.sum(dim=1).clamp_min(1.0)
    per_sample = -(mask * log_probs).sum(dim=1) / mask_sum
    return per_sample.mean()


def safe_item(value: torch.Tensor | float | int) -> float:
    """Convert a tensor-like scalar to a Python float for logs."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)

