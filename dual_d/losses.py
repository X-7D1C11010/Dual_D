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
    - batch_class_prototypes(...)
    - class_prototype_contrastive_loss(...)
    - safe_item(tensor)
"""

from __future__ import annotations

from typing import Optional, Tuple

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
    positive_labels: Optional[torch.Tensor] = None,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Class-aware paired contrastive loss.

    Args:
        anchor_features: Generated features to be pulled toward positives.
        positive_features: Real features used as positive candidates.
        labels: Optional class labels for anchors. If omitted, the diagonal
            pair in the batch is treated as the positive pair.
        positive_labels: Optional labels for positive candidates. If omitted,
            ``labels`` is reused for backward compatibility.
        temperature: Softmax temperature.

    Returns:
        Scalar contrastive loss.
    """

    if anchor_features.size(0) == 0:
        return anchor_features.new_tensor(0.0)

    anchor_norm = F.normalize(anchor_features, p=2, dim=1)
    positive_norm = F.normalize(positive_features, p=2, dim=1)
    logits = torch.matmul(anchor_norm, positive_norm.t()) / float(temperature)

    if labels is None:
        target = torch.arange(anchor_features.size(0), device=anchor_features.device)
        return F.cross_entropy(logits, target)

    anchor_labels = labels.view(-1)
    candidate_labels = (
        positive_labels.view(-1) if positive_labels is not None else anchor_labels
    )
    if anchor_labels.numel() != anchor_features.size(0):
        raise ValueError("Anchor label count does not match anchor feature count.")
    if candidate_labels.numel() != positive_features.size(0):
        raise ValueError("Candidate label count does not match positive feature count.")
    # Aggregate all same-class cross-domain positives into one probability
    # mass.  Averaging one log-probability per positive makes the unavoidable
    # loss floor depend on how often a class was sampled in the batch.  The
    # log-sum-exp form is invariant to that count and is therefore more stable
    # for the small, class-imbalanced weather domains used by this project.
    positive_mask = anchor_labels.unsqueeze(1).eq(candidate_labels.unsqueeze(0))
    valid_anchors = positive_mask.any(dim=1)
    if not bool(valid_anchors.any()):
        return anchor_features.new_tensor(0.0)
    valid_logits = logits[valid_anchors]
    valid_mask = positive_mask[valid_anchors]
    positive_log_mass = torch.logsumexp(
        valid_logits.masked_fill(~valid_mask, float("-inf")),
        dim=1,
    )
    all_log_mass = torch.logsumexp(valid_logits, dim=1)
    return (all_log_mass - positive_log_mass).mean()


def batch_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-class feature prototypes and a present-class mask.

    Prototypes are computed only from classes present in the current batch.
    Missing classes keep a zero vector and are masked out by downstream losses.
    """

    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    prototypes = features.new_zeros((num_classes, features.size(1)))
    counts = features.new_zeros((num_classes,))
    labels = labels.view(-1).long()
    valid = labels.ge(0) & labels.lt(num_classes)
    if bool(valid.any()):
        valid_labels = labels[valid]
        valid_features = features[valid]
        prototypes.index_add_(0, valid_labels, valid_features)
        counts.index_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=features.dtype))

    present_mask = counts.gt(0)
    if bool(present_mask.any()):
        prototypes[present_mask] = prototypes[present_mask] / counts[present_mask].unsqueeze(1)
    return prototypes, present_mask


def class_prototype_contrastive_loss(
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor,
    prototype_features: torch.Tensor,
    prototype_mask: torch.Tensor,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Pull anchors to the matching class prototype and away from other classes."""

    if anchor_features.size(0) == 0 or prototype_features.size(0) == 0:
        return anchor_features.new_tensor(0.0)

    num_classes = int(prototype_features.size(0))
    labels = anchor_labels.view(-1).long()
    in_range = labels.ge(0) & labels.lt(num_classes)
    valid = in_range & prototype_mask[labels.clamp(0, max(num_classes - 1, 0))].bool()
    if not bool(valid.any()):
        return anchor_features.new_tensor(0.0)

    anchor_norm = F.normalize(anchor_features[valid], p=2, dim=1)
    prototype_norm = F.normalize(prototype_features, p=2, dim=1)
    logits = torch.matmul(anchor_norm, prototype_norm.t()) / float(temperature)
    logits = logits.masked_fill(~prototype_mask.view(1, -1).bool(), -1e4)
    return F.cross_entropy(logits, labels[valid])


def safe_item(value: torch.Tensor | float | int) -> float:
    """Convert a tensor-like scalar to a Python float for logs."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
