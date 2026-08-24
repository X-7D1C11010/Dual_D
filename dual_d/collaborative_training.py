"""Cooperative module that joins twin generators and dual discriminators.

Module purpose:
    Provide the central feature-level twin adversarial mechanism inspired by
    TACL. It creates source-like and target-like features, applies two
    independent discriminators, and exposes discriminator-side and
    generator-side loss functions.

Public interfaces:
    - DualDForwardOutput: dataclass containing all generated feature tensors.
    - DualDiscriminatorCoordinator: nn.Module coordinating all Dual_D parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .auxiliary_discriminator import AuxiliaryTargetDiscriminator
from .config import DualDConfig
from .feature_generators import BidirectionalFeatureTranslator
from .losses import (
    batch_class_prototypes,
    class_prototype_contrastive_loss,
    cycle_consistency_loss,
    discriminator_real_fake_loss,
    generator_fooling_loss,
    identity_preservation_loss,
    paired_contrastive_loss,
    safe_item,
)
from .primary_discriminator import PrimarySourceDiscriminator


@dataclass
class DualDForwardOutput:
    """Container for all feature tensors produced by the twin mapper."""

    source_features: torch.Tensor
    target_features: torch.Tensor
    source_like: torch.Tensor
    target_reconstruction: torch.Tensor
    target_like: torch.Tensor
    source_reconstruction: torch.Tensor
    source_identity: torch.Tensor
    target_identity: torch.Tensor


class DualDiscriminatorCoordinator(nn.Module):
    """Coordinator for source and target direction-aware adversarial learning.

    Args:
        config: Dual_D configuration. The feature dimension must match the
            fused feature dimension produced by the original JMDA-Net pipeline.
    """

    def __init__(self, config: DualDConfig):
        super().__init__()
        self.config = config
        self.translator = BidirectionalFeatureTranslator(
            feature_dim=config.feature_dim,
            hidden_dim=config.generator.hidden_dim,
            num_layers=config.generator.num_layers,
            dropout=config.generator.dropout,
            residual_scale=config.generator.residual_scale,
            use_layer_norm=config.generator.use_layer_norm,
        )
        self.primary_discriminator = PrimarySourceDiscriminator(
            feature_dim=config.feature_dim,
            hidden_dims=config.primary_discriminator.hidden_dims,
            dropout=config.primary_discriminator.dropout,
            use_spectral_norm=config.primary_discriminator.use_spectral_norm,
        )
        self.auxiliary_discriminator = AuxiliaryTargetDiscriminator(
            feature_dim=config.feature_dim,
            hidden_dims=config.auxiliary_discriminator.hidden_dims,
            dropout=config.auxiliary_discriminator.dropout,
            use_spectral_norm=config.auxiliary_discriminator.use_spectral_norm,
        )
        # Persistent class prototypes are initialized lazily because the
        # number of classes belongs to the dataset rather than the model JSON.
        self.register_buffer("source_prototypes_ema", torch.empty(0, config.feature_dim))
        self.register_buffer("target_prototypes_ema", torch.empty(0, config.feature_dim))
        self.register_buffer("source_prototype_seen", torch.empty(0, dtype=torch.bool))
        self.register_buffer("target_prototype_seen", torch.empty(0, dtype=torch.bool))

    @torch.no_grad()
    def _update_ema_prototypes(
        self,
        source_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor,
        target_labels: torch.Tensor,
        num_classes: int,
    ) -> None:
        """Update class-balanced source and target feature prototypes."""

        if self.source_prototypes_ema.size(0) != num_classes:
            shape = (num_classes, self.config.feature_dim)
            self.source_prototypes_ema = source_features.new_zeros(shape)
            self.target_prototypes_ema = target_features.new_zeros(shape)
            self.source_prototype_seen = torch.zeros(
                num_classes, dtype=torch.bool, device=source_features.device
            )
            self.target_prototype_seen = torch.zeros(
                num_classes, dtype=torch.bool, device=target_features.device
            )

        momentum = max(0.0, min(float(self.config.prototype_momentum), 0.9999))
        for features, labels, stored, seen in (
            (source_features, source_labels, self.source_prototypes_ema, self.source_prototype_seen),
            (target_features, target_labels, self.target_prototypes_ema, self.target_prototype_seen),
        ):
            batch_prototypes, present = batch_class_prototypes(
                features.detach(), labels, num_classes
            )
            first_observation = present & ~seen
            repeated = present & seen
            stored[first_observation] = batch_prototypes[first_observation]
            stored[repeated] = (
                momentum * stored[repeated]
                + (1.0 - momentum) * batch_prototypes[repeated]
            )
            seen |= present

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> DualDForwardOutput:
        """Run bidirectional feature translation and identity paths."""

        source_like, target_reconstruction = self.translator.cycle_from_target(target_features)
        target_like, source_reconstruction = self.translator.cycle_from_source(source_features)
        source_identity, target_identity = self.translator.identities(
            source_features,
            target_features,
        )
        return DualDForwardOutput(
            source_features=source_features,
            target_features=target_features,
            source_like=source_like,
            target_reconstruction=target_reconstruction,
            target_like=target_like,
            source_reconstruction=source_reconstruction,
            source_identity=source_identity,
            target_identity=target_identity,
        )

    def set_discriminators_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients for both discriminators."""

        for module in (self.primary_discriminator, self.auxiliary_discriminator):
            for parameter in module.parameters():
                parameter.requires_grad = bool(trainable)

    def generator_parameters(self):
        """Return parameters owned by feature translators only."""

        return self.translator.parameters()

    def discriminator_parameters(self):
        """Return parameters owned by both discriminators."""

        return list(self.primary_discriminator.parameters()) + list(
            self.auxiliary_discriminator.parameters()
        )

    def compute_discriminator_loss(
        self,
        outputs: DualDForwardOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute discriminator-side loss using detached generated features."""

        real_source_logits = self.primary_discriminator(outputs.source_features.detach())
        primary_fakes = [outputs.source_like.detach()]
        if self.config.include_reconstruction_fakes:
            primary_fakes.append(outputs.source_reconstruction.detach())

        primary_losses = [
            discriminator_real_fake_loss(real_source_logits, self.primary_discriminator(fake))
            for fake in primary_fakes
        ]
        primary_loss = torch.stack(primary_losses).mean()

        real_target_logits = self.auxiliary_discriminator(outputs.target_features.detach())
        auxiliary_fakes = [outputs.target_like.detach()]
        if self.config.include_reconstruction_fakes:
            auxiliary_fakes.append(outputs.target_reconstruction.detach())

        auxiliary_losses = [
            discriminator_real_fake_loss(real_target_logits, self.auxiliary_discriminator(fake))
            for fake in auxiliary_fakes
        ]
        auxiliary_loss = torch.stack(auxiliary_losses).mean()

        total_loss = 0.5 * (primary_loss + auxiliary_loss)
        logs = {
            "dual_d/discriminator_total": safe_item(total_loss),
            "dual_d/discriminator_primary": safe_item(primary_loss),
            "dual_d/discriminator_auxiliary": safe_item(auxiliary_loss),
        }
        return total_loss, logs

    def compute_generator_loss(
        self,
        outputs: DualDForwardOutput,
        labels: Optional[torch.Tensor] = None,
        classifier: Optional[nn.Module] = None,
        criterion_cls: Optional[nn.Module] = None,
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        adversarial_scale: float = 1.0,
        module_c_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute generator-side cooperative loss.

        Classifier feedback is optional. When provided, source-like features are
        supervised with target labels and target-like features are supervised
        with source labels, preserving class identity across both directions.
        """

        weights = self.config.loss_weights
        adversarial_scale = max(0.0, min(float(adversarial_scale), 1.0))
        module_c_scale = max(0.0, min(float(module_c_scale), 1.0))

        primary_adv = generator_fooling_loss(
            self.primary_discriminator(outputs.source_like)
        )
        auxiliary_adv = generator_fooling_loss(
            self.auxiliary_discriminator(outputs.target_like)
        )

        cycle_loss = cycle_consistency_loss(
            source_reconstruction=outputs.source_reconstruction,
            source_features=outputs.source_features,
            target_reconstruction=outputs.target_reconstruction,
            target_features=outputs.target_features,
        )
        identity_loss = identity_preservation_loss(
            source_identity=outputs.source_identity,
            source_features=outputs.source_features,
            target_identity=outputs.target_identity,
            target_features=outputs.target_features,
        )

        contrast_source_positive = outputs.source_features
        contrast_target_positive = outputs.target_features
        if self.config.detach_contrastive_positives:
            contrast_source_positive = contrast_source_positive.detach()
            contrast_target_positive = contrast_target_positive.detach()

        source_anchor_labels = target_labels if target_labels is not None else labels
        source_candidate_labels = source_labels if source_labels is not None else labels
        target_anchor_labels = source_labels if source_labels is not None else labels
        target_candidate_labels = target_labels if target_labels is not None else labels

        contrast_loss = 0.5 * (
            paired_contrastive_loss(
                outputs.source_like,
                contrast_source_positive,
                labels=source_anchor_labels,
                positive_labels=source_candidate_labels,
                temperature=self.config.contrastive_temperature,
            )
            + paired_contrastive_loss(
                outputs.target_like,
                contrast_target_positive,
                labels=target_anchor_labels,
                positive_labels=target_candidate_labels,
                temperature=self.config.contrastive_temperature,
            )
        )

        prototype_contrastive_loss = outputs.source_features.new_tensor(0.0)
        if source_labels is not None and target_labels is not None:
            if num_classes is None:
                max_label = torch.cat([source_labels.view(-1), target_labels.view(-1)]).max()
                resolved_num_classes = int(max_label.detach().cpu().item()) + 1
            else:
                resolved_num_classes = int(num_classes)

            if resolved_num_classes > 0:
                if self.training:
                    self._update_ema_prototypes(
                        outputs.source_features,
                        source_labels,
                        outputs.target_features,
                        target_labels,
                        resolved_num_classes,
                    )
                source_prototypes = self.source_prototypes_ema.detach()
                target_prototypes = self.target_prototypes_ema.detach()
                source_prototype_mask = self.source_prototype_seen
                target_prototype_mask = self.target_prototype_seen
                prototype_contrastive_loss = 0.5 * (
                    class_prototype_contrastive_loss(
                        outputs.source_like,
                        target_labels,
                        source_prototypes,
                        source_prototype_mask,
                        temperature=self.config.contrastive_temperature,
                    )
                    + class_prototype_contrastive_loss(
                        outputs.target_like,
                        source_labels,
                        target_prototypes,
                        target_prototype_mask,
                        temperature=self.config.contrastive_temperature,
                    )
                )

        classification_loss = outputs.source_features.new_tensor(0.0)
        if classifier is not None and criterion_cls is not None:
            classifier_parameters = list(classifier.parameters())
            original_requires_grad = [parameter.requires_grad for parameter in classifier_parameters]
            freeze_classifier = bool(self.config.freeze_classifier_during_feedback)
            if freeze_classifier:
                for parameter in classifier_parameters:
                    parameter.requires_grad_(False)
            try:
                if target_labels is not None:
                    classification_loss = classification_loss + criterion_cls(
                        classifier(outputs.source_like),
                        target_labels,
                    )
                if source_labels is not None:
                    classification_loss = classification_loss + criterion_cls(
                        classifier(outputs.target_like),
                        source_labels,
                    )
            finally:
                if freeze_classifier:
                    for parameter, requires_grad in zip(
                        classifier_parameters, original_requires_grad
                    ):
                        parameter.requires_grad_(requires_grad)

        weighted_cycle = module_c_scale * weights.cycle * cycle_loss
        weighted_identity = module_c_scale * weights.identity * identity_loss
        weighted_contrast = module_c_scale * weights.contrastive * contrast_loss
        weighted_prototype = (
            module_c_scale * weights.prototype_contrastive * prototype_contrastive_loss
        )
        weighted_classification = (
            module_c_scale * weights.classification * classification_loss
        )

        total_loss = (
            adversarial_scale * weights.adv_primary * primary_adv
            + adversarial_scale * weights.adv_auxiliary * auxiliary_adv
            + weighted_cycle
            + weighted_identity
            + weighted_contrast
            + weighted_prototype
            + weighted_classification
        )
        logs = {
            "dual_d/generator_total": safe_item(total_loss),
            "dual_d/adv_primary": safe_item(primary_adv),
            "dual_d/adv_auxiliary": safe_item(auxiliary_adv),
            "dual_d/cycle": safe_item(cycle_loss),
            "dual_d/identity": safe_item(identity_loss),
            "dual_d/contrastive": safe_item(contrast_loss),
            "dual_d/prototype_contrastive": safe_item(prototype_contrastive_loss),
            "dual_d/classification_feedback": safe_item(classification_loss),
            "dual_d/adversarial_scale": adversarial_scale,
            "dual_d/module_c_scale": module_c_scale,
            "dual_d/weighted_cycle": safe_item(weighted_cycle),
            "dual_d/weighted_identity": safe_item(weighted_identity),
            "dual_d/weighted_contrastive": safe_item(weighted_contrast),
            "dual_d/weighted_prototype_contrastive": safe_item(weighted_prototype),
            "dual_d/weighted_classification_feedback": safe_item(weighted_classification),
        }
        return total_loss, logs

    @torch.no_grad()
    def source_like_features(self, target_features: torch.Tensor) -> torch.Tensor:
        """Translate target-domain features to source-like features for inference."""

        return self.translator.target_to_source(target_features)
