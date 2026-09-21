"""Tensor-based multimodal alignment module.

Module purpose:
    Provide a standalone implementation of the stable tensor alignment layer
    used before the Dual_D feature-level adversarial module. It projects source
    and target modality features into a shared low-dimensional space while
    maximizing class-prototype correlation between independently sampled
    domains. The tensor-product correlation is evaluated algebraically so a
    large multimodal outer-product tensor is never materialized.

Public interface:
    - TensorBasedAlignmentStable
    - PlainMultimodalProjection

Usage:
    >>> tal = TensorBasedAlignmentStable([512, 512, 512], [128, 128, 128], num_modalities=3)
    >>> projected_source, projected_target, loss = tal(
    ...     [s_vis, s_ir, s_ais], [t_vis, t_ir, t_ais]
    ... )
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TensorBasedAlignmentStable(nn.Module):
    """Stable tensor-based multimodal alignment.

    Args:
        input_dims: Input feature dimension for each modality.
        output_dims: Projected feature dimension for each modality.
        num_modalities: Number of modalities.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        num_modalities: int = 2,
        cross_domain_contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.15,
        orthogonality_weight: float = 0.0,
        orthogonalize_interval: int = 1,
        use_shared_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.num_modalities = int(num_modalities)
        self.cross_domain_contrastive_weight = float(
            cross_domain_contrastive_weight
        )
        self.contrastive_temperature = float(contrastive_temperature)
        self.orthogonality_weight = float(orthogonality_weight)
        self.orthogonalize_interval = int(orthogonalize_interval)
        self.use_shared_layer_norm = bool(use_shared_layer_norm)
        if self.cross_domain_contrastive_weight < 0.0:
            raise ValueError("cross_domain_contrastive_weight must be non-negative.")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if self.orthogonality_weight < 0.0:
            raise ValueError("orthogonality_weight must be non-negative.")
        if self.orthogonalize_interval < 0:
            raise ValueError("orthogonalize_interval must be non-negative.")

        self.U_matrices = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.input_dims[idx], self.output_dims[idx]))
                for idx in range(self.num_modalities)
            ]
        )
        self.V_matrices = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.input_dims[idx], self.output_dims[idx]))
                for idx in range(self.num_modalities)
            ]
        )
        self.shared_normalizers = nn.ModuleList(
            [nn.LayerNorm(dim) for dim in self.output_dims]
            if self.use_shared_layer_norm
            else []
        )
        self.last_diagnostics = {}
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Orthogonally initialize projection matrices."""

        for source_matrix, target_matrix in zip(self.U_matrices, self.V_matrices):
            nn.init.orthogonal_(source_matrix)
            nn.init.orthogonal_(target_matrix)

    @staticmethod
    def create_multimodal_tensor(modalities: List[torch.Tensor]) -> torch.Tensor:
        """Create a batch-wise outer-product tensor from modality features."""

        result = modalities[0]
        for idx in range(1, len(modalities)):
            modality = modalities[idx]
            result = result.unsqueeze(-1)
            modality = modality.unsqueeze(1)
            for _ in range(idx - 1):
                modality = modality.unsqueeze(1)
            result = result * modality
        return result

    @staticmethod
    def mode_n_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
        """Apply a mode-n product to a batch-first tensor."""

        tensor_mode = mode + 1
        dims = list(range(tensor.dim()))
        dims[tensor_mode], dims[-1] = dims[-1], dims[tensor_mode]
        tensor_permuted = tensor.permute(dims).contiguous()
        original_shape = tensor_permuted.shape
        mode_size = original_shape[-1]
        tensor_2d = tensor_permuted.view(-1, mode_size)
        if matrix.shape[0] != mode_size:
            raise ValueError(f"Dimension mismatch: {matrix.shape[0]} vs {mode_size}")
        result_2d = torch.matmul(tensor_2d, matrix)
        new_shape = original_shape[:-1] + (matrix.shape[1],)
        result_reshaped = result_2d.view(new_shape)
        inv_dims = [0] * len(dims)
        for idx, dim in enumerate(dims):
            inv_dims[dim] = idx
        return result_reshaped.permute(inv_dims).contiguous()

    @staticmethod
    def tensor_contraction(tensor: torch.Tensor, exclude_mode: int) -> torch.Tensor:
        """Contract all modality dimensions except one."""

        dims_to_contract = [
            dim for dim in range(1, tensor.dim()) if dim != exclude_mode + 1
        ]
        result = tensor
        for dim in sorted(dims_to_contract, reverse=True):
            result = torch.sum(result, dim=dim, keepdim=False)
        return result

    @staticmethod
    def compute_correlation_score(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute paired cosine correlation score."""

        source_norm = F.normalize(source_features, p=2, dim=1)
        target_norm = F.normalize(target_features, p=2, dim=1)
        similarity = torch.mm(source_norm, target_norm.t())
        return torch.diagonal(similarity).mean()

    @staticmethod
    def class_prototype_tensor_correlation(
        projected_source: List[torch.Tensor],
        projected_target: List[torch.Tensor],
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """Align unpaired domains through class-level tensor prototypes.

        Source and Target samples are never matched by row.  For every class
        present in both independently shuffled batches, this method forms one
        prototype per modality and compares the corresponding rank-one tensor
        products.  The tensor-product cosine factorizes into the product of
        per-modality cosines, so no large outer-product tensor is materialized.

        A per-modality mean term removes the sign ambiguity of an even-order
        tensor product (two anti-aligned modalities must not count as aligned).
        When labels are unavailable, each complete batch is treated as one
        unpaired domain prototype rather than using arbitrary diagonal pairs.
        """

        if not projected_source or len(projected_source) != len(projected_target):
            raise ValueError("Source and Target projected modalities must align.")
        device = projected_source[0].device
        if source_labels is None or target_labels is None:
            modality_scores = []
            for source_features, target_features in zip(
                projected_source,
                projected_target,
            ):
                modality_scores.append(
                    F.cosine_similarity(
                        source_features.mean(dim=0, keepdim=True),
                        target_features.mean(dim=0, keepdim=True),
                        dim=1,
                        eps=1e-6,
                    ).squeeze(0)
                )
            stacked = torch.stack(modality_scores)
            return 0.5 * (torch.prod(stacked) + stacked.mean())

        source_labels = source_labels.to(device=device, dtype=torch.long)
        target_labels = target_labels.to(device=device, dtype=torch.long)
        if num_classes is None:
            num_classes = int(
                torch.cat([source_labels, target_labels]).max().detach().cpu()
            ) + 1
        resolved_classes = int(num_classes)
        source_assignment = F.one_hot(
            source_labels,
            num_classes=resolved_classes,
        ).to(projected_source[0].dtype)
        target_assignment = F.one_hot(
            target_labels,
            num_classes=resolved_classes,
        ).to(projected_target[0].dtype)
        source_counts = source_assignment.sum(dim=0)
        target_counts = target_assignment.sum(dim=0)
        shared_mask = (source_counts > 0) & (target_counts > 0)
        shared_weight = shared_mask.to(projected_source[0].dtype)

        modality_scores = []
        for source_features, target_features in zip(
            projected_source,
            projected_target,
        ):
            source_prototypes = torch.mm(source_assignment.t(), source_features)
            target_prototypes = torch.mm(target_assignment.t(), target_features)
            source_prototypes = source_prototypes / source_counts.clamp_min(1).unsqueeze(1)
            target_prototypes = target_prototypes / target_counts.clamp_min(1).unsqueeze(1)
            modality_scores.append(
                F.cosine_similarity(
                    source_prototypes,
                    target_prototypes,
                    dim=1,
                    eps=1e-6,
                )
            )
        stacked = torch.stack(modality_scores)
        tensor_score = torch.prod(stacked, dim=0)
        class_score = 0.5 * (tensor_score + stacked.mean(dim=0))
        return (class_score * shared_weight).sum() / shared_weight.sum().clamp_min(1.0)

    @staticmethod
    def factorized_tensor_contraction(
        modalities: List[torch.Tensor],
        matrices: nn.ParameterList | List[torch.Tensor],
        exclude_mode: int,
    ) -> torch.Tensor:
        """Contract an outer-product tensor without explicitly constructing it.

        For one sample, the multimodal tensor is a rank-one outer product.  If
        all modes except ``exclude_mode`` are projected and summed, the exact
        result is the unprojected feature in the excluded mode multiplied by
        the product of the summed projected features from every other mode.
        This is mathematically equivalent to ``create_multimodal_tensor`` plus
        repeated mode products and ``tensor_contraction`` but uses linear
        rather than multiplicative memory in the modality dimensions.
        """

        result = modalities[exclude_mode]
        scale = result.new_ones((result.size(0), 1))
        for idx, (features, matrix) in enumerate(zip(modalities, matrices)):
            if idx == exclude_mode:
                continue
            projected = torch.mm(features, matrix)
            scale = scale * projected.sum(dim=1, keepdim=True)
        return result * scale

    @staticmethod
    def class_balanced_cross_domain_contrastive_loss(
        projected_source: List[torch.Tensor],
        projected_target: List[torch.Tensor],
        source_labels: Optional[torch.Tensor],
        target_labels: Optional[torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        """Pull same-class cross-domain samples together without pair matching.

        Every Source-modality sample treats *all* Target-modality samples of
        the same class as positives, and vice versa. All modality pairs are
        included, so the shared TAL space is encouraged to remove both domain
        and modality nuisance while different classes remain negatives. Losses
        are averaged within class before averaging across classes, preventing
        the very frequent M4-SAR classes from completely dominating TAL. Row
        order and physical ``pair_id`` are intentionally irrelevant.
        """

        if source_labels is None or target_labels is None:
            return projected_source[0].sum() * 0.0
        source_labels = source_labels.to(
            device=projected_source[0].device,
            dtype=torch.long,
        )
        target_labels = target_labels.to(
            device=projected_target[0].device,
            dtype=torch.long,
        )

        def directional_loss(
            anchors: torch.Tensor,
            candidates: torch.Tensor,
            anchor_labels: torch.Tensor,
            candidate_labels: torch.Tensor,
        ) -> torch.Tensor:
            logits = torch.mm(
                F.normalize(anchors, p=2, dim=1, eps=1e-6),
                F.normalize(candidates, p=2, dim=1, eps=1e-6).t(),
            ) / float(temperature)
            positive_mask = anchor_labels[:, None].eq(candidate_labels[None, :])
            valid_anchors = positive_mask.any(dim=1)
            if not bool(valid_anchors.any()):
                return logits.sum() * 0.0
            logits = logits[valid_anchors]
            positive_mask = positive_mask[valid_anchors]
            valid_labels = anchor_labels[valid_anchors]
            logits = logits - logits.max(dim=1, keepdim=True).values.detach()
            positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
            sample_losses = (
                torch.logsumexp(logits, dim=1)
                - torch.logsumexp(positive_logits, dim=1)
            )
            class_losses = []
            for class_id in torch.unique(valid_labels):
                class_mask = valid_labels.eq(class_id)
                class_losses.append(sample_losses[class_mask].mean())
            return torch.stack(class_losses).mean()

        cross_view_losses = []
        for source_features in projected_source:
            for target_features in projected_target:
                source_to_target = directional_loss(
                    source_features,
                    target_features,
                    source_labels,
                    target_labels,
                )
                target_to_source = directional_loss(
                    target_features,
                    source_features,
                    target_labels,
                    source_labels,
                )
                cross_view_losses.append(
                    0.5 * (source_to_target + target_to_source)
                )
        return torch.stack(cross_view_losses).mean()

    def orthogonality_penalty(self) -> torch.Tensor:
        """Return a smooth alternative to overwriting Adam updates with QR."""

        penalties = []
        for matrix in [*self.U_matrices, *self.V_matrices]:
            identity = torch.eye(
                matrix.size(1),
                device=matrix.device,
                dtype=matrix.dtype,
            )
            penalties.append(F.mse_loss(matrix.t().mm(matrix), identity))
        return torch.stack(penalties).mean()

    def project_source(self, source_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project source-domain modality features."""

        projected = [
            torch.mm(source_modalities[idx], self.U_matrices[idx])
            for idx in range(self.num_modalities)
        ]
        if self.use_shared_layer_norm:
            projected = [
                self.shared_normalizers[idx](features)
                for idx, features in enumerate(projected)
            ]
        return projected

    def project_target(self, target_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project target-domain modality features."""

        projected = [
            torch.mm(target_modalities[idx], self.V_matrices[idx])
            for idx in range(self.num_modalities)
        ]
        if self.use_shared_layer_norm:
            projected = [
                self.shared_normalizers[idx](features)
                for idx, features in enumerate(projected)
            ]
        return projected

    def forward(
        self,
        source_modalities: List[torch.Tensor],
        target_modalities: List[torch.Tensor],
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Project modalities and return an unpaired tensor-alignment loss."""

        if len(source_modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} source modalities, "
                f"got {len(source_modalities)}."
            )
        if len(target_modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} target modalities, "
                f"got {len(target_modalities)}."
            )

        projected_source = self.project_source(source_modalities)
        projected_target = self.project_target(target_modalities)
        tensor_correlation = self.class_prototype_tensor_correlation(
            projected_source,
            projected_target,
            source_labels,
            target_labels,
            num_classes,
        )
        prototype_loss = 1.0 - tensor_correlation
        alignment_loss = prototype_loss
        contrastive_loss = alignment_loss.new_zeros(())
        if self.cross_domain_contrastive_weight > 0.0:
            contrastive_loss = self.class_balanced_cross_domain_contrastive_loss(
                projected_source,
                projected_target,
                source_labels,
                target_labels,
                self.contrastive_temperature,
            )
            alignment_loss = (
                alignment_loss
                + self.cross_domain_contrastive_weight * contrastive_loss
            )
        orthogonality_penalty = alignment_loss.new_zeros(())
        if self.orthogonality_weight > 0.0:
            orthogonality_penalty = self.orthogonality_penalty()
            alignment_loss = (
                alignment_loss
                + self.orthogonality_weight * orthogonality_penalty
            )
        self.last_diagnostics = {
            "prototype_loss": prototype_loss.detach(),
            "tensor_correlation": tensor_correlation.detach(),
            "cross_domain_contrastive": contrastive_loss.detach(),
            "orthogonality_penalty": orthogonality_penalty.detach(),
        }
        return projected_source, projected_target, alignment_loss

    def apply_orthogonal_projection(self) -> None:
        """Project learned matrices back to an orthogonal basis with QR."""

        with torch.no_grad():
            for idx in range(self.num_modalities):
                q_source, _ = torch.linalg.qr(self.U_matrices[idx].data, mode="reduced")
                q_target, _ = torch.linalg.qr(self.V_matrices[idx].data, mode="reduced")
                self.U_matrices[idx].data = q_source[:, : self.output_dims[idx]]
                self.V_matrices[idx].data = q_target[:, : self.output_dims[idx]]


class PlainMultimodalProjection(nn.Module):
    """Non-tensor ablation: shared per-modality linear projection and concat.

    The same projection is used in Source and Target so the downstream fused
    dimension and modality block layout match TAL. No tensor contraction,
    domain-specific U/V matrices, correlation objective, or pair information
    is used.
    """

    def __init__(self, input_dims: List[int], output_dims: List[int]):
        super().__init__()
        if len(input_dims) != len(output_dims) or not input_dims:
            raise ValueError("Plain projection dimensions must be non-empty and aligned.")
        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.num_modalities = len(input_dims)
        self.projections = nn.ModuleList(
            [
                nn.Linear(input_dim, output_dim, bias=False)
                for input_dim, output_dim in zip(input_dims, output_dims)
            ]
        )
        for projection in self.projections:
            nn.init.orthogonal_(projection.weight)

    def _project(self, modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(modalities)}."
            )
        return [
            projection(features)
            for projection, features in zip(self.projections, modalities)
        ]

    def project_source(self, source_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._project(source_modalities)

    def project_target(self, target_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._project(target_modalities)

    def forward(
        self,
        source_modalities: List[torch.Tensor],
        target_modalities: List[torch.Tensor],
        source_labels: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        del source_labels, target_labels, num_classes
        projected_source = self.project_source(source_modalities)
        projected_target = self.project_target(target_modalities)
        return (
            projected_source,
            projected_target,
            projected_source[0].new_zeros(()),
        )

    def apply_orthogonal_projection(self) -> None:
        """Plain projections have no TAL-specific post-step constraint."""
