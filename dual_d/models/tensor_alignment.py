"""Tensor-based multimodal alignment module.

Module purpose:
    Provide a standalone implementation of the stable tensor alignment layer
    used before the Dual_D feature-level adversarial module. It projects source
    and target modality features into a shared low-dimensional space while
    maximizing paired correlation.

Public interface:
    - TensorBasedAlignmentStable

Usage:
    >>> tal = TensorBasedAlignmentStable([512, 512], [128, 128], num_modalities=2)
    >>> (p_s_vis, p_s_ir), (p_t_vis, p_t_ir), loss = tal([s_vis, s_ir], [t_vis, t_ir])
"""

from __future__ import annotations

from typing import List, Tuple

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
    ):
        super().__init__()
        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.num_modalities = int(num_modalities)

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

    def project_source(self, source_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project source-domain modality features."""

        return [
            torch.mm(source_modalities[idx], self.U_matrices[idx])
            for idx in range(self.num_modalities)
        ]

    def project_target(self, target_modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project target-domain modality features."""

        return [
            torch.mm(target_modalities[idx], self.V_matrices[idx])
            for idx in range(self.num_modalities)
        ]

    def forward(
        self,
        source_modalities: List[torch.Tensor],
        target_modalities: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Project modalities and return correlation alignment loss."""

        source_tensor = self.create_multimodal_tensor(source_modalities)
        target_tensor = self.create_multimodal_tensor(target_modalities)

        total_correlation = source_modalities[0].new_tensor(0.0)
        for mode in range(self.num_modalities):
            source_projected = source_tensor
            target_projected = target_tensor
            for idx in range(self.num_modalities):
                if idx != mode:
                    source_projected = self.mode_n_product(
                        source_projected,
                        self.U_matrices[idx],
                        idx,
                    )
                    target_projected = self.mode_n_product(
                        target_projected,
                        self.V_matrices[idx],
                        idx,
                    )
            source_contracted = self.tensor_contraction(source_projected, mode)
            target_contracted = self.tensor_contraction(target_projected, mode)
            total_correlation = total_correlation + self.compute_correlation_score(
                source_contracted,
                target_contracted,
            )

        projected_source = self.project_source(source_modalities)
        projected_target = self.project_target(target_modalities)
        alignment_loss = -total_correlation / self.num_modalities
        return projected_source, projected_target, alignment_loss

    def apply_orthogonal_projection(self) -> None:
        """Project learned matrices back to an orthogonal basis with QR."""

        with torch.no_grad():
            for idx in range(self.num_modalities):
                q_source, _ = torch.linalg.qr(self.U_matrices[idx].data, mode="reduced")
                q_target, _ = torch.linalg.qr(self.V_matrices[idx].data, mode="reduced")
                self.U_matrices[idx].data = q_source[:, : self.output_dims[idx]]
                self.V_matrices[idx].data = q_target[:, : self.output_dims[idx]]

