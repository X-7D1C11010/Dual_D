"""Gradient reversal utilities for adversarial feature learning.

Module purpose:
    Provide a reusable gradient reversal function compatible with the original
    JMDA-Net discriminator style. The forward pass is identity; the backward
    pass multiplies the incoming gradient by ``-alpha``.

Public interfaces:
    - gradient_reverse(x, alpha): functional API.
    - GradientReversalLayer: nn.Module wrapper.
"""

from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFunction(torch.autograd.Function):
    """Autograd implementation of gradient reversal."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def gradient_reverse(input_tensor: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Reverse gradients while keeping the forward value unchanged."""

    return _GradientReversalFunction.apply(input_tensor, float(alpha))


class GradientReversalLayer(nn.Module):
    """Module wrapper around gradient_reverse."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, input_tensor: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        """Apply gradient reversal to ``input_tensor``."""

        strength = self.alpha if alpha is None else float(alpha)
        return gradient_reverse(input_tensor, strength)

