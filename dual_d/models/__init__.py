"""Model components for the standalone Dual_D training pipeline."""

from .backbones import (
    Classifier,
    IRFeatureExtractor,
    LabelSmoothingCrossEntropy,
    VisualFeatureExtractor,
    set_requires_grad,
)
from .tensor_alignment import TensorBasedAlignmentStable

__all__ = [
    "Classifier",
    "IRFeatureExtractor",
    "LabelSmoothingCrossEntropy",
    "TensorBasedAlignmentStable",
    "VisualFeatureExtractor",
    "set_requires_grad",
]

