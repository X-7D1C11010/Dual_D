"""Model components for the standalone Dual_D training pipeline."""

from .backbones import (
    AISFeatureExtractor,
    AISMlpFeatureExtractor,
    Classifier,
    ComplexAISFeatureExtractor,
    ComplexConv1d,
    IRFeatureExtractor,
    LabelSmoothingCrossEntropy,
    OpticalResNet20Encoder,
    SARResNet20Encoder,
    SmallResNet20Encoder,
    VisualFeatureExtractor,
    set_requires_grad,
)
from .tensor_alignment import TensorBasedAlignmentStable

__all__ = [
    "AISFeatureExtractor",
    "AISMlpFeatureExtractor",
    "Classifier",
    "ComplexAISFeatureExtractor",
    "ComplexConv1d",
    "IRFeatureExtractor",
    "LabelSmoothingCrossEntropy",
    "OpticalResNet20Encoder",
    "SARResNet20Encoder",
    "SmallResNet20Encoder",
    "TensorBasedAlignmentStable",
    "VisualFeatureExtractor",
    "set_requires_grad",
]
