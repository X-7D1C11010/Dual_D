"""Dual discriminator extension package for JMDA-Net.

The package is intentionally self-contained. It does not import or modify
JMDA-Net scripts at module import time; integration is performed through
feature tensors produced by the original pipeline.
"""

from .config import DualDConfig, LossWeights, load_config
from .integration_adapter import DualDTrainingAdapter

__all__ = [
    "DualDConfig",
    "LossWeights",
    "DualDTrainingAdapter",
    "load_config",
]

