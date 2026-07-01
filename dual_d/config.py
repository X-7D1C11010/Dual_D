"""Configuration objects for the Dual_D extension.

Module purpose:
    Provide serializable dataclass configurations for the dual-discriminator
    feature adaptation module.

Public interfaces:
    - LossWeights: loss coefficients used by generator-side optimization.
    - DiscriminatorConfig: hidden-layer and regularization settings.
    - GeneratorConfig: feature mapper architecture settings.
    - DualDConfig: top-level configuration used by the adapter/coordinator.
    - load_config(path): read a JSON configuration file.
    - save_config(config, path): write a JSON configuration file.

Usage:
    >>> from dual_d.config import DualDConfig, load_config
    >>> cfg = DualDConfig(feature_dim=256)
    >>> cfg = load_config("configs/dual_d_default_config.json")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _tuple_from(value: Iterable[int] | Tuple[int, ...]) -> Tuple[int, ...]:
    """Normalize JSON list values into immutable tuples."""

    return tuple(int(v) for v in value)


@dataclass
class LossWeights:
    """Relative weights for the cooperative dual-discriminator objective.

    Attributes:
        classification: Weight for optional classifier feedback on generated
            source-like and target-like features.
        adv_primary: Weight for the primary discriminator fooling loss.
        adv_auxiliary: Weight for the auxiliary discriminator fooling loss.
        cycle: Weight for bidirectional cycle consistency.
        identity: Weight for identity preservation.
        contrastive: Weight for paired/class-aware feature contrast.
    """

    classification: float = 1.0
    adv_primary: float = 0.10
    adv_auxiliary: float = 0.10
    cycle: float = 0.50
    identity: float = 0.05
    contrastive: float = 0.10


@dataclass
class DiscriminatorConfig:
    """MLP discriminator configuration."""

    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.30
    use_spectral_norm: bool = False


@dataclass
class GeneratorConfig:
    """Bidirectional feature mapper configuration."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.10
    residual_scale: float = 0.50
    use_layer_norm: bool = True


@dataclass
class DualDConfig:
    """Top-level configuration for feature-level twin adversarial learning.

    Attributes:
        feature_dim: Dimension of fused source/target features. For the current
            JMDA-Net main path this is usually 256 because two 128-D TAL
            projections are concatenated.
        contrastive_temperature: Temperature for class-aware contrastive loss.
        include_reconstruction_fakes: If true, cycle-reconstructed features are
            also treated as fake samples for the corresponding discriminator.
        detach_contrastive_positives: If true, real positive features are used as
            fixed anchors for generated features during generator optimization.
    """

    feature_dim: int = 256
    contrastive_temperature: float = 0.20
    include_reconstruction_fakes: bool = True
    detach_contrastive_positives: bool = True
    loss_weights: LossWeights = field(default_factory=LossWeights)
    primary_discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    auxiliary_discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DualDConfig":
        """Create a configuration from a JSON-compatible dictionary."""

        loss_data = data.get("loss_weights", {})
        primary_data = data.get("primary_discriminator", {})
        auxiliary_data = data.get("auxiliary_discriminator", {})
        generator_data = data.get("generator", {})

        if "hidden_dims" in primary_data:
            primary_data["hidden_dims"] = _tuple_from(primary_data["hidden_dims"])
        if "hidden_dims" in auxiliary_data:
            auxiliary_data["hidden_dims"] = _tuple_from(auxiliary_data["hidden_dims"])

        return cls(
            feature_dim=int(data.get("feature_dim", 256)),
            contrastive_temperature=float(data.get("contrastive_temperature", 0.20)),
            include_reconstruction_fakes=bool(data.get("include_reconstruction_fakes", True)),
            detach_contrastive_positives=bool(data.get("detach_contrastive_positives", True)),
            loss_weights=LossWeights(**loss_data),
            primary_discriminator=DiscriminatorConfig(**primary_data),
            auxiliary_discriminator=DiscriminatorConfig(**auxiliary_data),
            generator=GeneratorConfig(**generator_data),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""

        return asdict(self)


def load_config(path: str | Path) -> DualDConfig:
    """Load a DualDConfig from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return DualDConfig.from_dict(data)


def save_config(config: DualDConfig, path: str | Path) -> None:
    """Save a DualDConfig as formatted JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        json.dump(config.to_dict(), file_obj, indent=2)
        file_obj.write("\n")

