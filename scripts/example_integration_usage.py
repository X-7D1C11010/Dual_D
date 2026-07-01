"""Example usage for the Dual_D extension.

Module purpose:
    Demonstrate how to instantiate the Dual_D adapter, feed TAL-aligned feature
    tensors, and compute discriminator/generator losses without touching any
    original JMDA-Net script.

Interface:
    Run directly from PowerShell:
        python D:/Code/Dual_D/scripts/example_integration_usage.py

Expected output:
    The script prints synthetic discriminator and generator loss logs. It uses
    random tensors only, so the numbers are not meaningful model metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_d.config import load_config  # noqa: E402
from dual_d.integration_adapter import DualDTrainingAdapter  # noqa: E402


class TinyClassifier(nn.Module):
    """Minimal classifier used only for the synthetic interface example."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits."""

        return self.net(features)


def main() -> None:
    """Run a synthetic Dual_D forward/loss pass."""

    config_path = PROJECT_ROOT / "configs" / "dual_d_default_config.json"
    config = load_config(config_path)

    batch_size = 8
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = DualDTrainingAdapter(config).to(device)
    classifier = TinyClassifier(config.feature_dim, num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss()

    feat_src = torch.randn(batch_size, config.feature_dim, device=device)
    feat_tgt = torch.randn(batch_size, config.feature_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    outputs = adapter.forward_features(feat_src, feat_tgt, labels=labels)
    loss_d, logs_d = adapter.compute_discriminator_loss(outputs)
    loss_g, logs_g = adapter.compute_generator_loss(
        outputs=outputs,
        labels=labels,
        classifier=classifier,
        criterion_cls=criterion_cls,
        source_labels=labels,
        target_labels=labels,
    )

    print("Dual_D discriminator loss:", float(loss_d.detach().cpu()))
    print(logs_d)
    print("Dual_D generator loss:", float(loss_g.detach().cpu()))
    print(logs_g)


if __name__ == "__main__":
    main()

