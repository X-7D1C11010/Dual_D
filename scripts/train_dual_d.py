"""Standalone Dual_D training entrypoint.

Module purpose:
    Start full Dual_D training from the independent ``Dual_D`` folder. This
    script does not import files from JMDA-Net or any other sibling project.
    It trains the visual/IR feature extractors, tensor alignment module,
    bidirectional feature translator, primary discriminator, auxiliary
    discriminator, and classifier.

Command examples:
    Linux:
        python scripts/train_dual_d.py \\
            --source-root /home/lixiang/lx/Data/晴天 \\
            --target-root /home/lixiang/lx/Data/雨天 \\
            --output-dir runs \\
            --epochs 100 \\
            --batch-size 32

    Windows:
        & 'D:\\Anaconda\\envs\\pytorch\\python.exe' D:\\Code\\Dual_D\\scripts\\train_dual_d.py `
            --source-root D:\\Code\\TADA\\Data\\晴天 `
            --target-root D:\\Code\\TADA\\Data\\雨天 `
            --output-dir D:\\Code\\Dual_D\\runs

Outputs:
    output_dir/run_name/train.log
    output_dir/run_name/metrics.csv
    output_dir/run_name/checkpoints/best_model.pt
    output_dir/run_name/checkpoints/last_model.pt
    output_dir/run_name/best_metrics.json
    output_dir/run_name/result_summary.json
    output_dir/run_name/resolved_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_d.training.trainer import run_training  # noqa: E402


def load_json_defaults(path: str | Path | None) -> Dict[str, Any]:
    """Load JSON defaults if a config path is provided."""

    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build the full CLI parser using optional JSON defaults."""

    def default(name: str, fallback):
        return defaults.get(name, fallback)

    parser = argparse.ArgumentParser(
        description="Train the standalone Dual_D dual-discriminator algorithm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument(
        "--dual-config",
        default=default(
            "dual_config",
            str(PROJECT_ROOT / "configs" / "dual_d_default_config.json"),
        ),
        help="Dual_D module config JSON.",
    )

    parser.add_argument("--source-root", default=default("source_root", ""))
    parser.add_argument("--target-root", default=default("target_root", ""))
    parser.add_argument("--output-dir", default=default("output_dir", str(PROJECT_ROOT / "runs")))
    parser.add_argument("--run-name", default=default("run_name", ""))

    parser.add_argument("--train-phase", default=default("train_phase", "train"))
    parser.add_argument("--val-phase", default=default("val_phase", "val"))
    parser.add_argument(
        "--source-layout",
        default=default("source_layout", "auto"),
        choices=["auto", "modality_first", "class_first"],
    )
    parser.add_argument(
        "--target-layout",
        default=default("target_layout", "auto"),
        choices=["auto", "modality_first", "class_first"],
    )
    parser.add_argument("--vis-folder", default=default("vis_folder", "可见光"))
    parser.add_argument("--ir-folder", default=default("ir_folder", "红外"))

    parser.add_argument("--epochs", type=int, default=default("epochs", 100))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 32))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 4))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--seed", type=int, default=default("seed", 42))

    parser.add_argument("--image-size", type=int, default=default("image_size", 224))
    parser.add_argument("--resize-size", type=int, default=default("resize_size", 256))
    parser.add_argument(
        "--augmentation-strength",
        type=float,
        default=default("augmentation_strength", 0.5),
        help="Modality-specific photometric jitter strength in [0, 1].",
    )
    parser.add_argument("--feature-dim", type=int, default=default("feature_dim", 512))
    parser.add_argument("--proj-dim", type=int, default=default("proj_dim", 128))

    parser.add_argument(
        "--pretrained-visual",
        action=argparse.BooleanOptionalAction,
        default=default("pretrained_visual", False),
        help="Use torchvision ImageNet weights for ResNet-18 visual extractor.",
    )
    parser.add_argument(
        "--freeze-visual-backbone",
        action=argparse.BooleanOptionalAction,
        default=default("freeze_visual_backbone", True),
        help="Freeze early visual backbone blocks and train late blocks/projection.",
    )
    parser.add_argument(
        "--val-augment",
        action=argparse.BooleanOptionalAction,
        default=default("val_augment", False),
    )

    parser.add_argument("--label-smoothing", type=float, default=default("label_smoothing", 0.1))
    parser.add_argument(
        "--classifier-dropout",
        type=float,
        default=default("classifier_dropout", 0.3),
    )
    parser.add_argument("--tal-weight", type=float, default=default("tal_weight", 0.3))
    parser.add_argument("--lr-main", type=float, default=default("lr_main", 5e-4))
    parser.add_argument("--lr-visual", type=float, default=default("lr_visual", 1e-5))
    parser.add_argument("--lr-discriminator", type=float, default=default("lr_discriminator", 5e-4))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 1e-4))
    parser.add_argument("--lr-factor", type=float, default=default("lr_factor", 0.5))
    parser.add_argument("--lr-patience", type=int, default=default("lr_patience", 10))
    parser.add_argument("--min-lr", type=float, default=default("min_lr", 1e-6))
    parser.add_argument(
        "--min-lr-discriminator",
        type=float,
        default=default("min_lr_discriminator", 1e-6),
    )
    parser.add_argument(
        "--discriminator-update-interval",
        type=int,
        default=default("discriminator_update_interval", 2),
    )
    parser.add_argument("--grad-clip", type=float, default=default("grad_clip", 1.0))
    parser.add_argument(
        "--adversarial-warmup-epochs",
        type=int,
        default=default("adversarial_warmup_epochs", 5),
        help="Classifier/TAL-only epochs before discriminator updates begin.",
    )
    parser.add_argument(
        "--adversarial-ramp-epochs",
        type=int,
        default=default("adversarial_ramp_epochs", 15),
        help="Epochs used to linearly ramp adversarial generator weights to 1.",
    )
    parser.add_argument(
        "--monitor-metric",
        default=default("monitor_metric", "val_f1_macro_present"),
        choices=["val_acc", "val_f1_macro_present", "val_loss"],
        help="Metric used by both LR schedulers, checkpointing, and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=default("early_stopping_patience", 15),
        help="Stop after this many unimproved epochs; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=default("early_stopping_min_epochs", 75),
        help="Do not stop before this epoch even if the patience counter is exhausted.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=default("early_stopping_min_delta", 0.001),
    )
    parser.add_argument(
        "--train-eval-interval",
        type=int,
        default=default("train_eval_interval", 1),
        help="Evaluate the complete deterministic target train split every N epochs.",
    )
    parser.add_argument(
        "--data-audit-hashes",
        action=argparse.BooleanOptionalAction,
        default=default("data_audit_hashes", True),
        help="Hash target train/validation images to detect exact duplicate content.",
    )
    parser.add_argument(
        "--strict-data-audit",
        action=argparse.BooleanOptionalAction,
        default=default("strict_data_audit", True),
        help="Abort training when the split audit finds leakage or invalid labels.",
    )
    parser.add_argument(
        "--eval-feature-mode",
        default=default("eval_feature_mode", "source_like"),
        choices=["raw", "source_like", "residual"],
        help="Target validation feature mode before classifier.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with optional JSON defaults."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    known_args, remaining = config_parser.parse_known_args()
    defaults = load_json_defaults(known_args.config)
    parser = build_parser(defaults)
    args = parser.parse_args(["--config", known_args.config] + remaining if known_args.config else remaining)

    if not args.source_root:
        parser.error("--source-root is required, or provide it in --config.")
    if not args.target_root:
        parser.error("--target-root is required, or provide it in --config.")
    if args.discriminator_update_interval <= 0:
        parser.error("--discriminator-update-interval must be positive.")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1).")
    if not 0.0 <= args.classifier_dropout < 1.0:
        parser.error("--classifier-dropout must be in [0, 1).")
    if not 0.0 <= args.augmentation_strength <= 1.0:
        parser.error("--augmentation-strength must be in [0, 1].")
    if args.train_eval_interval <= 0:
        parser.error("--train-eval-interval must be positive.")
    if args.adversarial_warmup_epochs < 0 or args.adversarial_ramp_epochs < 0:
        parser.error("Adversarial warmup/ramp epochs must be non-negative.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative.")
    if args.early_stopping_min_epochs < 0:
        parser.error("--early-stopping-min-epochs must be non-negative.")
    return args


def main() -> None:
    """CLI main function."""

    args = parse_args()
    summary = run_training(args)
    print("Dual_D training complete.")
    print(f"Run directory: {summary['run_dir']}")
    print(f"Best validation accuracy: {summary['best_acc']:.4f}")


if __name__ == "__main__":
    main()
