"""Standalone Dual_D training entrypoint.

Module purpose:
    Start full Dual_D training from the independent ``Dual_D`` folder. This
    script does not import files from JMDA-Net or any other sibling project.
    It trains the visual/IR/AIS feature extractors, tensor alignment module,
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
    output_dir/run_name/checkpoints/*.pt (when --save-checkpoints is enabled)
    output_dir/run_name/best_metrics.json
    output_dir/run_name/result_summary.json
    output_dir/run_name/resolved_config.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_d.training.trainer import run_training  # noqa: E402
from dual_d.training.checkpointing import save_json  # noqa: E402
from dual_d.data.ais_signal import resolve_reference_ais_file  # noqa: E402


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
    parser.add_argument(
        "--target-parent-root",
        default=default("target_parent_root", ""),
        help="Parent containing all adverse-weather target-domain folders.",
    )
    parser.add_argument(
        "--target-domains",
        nargs="+",
        default=default("target_domains", ["黑天", "逆光", "雾天", "雨天"]),
        help="Target-domain folder names used with --target-parent-root.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=default("iterations", 3),
        help="Independent training repetitions per target domain.",
    )
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
    parser.add_argument("--ais-folder", default=default("ais_folder", "AIS"))
    parser.add_argument(
        "--use-ais",
        action=argparse.BooleanOptionalAction,
        default=default("use_ais", False),
        help=(
            "Enable the AIS branch as an explicit ablation. A global MAT file is "
            "treated as an unpaired, label-independent prior."
        ),
    )
    parser.add_argument(
        "--source-ais-root",
        default=default("source_ais_root", ""),
        help="Optional separate source AIS root; omit when AIS is inside source-root.",
    )
    parser.add_argument(
        "--source-ais-data-path",
        default=default("source_ais_data_path", ""),
        help="Optional explicit JMDA-Net AIS .mat/.h5 file for the source domain.",
    )
    parser.add_argument(
        "--target-ais-root",
        default=default("target_ais_root", ""),
        help="Optional separate AIS root for one --target-root experiment.",
    )
    parser.add_argument(
        "--target-ais-data-path",
        default=default("target_ais_data_path", ""),
        help="Optional explicit JMDA-Net AIS .mat/.h5 file; can be shared by all targets.",
    )
    parser.add_argument(
        "--target-ais-parent-root",
        default=default("target_ais_parent_root", ""),
        help="Optional AIS parent with the same weather subfolders as target-parent-root.",
    )
    parser.add_argument(
        "--ais-match",
        choices=["auto", "stem", "index"],
        default=default("ais_match", "auto"),
        help="How each AIS file is matched to its VIS/IR sample.",
    )
    parser.add_argument(
        "--ais-sequence-length",
        type=int,
        default=default("ais_sequence_length", 128),
    )
    parser.add_argument(
        "--ais-encoder",
        choices=["complex", "mlp"],
        default=default("ais_encoder", "complex"),
        help="Complex I/Q encoder or MLP for pre-extracted numerical AIS features.",
    )
    parser.add_argument(
        "--ais-dropout",
        type=float,
        default=default("ais_dropout", 0.1),
    )
    parser.add_argument(
        "--ais-normalize",
        action=argparse.BooleanOptionalAction,
        default=default("ais_normalize", True),
        help="Apply per-sample I/Q standardization.",
    )

    parser.add_argument("--epochs", type=int, default=default("epochs", 100))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 32))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 4))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--seed", type=int, default=default("seed", 42))
    parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=default("multi_gpu", True),
        help=(
            "Use DataParallel for feature encoders/classifier when multiple CUDA "
            "GPUs exist; automatically fall back to the primary GPU if NCCL "
            "broadcast is unavailable."
        ),
    )
    parser.add_argument(
        "--min-steps-per-epoch",
        type=int,
        default=default("min_steps_per_epoch", 8),
        help="Minimum class-balanced optimizer steps for small target domains.",
    )

    parser.add_argument("--image-size", type=int, default=default("image_size", 224))
    parser.add_argument("--resize-size", type=int, default=default("resize_size", 256))
    parser.add_argument(
        "--augmentation-strength",
        type=float,
        default=default("augmentation_strength", 0.5),
        help="Modality-specific photometric jitter strength in [0, 1].",
    )
    parser.add_argument(
        "--synchronize-modalities",
        action=argparse.BooleanOptionalAction,
        default=default("synchronize_modalities", False),
        help="Reuse geometric augmentation only for genuinely registered VIS/IR pairs.",
    )
    parser.add_argument("--feature-dim", type=int, default=default("feature_dim", 512))
    parser.add_argument("--proj-dim", type=int, default=default("proj_dim", 128))

    parser.add_argument(
        "--pretrained-visual",
        action=argparse.BooleanOptionalAction,
        default=default("pretrained_visual", True),
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
        default=default("classifier_dropout", 0.4),
    )
    parser.add_argument(
        "--target-classification-weight",
        type=float,
        default=default("target_classification_weight", 1.0),
        help=(
            "Weight for direct target-domain classification in the main loss. "
            "Lower values reduce memorization on small target splits."
        ),
    )
    parser.add_argument("--tal-weight", type=float, default=default("tal_weight", 0.3))
    parser.add_argument("--lr-main", type=float, default=default("lr_main", 1e-4))
    parser.add_argument("--lr-visual", type=float, default=default("lr_visual", 1e-5))
    parser.add_argument("--lr-discriminator", type=float, default=default("lr_discriminator", 5e-5))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 5e-4))
    parser.add_argument("--lr-factor", type=float, default=default("lr_factor", 0.5))
    parser.add_argument("--lr-patience", type=int, default=default("lr_patience", 6))
    parser.add_argument("--min-lr", type=float, default=default("min_lr", 1e-6))
    parser.add_argument(
        "--min-lr-discriminator",
        type=float,
        default=default("min_lr_discriminator", 1e-6),
    )
    parser.add_argument(
        "--discriminator-update-interval",
        type=int,
        default=default("discriminator_update_interval", 3),
    )
    parser.add_argument("--grad-clip", type=float, default=default("grad_clip", 5.0))
    parser.add_argument(
        "--adversarial-warmup-epochs",
        type=int,
        default=default("adversarial_warmup_epochs", 10),
        help="Epochs before discriminator updates and generator adversarial terms begin.",
    )
    parser.add_argument(
        "--adversarial-ramp-epochs",
        type=int,
        default=default("adversarial_ramp_epochs", 30),
        help="Epochs used to linearly ramp adversarial generator weights to 1.",
    )
    parser.add_argument(
        "--monitor-metric",
        default=default("monitor_metric", "val_acc"),
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
        default=default("early_stopping_min_epochs", 25),
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
        help="Hash target train/validation VIS/IR/AIS files to detect duplicates.",
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
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=default("save_checkpoints", False),
        help="Write model checkpoint files; disabled by default.",
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
    if bool(args.target_root) == bool(args.target_parent_root):
        parser.error("Provide exactly one of --target-root or --target-parent-root.")
    if args.use_ais and args.target_parent_root and args.target_ais_root:
        parser.error("Use --target-ais-parent-root for multi-domain experiments.")
    if args.use_ais and args.target_root and args.target_ais_parent_root:
        parser.error("Use --target-ais-root for a single target-domain experiment.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.min_steps_per_epoch <= 0:
        parser.error("--min-steps-per-epoch must be positive.")
    if args.ais_sequence_length <= 0:
        parser.error("--ais-sequence-length must be positive.")
    if not 0.0 <= args.ais_dropout < 1.0:
        parser.error("--ais-dropout must be in [0, 1).")
    if args.discriminator_update_interval <= 0:
        parser.error("--discriminator-update-interval must be positive.")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1).")
    if not 0.0 <= args.classifier_dropout < 1.0:
        parser.error("--classifier-dropout must be in [0, 1).")
    if args.target_classification_weight < 0.0:
        parser.error("--target-classification-weight must be non-negative.")
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


def resolve_experiments(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    """Resolve one or four target domains and their optional AIS roots."""

    if args.target_root:
        target_root = Path(args.target_root)
        return [(target_root.name, target_root, str(args.target_ais_root or ""))]

    domains = args.target_domains
    if isinstance(domains, str):
        domains = [item.strip() for item in domains.split(",") if item.strip()]
    if not domains:
        raise ValueError("At least one --target-domains entry is required.")

    target_parent = Path(args.target_parent_root)
    ais_parent = Path(args.target_ais_parent_root) if args.target_ais_parent_root else None
    experiments = []
    for domain in domains:
        domain = str(domain)
        target_root = target_parent / domain
        target_ais_root = str(ais_parent / domain) if ais_parent is not None else ""
        experiments.append((domain, target_root, target_ais_root))
    return experiments


def run_experiment_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    """Run every target domain for the requested number of independent trials."""

    experiments = resolve_experiments(args)
    if not Path(args.source_root).exists():
        raise FileNotFoundError(f"Source domain does not exist: {args.source_root}")
    if args.use_ais and args.source_ais_root and not Path(args.source_ais_root).exists():
        raise FileNotFoundError(f"Source AIS root does not exist: {args.source_ais_root}")
    source_ais_data_path = getattr(args, "source_ais_data_path", "")
    target_ais_data_path = getattr(args, "target_ais_data_path", "")
    if args.use_ais and source_ais_data_path and not Path(source_ais_data_path).is_file():
        raise FileNotFoundError(f"Source AIS data file does not exist: {source_ais_data_path}")
    if args.use_ais and target_ais_data_path and not Path(target_ais_data_path).is_file():
        raise FileNotFoundError(f"Target AIS data file does not exist: {target_ais_data_path}")
    shared_ais_parent = (
        Path(args.target_ais_parent_root)
        if args.use_ais and args.target_ais_parent_root
        else None
    )
    for domain, target_root, target_ais_root in experiments:
        if not target_root.exists():
            raise FileNotFoundError(f"Target domain does not exist: {target_root}")
        if args.use_ais and target_ais_root and not Path(target_ais_root).exists():
            if shared_ais_parent is None or resolve_reference_ais_file(
                shared_ais_parent,
                ais_folder=getattr(args, "ais_folder", "AIS"),
            ) is None:
                raise FileNotFoundError(f"Target AIS domain does not exist: {target_ais_root}")

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_runs = len(experiments) * int(args.iterations)
    summaries: List[Dict[str, Any]] = []
    run_index = 0

    for domain, target_root, target_ais_root in experiments:
        for iteration in range(1, int(args.iterations) + 1):
            run_index += 1
            run_args = deepcopy(args)
            run_args.target_root = str(target_root)
            if (
                target_ais_root
                and not Path(target_ais_root).exists()
                and shared_ais_parent is not None
                and resolve_reference_ais_file(
                    shared_ais_parent,
                    ais_folder=getattr(args, "ais_folder", "AIS"),
                )
                is not None
            ):
                run_args.target_ais_root = str(shared_ais_parent)
            else:
                run_args.target_ais_root = target_ais_root
            run_args.seed = int(args.seed) + run_index - 1
            run_args.target_domain_name = domain
            run_args.iteration_index = iteration
            if total_runs > 1:
                prefix = args.run_name.strip() or "dual_d"
                safe_domain = Path(domain).name
                run_args.run_name = (
                    f"{prefix}_{safe_domain}_iter{iteration:02d}_{batch_timestamp}"
                )

            print(
                f"Starting run {run_index}/{total_runs}: "
                f"domain={domain}, iteration={iteration}, seed={run_args.seed}"
            )
            summary = run_training(run_args)
            summary = {
                **summary,
                "target_domain": domain,
                "target_root": str(target_root),
                "target_ais_root": target_ais_root,
                "iteration": iteration,
                "seed": run_args.seed,
            }
            summaries.append(summary)

    domain_statistics: Dict[str, Dict[str, float]] = {}
    for domain, _, _ in experiments:
        accuracies = [
            float(item["best_acc"])
            for item in summaries
            if item["target_domain"] == domain
        ]
        domain_statistics[domain] = {
            "runs": len(accuracies),
            "best_acc_mean": mean(accuracies),
            "best_acc_std": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
            "best_acc_min": min(accuracies),
            "best_acc_max": max(accuracies),
        }

    batch_summary = {
        "batch_timestamp": batch_timestamp,
        "iterations": int(args.iterations),
        "target_domains": [domain for domain, _, _ in experiments],
        "total_runs": total_runs,
        "domain_statistics": domain_statistics,
        "runs": summaries,
    }
    summary_path = Path(args.output_dir) / f"batch_summary_{batch_timestamp}.json"
    save_json(batch_summary, summary_path)
    batch_summary["summary_path"] = str(summary_path)
    return batch_summary


def main() -> None:
    """CLI main function."""

    args = parse_args()
    summary = run_experiment_matrix(args)
    print("Dual_D experiment matrix complete.")
    print(f"Batch summary: {summary['summary_path']}")
    for domain, metrics in summary["domain_statistics"].items():
        print(
            f"{domain}: best val acc {metrics['best_acc_mean']:.4f} "
            f"± {metrics['best_acc_std']:.4f} over {metrics['runs']} run(s)"
        )


if __name__ == "__main__":
    main()
