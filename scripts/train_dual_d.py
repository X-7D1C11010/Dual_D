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
            --epochs 60 \\
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


# These are the training knobs that may vary by target weather.  Keeping the
# allow-list narrow prevents a profile from silently changing data paths,
# model structure, or evaluation semantics.
WEATHER_PROFILE_KEYS = frozenset(
    {
        "batch_size",
        "num_workers",
        "min_steps_per_epoch",
        "augmentation_strength",
        "vis_augmentation_strength",
        "ir_augmentation_strength",
        "label_smoothing",
        "classifier_dropout",
        "freeze_frozen_batch_norm_stats",
        "target_classification_weight",
        "lr_main",
        "lr_visual",
        "lr_discriminator",
        "weight_decay",
        "lr_patience",
        "lr_scheduler_start_epoch",
        "discriminator_update_interval",
        "adversarial_warmup_epochs",
        "adversarial_ramp_epochs",
        "module_c_warmup_epochs",
        "module_c_ramp_epochs",
        "monitor_stability_window",
        "dual_loss_weights",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_delta",
    }
)


def load_json_defaults(path: str | Path | None) -> Dict[str, Any]:
    """Load JSON defaults if a config path is provided."""

    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_weather_profiles(path: str | Path | None) -> Dict[str, Dict[str, Any]]:
    """Load and validate optional per-weather training overrides.

    The file may either contain a top-level ``profiles`` mapping plus an
    optional ``default`` mapping, or map weather names directly to mappings.
    A top-level ``profile_files`` mapping may point each weather at a separate
    JSON override file. Values are applied to a per-run argument copy in
    ``run_experiment_matrix``.
    """

    if not path:
        return {"default": {}, "profiles": {}}
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Weather profile config does not exist: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("Weather profile config must be a JSON object.")
    profile_files = payload.get("profile_files", {})
    if "profiles" in payload or "profile_files" in payload:
        profiles = payload.get("profiles", {})
        default = payload.get("default", {})
    else:
        default = payload.get("default", {})
        profiles = {key: value for key, value in payload.items() if key != "default"}
    if (
        not isinstance(default, dict)
        or not isinstance(profiles, dict)
        or not isinstance(profile_files, dict)
    ):
        raise ValueError(
            "Weather profile 'default', 'profiles', and 'profile_files' must be objects."
        )

    profiles = dict(profiles)
    for name, referenced_path in profile_files.items():
        if str(name) in profiles:
            raise ValueError(
                f"Weather profile '{name}' is defined both inline and in profile_files."
            )
        if not isinstance(referenced_path, str) or not referenced_path.strip():
            raise ValueError(
                f"Weather profile file for '{name}' must be a non-empty path string."
            )
        referenced_profile_path = Path(referenced_path)
        if not referenced_profile_path.is_absolute():
            referenced_profile_path = profile_path.parent / referenced_profile_path
        if not referenced_profile_path.is_file():
            raise FileNotFoundError(
                f"Weather profile file for '{name}' does not exist: "
                f"{referenced_profile_path}"
            )
        with referenced_profile_path.open("r", encoding="utf-8") as file_obj:
            profiles[str(name)] = json.load(file_obj)

    def validate(name: str, values: Any) -> Dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError(f"Weather profile '{name}' must be an object.")
        unknown = sorted(set(values) - WEATHER_PROFILE_KEYS)
        if unknown:
            raise ValueError(
                f"Weather profile '{name}' contains unsupported keys: {', '.join(unknown)}"
            )
        validated_values = dict(values)
        dual_loss_weights = validated_values.get("dual_loss_weights")
        if dual_loss_weights is not None:
            if not isinstance(dual_loss_weights, dict):
                raise ValueError(
                    f"Weather profile '{name}' value 'dual_loss_weights' must be an object."
                )
            valid_loss_names = {
                "classification",
                "adv_primary",
                "adv_auxiliary",
                "cycle",
                "identity",
                "contrastive",
                "prototype_contrastive",
            }
            unknown_losses = sorted(set(dual_loss_weights) - valid_loss_names)
            if unknown_losses:
                raise ValueError(
                    f"Weather profile '{name}' contains unsupported Dual-D loss weights: "
                    + ", ".join(unknown_losses)
                )
            for loss_name, weight in dual_loss_weights.items():
                if not isinstance(weight, (int, float)) or weight < 0.0:
                    raise ValueError(
                        f"Weather profile '{name}' Dual-D weight '{loss_name}' "
                        "must be non-negative."
                    )
            validated_values["dual_loss_weights"] = dict(dual_loss_weights)
        return validated_values

    validated = {"default": validate("default", default), "profiles": {}}
    for name, values in profiles.items():
        validated["profiles"][str(name)] = validate(str(name), values)
    return validated


def apply_weather_profile(
    args: argparse.Namespace,
    domain: str,
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply one weather profile to a run-local argument namespace."""

    overrides = dict(profiles.get("default", {}))
    overrides.update(profiles.get("profiles", {}).get(str(domain), {}))
    positive_ints = {
        "batch_size",
        "num_workers",
        "min_steps_per_epoch",
        "lr_patience",
        "discriminator_update_interval",
        "monitor_stability_window",
    }
    nonnegative_ints = {
        "adversarial_warmup_epochs",
        "adversarial_ramp_epochs",
        "module_c_warmup_epochs",
        "module_c_ramp_epochs",
        "lr_scheduler_start_epoch",
        "early_stopping_patience",
        "early_stopping_min_epochs",
    }
    unit_floats = {
        "augmentation_strength",
        "vis_augmentation_strength",
        "ir_augmentation_strength",
    }
    probability_floats = {
        "label_smoothing",
        "classifier_dropout",
    }
    nonnegative_floats = {
        "target_classification_weight",
        "lr_main",
        "lr_visual",
        "lr_discriminator",
        "weight_decay",
        "early_stopping_min_delta",
    }
    boolean_keys = {"freeze_frozen_batch_norm_stats"}
    for key in positive_ints:
        if key in overrides and (
            not isinstance(overrides[key], int) or overrides[key] <= 0
        ):
            raise ValueError(f"Weather profile value '{key}' must be a positive integer.")
    for key in nonnegative_ints:
        if key in overrides and (
            not isinstance(overrides[key], int) or overrides[key] < 0
        ):
            raise ValueError(f"Weather profile value '{key}' must be a non-negative integer.")
    for key in unit_floats:
        if key in overrides and (
            not isinstance(overrides[key], (int, float))
            or not 0.0 <= overrides[key] <= 1.0
        ):
            raise ValueError(f"Weather profile value '{key}' must be in [0, 1].")
    for key in probability_floats:
        if key in overrides and (
            not isinstance(overrides[key], (int, float))
            or not 0.0 <= overrides[key] < 1.0
        ):
            raise ValueError(f"Weather profile value '{key}' must be in [0, 1).")
    for key in nonnegative_floats:
        if key in overrides and (
            not isinstance(overrides[key], (int, float)) or overrides[key] < 0.0
        ):
            raise ValueError(f"Weather profile value '{key}' must be non-negative.")
    for key in boolean_keys:
        if key in overrides and not isinstance(overrides[key], bool):
            raise ValueError(f"Weather profile value '{key}' must be boolean.")
    for key, value in overrides.items():
        setattr(args, key, value)
    args.weather_profile_name = str(domain) if overrides else "default"
    args.weather_profile_overrides = dict(overrides)
    return overrides


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
        "--weather-profile-config",
        default=default("weather_profile_config", ""),
        help=(
            "Optional JSON file with per-weather training overrides. Profiles are "
            "applied to each resolved target-domain run."
        ),
    )
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
    parser.add_argument(
        "--group-iterations",
        action=argparse.BooleanOptionalAction,
        default=default("group_iterations", False),
        help=(
            "Store repetitions for the same variant/weather in one directory, "
            "suffixing each artifact with its iteration number."
        ),
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

    parser.add_argument("--epochs", type=int, default=default("epochs", 60))
    parser.add_argument("--batch-size", type=int, default=default("batch_size", 32))
    parser.add_argument("--num-workers", type=int, default=default("num_workers", 4))
    parser.add_argument("--device", default=default("device", "auto"))
    parser.add_argument("--seed", type=int, default=default("seed", 42))
    parser.add_argument(
        "--deterministic-training",
        action=argparse.BooleanOptionalAction,
        default=default("deterministic_training", False),
        help="Prefer deterministic PyTorch/CUDA kernels for paired ablation runs.",
    )
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
        default=default("min_steps_per_epoch", 4),
        help="Minimum class-balanced optimizer steps for small target domains.",
    )

    parser.add_argument("--image-size", type=int, default=default("image_size", 224))
    parser.add_argument("--resize-size", type=int, default=default("resize_size", 256))
    parser.add_argument(
        "--augmentation-strength",
        type=float,
        default=default("augmentation_strength", 0.6),
        help="Modality-specific photometric jitter strength in [0, 1].",
    )
    parser.add_argument(
        "--vis-augmentation-strength",
        type=float,
        default=default("vis_augmentation_strength", None),
        help=(
            "Optional visible-light jitter strength in [0, 1]. When omitted, "
            "--augmentation-strength is used."
        ),
    )
    parser.add_argument(
        "--ir-augmentation-strength",
        type=float,
        default=default("ir_augmentation_strength", None),
        help=(
            "Optional infrared jitter strength in [0, 1]. When omitted, "
            "--augmentation-strength is used."
        ),
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
        "--freeze-frozen-batch-norm-stats",
        action=argparse.BooleanOptionalAction,
        default=default("freeze_frozen_batch_norm_stats", False),
        help=(
            "Keep running statistics fixed in BatchNorm layers whose affine "
            "parameters are frozen."
        ),
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
        default=default("classifier_dropout", 0.45),
    )
    parser.add_argument(
        "--target-classification-weight",
        type=float,
        default=default("target_classification_weight", 0.75),
        help=(
            "Weight for direct target-domain classification in the main loss. "
            "Lower values reduce memorization on small target splits."
        ),
    )
    parser.add_argument("--tal-weight", type=float, default=default("tal_weight", 0.3))
    parser.add_argument("--lr-main", type=float, default=default("lr_main", 7.5e-5))
    parser.add_argument("--lr-visual", type=float, default=default("lr_visual", 7.5e-6))
    parser.add_argument(
        "--lr-discriminator",
        type=float,
        default=default("lr_discriminator", 3.75e-5),
    )
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay", 7.5e-4))
    parser.add_argument("--lr-factor", type=float, default=default("lr_factor", 0.5))
    parser.add_argument("--lr-patience", type=int, default=default("lr_patience", 6))
    parser.add_argument(
        "--lr-scheduler-start-epoch",
        type=int,
        default=default("lr_scheduler_start_epoch", 1),
        help=(
            "Do not feed validation metrics to ReduceLROnPlateau before this "
            "epoch. Use this to keep learning rates fixed while warmup/ramp "
            "schedules become active."
        ),
    )
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
        "--module-c-warmup-epochs",
        type=int,
        default=default("module_c_warmup_epochs", 5),
        help="Epochs that train classification/TAL before Module-C constraints begin.",
    )
    parser.add_argument(
        "--module-c-ramp-epochs",
        type=int,
        default=default("module_c_ramp_epochs", 10),
        help="Epochs used to linearly ramp all Module-C constraint weights to 1.",
    )
    parser.add_argument(
        "--monitor-metric",
        default=default("monitor_metric", "val_acc"),
        choices=["val_acc", "val_f1_macro_present", "val_loss"],
        help="Metric used by both LR schedulers, checkpointing, and early stopping.",
    )
    parser.add_argument(
        "--monitor-stability-window",
        type=int,
        default=default("monitor_stability_window", 1),
        help=(
            "Select checkpoints by the worst monitored value over the latest N "
            "epochs. N=1 retains ordinary peak selection; larger windows reduce "
            "single-epoch optimism on very small validation sets."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=default("early_stopping_patience", 10),
        help="Stop after this many unimproved epochs; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=default("early_stopping_min_epochs", 20),
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
        "--raw-eval-interval",
        type=int,
        default=default("raw_eval_interval", 5),
        help=(
            "Evaluate the secondary raw-feature validation view every N epochs. "
            "It is also evaluated whenever the monitored validation metric improves."
        ),
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
    parser.add_argument(
        "--save-feature-embeddings",
        action=argparse.BooleanOptionalAction,
        default=default("save_feature_embeddings", False),
        help="Save best source/target raw and translated features for diagnostics.",
    )
    parser.add_argument(
        "--feature-visualization-samples",
        type=int,
        default=default("feature_visualization_samples", 512),
        help="Maximum validation samples stored in feature_embeddings.npz.",
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
    if (
        args.vis_augmentation_strength is not None
        and not 0.0 <= args.vis_augmentation_strength <= 1.0
    ):
        parser.error("--vis-augmentation-strength must be in [0, 1].")
    if (
        args.ir_augmentation_strength is not None
        and not 0.0 <= args.ir_augmentation_strength <= 1.0
    ):
        parser.error("--ir-augmentation-strength must be in [0, 1].")
    if args.train_eval_interval <= 0:
        parser.error("--train-eval-interval must be positive.")
    if args.raw_eval_interval <= 0:
        parser.error("--raw-eval-interval must be positive.")
    if args.adversarial_warmup_epochs < 0 or args.adversarial_ramp_epochs < 0:
        parser.error("Adversarial warmup/ramp epochs must be non-negative.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative.")
    if args.early_stopping_min_epochs < 0:
        parser.error("--early-stopping-min-epochs must be non-negative.")
    if args.lr_scheduler_start_epoch < 0:
        parser.error("--lr-scheduler-start-epoch must be non-negative.")
    if args.monitor_stability_window <= 0:
        parser.error("--monitor-stability-window must be positive.")
    if args.feature_visualization_samples <= 0:
        parser.error("--feature-visualization-samples must be positive.")
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
    weather_profiles = load_weather_profiles(
        getattr(args, "weather_profile_config", "")
    )
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
            profile_overrides = apply_weather_profile(run_args, domain, weather_profiles)
            if total_runs > 1:
                prefix = args.run_name.strip() or "dual_d"
                safe_domain = Path(domain).name
                if bool(getattr(args, "group_iterations", False)):
                    run_args.run_name = f"{prefix}_{safe_domain}_{batch_timestamp}"
                    run_args.artifact_suffix = f"iter{iteration:02d}"
                else:
                    run_args.run_name = (
                        f"{prefix}_{safe_domain}_iter{iteration:02d}_{batch_timestamp}"
                    )
                    run_args.artifact_suffix = ""

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
                "weather_profile": getattr(run_args, "weather_profile_name", "default"),
                "weather_profile_overrides": profile_overrides,
            }
            summaries.append(summary)

    domain_statistics: Dict[str, Dict[str, float]] = {}
    for domain, _, _ in experiments:
        domain_runs = [
            item for item in summaries if item["target_domain"] == domain
        ]
        selected_metrics = {
            "acc": [],
            "precision_macro_present": [],
            "recall_macro_present": [],
            "f1_macro_present": [],
        }
        best_accuracies = []
        for item in domain_runs:
            best_accuracies.append(float(item["best_acc"]))
            validation = item.get("best_metrics", {}).get("val", {})
            selected_metrics["acc"].append(
                float(validation.get("accuracy", item["best_acc"]))
            )
            for name in (
                "precision_macro_present",
                "recall_macro_present",
                "f1_macro_present",
            ):
                selected_metrics[name].append(float(validation[name]))

        statistics: Dict[str, float] = {"runs": len(domain_runs)}
        for name, values in selected_metrics.items():
            statistics[f"val_{name}_mean"] = mean(values)
            statistics[f"val_{name}_std"] = (
                pstdev(values) if len(values) > 1 else 0.0
            )
            statistics[f"val_{name}_min"] = min(values)
            statistics[f"val_{name}_max"] = max(values)
        # Retain the historical maximum-accuracy fields for compatibility.
        # The four ``val_*`` groups above all come from the one epoch selected
        # by ``monitor_metric`` and are the values that should be reported.
        statistics.update(
            {
                "best_acc_mean": mean(best_accuracies),
                "best_acc_std": (
                    pstdev(best_accuracies) if len(best_accuracies) > 1 else 0.0
                ),
                "best_acc_min": min(best_accuracies),
                "best_acc_max": max(best_accuracies),
            }
        )
        domain_statistics[domain] = statistics

    batch_summary = {
        "batch_timestamp": batch_timestamp,
        "iterations": int(args.iterations),
        "target_domains": [domain for domain, _, _ in experiments],
        "total_runs": total_runs,
        "group_iterations": bool(getattr(args, "group_iterations", False)),
        "weather_profile_config": getattr(args, "weather_profile_config", ""),
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
            f"{domain}: selected-epoch val "
            f"ACC {metrics['val_acc_mean']:.4f} ± {metrics['val_acc_std']:.4f} | "
            f"Precision {metrics['val_precision_macro_present_mean']:.4f} | "
            f"Recall {metrics['val_recall_macro_present_mean']:.4f} | "
            f"F1 {metrics['val_f1_macro_present_mean']:.4f} "
            f"over {metrics['runs']} run(s)"
        )


if __name__ == "__main__":
    main()
