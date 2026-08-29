"""Run the complete, non-duplicated Module-C ablation matrix in one command.

``full`` is already one member of the leave-one-constraint-out matrix, so a
separate Full stage would train it twice and could accidentally use a different
domain list or seed assignment.  This launcher therefore runs exactly seven
variants for the requested weather domains: Full Module C, five individual
constraint removals, and all Module-C constraints removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_dual_d_default.json"
DEFAULT_DUAL_CONFIG = PROJECT_ROOT / "configs" / "dual_d_default_config.json"
DEFAULT_PROFILE_CONFIG = PROJECT_ROOT / "configs" / "module_c_weather_profiles_v8.json"
EXPERIMENT_VARIANTS = [
    "full",
    "no_cycle",
    "no_identity",
    "no_paired_contrastive",
    "no_prototype_contrastive",
    "no_classification_feedback",
    "no_module_c",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Full and every required Module-C ablation exactly once."
    )
    parser.add_argument("--base-train-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-dual-config", default=str(DEFAULT_DUAL_CONFIG))
    parser.add_argument("--weather-profile-config", default=str(DEFAULT_PROFILE_CONFIG))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-parent-root", required=True)
    parser.add_argument(
        "--target-domains",
        nargs="+",
        default=["黑天", "逆光", "雨天"],
        help="Target weather domains trained by the experiment matrix.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "combined_experiments"))
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--group-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store all repetitions for one variant/weather in one directory.",
    )
    return parser


def _run(command: list[str], label: str) -> None:
    print(f"=== Starting {label} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    print(f"=== Finished {label} ===", flush=True)


def build_ablation_command(args: argparse.Namespace) -> list[str]:
    """Build the sole training subprocess used by this launcher."""

    group_flag = "--group-iterations" if args.group_iterations else "--no-group-iterations"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ablate_module_c.py"),
        "--run",
        "--base-train-config",
        str(args.base_train_config),
        "--base-dual-config",
        str(args.base_dual_config),
        "--weather-profile-config",
        str(args.weather_profile_config),
        "--source-root",
        str(args.source_root),
        "--target-parent-root",
        str(args.target_parent_root),
        "--target-domains",
        *[str(domain) for domain in args.target_domains],
        "--variants",
        *EXPERIMENT_VARIANTS,
        "--iterations",
        str(args.iterations),
        "--epochs",
        str(args.epochs),
        "--device",
        str(args.device),
        group_flag,
        "--no-pca-feature-view",
        "--no-tsne-feature-view",
        "--output-dir",
        str(Path(args.output_dir)),
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    return command


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    _run(build_ablation_command(args), "Full and Module-C ablation matrix")


if __name__ == "__main__":
    main()
