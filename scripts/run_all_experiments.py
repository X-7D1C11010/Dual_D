"""Run the standalone Full experiment and Module-C ablation in one command.

The Full baseline defaults to Backlight/Rain, while the ablation matrix covers
all four weather domains.  Both stages use the same 60-epoch setting and keep
iterations grouped by variant/weather when ``--group-iterations`` is enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_dual_d_default.json"
DEFAULT_DUAL_CONFIG = PROJECT_ROOT / "configs" / "dual_d_default_config.json"
DEFAULT_PROFILE_CONFIG = PROJECT_ROOT / "configs" / "module_c_weather_profiles.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Full and Module-C ablation experiments sequentially."
    )
    parser.add_argument("--base-train-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-dual-config", default=str(DEFAULT_DUAL_CONFIG))
    parser.add_argument("--weather-profile-config", default=str(DEFAULT_PROFILE_CONFIG))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-parent-root", required=True)
    parser.add_argument("--full-target-domains", nargs="+", default=["逆光", "雨天"])
    parser.add_argument("--ablation-target-domains", nargs="+", default=["黑天", "逆光", "雾天", "雨天"])
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


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    output_root = Path(args.output_dir)
    full_output = output_root / "full"
    ablation_output = output_root / "module_c_ablation"
    group_flag = "--group-iterations" if args.group_iterations else "--no-group-iterations"

    full_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_dual_d.py"),
        "--config",
        str(args.base_train_config),
        "--dual-config",
        str(args.base_dual_config),
        "--weather-profile-config",
        str(args.weather_profile_config),
        "--source-root",
        str(args.source_root),
        "--target-parent-root",
        str(args.target_parent_root),
        "--target-domains",
        *[str(domain) for domain in args.full_target_domains],
        "--iterations",
        str(args.iterations),
        "--epochs",
        str(args.epochs),
        "--device",
        str(args.device),
        "--no-multi-gpu",
        "--no-use-ais",
        group_flag,
        "--output-dir",
        str(full_output),
    ]
    if args.seed is not None:
        full_command.extend(["--seed", str(args.seed)])
    if args.num_workers is not None:
        full_command.extend(["--num-workers", str(args.num_workers)])
    _run(full_command, "Full experiment")

    ablation_command = [
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
        *[str(domain) for domain in args.ablation_target_domains],
        "--iterations",
        str(args.iterations),
        "--epochs",
        str(args.epochs),
        "--device",
        str(args.device),
        group_flag,
        "--output-dir",
        str(ablation_output),
    ]
    if args.seed is not None:
        ablation_command.extend(["--seed", str(args.seed)])
    if args.num_workers is not None:
        ablation_command.extend(["--num-workers", str(args.num_workers)])
    _run(ablation_command, "Module-C ablation")


if __name__ == "__main__":
    main()
