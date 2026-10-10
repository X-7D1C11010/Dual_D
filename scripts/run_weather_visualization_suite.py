#!/usr/bin/env python3
"""Train full/no-drift once per weather and create reusable visualizations.

Sunny weather is the Source domain. Every other selected weather is trained as
an independent Target domain with seed 42 by default. AIS is explicitly
disabled. Runs are sequential, and each training invocation has iterations=1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_dual_d_default.json"
DEFAULT_WEATHER_PROFILES = (
    PROJECT_ROOT / "configs" / "module_c_weather_profiles_v13.json"
)
VARIANTS = {
    "full": ["--modality-drift-enabled"],
    "no_modality_drift": ["--no-modality-drift-enabled"],
}


def _targets(data_root: Path, source_weather: str, requested: Sequence[str]) -> List[str]:
    if requested:
        values = list(dict.fromkeys(requested))
    else:
        values = sorted(
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and path.name != source_weather
        )
    values = [
        value
        for value in values
        if value != source_weather and value.lower() != "ais"
    ]
    missing = [value for value in values if not (data_root / value).is_dir()]
    if missing:
        raise FileNotFoundError(f"目标天气目录不存在：{missing}")
    if not values:
        raise ValueError("没有发现目标天气目录")
    return values


def _run(command: Sequence[str], log_path: Path, env: Dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat()}] {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"命令失败（exit={completed.returncode}），日志：{log_path}")


def _training_command(
    args: argparse.Namespace,
    target: str,
    variant: str,
    run_name: str,
    run_root: Path,
) -> List[str]:
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "train_dual_d.py"),
        "--config",
        str(args.config),
        "--weather-profile-config",
        str(args.weather_profile_config),
        "--source-root",
        str(args.data_root / args.source_weather),
        "--target-root",
        str(args.data_root / target),
        "--output-dir",
        str(run_root),
        "--run-name",
        run_name,
        "--iterations",
        "1",
        "--seed",
        str(args.seed),
        "--model-mode",
        "dual_d",
        "--alignment-mode",
        "tal",
        "--translation-enabled",
        "--module-c-enabled",
        "--no-use-ais",
        "--synchronize-modalities",
        "--no-multi-gpu",
        "--deterministic-training",
        "--no-save-checkpoints",
        "--no-evaluate-target-test",
        "--save-feature-embeddings",
        "--feature-visualization-samples",
        str(args.feature_visualization_samples),
        "--device",
        args.device,
        *VARIANTS[variant],
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--weather-profile-config",
        type=Path,
        default=DEFAULT_WEATHER_PROFILES,
        help="默认使用当前最新的逐天气训练配置；仍只训练一个 seed",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/lixiang/data/Data"))
    parser.add_argument("--source-weather", default="晴天")
    parser.add_argument("--target-weathers", nargs="*", default=[])
    parser.add_argument("--experiment-name", default="weather_modules_seed42")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "weather_visualization",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--feature-visualization-samples", type=int, default=1024)
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用已有 result_summary.json 和 feature_embeddings.npz",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"训练配置不存在：{args.config}")
    if not args.weather_profile_config.is_file():
        raise FileNotFoundError(f"天气配置不存在：{args.weather_profile_config}")
    if args.data_root.is_dir():
        targets = _targets(args.data_root, args.source_weather, args.target_weathers)
        if not (args.data_root / args.source_weather).is_dir():
            raise FileNotFoundError(f"源域不存在：{args.data_root / args.source_weather}")
    elif args.dry_run and args.target_weathers:
        targets = list(dict.fromkeys(args.target_weathers))
    else:
        raise FileNotFoundError(f"数据根目录不存在：{args.data_root}")

    suite_root = args.output_root / args.experiment_name
    run_root = suite_root / "runs"
    visual_root = suite_root / "visualizations"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    summary: Dict[str, object] = {
        "source_weather": args.source_weather,
        "target_weathers": targets,
        "variants": list(VARIANTS),
        "seed": args.seed,
        "iterations_per_variant_weather": 1,
        "ais_used": False,
        "target_test_used_for_selection_or_visualization": False,
        "runs": {},
    }

    for target in targets:
        summary["runs"][target] = {}
        embeddings: Dict[str, Path] = {}
        for variant in VARIANTS:
            run_name = f"{target}_{variant}_seed{args.seed}"
            run_dir = run_root / run_name
            result_path = run_dir / "result_summary.json"
            embedding_path = run_dir / "feature_embeddings.npz"
            if args.resume and result_path.is_file() and embedding_path.is_file():
                print(f"[复用] {run_name}", flush=True)
            else:
                command = _training_command(args, target, variant, run_name, run_root)
                _run(command, suite_root / "logs" / f"{run_name}.log", env, args.dry_run)
            if args.dry_run:
                continue
            if not embedding_path.is_file():
                raise FileNotFoundError(
                    f"{run_name} 完成后未找到 {embedding_path}；请检查对应日志"
                )
            embeddings[variant] = embedding_path
            summary["runs"][target][variant] = {
                "run_dir": str(run_dir.resolve()),
                "feature_embeddings": str(embedding_path.resolve()),
            }

        if args.dry_run:
            continue
        visual_command = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "visualize_weather_modules.py"),
            "--full-run",
            str(embeddings["full"]),
            "--no-drift-run",
            str(embeddings["no_modality_drift"]),
            "--output-dir",
            str(visual_root / target),
            "--max-per-class",
            str(args.max_per_class),
            "--seed",
            str(args.seed),
            "--perplexity",
            str(args.perplexity),
        ]
        _run(
            visual_command,
            suite_root / "logs" / f"{target}_visualization.log",
            env,
            False,
        )

    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "suite_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
