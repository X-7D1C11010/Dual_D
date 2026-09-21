"""Run the fixed seed-43 optimized M4-SAR experiment and TAL visualizations.

This wrapper deliberately launches exactly one training run. The training
entrypoint retains its leakage guard: Target test is constructed only after
Target-validation checkpoint selection has completed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "m4sar_classification_full_optimized_v3.json"


def _load_and_validate_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "dataset_type": "m4sar_classification",
        "model_mode": "dual_d",
        "alignment_mode": "tal",
        "iterations": 1,
        "seed": 43,
        "run_name": "full_seed43",
        "evaluate_target_test": True,
        "save_feature_embeddings": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "The one-run protocol config was changed incompatibly: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--font-path",
        default="",
        help="Optional Chinese font file passed to the visualization script.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Train only; the default automatically renders TAL diagnostics.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the default disposable train-only gradient smoke step.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_and_validate_config(config_path)
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.skip_preflight:
        preflight_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "preflight_m4sar.py"),
            "--config",
            str(config_path),
            "--batch-size",
            str(int(config["batch_size"])),
            "--num-workers",
            "0",
        ]
        print("执行一次可丢弃的 train-only preflight；不会打开 Target test。")
        subprocess.run(
            preflight_command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
    train_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_dual_d.py"),
        "--config",
        str(config_path),
    ]
    print("启动固定协议：1 次完整训练，seed=43；Target test 仅在选定 checkpoint 后打开。")
    subprocess.run(
        train_command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    if args.skip_visualization:
        return
    visualization_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "visualize_m4sar_ablation.py"),
        "--experiment-dir",
        str(output_dir),
    ]
    if args.font_path:
        visualization_command.extend(["--font-path", args.font_path])
    subprocess.run(visualization_command, cwd=PROJECT_ROOT, check=True)
    print(f"训练与 TAL 可视化全部完成：{output_dir}")


if __name__ == "__main__":
    main()
