"""Run Full M4-SAR Dual-D and four structural ablations sequentially.

One invocation launches every variant/seed with a fixed batch configuration,
then generates the Chinese summary figures. Completed runs can be reused with
``--resume``. Target test remains inside the trainer and is opened only after
Target-validation checkpoint selection has completed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean, pstdev
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "m4sar_classification.json"

VARIANTS = {
    "full": [
        "--alignment-mode", "tal",
        "--translation-enabled",
        "--module-c-enabled",
        "--modality-drift-enabled",
    ],
    "no_tal": [
        "--alignment-mode", "plain",
        "--translation-enabled",
        "--module-c-enabled",
        "--modality-drift-enabled",
    ],
    "no_translation_stack": [
        "--alignment-mode", "tal",
        "--no-translation-enabled",
        "--no-module-c-enabled",
        "--no-modality-drift-enabled",
        "--eval-feature-mode", "raw",
    ],
    "no_module_c": [
        "--alignment-mode", "tal",
        "--translation-enabled",
        "--no-module-c-enabled",
        "--modality-drift-enabled",
    ],
    "no_modality_drift": [
        "--alignment-mode", "tal",
        "--translation-enabled",
        "--module-c-enabled",
        "--no-modality-drift-enabled",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runs" / "m4sar_ablation"),
    )
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--wait-for-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait before every run and select the GPU with most free memory.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a run only when its result_summary.json already exists.",
    )
    parser.add_argument(
        "--evaluate-target-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate Target test once per selected validation checkpoint.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--font-path",
        default="",
        help="Optional SimSun/Chinese TTF or TTC path used by all figures.",
    )
    return parser


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _wait_for_gpu(min_free_gb: float, poll_seconds: float) -> int:
    """Wait without killing other jobs; return one physical GPU index."""

    while True:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        devices = []
        for line in completed.stdout.splitlines():
            if line.strip():
                index, free_mib = (int(value.strip()) for value in line.split(","))
                devices.append((index, free_mib / 1024.0))
        if not devices:
            raise RuntimeError("nvidia-smi returned no GPUs.")
        selected = max(devices, key=lambda item: item[1])
        status = " | ".join(f"GPU {index}: {free:.2f} GiB" for index, free in devices)
        print(f"[显存检查] {status}", flush=True)
        if selected[1] >= min_free_gb:
            return selected[0]
        time.sleep(poll_seconds)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _summary_rows(experiment_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(experiment_dir.glob("*_seed*/result_summary.json")):
        summary = _load_json(path)
        run_name = path.parent.name
        variant, seed_text = run_name.rsplit("_seed", 1)
        target_test = summary.get("target_test") or {}
        best = summary.get("best_metrics", {}).get("val", {})
        rows.append(
            {
                "variant": variant,
                "seed": int(seed_text),
                "selected_epoch": summary.get("best_metrics", {}).get("epoch"),
                "val_accuracy": best.get("accuracy"),
                "val_precision_macro": best.get("precision_macro_present"),
                "val_recall_macro": best.get("recall_macro_present"),
                "val_f1_macro": best.get("f1_macro_present"),
                "test_accuracy": target_test.get("accuracy"),
                "test_precision_macro": target_test.get("precision_macro_present"),
                "test_recall_macro": target_test.get("recall_macro_present"),
                "test_f1_macro": target_test.get("f1_macro_present"),
                "run_dir": str(path.parent),
            }
        )
    return rows


def _write_summary_csv(experiment_dir: Path) -> None:
    rows = _summary_rows(experiment_dir)
    if not rows:
        return
    with (experiment_dir / "ablation_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        aggregate = {"variant": variant, "runs": len(selected)}
        for metric in (
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
        ):
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            aggregate[f"{metric}_mean"] = mean(values) if values else None
            aggregate[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate)
    with (experiment_dir / "ablation_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    per_class_buckets: dict[tuple[str, int, str], list[float]] = {}
    class_names: dict[int, str] = {}
    for row in rows:
        metrics_path = Path(str(row["run_dir"])) / "target_test_metrics.json"
        label_path = Path(str(row["run_dir"])) / "label_map.json"
        if not metrics_path.is_file() or not label_path.is_file():
            continue
        metrics = _load_json(metrics_path)
        label_map = _load_json(label_path)
        id_to_name = {int(class_id): name for name, class_id in label_map.items()}
        class_names.update(id_to_name)
        for class_id in range(int(metrics["num_classes"])):
            for metric, source_key in (
                ("accuracy_ovr", "per_class_accuracy_ovr"),
                ("precision", "per_class_precision"),
                ("recall", "per_class_recall"),
                ("f1", "per_class_f1"),
            ):
                per_class_buckets.setdefault(
                    (str(row["variant"]), class_id, metric), []
                ).append(float(metrics[source_key][class_id]))
    per_class_rows = []
    for variant in sorted({key[0] for key in per_class_buckets}):
        for class_id in sorted({key[1] for key in per_class_buckets if key[0] == variant}):
            output = {
                "variant": variant,
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
            }
            for metric in ("accuracy_ovr", "precision", "recall", "f1"):
                values = per_class_buckets[(variant, class_id, metric)]
                output[f"{metric}_mean"] = mean(values)
                output[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            per_class_rows.append(output)
    if per_class_rows:
        with (experiment_dir / "per_class_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_class_rows[0]))
            writer.writeheader()
            writer.writerows(per_class_rows)


def main() -> None:
    parser = build_parser()
    args, forwarded = parser.parse_known_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    if not args.seeds:
        parser.error("At least one seed is required.")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.min_free_gb <= 0 or args.poll_seconds <= 0:
        parser.error("GPU memory threshold and poll interval must be positive.")

    experiment_name = args.experiment_name or datetime.now().strftime(
        "m4sar_full_ablation_%Y%m%d_%H%M%S"
    )
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": experiment_name,
        "config": str(config_path.resolve()),
        "variants": args.variants,
        "seeds": args.seeds,
        "target_test_policy": "validation-selected checkpoint; final evaluation only",
        "loader_policy": "independent Source/Target shuffle; no pair correspondence",
        "fixed_batch_size": args.batch_size,
    }
    _write_json(experiment_dir / "experiment_manifest.json", manifest)

    if args.visualize and not args.dry_run:
        visualization_check = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "visualize_m4sar_ablation.py"),
            "--check-only",
        ]
        if args.font_path:
            visualization_check.extend(["--font-path", args.font_path])
        subprocess.run(visualization_check, cwd=PROJECT_ROOT, check=True)

    if args.preflight and not args.dry_run:
        preflight_environment = os.environ.copy()
        if args.wait_for_gpu and args.device in {None, "auto", "cuda"}:
            gpu_index = _wait_for_gpu(args.min_free_gb, args.poll_seconds)
            preflight_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "preflight_m4sar.py"),
                "--config", str(config_path),
            ],
            cwd=PROJECT_ROOT,
            env=preflight_environment,
            check=True,
        )

    for variant in args.variants:
        for seed in args.seeds:
            run_name = f"{variant}_seed{seed}"
            run_dir = experiment_dir / run_name
            result_path = run_dir / "result_summary.json"
            if args.resume and result_path.is_file():
                print(f"[跳过已完成] {run_name}", flush=True)
                continue
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_dual_d.py"),
                "--config", str(config_path),
                "--output-dir", str(experiment_dir),
                "--run-name", run_name,
                "--iterations", "1",
                "--seed", str(seed),
                "--model-mode", "dual_d",
                "--deterministic-training",
                "--save-checkpoints",
                "--save-feature-embeddings",
                (
                    "--evaluate-target-test"
                    if args.evaluate_target_test
                    else "--no-evaluate-target-test"
                ),
                *VARIANTS[variant],
            ]
            if args.epochs is not None:
                command.extend(["--epochs", str(args.epochs)])
            if args.batch_size is not None:
                command.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            if args.device is not None:
                command.extend(["--device", str(args.device)])
            command.extend(forwarded)
            print(f"\n[开始] {run_name}\n{' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            environment = os.environ.copy()
            if args.wait_for_gpu and args.device in {None, "auto", "cuda"}:
                gpu_index = _wait_for_gpu(args.min_free_gb, args.poll_seconds)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                environment.setdefault(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                )
                print(f"[选择 GPU] 物理 GPU {gpu_index}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    if args.dry_run:
        print("Dry run complete; no training or visualization was started.")
        return

    _write_summary_csv(experiment_dir)
    if args.visualize:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "visualize_m4sar_ablation.py"),
            "--experiment-dir", str(experiment_dir),
        ]
        if args.font_path:
            command.extend(["--font-path", args.font_path])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"\n全部实验完成：{experiment_dir}", flush=True)


if __name__ == "__main__":
    main()
