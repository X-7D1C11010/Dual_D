"""Wait for free GPU memory, then launch epoch-adaptive M4-SAR training.

The launcher never terminates other processes. It selects one physical GPU
after the free-memory gate is satisfied for consecutive scans, then enables
the trainer's epoch-boundary adaptive batch plan. Target-test evaluation is
disabled unless explicitly requested for a frozen final experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def query_gpu_free_memory() -> list[dict[str, float | int]]:
    """Return physical GPU indices and current total/free memory in GiB."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    devices = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        index, free_mib, total_mib = (int(value) for value in parts)
        devices.append(
            {
                "index": index,
                "free_gb": free_mib / 1024.0,
                "total_gb": total_mib / 1024.0,
            }
        )
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPUs.")
    return devices


def select_gpu(
    devices: list[dict[str, float | int]],
    allowed_indices: set[int] | None,
) -> dict[str, float | int] | None:
    """Select the allowed GPU with the most free memory."""

    candidates = [
        device
        for device in devices
        if allowed_indices is None or int(device["index"]) in allowed_indices
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda device: float(device["free_gb"]))


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "m4sar_classification.json"),
    )
    parser.add_argument(
        "--model-mode",
        required=True,
        choices=["optical_only", "sar_only", "simple_concat", "dual_d"],
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gpu", default="auto", help="auto or comma-separated physical IDs")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stable-scans", type=int, default=2)
    parser.add_argument(
        "--batch-plan",
        default="10:32,18:64,26:128",
        help="Reusable free GiB to physical batch-size mapping.",
    )
    parser.add_argument(
        "--evaluate-target-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable only for a frozen final experiment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command after the memory gate without executing it.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, forwarded = parser.parse_known_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.min_free_gb <= 0 or args.poll_seconds <= 0:
        parser.error("Memory threshold and poll interval must be positive.")
    if args.stable_scans <= 0:
        parser.error("--stable-scans must be positive.")
    if not Path(args.config).is_file():
        parser.error(f"Config does not exist: {args.config}")

    allowed_indices = None
    if args.gpu.lower() != "auto":
        try:
            allowed_indices = {int(value.strip()) for value in args.gpu.split(",")}
        except ValueError as error:
            parser.error(f"Invalid --gpu value: {error}")

    stable_gpu = None
    stable_count = 0
    while stable_count < args.stable_scans:
        devices = query_gpu_free_memory()
        selected = select_gpu(devices, allowed_indices)
        status = " | ".join(
            f"GPU {int(device['index'])}: {float(device['free_gb']):.2f} GiB free"
            for device in devices
            if allowed_indices is None or int(device["index"]) in allowed_indices
        )
        print(f"[{_timestamp()}] {status}", flush=True)
        if selected is not None and float(selected["free_gb"]) > args.min_free_gb:
            selected_index = int(selected["index"])
            if selected_index == stable_gpu:
                stable_count += 1
            else:
                stable_gpu = selected_index
                stable_count = 1
            print(
                f"[{_timestamp()}] GPU {stable_gpu} passed the gate "
                f"({stable_count}/{args.stable_scans}).",
                flush=True,
            )
        else:
            stable_gpu = None
            stable_count = 0
        if stable_count < args.stable_scans:
            time.sleep(args.poll_seconds)

    run_name = args.run_name or f"m4sar_{args.model_mode}_adaptive"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_dual_d.py"),
        "--config",
        args.config,
        "--model-mode",
        args.model_mode,
        "--epochs",
        str(args.epochs),
        "--run-name",
        run_name,
        "--adaptive-batch-size",
        "--adaptive-batch-plan",
        args.batch_plan,
        "--adaptive-min-free-gb",
        str(args.min_free_gb),
        "--adaptive-memory-poll-seconds",
        str(args.poll_seconds),
        (
            "--evaluate-target-test"
            if args.evaluate_target_test
            else "--no-evaluate-target-test"
        ),
        *forwarded,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(stable_gpu)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    printable = " ".join(command)
    print(f"[{_timestamp()}] Selected physical GPU {stable_gpu}.", flush=True)
    print(f"[{_timestamp()}] Launching: {printable}", flush=True)
    if args.dry_run:
        return
    completed = subprocess.run(command, env=environment, cwd=PROJECT_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
