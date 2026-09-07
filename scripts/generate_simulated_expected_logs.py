"""Generate clearly labelled simulated Module-C training logs.

The real experiment logs under ``runs/`` are treated as read-only templates.
Generated files are written to ``result/`` and are explicitly marked as
simulated expected curves, not observed experimental evidence.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "module_c_v13_ablation_60"
    / "module_c_20260830_203640"
)
OUTPUT_ROOT = PROJECT_ROOT / "result"

VARIANTS = (
    "no_identity",
    "no_cycle",
    "no_paired_contrastive",
    "no_prototype_contrastive",
    "no_classification_feedback",
    "no_module_c",
    "full",
)
DOMAINS = ("黑天", "逆光", "雾天", "雨天")

# Values are the user-specified expected peak validation accuracies. Missing
# variant/domain pairs intentionally keep the observed curve unchanged.
EXPECTED_PEAKS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ("no_identity", "黑天"): (0.9917, 0.9833, 0.9750),
    ("no_identity", "雾天"): (1.0000, 0.9730, 0.9730),
    ("no_cycle", "黑天"): (0.9917, 0.9833, 0.9750),
    ("no_cycle", "雾天"): (0.9773, 0.9697, 0.9621),
    ("no_paired_contrastive", "黑天"): (0.9833, 0.9750, 0.9833),
    ("no_paired_contrastive", "逆光"): (0.9730, 1.0000, 0.9459),
    ("no_paired_contrastive", "雾天"): (0.9697, 0.9621, 0.9621),
    ("no_paired_contrastive", "雨天"): (1.0000, 0.9615, 0.9615),
    ("no_prototype_contrastive", "黑天"): (0.9833, 0.9750, 0.9667),
    ("no_prototype_contrastive", "逆光"): (1.0000, 0.9730, 0.9459),
    ("no_prototype_contrastive", "雾天"): (0.9621, 0.9621, 0.9545),
    ("no_prototype_contrastive", "雨天"): (1.0000, 0.9615, 0.9615),
    ("no_classification_feedback", "黑天"): (0.9667, 0.9417, 0.9500),
    ("no_classification_feedback", "逆光"): (0.9459, 0.9459, 0.9730),
    ("no_classification_feedback", "雾天"): (0.9545, 0.9470, 0.9394),
    ("no_classification_feedback", "雨天"): (0.9231, 1.0000, 0.9615),
    ("no_module_c", "黑天"): (0.9333, 0.9417, 0.9333),
    ("no_module_c", "逆光"): (0.9730, 0.9189, 0.8919),
    ("no_module_c", "雾天"): (0.9470, 0.9318, 0.9318),
    ("no_module_c", "雨天"): (0.9231, 0.8846, 1.0000),
}

TIMESTAMP_PREFIX = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\|\s*"
)
EPOCH_NUMBER = re.compile(r"\bEpoch\s+(\d{3})/(\d{3})\b")
VAL_ACC = re.compile(r"(?<=\bval_acc\s)(\d+(?:\.\d+)?)")
VAL_F1 = re.compile(r"(?<=\bval_f1\s)(\d+(?:\.\d+)?)")
RUNTIME_PROFILE = re.compile(
    r"stability_window=(\d+).*?checkpoint_select_from=(\d+)"
)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    val_acc: float
    val_f1: float


def _strip_timestamp(line: str) -> str:
    return TIMESTAMP_PREFIX.sub("", line.rstrip("\r\n"))


def _source_directory(variant: str, domain: str) -> Path:
    matches = sorted(SOURCE_ROOT.glob(f"module_c_{variant}_{domain}_*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one source directory for {variant}/{domain}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _parse_runtime(lines: Iterable[str]) -> Tuple[int, int]:
    for line in lines:
        match = RUNTIME_PROFILE.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 1, 1


def _parse_epochs(lines: Iterable[str]) -> List[EpochMetrics]:
    epochs: List[EpochMetrics] = []
    for line in lines:
        epoch_match = EPOCH_NUMBER.search(line)
        acc_match = VAL_ACC.search(line)
        f1_match = VAL_F1.search(line)
        if epoch_match and acc_match and f1_match:
            epochs.append(
                EpochMetrics(
                    epoch=int(epoch_match.group(1)),
                    val_acc=float(acc_match.group(1)),
                    val_f1=float(f1_match.group(1)),
                )
            )
    if not epochs:
        raise RuntimeError("Training log contains no epoch metric lines")
    return epochs


def _replace_metric(line: str, pattern: re.Pattern[str], value: float) -> str:
    return pattern.sub(f"{value:.4f}", line, count=1)


def _transform_epoch_metrics(
    epochs: List[EpochMetrics], target_peak: float
) -> Dict[int, EpochMetrics]:
    observed_peak = max(item.val_acc for item in epochs)
    if observed_peak <= 0.0:
        raise RuntimeError("Observed validation peak must be positive")

    transformed: Dict[int, EpochMetrics] = {}
    peak_epochs = {
        item.epoch for item in epochs if abs(item.val_acc - observed_peak) < 1e-12
    }
    for item in epochs:
        # Preserve the observed convergence and oscillation shape. Only the
        # vertical accuracy scale changes, so the requested value is attained
        # exactly wherever the real template reached its own peak.
        new_acc = target_peak * (item.val_acc / observed_peak)
        if item.epoch in peak_epochs:
            new_acc = target_peak
        new_acc = min(max(new_acc, 0.0), target_peak)

        # Preserve most of the observed macro-F1-versus-accuracy gap while
        # moving F1 consistently in the same direction as validation accuracy.
        delta = new_acc - item.val_acc
        new_f1 = item.val_f1 + 0.85 * delta
        new_f1 = min(max(new_f1, 0.0), 1.0)
        transformed[item.epoch] = EpochMetrics(item.epoch, new_acc, new_f1)
    return transformed


def _label_header(
    variant: str,
    domain: str,
    iteration: int,
    target_peak: Optional[float],
) -> List[str]:
    mode = "expected-curve simulation" if target_peak is not None else "de-timestamped template copy"
    target = f"{target_peak:.4f}" if target_peak is not None else "unchanged from source template"
    return [
        "=" * 78,
        "SIMULATED EXPECTED TRAINING LOG - NOT AN OBSERVED EXPERIMENT RESULT",
        "仅用于预期曲线可视化，不得作为真实实验结果或论文证据。",
        f"variant={variant} | domain={domain} | iteration={iteration:02d}",
        f"mode={mode} | requested_peak_val_acc={target}",
        "=" * 78,
    ]


def _copy_without_timestamps(
    source_lines: List[str], variant: str, domain: str, iteration: int
) -> Tuple[List[str], Dict[str, object]]:
    cleaned: List[str] = []
    for original in source_lines:
        line = _strip_timestamp(original)
        if line.startswith("Run directory:"):
            line = f"Simulated run: {variant}/{domain}/iter{iteration:02d}"
        cleaned.append(line)
    epochs = _parse_epochs(cleaned)
    peak = max(item.val_acc for item in epochs)
    return (
        _label_header(variant, domain, iteration, None) + cleaned,
        {
            "requested_peak_val_acc": "",
            "simulated_peak_val_acc": peak,
            "peak_val_f1": max(item.val_f1 for item in epochs),
            "selected_val_f1": "",
            "mode": "copied_without_timestamps",
        },
    )


def _simulate_expected_curve(
    source_lines: List[str],
    variant: str,
    domain: str,
    iteration: int,
    target_peak: float,
) -> Tuple[List[str], Dict[str, object]]:
    stripped = [_strip_timestamp(line) for line in source_lines]
    epochs = _parse_epochs(stripped)
    transformed = _transform_epoch_metrics(epochs, target_peak)
    stability_window, selection_floor = _parse_runtime(stripped)

    output = _label_header(variant, domain, iteration, target_peak)
    recent_f1: List[float] = []
    best_stable_score = float("-inf")
    selected_f1 = transformed[epochs[-1].epoch].val_f1

    for line in stripped:
        if line.startswith("Run directory:"):
            output.append(f"Simulated run: {variant}/{domain}/iter{iteration:02d}")
            continue
        if line.startswith("New best val_"):
            # Rebuilt below from the transformed validation trajectory.
            continue
        if line.startswith("Training complete."):
            continue

        epoch_match = EPOCH_NUMBER.search(line)
        if not epoch_match:
            output.append(line)
            continue

        epoch = int(epoch_match.group(1))
        item = transformed[epoch]
        line = _replace_metric(line, VAL_ACC, item.val_acc)
        line = _replace_metric(line, VAL_F1, item.val_f1)
        output.append(line)

        recent_f1.append(item.val_f1)
        recent_f1 = recent_f1[-stability_window:]
        if epoch >= selection_floor:
            stable_score = min(recent_f1)
            if stable_score > best_stable_score + 0.001:
                best_stable_score = stable_score
                selected_f1 = item.val_f1
                output.append(
                    "New best val_f1_macro_present: "
                    f"raw={item.val_f1:.4f} stable_score={stable_score:.4f} "
                    f"at epoch {epoch}"
                )

    output.append(
        "Training complete. "
        f"Peak validation accuracy: {target_peak:.4f} | "
        f"selected val_f1_macro_present: {selected_f1:.4f}"
    )
    simulated_peak = max(item.val_acc for item in transformed.values())
    return (
        output,
        {
            "requested_peak_val_acc": target_peak,
            "simulated_peak_val_acc": simulated_peak,
            "peak_val_f1": max(item.val_f1 for item in transformed.values()),
            "selected_val_f1": selected_f1,
            "mode": "simulated_expected_curve",
        },
    )


def _write_readme() -> None:
    text = """# 模拟的 Module C 预期训练日志

本目录中的日志均为 **SIMULATED / EXPECTED** 曲线，不是真实训练结果。

- 原始 `runs/` 实验文件未被修改。
- 用户指定目标 ACC 的日志，以同实验、同天气的真实逐轮曲线为模板，只改变验证 ACC/F1 的纵向尺度。
- 未指定目标 ACC 的日志保留原逐轮过程，仅删除时间戳。
- 每个日志开头均有模拟标识，不得将其作为真实实验结果、论文证据或正式消融数据。
- `simulated_summary.csv` 记录每个文件的目标峰值与生成后实际峰值。
"""
    (OUTPUT_ROOT / "README_SIMULATED.md").write_text(text, encoding="utf-8")


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Source experiment directory not found: {SOURCE_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_ROOT.iterdir())
    if existing:
        raise RuntimeError(
            f"Refusing to overwrite non-empty simulated output directory: {OUTPUT_ROOT}"
        )

    summary_rows: List[Dict[str, object]] = []
    for variant in VARIANTS:
        for domain in DOMAINS:
            source_dir = _source_directory(variant, domain)
            destination = OUTPUT_ROOT / f"module_c_{variant}_{domain}"
            destination.mkdir(parents=True, exist_ok=False)
            targets = EXPECTED_PEAKS.get((variant, domain))
            for iteration in range(1, 4):
                filename = f"train_iter{iteration:02d}.log"
                source_path = source_dir / filename
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing source log: {source_path}")
                source_lines = source_path.read_text(encoding="utf-8").splitlines()
                target_peak = targets[iteration - 1] if targets is not None else None
                if target_peak is None:
                    generated, metrics = _copy_without_timestamps(
                        source_lines, variant, domain, iteration
                    )
                else:
                    generated, metrics = _simulate_expected_curve(
                        source_lines, variant, domain, iteration, target_peak
                    )
                destination_path = destination / filename
                destination_path.write_text("\n".join(generated) + "\n", encoding="utf-8")
                summary_rows.append(
                    {
                        "variant": variant,
                        "domain": domain,
                        "iteration": iteration,
                        "log_path": str(destination_path.relative_to(OUTPUT_ROOT)),
                        **metrics,
                    }
                )

    _write_readme()
    fieldnames = [
        "variant",
        "domain",
        "iteration",
        "mode",
        "requested_peak_val_acc",
        "simulated_peak_val_acc",
        "peak_val_f1",
        "selected_val_f1",
        "log_path",
    ]
    with (OUTPUT_ROOT / "simulated_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Generated {len(summary_rows)} simulated logs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
