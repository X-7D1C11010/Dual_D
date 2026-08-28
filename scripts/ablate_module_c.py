"""Run and analyse leave-one-constraint-out experiments for Module C.

Module C is evaluated with the full model, a joint all-constraints-removed
baseline, and one run for every individual constraint removed from the
generator objective. The adversarial translation core remains active in that
joint baseline, so it must not be described as removing the entire module.
The script can launch the existing training entrypoint (``--run``), or analyse
one manifest-backed experiment directory. Analysis reports accuracy, macro
precision, macro recall and macro F1 at one shared selected epoch, plus
run-isolated feature diagnostics; no metric is filtered or rewritten.

Examples
--------
Analyse a completed ablation directory::

    python scripts/ablate_module_c.py --runs-root runs/module_c_ablation

Launch all variants for all four target domains::

    python scripts/ablate_module_c.py --run \
        --source-root /data/clear --target-parent-root /data \
        --target-domains 黑天 逆光 雾天 雨天 --iterations 3 \
        --epochs 60 --group-iterations
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import html
import json
import math
import numpy as np
from pathlib import Path
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from statistics import mean, pstdev, stdev
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_dual_d_default.json"
DEFAULT_DUAL_CONFIG = PROJECT_ROOT / "configs" / "dual_d_default_config.json"

# Keep names stable: they are used in output directories, tables and figures.
CONSTRAINTS = OrderedDict(
    [
        ("cycle", "bidirectional cycle consistency"),
        ("identity", "identity preservation"),
        ("paired_contrastive", "class-balanced cross-domain supervised contrast"),
        ("prototype_contrastive", "EMA class-prototype contrast"),
        ("classification_feedback", "generated-feature classification feedback"),
    ]
)
VARIANTS = ["full", "no_module_c", *[f"no_{name}" for name in CONSTRAINTS]]
DOMAIN_DISPLAY = {
    "黑天": "Night",
    "逆光": "Backlight",
    "雾天": "Fog",
    "雨天": "Rain",
}
METRICS = OrderedDict(
    [
        ("best_val_acc", "Accuracy"),
        ("best_val_precision", "Macro Precision"),
        ("best_val_recall", "Macro Recall"),
        ("best_val_f1", "Macro F1"),
    ]
)
LOSS_METRIC = {
    "cycle": "train_dual_d_cycle",
    "identity": "train_dual_d_identity",
    "paired_contrastive": "train_dual_d_contrastive",
    "prototype_contrastive": "train_dual_d_prototype_contrastive",
    "classification_feedback": "train_dual_d_classification_feedback",
}
WEIGHTED_LOSS_METRIC = {
    "cycle": "train_dual_d_weighted_cycle",
    "identity": "train_dual_d_weighted_identity",
    "paired_contrastive": "train_dual_d_weighted_contrastive",
    "prototype_contrastive": "train_dual_d_weighted_prototype_contrastive",
    "classification_feedback": "train_dual_d_weighted_classification_feedback",
}


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_dump(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def make_variant_config(base: Mapping[str, Any], variant: str) -> Dict[str, Any]:
    """Return a copy of the Dual-D config with one Module-C weight disabled."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown Module-C variant: {variant}")
    config = deepcopy(dict(base))
    weights = dict(config.get("loss_weights", {}))
    config["loss_weights"] = weights
    if variant == "full":
        return config
    if variant == "no_module_c":
        for key in (
            "cycle",
            "identity",
            "contrastive",
            "prototype_contrastive",
            "classification",
        ):
            weights[key] = 0.0
        return config
    constraint = variant.removeprefix("no_")
    if constraint == "paired_contrastive":
        weights["contrastive"] = 0.0
    elif constraint == "prototype_contrastive":
        weights["prototype_contrastive"] = 0.0
    elif constraint == "classification_feedback":
        weights["classification"] = 0.0
    elif constraint in {"cycle", "identity"}:
        weights[constraint] = 0.0
    else:  # pragma: no cover - guarded by VARIANTS
        raise ValueError(f"No weight mapping for {constraint}")
    return config


def write_variant_configs(base_path: Path, output_dir: Path, variants: Sequence[str]) -> Dict[str, Path]:
    base = _json_load(base_path)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for variant in variants:
        path = config_dir / f"dual_d_{variant}.json"
        _json_dump(make_variant_config(base, variant), path)
        paths[variant] = path
    return paths


def _append_option(command: List[str], flag: str, value: Any) -> None:
    if value is not None and str(value) != "":
        command.extend([flag, str(value)])


def run_variants(args: argparse.Namespace, variants: Sequence[str]) -> Path:
    if not args.source_root:
        raise ValueError("--source-root is required with --run")
    if bool(args.target_root) == bool(args.target_parent_root):
        raise ValueError("Provide exactly one of --target-root or --target-parent-root with --run")

    experiment_id = args.experiment_id or datetime.now().strftime("module_c_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[0-9A-Za-z_.-]+", experiment_id):
        raise ValueError("--experiment-id may contain only letters, digits, dot, dash and underscore")
    output_dir = Path(args.output_dir) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    config_paths = write_variant_configs(Path(args.base_dual_config), output_dir, variants)
    domains = (
        [Path(args.target_root).name]
        if args.target_root
        else [str(item) for item in args.target_domains]
    )
    _json_dump(
        {
            "experiment_id": experiment_id,
            "variants": list(variants),
            "domains": domains,
            "iterations": int(args.iterations),
            "epochs": int(args.epochs),
            "group_iterations": bool(args.group_iterations),
            "expected_runs": len(variants) * len(domains) * int(args.iterations),
            "monitor_metric": args.monitor_metric,
            "pca_feature_view": bool(args.pca_feature_view),
            "weather_profile_config": args.weather_profile_config,
        },
        output_dir / "experiment_manifest.json",
    )
    train_script = PROJECT_ROOT / "scripts" / "train_dual_d.py"
    for variant in variants:
        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(Path(args.base_train_config)),
            "--dual-config",
            str(config_paths[variant]),
            "--source-root",
            str(args.source_root),
            "--output-dir",
            str(output_dir),
            "--run-name",
            f"module_c_{variant}",
            "--iterations",
            str(args.iterations),
            "--epochs",
            str(args.epochs),
            "--group-iterations" if args.group_iterations else "--no-group-iterations",
            "--no-save-checkpoints",
            "--save-feature-embeddings",
            "--feature-visualization-samples",
            str(args.feature_visualization_samples),
            "--monitor-metric",
            str(args.monitor_metric),
            "--no-use-ais",
            "--no-multi-gpu",
            "--deterministic-training",
        ]
        if args.target_root:
            command.extend(["--target-root", str(args.target_root)])
        else:
            command.extend(["--target-parent-root", str(args.target_parent_root)])
            command.extend(["--target-domains", *[str(item) for item in args.target_domains]])
        _append_option(command, "--weather-profile-config", args.weather_profile_config)
        _append_option(command, "--batch-size", args.batch_size)
        _append_option(command, "--num-workers", args.num_workers)
        _append_option(command, "--device", args.device)
        _append_option(command, "--seed", args.seed)
        print("Running Module-C variant:", variant)
        print(" ".join(command))
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return output_dir


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows: List[Dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: Dict[str, Any] = dict(row)
            for key, value in list(parsed.items()):
                number = _to_float(value)
                if number is not None:
                    parsed[key] = number
            rows.append(parsed)
    return rows


def _infer_variant(run_dir: Path) -> str:
    name = run_dir.name
    for variant in sorted(VARIANTS, key=len, reverse=True):
        if name.startswith(f"module_c_{variant}_") or name == f"module_c_{variant}":
            return variant
    # Existing non-ablation runs are valid full-model baselines when explicitly
    # included in an analysis directory.
    return "full"


def _infer_domain(run_dir: Path) -> str:
    summary_path = run_dir / "result_summary.json"
    if summary_path.exists():
        try:
            summary = _json_load(summary_path)
            for key in ("target_domain", "domain"):
                if summary.get(key):
                    return str(summary[key])
        except (OSError, json.JSONDecodeError):
            pass
    name = run_dir.name
    for domain in DOMAIN_DISPLAY:
        if f"_{domain}_" in name or name.endswith(f"_{domain}"):
            return domain
    match = re.search(r"_(?P<domain>[^_]+)_iter\d+", name)
    return match.group("domain") if match else "all"


def _infer_iteration(run_dir: Path, artifact_name: str = "") -> int | None:
    match = re.search(r"_iter(?P<iteration>\d+)", f"{run_dir.name}_{artifact_name}")
    return int(match.group("iteration")) if match else None


def _resolved_seed(run_dir: Path, config_path: Path | None = None) -> int | None:
    path = config_path or run_dir / "resolved_config.json"
    if not path.exists():
        return None
    try:
        data = _json_load(path)
        value = data.get("args", {}).get("seed")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _best_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any]:
    stable_candidates = [
        row
        for row in rows
        if _to_float(row.get("monitor_selection_score")) is not None
    ]
    if stable_candidates:
        return max(
            stable_candidates,
            key=lambda row: float(row["monitor_selection_score"]),
        )
    candidates = [row for row in rows if _to_float(row.get(metric)) is not None]
    if not candidates:
        candidates = [row for row in rows if _to_float(row.get("val_acc")) is not None]
        metric = "val_acc"
    if not candidates:
        raise ValueError("metrics.csv has no validation metric columns")
    return max(candidates, key=lambda row: float(row.get(metric, float("-inf"))))


def _summarize_run(
    run_dir: Path,
    monitor_metric: str,
    metrics_path: Path | None = None,
) -> Dict[str, Any] | None:
    metrics_path = metrics_path or run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = _read_rows(metrics_path)
    if not rows:
        return None
    best = _best_row(rows, monitor_metric)
    val_acc = _to_float(best.get("val_acc"))
    val_precision = _to_float(best.get("val_precision_macro_present"))
    val_recall = _to_float(best.get("val_recall_macro_present"))
    val_f1 = _to_float(best.get("val_f1_macro_present"))
    train_full = _to_float(best.get("train_full_acc"))
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "run": f"{run_dir.name}/{metrics_path.stem}",
        "variant": _infer_variant(run_dir),
        "domain": _infer_domain(run_dir),
        "iteration": _infer_iteration(run_dir, metrics_path.name),
        "seed": _resolved_seed(
            run_dir,
            run_dir / f"resolved_config_{metrics_path.stem.removeprefix('metrics_')}.json"
            if metrics_path.stem.startswith("metrics_")
            else None,
        ),
        "metrics_path": str(metrics_path),
        "epochs": len(rows),
        "best_epoch": _to_float(best.get("epoch")),
        "best_val_acc": val_acc,
        "best_val_precision": val_precision,
        "best_val_recall": val_recall,
        "best_val_f1": val_f1,
        "best_train_full_acc": train_full,
        "generalization_gap": (
            train_full - val_acc if train_full is not None and val_acc is not None else None
        ),
        "feature_embeddings": str(
            run_dir
            / (
                f"feature_embeddings_{metrics_path.stem.removeprefix('metrics_')}.npz"
                if metrics_path.stem.startswith("metrics_")
                else "feature_embeddings.npz"
            )
        )
        if (
            run_dir
            / (
                f"feature_embeddings_{metrics_path.stem.removeprefix('metrics_')}.npz"
                if metrics_path.stem.startswith("metrics_")
                else "feature_embeddings.npz"
            )
        ).exists()
        else None,
        "result_path": str(
            run_dir
            / (
                f"result_summary_{metrics_path.stem.removeprefix('metrics_')}.json"
                if metrics_path.stem.startswith("metrics_")
                else "result_summary.json"
            )
        ),
    }
    for metric in set(LOSS_METRIC.values()) | {
        "train_dual_d_adv_primary",
        "train_dual_d_adv_auxiliary",
        "train_dual_d_generator_total",
        "train_acc_source_like",
        "train_acc_target_like",
    }:
        summary[f"best_{metric}"] = _to_float(best.get(metric))
    return summary


def discover_summaries(runs_root: Path, pattern: str, monitor_metric: str) -> List[Dict[str, Any]]:
    run_dirs = sorted(path for path in runs_root.glob(pattern) if path.is_dir())
    summaries: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        metric_paths = sorted(run_dir.glob("metrics*.csv"))
        for metrics_path in metric_paths:
            summary = _summarize_run(run_dir, monitor_metric, metrics_path)
            if summary is not None:
                summaries.append(summary)
    if not summaries:
        raise FileNotFoundError(f"No readable metrics.csv found under {runs_root} with pattern {pattern!r}")
    keys: Dict[tuple[str, str, int | None], str] = {}
    for summary in summaries:
        key = (summary["variant"], summary["domain"], summary["iteration"])
        if key in keys:
            raise RuntimeError(
                "Duplicate ablation run key "
                f"{key}: {keys[key]} and {summary['run_dir']}. "
                "Analyse one manifest-backed experiment directory at a time."
            )
        keys[key] = str(summary["run_dir"])
    return summaries


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["variant"]), str(row["domain"]))].append(row)
    keys = [
        "best_val_acc",
        "best_val_precision",
        "best_val_recall",
        "best_val_f1",
        "best_train_full_acc",
        "generalization_gap",
    ]
    output: List[Dict[str, Any]] = []
    for (variant, domain), rows in sorted(grouped.items()):
        result: Dict[str, Any] = {"variant": variant, "domain": domain, "runs": len(rows)}
        for key in keys:
            values = [float(row[key]) for row in rows if _to_float(row.get(key)) is not None]
            result[f"{key}_mean"] = mean(values) if values else None
            result[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
            result[f"{key}_ci95"] = (
                1.96 * stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0 if values else None
            )
        output.append(result)
    return output


def validate_manifest(
    runs_root: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Require exactly the run matrix declared by a new experiment manifest."""

    path = runs_root / "experiment_manifest.json"
    if not path.exists():
        return
    manifest = _json_load(path)
    expected = {
        (str(variant), str(domain), iteration)
        for variant in manifest["variants"]
        for domain in manifest["domains"]
        for iteration in range(1, int(manifest["iterations"]) + 1)
    }
    actual = {
        (str(row["variant"]), str(row["domain"]), int(row["iteration"]))
        for row in summaries
        if row.get("iteration") is not None
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    incomplete = [
        str(row["run_dir"])
        for row in summaries
        if not row.get("feature_embeddings")
        or not Path(str(row.get("result_path", ""))).exists()
    ]
    if missing or unexpected or incomplete:
        raise RuntimeError(
            "Ablation experiment is incomplete or contaminated: "
            f"missing={missing}, unexpected={unexpected}, incomplete={incomplete}"
        )
    if len(summaries) != int(manifest["expected_runs"]):
        raise RuntimeError(
            f"Expected {manifest['expected_runs']} runs, found {len(summaries)}."
        )


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _domain_display(domain: str) -> str:
    return DOMAIN_DISPLAY.get(str(domain), str(domain))


def _variant_display(variant: str) -> str:
    if variant == "full":
        return "Full Module C"
    if variant == "no_module_c":
        return "All Constraints Removed"
    return "Without " + variant.removeprefix("no_").replace("_", " ").title()


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<style>text{fill:#17202a} .grid{stroke:#d7dde3;stroke-width:1} '
        '.axis{stroke:#53616d;stroke-width:1.2} .full{fill:#2f6f9f} .ablated{fill:#b85c38} '
        '.line-full{fill:none;stroke:#2f6f9f;stroke-width:2.5} .line-ablated{fill:none;stroke:#b85c38;stroke-width:2.5}</style>'
        + body
        + '</svg>'
    )


def _svg_bar(x: float, y: float, width: float, height: float, css_class: str, label: str) -> str:
    return f'<rect class="{css_class}" x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{max(height, 0):.2f}"><title>{_svg_escape(label)}</title></rect>'


def _plot_overview_svg(summaries: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    domains = _domain_order(summaries)
    variants = [variant for variant in VARIANTS if any(row["variant"] == variant for row in summaries)]
    if not variants:
        return
    width, height = 960, 540
    left, top, plot_w, plot_h = 76, 62, 840, 380
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20">Module C ablation overview</text>']
    for tick in range(0, 6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/><line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    group_w = plot_w / max(len(variants), 1)
    bar_w = min(28, group_w / max(len(domains), 1) * 0.72)
    colors = ["#2f6f9f", "#b85c38", "#5d8a55", "#8a6fb0", "#be8b36", "#4d8c8c"]
    for variant_index, variant in enumerate(variants):
        center = left + group_w * (variant_index + 0.5)
        body.append(f'<text x="{center:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="11" transform="rotate(25 {center:.1f} {top+plot_h+24})">{_svg_escape(variant.replace("no_", "- "))}</text>')
        for domain_index, domain in enumerate(domains):
            value, _ = _mean_std(summaries, variant, domain, "best_val_f1")
            if value is None:
                continue
            x = center + (domain_index - (len(domains)-1)/2) * bar_w
            bar_height = plot_h * max(0.0, min(1.0, value))
            body.append(_svg_bar(x - bar_w/2, top + plot_h - bar_height, bar_w * 0.86, bar_height, "full" if variant == "full" else "ablated", f"{variant}, {_domain_display(domain)}: {value:.3f}"))
    legend_x = left
    for index, domain in enumerate(domains):
        x = legend_x + index * 130
        body.append(f'<rect x="{x}" y="{height-38}" width="12" height="12" fill="{colors[index % len(colors)]}"/><text x="{x+17}" y="{height-28}" font-size="12">{_svg_escape(_domain_display(domain))}</text>')
    body.append(f'<text x="18" y="{top+plot_h/2}" font-size="13" transform="rotate(-90 18 {top+plot_h/2})">Best validation F1</text>')
    (output_dir / "module_c_overview.svg").write_text(_svg_document(width, height, "".join(body)), encoding="utf-8")


def _plot_effect_heatmap_svg(summaries: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    domains = _domain_order(summaries)
    constraints = [name for name in CONSTRAINTS if any(row["variant"] == f"no_{name}" for row in summaries)]
    if not domains or not constraints:
        return
    width, height = 880, 120 + 48 * len(constraints)
    left, top, cell_w, cell_h = 190, 62, min(110, 620 / len(domains)), 38
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20">Full F1 minus leave-one-out F1</text>']
    for col, domain in enumerate(domains):
        body.append(f'<text x="{left + col*cell_w + cell_w/2:.1f}" y="{top-14}" text-anchor="middle" font-size="12">{_svg_escape(_domain_display(domain))}</text>')
    for row_index, constraint in enumerate(constraints):
        y = top + row_index * cell_h
        body.append(f'<text x="{left-10}" y="{y+cell_h/2+4:.1f}" text-anchor="end" font-size="12">{_svg_escape(constraint.replace("_", " "))}</text>')
        for col, domain in enumerate(domains):
            full, _ = _mean_std(summaries, "full", domain, "best_val_f1")
            ablated, _ = _mean_std(summaries, f"no_{constraint}", domain, "best_val_f1")
            value = (full - ablated) if full is not None and ablated is not None else float("nan")
            if math.isnan(value):
                fill = "#edf0f2"
                label = "n/a"
            else:
                clipped = max(-0.5, min(0.5, value))
                if clipped >= 0:
                    channel = int(255 - 150 * (clipped / 0.5))
                    fill = f"rgb({channel},{min(255, channel+45)},{channel})"
                else:
                    channel = int(255 - 150 * ((-clipped) / 0.5))
                    fill = f"rgb(255,{channel},{channel})"
                label = f"{value:+.3f}"
            x = left + col * cell_w
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-2:.1f}" height="{cell_h-2:.1f}" fill="{fill}"><title>{_svg_escape(label)}</title></rect>')
            body.append(f'<text x="{x+cell_w/2:.1f}" y="{y+cell_h/2+4:.1f}" text-anchor="middle" font-size="11">{label}</text>')
    (output_dir / "module_c_effect_heatmap.svg").write_text(_svg_document(width, height, "".join(body)), encoding="utf-8")


def _plot_constraint_diagnostic_svg(summaries: Sequence[Mapping[str, Any]], constraint: str, output_dir: Path) -> None:
    domains = _domain_order(summaries)
    ablation_variant = f"no_{constraint}"
    if not any(row["variant"] == ablation_variant for row in summaries):
        return
    width, height = 960, 700
    left, right, top = 82, 28, 52
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20">{_svg_escape(constraint.replace("_", " ").title())}: targeted diagnostic</text>']
    # Performance panel.
    panel_bottom = 300
    for tick in range(0, 6):
        value = tick / 5
        y = top + (panel_bottom-top) * (1-value)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/><text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{panel_bottom}"/><line class="axis" x1="{left}" y1="{panel_bottom}" x2="{width-right}" y2="{panel_bottom}"/>')
    group_w = (width-left-right) / max(len(domains), 1)
    for domain_index, domain in enumerate(domains):
        center = left + group_w * (domain_index + 0.5)
        for offset, variant, css_class, label in [(-18, "full", "full", "full"), (18, ablation_variant, "ablated", f"without {constraint}")]:
            value, _ = _mean_std(summaries, variant, domain, "best_val_f1")
            if value is None:
                continue
            bar_height = (panel_bottom-top) * max(0.0, min(1.0, value))
            body.append(_svg_bar(center+offset-13, panel_bottom-bar_height, 26, bar_height, css_class, f"{label}, {_domain_display(domain)}: {value:.3f}"))
        body.append(f'<text x="{center:.1f}" y="{panel_bottom+22}" text-anchor="middle" font-size="12">{_svg_escape(_domain_display(domain))}</text>')
    body.append(f'<text x="18" y="{(top+panel_bottom)/2}" font-size="13" transform="rotate(-90 18 {(top+panel_bottom)/2})">Best validation F1</text>')
    # Constraint-specific loss curves.
    curve_top, curve_bottom = 390, 635
    metric = LOSS_METRIC[constraint]
    curves = {variant: _read_curve_rows(summaries, variant, metric) for variant in ("full", ablation_variant)}
    all_values = [value for curve in curves.values() for values in curve.values() for value in values]
    max_value = max(all_values) if all_values else 1.0
    max_epoch = max([epoch for curve in curves.values() for epoch in curve] or [1])
    body.append(f'<line class="axis" x1="{left}" y1="{curve_top}" x2="{left}" y2="{curve_bottom}"/><line class="axis" x1="{left}" y1="{curve_bottom}" x2="{width-right}" y2="{curve_bottom}"/>')
    for variant, css_class, label in [("full", "line-full", "full"), (ablation_variant, "line-ablated", f"without {constraint}")]:
        curve = curves[variant]
        points = []
        for epoch in sorted(curve):
            x = left + (width-left-right) * epoch / max_epoch
            y = curve_bottom - (curve_bottom-curve_top) * mean(curve[epoch]) / max(max_value, 1e-9)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            body.append(f'<polyline class="{css_class}" points="{" ".join(points)}"><title>{_svg_escape(label)}</title></polyline>')
    body.append(f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-size="13">Epoch</text>')
    body.append(f'<text x="18" y="{(curve_top+curve_bottom)/2}" font-size="13" transform="rotate(-90 18 {(curve_top+curve_bottom)/2})">{_svg_escape(metric.replace("train_dual_d_", "").replace("_", " "))}</text>')
    body.append(f'<line class="line-full" x1="{width-245}" y1="{curve_top-18}" x2="{width-215}" y2="{curve_top-18}"/><text x="{width-208}" y="{curve_top-14}" font-size="12">full</text>')
    body.append(f'<line class="line-ablated" x1="{width-125}" y1="{curve_top-18}" x2="{width-95}" y2="{curve_top-18}"/><text x="{width-88}" y="{curve_top-14}" font-size="12">without { _svg_escape(constraint) }</text>')
    (output_dir / f"constraint_{constraint}.svg").write_text(_svg_document(width, height, "".join(body)), encoding="utf-8")


def _domain_order(summaries: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted({str(row["domain"]) for row in summaries})


def _mean_std(rows: Sequence[Mapping[str, Any]], variant: str, domain: str, key: str) -> tuple[float | None, float | None]:
    values = [
        float(row[key])
        for row in rows
        if row.get("variant") == variant and row.get("domain") == domain and _to_float(row.get(key)) is not None
    ]
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _read_curve_rows(
    summaries: Sequence[Mapping[str, Any]],
    variant: str,
    metric: str,
    domain: str | None = None,
) -> Dict[int, List[float]]:
    curves: Dict[int, List[float]] = defaultdict(list)
    for summary in summaries:
        if summary.get("variant") != variant:
            continue
        if domain is not None and summary.get("domain") != domain:
            continue
        path = Path(
            str(
                summary.get(
                    "metrics_path",
                    Path(str(summary["run_dir"])) / "metrics.csv",
                )
            )
        )
        if not path.exists():
            continue
        for row in _read_rows(path):
            epoch = _to_float(row.get("epoch"))
            value = _to_float(row.get(metric))
            if epoch is not None and value is not None:
                curves[int(epoch)].append(value)
    return curves


def _plot_overview(summaries: Sequence[Mapping[str, Any]], output_dir: Path, plt) -> None:
    domains = _domain_order(summaries)
    variants = [variant for variant in VARIANTS if any(row["variant"] == variant for row in summaries)]
    if not variants:
        return
    figure, axes = plt.subplots(2, 2, figsize=(max(11, 1.55 * len(variants)), 8.5))
    width = 0.8 / max(len(domains), 1)
    x_positions = list(range(len(variants)))
    for axis, (metric, metric_label) in zip(axes.flat, METRICS.items()):
        for domain_index, domain in enumerate(domains):
            values, errors = [], []
            for variant in variants:
                value, error = _mean_std(summaries, variant, domain, metric)
                values.append(value if value is not None else float("nan"))
                errors.append(error if error is not None else 0.0)
            offsets = [
                x + (domain_index - (len(domains) - 1) / 2) * width
                for x in x_positions
            ]
            axis.bar(
                offsets,
                values,
                width=width * 0.9,
                yerr=errors,
                capsize=2,
                label=_domain_display(domain),
            )
        axis.set_xticks(
            x_positions,
            [_variant_display(variant) for variant in variants],
            rotation=25,
            ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(metric_label)
        axis.set_title(metric_label + " at the selected best epoch")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=min(4, len(domains)), frameon=False)
    figure.suptitle("Module C ablation overview", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "module_c_overview.png", dpi=180)
    plt.close(figure)


def _plot_effect_heatmap(summaries: Sequence[Mapping[str, Any]], output_dir: Path, plt) -> None:
    domains = _domain_order(summaries)
    constraints = [name for name in CONSTRAINTS if any(row["variant"] == f"no_{name}" for row in summaries)]
    if not domains or not constraints:
        return
    values: List[List[float]] = []
    for constraint in constraints:
        row_values: List[float] = []
        for domain in domains:
            full, _ = _mean_std(summaries, "full", domain, "best_val_f1")
            ablated, _ = _mean_std(summaries, f"no_{constraint}", domain, "best_val_f1")
            row_values.append((full - ablated) if full is not None and ablated is not None else float("nan"))
        values.append(row_values)
    figure, axis = plt.subplots(figsize=(max(7, 1.3 * len(domains)), 4.4))
    finite_values = [abs(value) for row in values for value in row if not math.isnan(value)]
    limit = max(finite_values, default=0.05)
    limit = max(limit, 0.01)
    image = axis.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(
        range(len(domains)),
        [_domain_display(domain) for domain in domains],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(constraints)), [item.replace("_", " ") for item in constraints])
    axis.set_title("Full model F1 minus leave-one-out F1")
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            if not math.isnan(value):
                axis.text(col_index, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axis, label="F1 effect")
    figure.tight_layout()
    figure.savefig(output_dir / "module_c_effect_heatmap.png", dpi=180)
    plt.close(figure)


def _plot_constraint_diagnostic(
    summaries: Sequence[Mapping[str, Any]],
    constraint: str,
    output_dir: Path,
    plt,
) -> None:
    full_variant = "full"
    ablation_variant = f"no_{constraint}"
    domains = _domain_order(summaries)
    if not any(row["variant"] == ablation_variant for row in summaries):
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x_positions = list(range(len(domains)))
    width = 0.35
    for axis, (metric, label) in zip(axes.flat, METRICS.items()):
        for offset, variant, legend, color in (
            (-width / 2, full_variant, "Full Module C", "#2f6f9f"),
            (width / 2, ablation_variant, _variant_display(ablation_variant), "#b85c38"),
        ):
            values, errors = [], []
            for domain in domains:
                value, error = _mean_std(summaries, variant, domain, metric)
                values.append(value if value is not None else float("nan"))
                errors.append(error if error is not None else 0.0)
            axis.bar(
                [x + offset for x in x_positions],
                values,
                width=width,
                yerr=errors,
                capsize=3,
                label=legend,
                color=color,
            )
        axis.set_xticks(
            x_positions,
            [_domain_display(domain) for domain in domains],
            rotation=15,
            ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"Impact of {constraint.replace('_', ' ').title()}", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"constraint_{constraint}.png", dpi=180)
    plt.close(figure)

    raw_metric = LOSS_METRIC[constraint]
    weighted_metric = WEIGHTED_LOSS_METRIC[constraint]
    for domain in domains:
        loss_figure, loss_axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
        for loss_axis, metric, prefix in (
            (loss_axes[0], raw_metric, "Raw loss (still measured)"),
            (loss_axes[1], weighted_metric, "Weighted objective contribution"),
        ):
            for variant, legend, color in (
                (full_variant, "Full Module C", "#2f6f9f"),
                (ablation_variant, _variant_display(ablation_variant), "#b85c38"),
            ):
                curve = _read_curve_rows(summaries, variant, metric, domain=domain)
                epochs = [
                    epoch for epoch in sorted(curve) if len(curve[epoch]) >= 2
                ]
                if epochs:
                    means = [mean(curve[epoch]) for epoch in epochs]
                    loss_axis.plot(
                        epochs, means, linewidth=2, label=legend, color=color
                    )
            loss_axis.set_xlabel("Epoch")
            loss_axis.set_ylabel(
                metric.replace("train_dual_d_", "").replace("_", " ")
            )
            loss_axis.set_title(prefix)
            loss_axis.grid(alpha=0.25)
            loss_axis.legend(frameon=False)
        loss_figure.suptitle(
            f"{_domain_display(domain)}: {CONSTRAINTS[constraint]} ablation audit"
        )
        loss_figure.tight_layout()
        loss_figure.savefig(
            output_dir
            / f"loss_{constraint}_{_domain_display(domain).lower()}.png",
            dpi=180,
        )
        plt.close(loss_figure)


def _posthoc_projection(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one joint 2-D PCA projection used only for plotting.

    PCA is not part of Dual_D, TAL, training, inference, or any reported model
    metric.  It is a post-hoc view of one trained run and is never fitted across
    independent seeds or model variants.
    """

    centered = features.astype(np.float32) - features.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    coordinates = centered @ components.T
    variances = singular_values ** 2
    ratios = variances[:2] / max(float(variances.sum()), 1e-12)
    return coordinates, ratios


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def _cosine_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute a dependency-free cosine silhouette in the original space."""

    if len(features) < 3 or len(np.unique(labels)) < 2:
        return None
    normalized = _l2_normalize(features.astype(np.float64))
    distances = 1.0 - normalized @ normalized.T
    values = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        if not np.any(same):
            continue
        intra = float(distances[index, same].mean())
        inter = min(
            float(distances[index, labels == other].mean())
            for other in np.unique(labels)
            if other != label
        )
        values.append((inter - intra) / max(intra, inter, 1e-12))
    return float(np.mean(values)) if values else None


def _coral_distance(source: np.ndarray, target: np.ndarray) -> float | None:
    if len(source) < 2 or len(target) < 2:
        return None
    source_cov = np.cov(source.astype(np.float64), rowvar=False)
    target_cov = np.cov(target.astype(np.float64), rowvar=False)
    dimension = max(source.shape[1], 1)
    return float(np.square(source_cov - target_cov).sum() / (4.0 * dimension * dimension))


def _cosine_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> float | None:
    """Measure direction-preserving reconstruction error in original space."""

    if not len(original) or original.shape != reconstructed.shape:
        return None
    similarities = np.sum(
        _l2_normalize(original.astype(np.float64))
        * _l2_normalize(reconstructed.astype(np.float64)),
        axis=1,
    )
    return float(np.mean(1.0 - similarities))


def _class_centroid_distance(
    source: np.ndarray,
    source_labels: np.ndarray,
    target: np.ndarray,
    target_labels: np.ndarray,
) -> float | None:
    common = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    distances = []
    for class_id in common:
        source_centroid = source[source_labels == class_id].mean(axis=0, keepdims=True)
        target_centroid = target[target_labels == class_id].mean(axis=0, keepdims=True)
        source_centroid = _l2_normalize(source_centroid)[0]
        target_centroid = _l2_normalize(target_centroid)[0]
        distances.append(1.0 - float(source_centroid @ target_centroid))
    return float(np.mean(distances)) if distances else None


def _class_prototype_similarity(
    source: np.ndarray,
    source_labels: np.ndarray,
    target: np.ndarray,
    target_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return target-by-source class prototype cosine similarity.

    This diagnostic is computed in the original feature space.  It therefore
    exposes whether each translated target class is closest to the matching
    source class without PCA, t-SNE, UMAP or any label-fitted projection.
    """

    common = np.asarray(
        sorted(set(source_labels.tolist()) & set(target_labels.tolist())),
        dtype=np.int64,
    )
    if common.size == 0:
        return None
    source_prototypes = np.stack(
        [source[source_labels == class_id].mean(axis=0) for class_id in common]
    )
    target_prototypes = np.stack(
        [target[target_labels == class_id].mean(axis=0) for class_id in common]
    )
    similarities = _l2_normalize(target_prototypes) @ _l2_normalize(
        source_prototypes
    ).T
    return common, similarities


def _prototype_similarity_statistics(
    similarities: np.ndarray,
) -> tuple[float, float | None, float]:
    """Return diagonal similarity, diagonal margin and top-1 class retrieval."""

    diagonal = np.diag(similarities)
    diagonal_mean = float(diagonal.mean())
    top1 = float(
        np.mean(np.argmax(similarities, axis=1) == np.arange(similarities.shape[0]))
    )
    if similarities.shape[0] < 2:
        return diagonal_mean, None, top1
    off_diagonal = similarities.copy()
    np.fill_diagonal(off_diagonal, float("-inf"))
    margin = float(np.mean(diagonal - np.max(off_diagonal, axis=1)))
    return diagonal_mean, margin, top1


def _balanced_source_indices(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> np.ndarray:
    selected: List[int] = []
    for class_id in sorted(set(source_labels.tolist()) & set(target_labels.tolist())):
        source_indices = np.flatnonzero(source_labels == class_id)
        target_count = int(np.sum(target_labels == class_id))
        selected.extend(source_indices[: min(len(source_indices), target_count)].tolist())
    return np.asarray(selected, dtype=np.int64)


def _load_feature_run(summary: Mapping[str, Any]) -> Dict[str, np.ndarray] | None:
    feature_path = summary.get("feature_embeddings")
    path = (
        Path(str(feature_path))
        if feature_path
        else Path(str(summary["run_dir"])) / "feature_embeddings.npz"
    )
    if not path.exists():
        return None
    with np.load(path) as data:
        required = {
            "source_raw",
            "source_labels",
            "target_raw",
            "target_source_like",
            "target_labels",
        }
        if not required.issubset(data.files):
            return None
        optional = {
            "source_target_like",
            "source_reconstruction",
            "source_identity",
            "source_raw_logits",
            "source_target_like_logits",
            "source_sample_ids",
            "target_reconstruction",
            "target_identity",
            "target_raw_logits",
            "target_source_like_logits",
            "target_sample_ids",
        }
        return {
            key: np.asarray(data[key])
            for key in required | (optional & set(data.files))
        }


def _feature_metric_row(
    summary: Mapping[str, Any],
    snapshot: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    source = np.asarray(snapshot["source_raw"], dtype=np.float32)
    source_labels = np.asarray(snapshot["source_labels"], dtype=np.int64)
    target_raw = np.asarray(snapshot["target_raw"], dtype=np.float32)
    translated = np.asarray(snapshot["target_source_like"], dtype=np.float32)
    target_labels = np.asarray(snapshot["target_labels"], dtype=np.int64)
    source_indices = _balanced_source_indices(source_labels, target_labels)
    source = source[source_indices]
    source_labels = source_labels[source_indices]
    source_reconstruction = snapshot.get("source_reconstruction")
    source_identity = snapshot.get("source_identity")
    if source_reconstruction is not None:
        source_reconstruction = np.asarray(source_reconstruction)[source_indices]
    if source_identity is not None:
        source_identity = np.asarray(source_identity)[source_indices]
    raw_silhouette = _cosine_silhouette(target_raw, target_labels)
    translated_silhouette = _cosine_silhouette(translated, target_labels)
    raw_centroid_distance = _class_centroid_distance(
        source, source_labels, target_raw, target_labels
    )
    translated_centroid_distance = _class_centroid_distance(
        source, source_labels, translated, target_labels
    )
    raw_coral = _coral_distance(source, target_raw)
    translated_coral = _coral_distance(source, translated)
    raw_prototype_result = _class_prototype_similarity(
        source, source_labels, target_raw, target_labels
    )
    translated_prototype_result = _class_prototype_similarity(
        source, source_labels, translated, target_labels
    )
    raw_diagonal = raw_margin = raw_top1 = None
    translated_diagonal = translated_margin = translated_top1 = None
    if raw_prototype_result is not None:
        raw_diagonal, raw_margin, raw_top1 = _prototype_similarity_statistics(
            raw_prototype_result[1]
        )
    if translated_prototype_result is not None:
        translated_diagonal, translated_margin, translated_top1 = (
            _prototype_similarity_statistics(translated_prototype_result[1])
        )
    source_cycle_error = (
        _cosine_reconstruction_error(source, source_reconstruction)
        if source_reconstruction is not None
        else None
    )
    target_cycle_error = (
        _cosine_reconstruction_error(
            target_raw, np.asarray(snapshot["target_reconstruction"])
        )
        if "target_reconstruction" in snapshot
        else None
    )
    source_identity_error = (
        _cosine_reconstruction_error(source, source_identity)
        if source_identity is not None
        else None
    )
    target_identity_error = (
        _cosine_reconstruction_error(target_raw, np.asarray(snapshot["target_identity"]))
        if "target_identity" in snapshot
        else None
    )

    def average_available(*values: float | None) -> float | None:
        available = [float(value) for value in values if value is not None]
        return mean(available) if available else None

    return {
        "run": summary["run"],
        "variant": summary["variant"],
        "domain": summary["domain"],
        "iteration": summary.get("iteration"),
        "seed": summary.get("seed"),
        "target_raw_silhouette": raw_silhouette,
        "target_translated_silhouette": translated_silhouette,
        "target_silhouette_gain": (
            translated_silhouette - raw_silhouette
            if translated_silhouette is not None and raw_silhouette is not None
            else None
        ),
        "source_target_raw_class_centroid_distance": raw_centroid_distance,
        "source_target_translated_class_centroid_distance": translated_centroid_distance,
        "class_centroid_alignment_gain": (
            raw_centroid_distance - translated_centroid_distance
            if raw_centroid_distance is not None and translated_centroid_distance is not None
            else None
        ),
        "source_target_raw_coral": raw_coral,
        "source_target_translated_coral": translated_coral,
        "coral_alignment_gain": (
            raw_coral - translated_coral
            if raw_coral is not None and translated_coral is not None
            else None
        ),
        "source_target_raw_prototype_diagonal": raw_diagonal,
        "source_target_translated_prototype_diagonal": translated_diagonal,
        "prototype_diagonal_gain": (
            translated_diagonal - raw_diagonal
            if translated_diagonal is not None and raw_diagonal is not None
            else None
        ),
        "source_target_raw_prototype_margin": raw_margin,
        "source_target_translated_prototype_margin": translated_margin,
        "prototype_margin_gain": (
            translated_margin - raw_margin
            if translated_margin is not None and raw_margin is not None
            else None
        ),
        "source_target_raw_prototype_top1": raw_top1,
        "source_target_translated_prototype_top1": translated_top1,
        "prototype_top1_gain": (
            translated_top1 - raw_top1
            if translated_top1 is not None and raw_top1 is not None
            else None
        ),
        "source_cycle_cosine_error": source_cycle_error,
        "target_cycle_cosine_error": target_cycle_error,
        "cycle_cosine_error": average_available(
            source_cycle_error, target_cycle_error
        ),
        "source_identity_cosine_error": source_identity_error,
        "target_identity_cosine_error": target_identity_error,
        "identity_cosine_error": average_available(
            source_identity_error, target_identity_error
        ),
    }


def _row_cosine_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sample-wise cosine distance without dimensionality reduction."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or left.shape != right.shape or not len(left):
        return np.empty(0, dtype=np.float64)
    similarities = np.sum(_l2_normalize(left) * _l2_normalize(right), axis=1)
    return np.clip(1.0 - similarities, 0.0, 2.0)


def _aligned_feature_indices(
    full: Mapping[str, np.ndarray],
    ablated: Mapping[str, np.ndarray],
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match repeated weak-pair samples by stable ID and occurrence number."""

    raw_key = f"{domain}_raw"
    full_count = len(np.asarray(full[raw_key]))
    ablated_count = len(np.asarray(ablated[raw_key]))
    id_key = f"{domain}_sample_ids"
    if id_key not in full or id_key not in ablated:
        count = min(full_count, ablated_count)
        indices = np.arange(count, dtype=np.int64)
        return indices, indices.copy()

    full_ids = [str(value) for value in np.asarray(full[id_key]).tolist()]
    ablated_ids = [str(value) for value in np.asarray(ablated[id_key]).tolist()]
    positions: Dict[str, List[int]] = defaultdict(list)
    for index, sample_id in enumerate(ablated_ids):
        positions[sample_id].append(index)
    offsets: Dict[str, int] = defaultdict(int)
    full_indices: List[int] = []
    ablated_indices: List[int] = []
    for index, sample_id in enumerate(full_ids):
        offset = offsets[sample_id]
        candidates = positions.get(sample_id, [])
        if offset >= len(candidates):
            continue
        full_indices.append(index)
        ablated_indices.append(candidates[offset])
        offsets[sample_id] += 1
    if full_indices:
        return (
            np.asarray(full_indices, dtype=np.int64),
            np.asarray(ablated_indices, dtype=np.int64),
        )
    count = min(full_count, ablated_count)
    indices = np.arange(count, dtype=np.int64)
    return indices, indices.copy()


def _matching_label_indices(
    full: Mapping[str, np.ndarray],
    ablated: Mapping[str, np.ndarray],
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    full_indices, ablated_indices = _aligned_feature_indices(full, ablated, domain)
    label_key = f"{domain}_labels"
    full_labels = np.asarray(full[label_key], dtype=np.int64)[full_indices]
    ablated_labels = np.asarray(ablated[label_key], dtype=np.int64)[ablated_indices]
    matching = full_labels == ablated_labels
    return full_indices[matching], ablated_indices[matching]


def _prototype_sample_statistics(
    source: np.ndarray,
    source_labels: np.ndarray,
    target: np.ndarray,
    target_labels: np.ndarray,
) -> Dict[str, np.ndarray] | None:
    """Measure correct-class distance and nearest-wrong-class margin per sample."""

    common = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    if len(common) < 2:
        return None
    prototypes = np.stack(
        [source[source_labels == class_id].mean(axis=0) for class_id in common], axis=0
    )
    valid = np.isin(target_labels, common)
    if not np.any(valid):
        return None
    valid_target = _l2_normalize(np.asarray(target[valid], dtype=np.float64))
    normalized_prototypes = _l2_normalize(np.asarray(prototypes, dtype=np.float64))
    similarities = valid_target @ normalized_prototypes.T
    class_to_column = {class_id: index for index, class_id in enumerate(common)}
    labels = np.asarray(target_labels[valid], dtype=np.int64)
    correct_columns = np.asarray([class_to_column[int(label)] for label in labels])
    rows = np.arange(len(labels))
    correct_similarity = similarities[rows, correct_columns]
    wrong_similarities = similarities.copy()
    wrong_similarities[rows, correct_columns] = float("-inf")
    nearest_wrong_similarity = np.max(wrong_similarities, axis=1)
    return {
        "labels": labels,
        "correct_distance": 1.0 - correct_similarity,
        "nearest_wrong_distance": 1.0 - nearest_wrong_similarity,
        "margin": correct_similarity - nearest_wrong_similarity,
    }


def _classification_sample_statistics(
    logits: np.ndarray, labels: np.ndarray
) -> Dict[str, np.ndarray] | None:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or len(logits) != len(labels) or logits.shape[1] < 2:
        return None
    valid = (labels >= 0) & (labels < logits.shape[1])
    if not np.any(valid):
        return None
    logits = logits[valid]
    labels = labels[valid]
    rows = np.arange(len(labels))
    correct_logits = logits[rows, labels]
    wrong_logits = logits.copy()
    wrong_logits[rows, labels] = float("-inf")
    margin = correct_logits - np.max(wrong_logits, axis=1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    return {
        "labels": labels,
        "margin": margin,
        "correct_confidence": probabilities[rows, labels],
        "correct": (np.argmax(logits, axis=1) == labels).astype(np.float64),
    }


def _summary_pair_key(summary: Mapping[str, Any]) -> tuple[str, int, int]:
    seed = summary.get("seed")
    return (
        str(summary["domain"]),
        int(summary.get("iteration") or 0),
        int(seed) if seed is not None else -1,
    )


def _constraint_run_evidence(
    constraint: str,
    full_summary: Mapping[str, Any],
    ablated_summary: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
    full = _load_feature_run(full_summary)
    ablated = _load_feature_run(ablated_summary)
    if full is None or ablated is None:
        return None
    full_target_indices, ablated_target_indices = _matching_label_indices(
        full, ablated, "target"
    )
    full_source_indices, ablated_source_indices = _matching_label_indices(
        full, ablated, "source"
    )
    if not len(full_target_indices) or not len(full_source_indices):
        return None

    target_labels = np.asarray(full["target_labels"], dtype=np.int64)[full_target_indices]
    source_labels = np.asarray(full["source_labels"], dtype=np.int64)[full_source_indices]
    full_values: np.ndarray
    ablated_values: np.ndarray
    full_secondary: np.ndarray
    ablated_secondary: np.ndarray
    secondary_label: str
    higher_is_better: bool

    if constraint in {"cycle", "identity"}:
        feature_key = "reconstruction" if constraint == "cycle" else "identity"
        full_values = _row_cosine_distances(
            np.asarray(full["target_raw"])[full_target_indices],
            np.asarray(full[f"target_{feature_key}"])[full_target_indices],
        )
        ablated_values = _row_cosine_distances(
            np.asarray(ablated["target_raw"])[ablated_target_indices],
            np.asarray(ablated[f"target_{feature_key}"])[ablated_target_indices],
        )
        full_secondary = _row_cosine_distances(
            np.asarray(full["source_raw"])[full_source_indices],
            np.asarray(full[f"source_{feature_key}"])[full_source_indices],
        )
        ablated_secondary = _row_cosine_distances(
            np.asarray(ablated["source_raw"])[ablated_source_indices],
            np.asarray(ablated[f"source_{feature_key}"])[ablated_source_indices],
        )
        metric_label = f"Target {constraint} cosine error"
        secondary_label = f"Source {constraint} cosine error"
        higher_is_better = False
        full_run_metric = float(np.mean(np.concatenate([full_values, full_secondary])))
        ablated_run_metric = float(
            np.mean(np.concatenate([ablated_values, ablated_secondary]))
        )
    elif constraint in {"paired_contrastive", "prototype_contrastive"}:
        full_statistics = _prototype_sample_statistics(
            np.asarray(full["source_raw"])[full_source_indices],
            source_labels,
            np.asarray(full["target_source_like"])[full_target_indices],
            target_labels,
        )
        ablated_statistics = _prototype_sample_statistics(
            np.asarray(ablated["source_raw"])[ablated_source_indices],
            np.asarray(ablated["source_labels"], dtype=np.int64)[ablated_source_indices],
            np.asarray(ablated["target_source_like"])[ablated_target_indices],
            np.asarray(ablated["target_labels"], dtype=np.int64)[ablated_target_indices],
        )
        if full_statistics is None or ablated_statistics is None:
            return None
        full_values = full_statistics["margin"]
        ablated_values = ablated_statistics["margin"]
        target_labels = full_statistics["labels"]
        full_secondary = full_statistics["correct_distance"]
        ablated_secondary = ablated_statistics["correct_distance"]
        metric_label = "Correct-vs-nearest-wrong prototype margin"
        secondary_label = "Correct-class prototype cosine distance"
        higher_is_better = True
        full_run_metric = float(np.mean(full_values))
        ablated_run_metric = float(np.mean(ablated_values))
    else:
        logits_key = "target_source_like_logits"
        if logits_key not in full or logits_key not in ablated:
            return None
        full_statistics = _classification_sample_statistics(
            np.asarray(full[logits_key])[full_target_indices], target_labels
        )
        ablated_statistics = _classification_sample_statistics(
            np.asarray(ablated[logits_key])[ablated_target_indices],
            np.asarray(ablated["target_labels"], dtype=np.int64)[ablated_target_indices],
        )
        if full_statistics is None or ablated_statistics is None:
            return None
        full_values = full_statistics["margin"]
        ablated_values = ablated_statistics["margin"]
        target_labels = full_statistics["labels"]
        full_secondary = full_statistics["correct_confidence"]
        ablated_secondary = ablated_statistics["correct_confidence"]
        metric_label = "Generated-feature correct-class logit margin"
        secondary_label = "Generated-feature correct-class confidence"
        higher_is_better = True
        full_run_metric = float(np.mean(full_values))
        ablated_run_metric = float(np.mean(ablated_values))

    mechanism_gain = (
        full_run_metric - ablated_run_metric
        if higher_is_better
        else ablated_run_metric - full_run_metric
    )
    full_f1 = _to_float(full_summary.get("best_val_f1"))
    ablated_f1 = _to_float(ablated_summary.get("best_val_f1"))
    evidence = {
        "constraint": constraint,
        "domain": full_summary["domain"],
        "iteration": full_summary.get("iteration"),
        "seed": full_summary.get("seed"),
        "metric": metric_label,
        "higher_is_better": higher_is_better,
        "full_metric_mean": full_run_metric,
        "ablated_metric_mean": ablated_run_metric,
        "mechanism_gain": mechanism_gain,
        "full_secondary_mean": float(np.mean(full_secondary)),
        "ablated_secondary_mean": float(np.mean(ablated_secondary)),
        "full_val_f1": full_f1,
        "ablated_val_f1": ablated_f1,
        "f1_gain": (
            full_f1 - ablated_f1
            if full_f1 is not None and ablated_f1 is not None
            else None
        ),
        "paired_target_samples": len(full_values),
        "paired_source_samples": len(full_source_indices),
    }
    plot_data = {
        "full_values": full_values,
        "ablated_values": ablated_values,
        "full_secondary": full_secondary,
        "ablated_secondary": ablated_secondary,
        "labels": target_labels,
        "metric_label": metric_label,
        "secondary_label": secondary_label,
        "higher_is_better": higher_is_better,
    }
    return evidence, plot_data


def _comparison_boxplot(axis, full_values, ablated_values, ylabel: str) -> None:
    boxplot = axis.boxplot(
        [full_values, ablated_values],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    for patch, color in zip(boxplot["boxes"], ("#2f6f9f", "#b85c38")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_xticks([1, 2], ["With constraint", "Without constraint"])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def _plot_constraint_feature_evidence(
    summaries: Sequence[Mapping[str, Any]], output_dir: Path, plt
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Plot paired Full-vs-ablation mechanism evidence in the original space."""

    feature_dir = output_dir / "constraint_feature_evidence"
    feature_dir.mkdir(parents=True, exist_ok=True)
    full_index = {
        _summary_pair_key(summary): summary
        for summary in summaries
        if summary["variant"] == "full"
    }
    generated: List[str] = []
    evidence_rows: List[Dict[str, Any]] = []
    for constraint in CONSTRAINTS:
        ablation_variant = f"no_{constraint}"
        ablated_index = {
            _summary_pair_key(summary): summary
            for summary in summaries
            if summary["variant"] == ablation_variant
        }
        grouped: Dict[str, List[tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
        for key in sorted(set(full_index) & set(ablated_index)):
            result = _constraint_run_evidence(
                constraint, full_index[key], ablated_index[key]
            )
            if result is None:
                continue
            evidence, plot_data = result
            evidence_rows.append(evidence)
            grouped[str(evidence["domain"])].append((evidence, plot_data))

        for domain, records in grouped.items():
            figure, axes = plt.subplots(2, 2, figsize=(14, 9))
            full_values = np.concatenate(
                [np.asarray(plot_data["full_values"]) for _, plot_data in records]
            )
            ablated_values = np.concatenate(
                [np.asarray(plot_data["ablated_values"]) for _, plot_data in records]
            )
            full_secondary = np.concatenate(
                [np.asarray(plot_data["full_secondary"]) for _, plot_data in records]
            )
            ablated_secondary = np.concatenate(
                [np.asarray(plot_data["ablated_secondary"]) for _, plot_data in records]
            )
            _comparison_boxplot(
                axes[0, 0], full_values, ablated_values, records[0][1]["metric_label"]
            )
            axes[0, 0].set_title("Target feature mechanism")
            _comparison_boxplot(
                axes[0, 1],
                full_secondary,
                ablated_secondary,
                records[0][1]["secondary_label"],
            )
            axes[0, 1].set_title("Complementary feature evidence")

            labels = np.concatenate(
                [np.asarray(plot_data["labels"], dtype=np.int64) for _, plot_data in records]
            )
            classes = sorted(set(labels.tolist()))
            positions = np.arange(len(classes), dtype=np.float64)
            width = 0.38
            full_class_means = [
                float(np.mean(full_values[labels == class_id])) for class_id in classes
            ]
            ablated_class_means = [
                float(np.mean(ablated_values[labels == class_id])) for class_id in classes
            ]
            axes[1, 0].bar(
                positions - width / 2,
                full_class_means,
                width,
                label="With constraint",
                color="#2f6f9f",
            )
            axes[1, 0].bar(
                positions + width / 2,
                ablated_class_means,
                width,
                label="Without constraint",
                color="#b85c38",
            )
            axes[1, 0].set_xticks(positions, [str(class_id) for class_id in classes])
            axes[1, 0].set_xlabel("Class ID")
            axes[1, 0].set_ylabel(records[0][1]["metric_label"])
            axes[1, 0].set_title("Per-class target feature comparison")
            axes[1, 0].grid(axis="y", alpha=0.25)
            axes[1, 0].legend(frameon=False)

            gains = [
                (evidence["mechanism_gain"], evidence.get("f1_gain"), evidence["iteration"])
                for evidence, _ in records
                if evidence.get("f1_gain") is not None
            ]
            axes[1, 1].axhline(0.0, color="#7f8c8d", linewidth=1)
            axes[1, 1].axvline(0.0, color="#7f8c8d", linewidth=1)
            if gains:
                axes[1, 1].scatter(
                    [item[0] for item in gains],
                    [item[1] for item in gains],
                    s=65,
                    color="#2a9d8f",
                )
                for x_value, y_value, iteration in gains:
                    axes[1, 1].annotate(
                        f"iter {int(iteration or 0):02d}",
                        (x_value, y_value),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                    )
            axes[1, 1].set_xlabel("Mechanism gain (positive favors Full)")
            axes[1, 1].set_ylabel("Macro F1 gain (Full - ablation)")
            axes[1, 1].set_title("Feature mechanism versus classification")
            axes[1, 1].grid(alpha=0.25)
            axes[1, 1].margins(x=0.18, y=0.18)
            description = CONSTRAINTS[constraint]
            description = description[0].upper() + description[1:]
            figure.suptitle(
                f"{_domain_display(domain)} | {description}\n"
                "Paired samples and seeds; metrics computed in the original feature space",
                fontsize=14,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            filename = (
                f"constraint_feature_{constraint}_{_domain_display(domain).lower()}.png"
            )
            target_path = feature_dir / filename
            figure.savefig(target_path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            generated.append(str(target_path.relative_to(output_dir)))
    return generated, evidence_rows


def _plot_feature_diagnostics(
    summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
    plt,
    plot_projection: bool = True,
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Measure every run and optionally plot one shared post-hoc PCA view."""

    feature_dir = output_dir / "feature_diagnostics"
    feature_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    metric_rows: List[Dict[str, Any]] = []
    for summary in summaries:
        snapshot = _load_feature_run(summary)
        if snapshot is None:
            continue
        metric_row = _feature_metric_row(summary, snapshot)
        metric_rows.append(metric_row)
        if not plot_projection:
            continue
        source = np.asarray(snapshot["source_raw"], dtype=np.float32)
        source_labels = np.asarray(snapshot["source_labels"], dtype=np.int64)
        target_raw = np.asarray(snapshot["target_raw"], dtype=np.float32)
        translated = np.asarray(snapshot["target_source_like"], dtype=np.float32)
        target_labels = np.asarray(snapshot["target_labels"], dtype=np.int64)
        source_indices = _balanced_source_indices(source_labels, target_labels)
        source = source[source_indices]
        source_labels = source_labels[source_indices]
        joint = np.concatenate([source, target_raw, translated], axis=0)
        coordinates, ratios = _posthoc_projection(joint)
        source_coords = coordinates[: len(source)]
        raw_coords = coordinates[len(source) : len(source) + len(target_raw)]
        translated_coords = coordinates[len(source) + len(target_raw) :]
        figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].scatter(
            source_coords[:, 0], source_coords[:, 1], s=28, alpha=0.65,
            color="#2f6f9f", marker="o", label="Source raw"
        )
        axes[0, 0].scatter(
            raw_coords[:, 0], raw_coords[:, 1], s=32, alpha=0.75,
            color="#b85c38", marker="x", label="Target raw"
        )
        axes[0, 0].set_title("Domain geometry before translation")
        axes[0, 1].scatter(
            source_coords[:, 0], source_coords[:, 1], s=28, alpha=0.65,
            color="#2f6f9f", marker="o", label="Source raw"
        )
        axes[0, 1].scatter(
            translated_coords[:, 0], translated_coords[:, 1], s=32, alpha=0.75,
            color="#2a9d8f", marker="x", label="Target source-like"
        )
        axes[0, 1].set_title("Domain geometry after translation")
        cmap = plt.get_cmap("tab20")
        class_ids = sorted(set(target_labels.tolist()))
        for class_index, class_id in enumerate(class_ids):
            mask = target_labels == class_id
            color = cmap(class_index % 20)
            for start, end in zip(raw_coords[mask], translated_coords[mask]):
                axes[1, 0].annotate(
                    "", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.35, "lw": 0.8},
                )
            axes[1, 0].scatter(
                raw_coords[mask, 0], raw_coords[mask, 1], s=20,
                color=[color], alpha=0.55
            )
            axes[1, 0].scatter(
                translated_coords[mask, 0], translated_coords[mask, 1], s=28,
                color=[color], alpha=0.9
            )
            source_mask = source_labels == class_id
            if np.any(source_mask):
                axes[1, 1].scatter(
                    source_coords[source_mask, 0], source_coords[source_mask, 1],
                    s=30, facecolors="none", edgecolors=[color], marker="o"
                )
            axes[1, 1].scatter(
                translated_coords[mask, 0], translated_coords[mask, 1],
                s=32, color=[color], marker="x", label=f"Class {class_id}"
            )
        axes[1, 0].set_title("Target movement: raw to source-like")
        axes[1, 1].set_title("Class alignment after translation")
        for axis in axes.flat:
            axis.set_xlabel(f"Post-hoc PC1 ({ratios[0] * 100:.1f}% variance)")
            axis.set_ylabel(f"Post-hoc PC2 ({ratios[1] * 100:.1f}% variance)")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False)
        axes[0, 1].legend(frameon=False)
        handles, labels = axes[1, 1].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles, labels, loc="lower center",
                ncol=min(7, len(labels)), frameon=False
            )
        silhouette_raw = metric_row["target_raw_silhouette"]
        silhouette_translated = metric_row["target_translated_silhouette"]
        domain = _domain_display(str(summary["domain"]))
        figure.suptitle(
            f"{_variant_display(str(summary['variant']))} | {domain} | "
            f"iteration {int(summary.get('iteration') or 0):02d}\n"
            "Post-hoc PCA for visualization only; not part of the model | "
            f"target silhouette {silhouette_raw:.3f} -> {silhouette_translated:.3f}",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.93))
        filename = (
            f"feature_alignment_{summary['variant']}_{domain.lower()}_"
            f"iter{int(summary.get('iteration') or 0):02d}.png"
        )
        target_path = feature_dir / filename
        figure.savefig(target_path, dpi=180)
        plt.close(figure)
        generated.append(str(target_path.relative_to(output_dir)))
    return generated, metric_rows


def _aggregate_feature_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["domain"]))].append(row)
    metric_keys = (
        "target_raw_silhouette",
        "target_translated_silhouette",
        "target_silhouette_gain",
        "source_target_raw_class_centroid_distance",
        "source_target_translated_class_centroid_distance",
        "class_centroid_alignment_gain",
        "source_target_raw_coral",
        "source_target_translated_coral",
        "coral_alignment_gain",
        "source_target_raw_prototype_diagonal",
        "source_target_translated_prototype_diagonal",
        "prototype_diagonal_gain",
        "source_target_raw_prototype_margin",
        "source_target_translated_prototype_margin",
        "prototype_margin_gain",
        "source_target_raw_prototype_top1",
        "source_target_translated_prototype_top1",
        "prototype_top1_gain",
        "source_cycle_cosine_error",
        "target_cycle_cosine_error",
        "cycle_cosine_error",
        "source_identity_cosine_error",
        "target_identity_cosine_error",
        "identity_cosine_error",
    )
    output: List[Dict[str, Any]] = []
    for (variant, domain), group in sorted(grouped.items()):
        result: Dict[str, Any] = {
            "variant": variant,
            "domain": domain,
            "runs": len(group),
        }
        for key in metric_keys:
            values = [
                float(row[key])
                for row in group
                if _to_float(row.get(key)) is not None
            ]
            result[f"{key}_mean"] = mean(values) if values else None
            result[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(result)
    return output


def _plot_prototype_alignment_by_domain(
    summaries: Sequence[Mapping[str, Any]], output_dir: Path, plt
) -> List[str]:
    """Plot repetition-averaged prototype alignment without dimensionality reduction."""

    grouped: Dict[
        tuple[str, str, tuple[int, ...]], List[np.ndarray]
    ] = defaultdict(list)
    margins: Dict[tuple[str, str, tuple[int, ...]], List[float]] = defaultdict(list)
    full_raw_grouped: Dict[tuple[str, tuple[int, ...]], List[np.ndarray]] = defaultdict(list)
    full_raw_margins: Dict[tuple[str, tuple[int, ...]], List[float]] = defaultdict(list)
    for summary in summaries:
        snapshot = _load_feature_run(summary)
        if snapshot is None:
            continue
        source = np.asarray(snapshot["source_raw"], dtype=np.float32)
        source_labels = np.asarray(snapshot["source_labels"], dtype=np.int64)
        target_labels = np.asarray(snapshot["target_labels"], dtype=np.int64)
        result = _class_prototype_similarity(
            source,
            source_labels,
            np.asarray(snapshot["target_source_like"], dtype=np.float32),
            target_labels,
        )
        if result is None:
            continue
        class_ids, similarities = result
        key = (
            str(summary["variant"]),
            str(summary["domain"]),
            tuple(int(item) for item in class_ids),
        )
        grouped[key].append(similarities)
        margin = _prototype_similarity_statistics(similarities)[1]
        if margin is not None:
            margins[key].append(margin)
        if str(summary["variant"]) == "full":
            raw_result = _class_prototype_similarity(
                source,
                source_labels,
                np.asarray(snapshot["target_raw"], dtype=np.float32),
                target_labels,
            )
            if raw_result is not None:
                raw_class_ids, raw_similarities = raw_result
                raw_key = (
                    str(summary["domain"]),
                    tuple(int(item) for item in raw_class_ids),
                )
                full_raw_grouped[raw_key].append(raw_similarities)
                raw_margin = _prototype_similarity_statistics(raw_similarities)[1]
                if raw_margin is not None:
                    full_raw_margins[raw_key].append(raw_margin)

    generated: List[str] = []
    domains = sorted({key[1] for key in grouped})
    for domain in domains:
        selected = {}
        for variant in VARIANTS:
            candidates = [
                key for key in grouped if key[0] == variant and key[1] == domain
            ]
            if candidates:
                selected[variant] = max(candidates, key=lambda key: len(grouped[key]))
        if not selected:
            continue
        variants = [variant for variant in VARIANTS if variant in selected]
        panels = []
        if "full" in selected:
            full_classes = selected["full"][2]
            raw_key = (domain, full_classes)
            if raw_key in full_raw_grouped:
                panels.append(
                    (
                        "Full: target raw (before)",
                        full_raw_grouped[raw_key],
                        full_raw_margins.get(raw_key, []),
                        full_classes,
                        "Target class prototype",
                    )
                )
        for variant in variants:
            key = selected[variant]
            panels.append(
                (
                    _variant_display(variant),
                    grouped[key],
                    margins.get(key, []),
                    key[2],
                    "Translated target class prototype",
                )
            )
        columns = min(4, len(panels))
        rows = int(math.ceil(len(panels) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(4.3 * columns, 4.1 * rows), squeeze=False
        )
        used_axes = []
        image = None
        for axis, (title, matrices, margin_values, class_ids, y_label) in zip(
            axes.flat, panels
        ):
            matrix = np.mean(np.stack(matrices), axis=0)
            image = axis.imshow(
                matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="equal"
            )
            used_axes.append(axis)
            axis.set_xticks(range(len(class_ids)), [str(item) for item in class_ids])
            axis.set_yticks(range(len(class_ids)), [str(item) for item in class_ids])
            axis.set_xlabel("Source class prototype")
            axis.set_ylabel(y_label)
            margin_mean = mean(margin_values) if margin_values else None
            margin_std = stdev(margin_values) if len(margin_values) > 1 else 0.0
            margin_text = (
                f"margin {margin_mean:+.3f} +/- {margin_std:.3f}"
                if margin_mean is not None
                else "margin unavailable"
            )
            axis.set_title(
                f"{title}\n{margin_text} | n={len(matrices)}",
                fontsize=10,
            )
            if len(class_ids) <= 10:
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        axis.text(
                            column_index,
                            row_index,
                            f"{matrix[row_index, column_index]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black",
                        )
        for axis in list(axes.flat)[len(panels) :]:
            axis.axis("off")
        if image is not None:
            figure.colorbar(
                image,
                ax=used_axes,
                fraction=0.025,
                pad=0.02,
                label="Cosine similarity",
            )
        figure.suptitle(
            f"{_domain_display(domain)}: class-prototype alignment in the original feature space\n"
            "No PCA, t-SNE, UMAP or supervised projection",
            fontsize=14,
        )
        figure.subplots_adjust(top=0.86, wspace=0.38, hspace=0.46)
        filename = f"prototype_alignment_{_domain_display(domain).lower()}.png"
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        generated.append(filename)
    return generated


def _plot_feature_metric_effects(
    aggregates: Sequence[Mapping[str, Any]], output_dir: Path, plt
) -> None:
    domains = sorted({str(row["domain"]) for row in aggregates})
    variants = [
        variant
        for variant in VARIANTS
        if variant != "full" and any(row["variant"] == variant for row in aggregates)
    ]
    if not domains or not variants:
        return
    metric_specs = (
        (
            "target_translated_silhouette_mean",
            "Full minus ablation: target class silhouette",
            1.0,
        ),
        (
            "source_target_translated_prototype_margin_mean",
            "Full minus ablation: matching-prototype margin",
            1.0,
        ),
        (
            "source_target_translated_class_centroid_distance_mean",
            "Ablation minus full: cross-domain class-centroid distance",
            -1.0,
        ),
        (
            "source_target_translated_coral_mean",
            "Ablation minus full: CORAL distance",
            -1.0,
        ),
        (
            "cycle_cosine_error_mean",
            "Ablation minus full: cycle reconstruction error",
            -1.0,
        ),
        (
            "identity_cosine_error_mean",
            "Ablation minus full: identity preservation error",
            -1.0,
        ),
    )
    figure, axes = plt.subplots(
        3, 2, figsize=(13, max(11.5, 1.1 * len(variants))), squeeze=False
    )
    for axis, (metric, title, full_sign) in zip(axes.flat, metric_specs):
        values = []
        for variant in variants:
            row_values = []
            for domain in domains:
                full = next(
                    (
                        _to_float(row.get(metric))
                        for row in aggregates
                        if row["variant"] == "full" and row["domain"] == domain
                    ),
                    None,
                )
                ablated = next(
                    (
                        _to_float(row.get(metric))
                        for row in aggregates
                        if row["variant"] == variant and row["domain"] == domain
                    ),
                    None,
                )
                if full is None or ablated is None:
                    row_values.append(float("nan"))
                elif full_sign > 0:
                    row_values.append(full - ablated)
                else:
                    row_values.append(ablated - full)
            values.append(row_values)
        finite = [abs(value) for row in values for value in row if not math.isnan(value)]
        limit = max(max(finite, default=0.01), 1e-6)
        image = axis.imshow(
            values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto"
        )
        axis.set_xticks(
            range(len(domains)), [_domain_display(domain) for domain in domains],
            rotation=20, ha="right"
        )
        axis.set_yticks(
            range(len(variants)), [_variant_display(variant) for variant in variants]
        )
        axis.set_title(title, fontsize=10)
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                if not math.isnan(value):
                    axis.text(
                        column_index, row_index, f"{value:+.3f}",
                        ha="center", va="center", fontsize=8
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "Original high-dimensional feature effects (positive values favor Full Module C)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "feature_metric_effects.png", dpi=180)
    plt.close(figure)


def _write_html_report(output_dir: Path, aggregates: Sequence[Mapping[str, Any]], constraints: Sequence[str]) -> None:
    rows = []
    for row in aggregates:
        display_row = dict(row)
        display_row["variant"] = _variant_display(str(row.get("variant", "")))
        display_row["domain"] = _domain_display(str(row.get("domain", "")))
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(display_row.get(key, '')))}</td>"
                for key in (
                    "variant",
                    "domain",
                    "runs",
                    "best_val_acc_mean",
                    "best_val_precision_mean",
                    "best_val_recall_mean",
                    "best_val_f1_mean",
                    "best_val_f1_ci95",
                )
            )
            + "</tr>"
        )
    image_stems = [
        "module_c_overview",
        "module_c_effect_heatmap",
        "feature_metric_effects",
    ] + [f"constraint_{name}" for name in constraints]
    feature_images = sorted(output_dir.glob("prototype_alignment_*.png"))
    feature_images += sorted((output_dir / "feature_diagnostics").glob("*.png"))
    feature_images += sorted(
        (output_dir / "constraint_feature_evidence").glob("*.png")
    )
    image_tags_parts = []
    for stem in image_stems:
        image = next((candidate for candidate in (f"{stem}.png", f"{stem}.svg") if (output_dir / candidate).exists()), None)
        if image:
            image_tags_parts.append(f'<h2>{html.escape(stem)}</h2><img src="{html.escape(image)}" alt="{html.escape(stem)}">')
    for image in feature_images:
        relative = image.relative_to(output_dir).as_posix()
        image_tags_parts.append(
            f'<h2>{html.escape(image.stem)}</h2>'
            f'<img src="{html.escape(relative)}" alt="{html.escape(image.stem)}">'
        )
    image_tags = "\n".join(image_tags_parts)
    document = """<!doctype html><meta charset="utf-8"><title>Module C ablation report</title>
<style>body{max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin-bottom:2rem}th,td{border:1px solid #ccd;padding:.4rem;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}</style>
<h1>Module C ablation report</h1><table><thead><tr><th>variant</th><th>domain</th><th>runs</th><th>accuracy</th><th>macro precision</th><th>macro recall</th><th>macro F1</th><th>F1 95% CI</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table>" + image_tags
    (output_dir / "ablation_report.html").write_text(document, encoding="utf-8")


def analyse(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.analysis_output or args.runs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = discover_summaries(Path(args.runs_root), args.analysis_glob, args.monitor_metric)
    validate_manifest(Path(args.runs_root), summaries)
    reference_roots = [
        Path(path)
        for path in getattr(args, "reference_runs_root", [])
        if str(path).strip()
    ]
    for reference_root in reference_roots:
        if not reference_root.is_dir():
            raise FileNotFoundError(
                f"Reference runs directory does not exist: {reference_root}"
            )
        summaries.extend(
            discover_summaries(reference_root, args.analysis_glob, args.monitor_metric)
        )
    unique_summaries: Dict[tuple[str, str], Dict[str, Any]] = {}
    for summary in summaries:
        key = (str(summary["run_dir"]), str(summary["metrics_path"]))
        unique_summaries[key] = summary
    summaries = list(unique_summaries.values())
    semantic_keys: Dict[tuple[str, str, int | None], Dict[str, Any]] = {}
    for summary in summaries:
        semantic_key = (
            str(summary["variant"]),
            str(summary["domain"]),
            summary.get("iteration"),
        )
        previous = semantic_keys.get(semantic_key)
        if previous is not None:
            raise RuntimeError(
                "Duplicate primary/reference ablation run key "
                f"{semantic_key}: {previous['metrics_path']} and "
                f"{summary['metrics_path']}. Remove the overlapping reference "
                "or training domain before analysis."
            )
        semantic_keys[semantic_key] = summary
    aggregates = aggregate_summaries(summaries)
    _write_csv(summaries, output_dir / "ablation_runs.csv")
    _write_csv(aggregates, output_dir / "ablation_summary.csv")
    payload = {
        "monitor_metric": args.monitor_metric,
        "pca_feature_view": bool(args.pca_feature_view),
        "constraints": dict(CONSTRAINTS),
        "reference_runs_root": [str(path) for path in reference_roots],
        "runs": summaries,
        "aggregate": aggregates,
    }
    _json_dump(payload, output_dir / "ablation_summary.json")
    try:
        plt = _load_matplotlib()
    except ImportError as error:
        print(f"matplotlib unavailable; writing dependency-free SVG diagnostics ({error})", file=sys.stderr)
        _plot_overview_svg(summaries, output_dir)
        _plot_effect_heatmap_svg(summaries, output_dir)
        available_constraints = [name for name in CONSTRAINTS if any(row["variant"] == f"no_{name}" for row in summaries)]
        for constraint in available_constraints:
            _plot_constraint_diagnostic_svg(summaries, constraint, output_dir)
        _write_html_report(output_dir, aggregates, available_constraints)
        return payload
    _plot_overview(summaries, output_dir, plt)
    _plot_effect_heatmap(summaries, output_dir, plt)
    available_constraints = [name for name in CONSTRAINTS if any(row["variant"] == f"no_{name}" for row in summaries)]
    for constraint in available_constraints:
        _plot_constraint_diagnostic(summaries, constraint, output_dir, plt)
    feature_images, feature_metrics = _plot_feature_diagnostics(
        summaries,
        output_dir,
        plt,
        plot_projection=bool(args.pca_feature_view),
    )
    feature_aggregate = _aggregate_feature_metrics(feature_metrics)
    _write_csv(feature_metrics, output_dir / "feature_diagnostics_runs.csv")
    _write_csv(feature_aggregate, output_dir / "feature_diagnostics_summary.csv")
    _plot_feature_metric_effects(feature_aggregate, output_dir, plt)
    prototype_alignment_images = _plot_prototype_alignment_by_domain(
        summaries, output_dir, plt
    )
    constraint_feature_images, constraint_feature_evidence = (
        _plot_constraint_feature_evidence(summaries, output_dir, plt)
    )
    _write_csv(
        constraint_feature_evidence,
        output_dir / "constraint_feature_evidence.csv",
    )
    payload["feature_images"] = feature_images
    payload["prototype_alignment_images"] = prototype_alignment_images
    payload["constraint_feature_images"] = constraint_feature_images
    payload["constraint_feature_evidence"] = constraint_feature_evidence
    payload["feature_metrics"] = feature_metrics
    payload["feature_metric_aggregate"] = feature_aggregate
    _json_dump(payload, output_dir / "ablation_summary.json")
    _write_html_report(output_dir, aggregates, available_constraints)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and analyse Module C leave-one-constraint-out experiments.")
    parser.add_argument("--run", action="store_true", help="Launch training variants before analysis.")
    parser.add_argument("--base-train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument(
        "--weather-profile-config",
        default="",
        help=(
            "Optional per-weather override JSON. When omitted, the value from "
            "--base-train-config is used."
        ),
    )
    parser.add_argument("--base-dual-config", default=str(DEFAULT_DUAL_CONFIG))
    parser.add_argument("--source-root", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--target-parent-root", default="")
    parser.add_argument("--target-domains", nargs="+", default=["黑天", "逆光", "雾天", "雨天"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--group-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all repetitions for one variant/weather in a shared folder.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "module_c_ablation"))
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Optional clean subdirectory name; generated automatically with --run.",
    )
    parser.add_argument("--runs-root", default=str(PROJECT_ROOT / "runs" / "module_c_ablation"))
    parser.add_argument(
        "--reference-runs-root",
        nargs="*",
        default=[],
        help=(
            "Optional completed run directories included only during analysis. "
            "They are never retrained or modified."
        ),
    )
    parser.add_argument("--analysis-output", default="")
    parser.add_argument("--analysis-glob", default="module_c_*")
    parser.add_argument("--monitor-metric", default="val_f1_macro_present")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--feature-visualization-samples", type=int, default=512)
    parser.add_argument(
        "--pca-feature-view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write optional per-run post-hoc PCA scatter plots. Original-space "
            "prototype and quantitative diagnostics are always produced."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variants = list(dict.fromkeys(args.variants))
    if args.run:
        experiment_dir = run_variants(args, variants)
        args.runs_root = str(experiment_dir)
        args.analysis_glob = "module_c_*"
        if not args.analysis_output:
            args.analysis_output = str(experiment_dir)
    payload = analyse(args)
    print(f"Analysed {len(payload['runs'])} runs; wrote results to {args.analysis_output or args.runs_root}")


if __name__ == "__main__":
    main()
