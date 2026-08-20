"""Run and analyse leave-one-constraint-out experiments for Module C.

Module C is evaluated with one complete run plus one run for every individual
constraint removed from the generator objective.  The script can launch the
existing training entrypoint (``--run``), or analyse already written
``metrics.csv`` files.  Analysis writes machine-readable summaries and
constraint-specific PNG diagnostics; no metric is filtered or rewritten.

Examples
--------
Analyse a completed ablation directory::

    python scripts/ablate_module_c.py --runs-root runs/module_c_ablation

Launch all variants for all four target domains::

    python scripts/ablate_module_c.py --run \
        --source-root /data/clear --target-parent-root /data \
        --target-domains 黑天 逆光 雾天 雨天 --iterations 3
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import html
import json
import math
import numpy as np
from pathlib import Path
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_dual_d_default.json"
DEFAULT_DUAL_CONFIG = PROJECT_ROOT / "configs" / "dual_d_default_config.json"

# Keep names stable: they are used in output directories, tables and figures.
CONSTRAINTS = OrderedDict(
    [
        ("cycle", "bidirectional cycle consistency"),
        ("identity", "identity preservation"),
        ("paired_contrastive", "class-aware multi-positive contrast"),
        ("prototype_contrastive", "batch class-prototype contrast"),
        ("classification_feedback", "generated-feature classification feedback"),
    ]
)
VARIANTS = ["full", *[f"no_{name}" for name in CONSTRAINTS]]
LOSS_METRIC = {
    "cycle": "train_dual_d_cycle",
    "identity": "train_dual_d_identity",
    "paired_contrastive": "train_dual_d_contrastive",
    "prototype_contrastive": "train_dual_d_prototype_contrastive",
    "classification_feedback": "train_dual_d_classification_feedback",
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


def run_variants(args: argparse.Namespace, variants: Sequence[str]) -> None:
    if not args.source_root:
        raise ValueError("--source-root is required with --run")
    if bool(args.target_root) == bool(args.target_parent_root):
        raise ValueError("Provide exactly one of --target-root or --target-parent-root with --run")

    output_dir = Path(args.output_dir)
    config_paths = write_variant_configs(Path(args.base_dual_config), output_dir, variants)
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
            "--no-save-checkpoints",
            "--save-feature-embeddings",
            "--feature-visualization-samples",
            str(args.feature_visualization_samples),
            "--no-use-ais",
        ]
        if args.target_root:
            command.extend(["--target-root", str(args.target_root)])
        else:
            command.extend(["--target-parent-root", str(args.target_parent_root)])
            command.extend(["--target-domains", *[str(item) for item in args.target_domains]])
        _append_option(command, "--epochs", args.epochs)
        _append_option(command, "--batch-size", args.batch_size)
        _append_option(command, "--num-workers", args.num_workers)
        _append_option(command, "--device", args.device)
        _append_option(command, "--seed", args.seed)
        print("Running Module-C variant:", variant)
        print(" ".join(command))
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


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
    match = re.search(r"_(?P<domain>[^_]+)_iter\d+", name)
    return match.group("domain") if match else "all"


def _best_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any]:
    candidates = [row for row in rows if _to_float(row.get(metric)) is not None]
    if not candidates:
        candidates = [row for row in rows if _to_float(row.get("val_acc")) is not None]
        metric = "val_acc"
    if not candidates:
        raise ValueError("metrics.csv has no validation metric columns")
    return max(candidates, key=lambda row: float(row.get(metric, float("-inf"))))


def _summarize_run(run_dir: Path, monitor_metric: str) -> Dict[str, Any] | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = _read_rows(metrics_path)
    if not rows:
        return None
    best = _best_row(rows, monitor_metric)
    val_acc = _to_float(best.get("val_acc"))
    val_f1 = _to_float(best.get("val_f1_macro_present"))
    train_full = _to_float(best.get("train_full_acc"))
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "run": run_dir.name,
        "variant": _infer_variant(run_dir),
        "domain": _infer_domain(run_dir),
        "epochs": len(rows),
        "best_epoch": _to_float(best.get("epoch")),
        "best_val_acc": val_acc,
        "best_val_f1": val_f1,
        "best_train_full_acc": train_full,
        "generalization_gap": (
            train_full - val_acc if train_full is not None and val_acc is not None else None
        ),
        "feature_embeddings": str(run_dir / "feature_embeddings.npz") if (run_dir / "feature_embeddings.npz").exists() else None,
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
        summary = _summarize_run(run_dir, monitor_metric)
        if summary is not None:
            summaries.append(summary)
    if not summaries:
        raise FileNotFoundError(f"No readable metrics.csv found under {runs_root} with pattern {pattern!r}")
    return summaries


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["variant"]), str(row["domain"]))].append(row)
    keys = ["best_val_acc", "best_val_f1", "best_train_full_acc", "generalization_gap"]
    output: List[Dict[str, Any]] = []
    for (variant, domain), rows in sorted(grouped.items()):
        result: Dict[str, Any] = {"variant": variant, "domain": domain, "runs": len(rows)}
        for key in keys:
            values = [float(row[key]) for row in rows if _to_float(row.get(key)) is not None]
            result[f"{key}_mean"] = mean(values) if values else None
            result[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(result)
    return output


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
    # Explicit CJK font fallback prevents Chinese labels from becoming boxes.
    from matplotlib import font_manager

    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    matplotlib.rcParams["font.sans-serif"] = [selected, *[name for name in candidates if name != selected]]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<style>text{font-family:Arial,sans-serif;fill:#17202a} .grid{stroke:#d7dde3;stroke-width:1} '
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
            body.append(_svg_bar(x - bar_w/2, top + plot_h - bar_height, bar_w * 0.86, bar_height, "full" if variant == "full" else "ablated", f"{variant}, {domain}: {value:.3f}"))
    legend_x = left
    for index, domain in enumerate(domains):
        x = legend_x + index * 130
        body.append(f'<rect x="{x}" y="{height-38}" width="12" height="12" fill="{colors[index % len(colors)]}"/><text x="{x+17}" y="{height-28}" font-size="12">{_svg_escape(domain)}</text>')
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
        body.append(f'<text x="{left + col*cell_w + cell_w/2:.1f}" y="{top-14}" text-anchor="middle" font-size="12">{_svg_escape(domain)}</text>')
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
            body.append(_svg_bar(center+offset-13, panel_bottom-bar_height, 26, bar_height, css_class, f"{label}, {domain}: {value:.3f}"))
        body.append(f'<text x="{center:.1f}" y="{panel_bottom+22}" text-anchor="middle" font-size="12">{_svg_escape(domain)}</text>')
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


def _read_curve_rows(summaries: Sequence[Mapping[str, Any]], variant: str, metric: str) -> Dict[int, List[float]]:
    curves: Dict[int, List[float]] = defaultdict(list)
    for summary in summaries:
        if summary.get("variant") != variant:
            continue
        path = Path(str(summary["run_dir"])) / "metrics.csv"
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
    figure, axis = plt.subplots(figsize=(max(8, 1.4 * len(variants)), 5.2))
    width = 0.8 / max(len(domains), 1)
    x_positions = list(range(len(variants)))
    for domain_index, domain in enumerate(domains):
        values, errors = [], []
        for variant in variants:
            value, error = _mean_std(summaries, variant, domain, "best_val_f1")
            values.append(value if value is not None else float("nan"))
            errors.append(error if error is not None else 0.0)
        offsets = [x + (domain_index - (len(domains) - 1) / 2) * width for x in x_positions]
        axis.bar(offsets, values, width=width * 0.9, yerr=errors, capsize=3, label=domain)
    axis.set_xticks(x_positions, [variant.replace("no_", "- ") for variant in variants], rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Best validation F1 (present classes)")
    axis.set_title("Module C ablation overview")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=min(3, len(domains)), frameon=False)
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
    image = axis.imshow(values, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    axis.set_xticks(range(len(domains)), domains, rotation=25, ha="right")
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
    figure, (performance_axis, curve_axis) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1.35]})
    x_positions = list(range(len(domains)))
    width = 0.35
    for offset, variant, label, color in [(-width / 2, full_variant, "full", "#2f6f9f"), (width / 2, ablation_variant, f"without {constraint}", "#b85c38")]:
        values, errors = [], []
        for domain in domains:
            value, error = _mean_std(summaries, variant, domain, "best_val_f1")
            values.append(value if value is not None else float("nan"))
            errors.append(error if error is not None else 0.0)
        performance_axis.bar([x + offset for x in x_positions], values, width=width, yerr=errors, capsize=3, label=label, color=color)
    performance_axis.set_xticks(x_positions, domains, rotation=20, ha="right")
    performance_axis.set_ylim(0, 1.05)
    performance_axis.set_ylabel("Best val F1")
    performance_axis.set_title(f"{constraint.replace('_', ' ').title()}: performance impact")
    performance_axis.grid(axis="y", alpha=0.25)
    performance_axis.legend(frameon=False)

    metric = LOSS_METRIC[constraint]
    for variant, label, color in [(full_variant, "full", "#2f6f9f"), (ablation_variant, f"without {constraint}", "#b85c38")]:
        curve = _read_curve_rows(summaries, variant, metric)
        if curve:
            epochs = sorted(curve)
            means = [mean(curve[epoch]) for epoch in epochs]
            curve_axis.plot(epochs, means, linewidth=2, label=label, color=color)
    curve_axis.set_xlabel("Epoch")
    curve_axis.set_ylabel(metric.replace("train_dual_d_", "").replace("_", " "))
    curve_axis.set_title(f"Constraint diagnostic: {CONSTRAINTS[constraint]}")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / f"constraint_{constraint}.png", dpi=180)
    plt.close(figure)


def _pca_2d(features: np.ndarray) -> np.ndarray:
    """Project feature rows to two comparable PCA coordinates."""

    if features.ndim != 2 or features.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    centered = features.astype(np.float32) - features.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return np.concatenate([centered, np.zeros_like(centered)], axis=1)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    if components.shape[0] == 1:
        components = np.vstack([components, np.zeros_like(components)])
    return centered @ components.T


def _feature_snapshot(summaries: Sequence[Mapping[str, Any]], variant: str, domain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    raw_parts, translated_parts, label_parts = [], [], []
    for summary in summaries:
        if summary.get("variant") != variant or summary.get("domain") != domain:
            continue
        path = Path(str(summary["run_dir"])) / "feature_embeddings.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            raw_parts.append(np.asarray(data["raw"], dtype=np.float32))
            translated_parts.append(np.asarray(data["source_like"], dtype=np.float32))
            label_parts.append(np.asarray(data["labels"], dtype=np.int64))
    if not label_parts:
        return None
    raw = np.concatenate(raw_parts, axis=0)
    translated = np.concatenate(translated_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    joint = np.concatenate([raw, translated], axis=0)
    coordinates = _pca_2d(joint)
    return coordinates[: len(raw)], coordinates[len(raw):], labels, joint


def _plot_feature_diagnostics(summaries: Sequence[Mapping[str, Any]], output_dir: Path, plt) -> List[str]:
    """Compare full vs leave-one-out feature geometry for every domain/constraint."""

    domains = _domain_order(summaries)
    generated: List[str] = []
    for constraint in CONSTRAINTS:
        ablated_variant = f"no_{constraint}"
        for domain in domains:
            full = _feature_snapshot(summaries, "full", domain)
            ablated = _feature_snapshot(summaries, ablated_variant, domain)
            if full is None or ablated is None:
                continue
            full_raw, full_translated, full_labels, _ = full
            abl_raw, abl_translated, abl_labels, _ = ablated
            # Refit PCA jointly so the two panels share a coordinate system.
            raw_joint = np.concatenate([full_raw, abl_raw], axis=0)
            translated_joint = np.concatenate([full_translated, abl_translated], axis=0)
            raw_coords = _pca_2d(raw_joint)
            translated_coords = _pca_2d(translated_joint)
            figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
            panels = [
                (axes[0, 0], raw_coords[: len(full_labels)], full_labels, "完整模块 C：原始投影"),
                (axes[0, 1], translated_coords[: len(full_labels)], full_labels, "完整模块 C：source-like"),
                (axes[1, 0], raw_coords[len(full_labels):], abl_labels, f"去除 {CONSTRAINTS[constraint]}：原始投影"),
                (axes[1, 1], translated_coords[len(full_labels):], abl_labels, f"去除 {CONSTRAINTS[constraint]}：source-like"),
            ]
            class_ids = sorted(set(full_labels.tolist()) | set(abl_labels.tolist()))
            cmap = plt.get_cmap("tab20")
            for axis, coordinates, labels, title in panels:
                for index, class_id in enumerate(class_ids):
                    mask = labels == class_id
                    if np.any(mask):
                        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=24, alpha=0.78, color=cmap(index % 20), label=f"类 {class_id}")
                axis.set_title(title, fontsize=10)
                axis.set_xlabel("主成分 1")
                axis.set_ylabel("主成分 2")
                axis.grid(alpha=0.2)
            handles, labels = axes[0, 1].get_legend_handles_labels()
            if handles:
                figure.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)), frameon=False)
            figure.suptitle(f"{domain}：模块 C 约束特征可视化", fontsize=14)
            figure.tight_layout(rect=(0, 0.06, 1, 0.94))
            filename = f"feature_{constraint}_{re.sub(r'[^0-9A-Za-z一-龥]+', '_', domain)}.png"
            figure.savefig(output_dir / filename, dpi=180)
            plt.close(figure)
            generated.append(filename)
    return generated


def _write_html_report(output_dir: Path, aggregates: Sequence[Mapping[str, Any]], constraints: Sequence[str]) -> None:
    rows = []
    for row in aggregates:
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(key, '')))}</td>"
                for key in ("variant", "domain", "runs", "best_val_f1_mean", "best_val_f1_std", "generalization_gap_mean")
            )
            + "</tr>"
        )
    image_stems = ["module_c_overview", "module_c_effect_heatmap"] + [f"constraint_{name}" for name in constraints]
    image_stems.extend(path.stem for path in output_dir.glob("feature_*.png"))
    image_tags_parts = []
    for stem in image_stems:
        image = next((candidate for candidate in (f"{stem}.png", f"{stem}.svg") if (output_dir / candidate).exists()), None)
        if image:
            image_tags_parts.append(f'<h2>{html.escape(stem)}</h2><img src="{html.escape(image)}" alt="{html.escape(stem)}">')
    image_tags = "\n".join(image_tags_parts)
    document = """<!doctype html><meta charset="utf-8"><title>Module C ablation report</title>
<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin-bottom:2rem}th,td{border:1px solid #ccd;padding:.4rem;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}</style>
<h1>Module C ablation report</h1><table><thead><tr><th>variant</th><th>domain</th><th>runs</th><th>best F1 mean</th><th>F1 std</th><th>generalization gap</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table>" + image_tags
    (output_dir / "ablation_report.html").write_text(document, encoding="utf-8")


def analyse(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.analysis_output or args.runs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = discover_summaries(Path(args.runs_root), args.analysis_glob, args.monitor_metric)
    aggregates = aggregate_summaries(summaries)
    _write_csv(summaries, output_dir / "ablation_runs.csv")
    _write_csv(aggregates, output_dir / "ablation_summary.csv")
    payload = {
        "monitor_metric": args.monitor_metric,
        "constraints": dict(CONSTRAINTS),
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
    _plot_feature_diagnostics(summaries, output_dir, plt)
    _write_html_report(output_dir, aggregates, available_constraints)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and analyse Module C leave-one-constraint-out experiments.")
    parser.add_argument("--run", action="store_true", help="Launch training variants before analysis.")
    parser.add_argument("--base-train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--base-dual-config", default=str(DEFAULT_DUAL_CONFIG))
    parser.add_argument("--source-root", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--target-parent-root", default="")
    parser.add_argument("--target-domains", nargs="+", default=["黑天", "逆光", "雾天", "雨天"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "module_c_ablation"))
    parser.add_argument("--runs-root", default=str(PROJECT_ROOT / "runs" / "module_c_ablation"))
    parser.add_argument("--analysis-output", default="")
    parser.add_argument("--analysis-glob", default="module_c_*")
    parser.add_argument("--monitor-metric", default="val_f1_macro_present")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--feature-visualization-samples", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variants = list(dict.fromkeys(args.variants))
    if args.run:
        run_variants(args, variants)
        args.runs_root = args.output_dir
        args.analysis_glob = "module_c_*"
        if not args.analysis_output:
            args.analysis_output = args.output_dir
    payload = analyse(args)
    print(f"Analysed {len(payload['runs'])} runs; wrote results to {args.analysis_output or args.runs_root}")


if __name__ == "__main__":
    main()
