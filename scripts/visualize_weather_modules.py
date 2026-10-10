#!/usr/bin/env python3
"""Cache and plot module representations for legacy weather-domain runs.

This script never trains or evaluates Target test. It reads the compact
feature_embeddings.npz snapshots written when validation selects a new best
checkpoint. Feature arrays, fixed t-SNE coordinates, metrics and plot style
are stored separately, so cosmetic changes never require model inference or a
new t-SNE fit.

The drift constraint is a loss rather than a forward layer. Its scientifically
valid view is therefore a matched full-vs-no-drift checkpoint comparison.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


DOMAINS = ("source", "target")
MODALITIES = ("vis", "ir")
GROUPS = tuple(f"{domain}_{modality}" for domain in DOMAINS for modality in MODALITIES)
COLORS = {
    "source_vis": "#1f77b4",
    "source_ir": "#ff7f0e",
    "target_vis": "#2ca02c",
    "target_ir": "#d62728",
}
GROUP_LABELS = {
    "source_vis": "源域-可见光",
    "source_ir": "源域-红外",
    "target_vis": "目标域-可见光",
    "target_ir": "目标域-红外",
}


class Snapshot:
    """Read current VIS/IR names and old satellite-named legacy snapshots."""

    def __init__(self, path: Path):
        self.path = path
        self.data = np.load(path, allow_pickle=True)

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self.data.files)

    def labels(self, domain: str) -> np.ndarray:
        return np.asarray(self.data[f"{domain}_labels"]).reshape(-1).astype(np.int64)

    def ids(self, domain: str) -> np.ndarray:
        key = f"{domain}_sample_ids"
        if key not in self.data:
            return np.asarray([f"{domain}_{i}" for i in range(len(self.labels(domain)))])
        return np.asarray(self.data[key]).reshape(-1).astype(str)

    def modality(self, domain: str, modality: str, stage: str) -> np.ndarray:
        # Before the dataset-aware naming fix, legacy [VIS, IR] arrays were
        # accidentally written as [sar, optical]. New snapshots use vis/ir.
        # This weather-only reader maps that historical order back correctly.
        current = f"{domain}_{modality}_{stage}_alignment"
        legacy_name = "sar" if modality == "vis" else "optical"
        keys = [current, f"{domain}_{legacy_name}_{stage}_alignment"]
        key = next((item for item in keys if item in self.data), None)
        if key is None:
            raise KeyError(f"{self.path} 缺少 {keys}；实际字段={list(self.keys)}")
        value = np.asarray(self.data[key], dtype=np.float32)
        return value.reshape(value.shape[0], -1)

    def translated(self, domain: str) -> Tuple[np.ndarray, np.ndarray]:
        key = "source_target_like" if domain == "source" else "target_source_like"
        if key not in self.data:
            raise KeyError(f"{self.path} 缺少翻译特征 {key}")
        value = np.asarray(self.data[key], dtype=np.float32)
        value = value.reshape(value.shape[0], -1)
        if value.shape[1] % 2:
            raise ValueError(f"{key} 的维度 {value.shape[1]} 不能拆成可见光/红外两块")
        return tuple(np.split(value, 2, axis=1))  # type: ignore[return-value]

    def schema(self) -> Dict[str, Sequence[int]]:
        return {key: list(np.asarray(self.data[key]).shape) for key in self.keys}


def _resolve_snapshot(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(
        path.rglob("feature_embeddings*.npz"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"{path} 下没有 feature_embeddings*.npz")
    return matches[0]


def _balanced_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for class_id in sorted(np.unique(labels).tolist()):
        current = np.flatnonzero(labels == class_id)
        if len(current) > limit:
            current = np.sort(rng.choice(current, limit, replace=False))
        parts.append(current)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _indices_for_ids(snapshot: Snapshot, domain: str, wanted: np.ndarray) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(snapshot.ids(domain).tolist())}
    missing = [sample_id for sample_id in wanted.tolist() if sample_id not in lookup]
    if missing:
        raise ValueError(f"{domain} 缺少 {len(missing)} 个待匹配 sample_id，例如 {missing[:3]}")
    return np.asarray([lookup[item] for item in wanted.tolist()], dtype=np.int64)


def _matched_drift_indices(
    full: Snapshot, no_drift: Snapshot, domain: str, limit: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    full_ids, other_ids = full.ids(domain), set(no_drift.ids(domain).tolist())
    common = np.asarray([i for i, sample_id in enumerate(full_ids) if sample_id in other_ids])
    if not len(common):
        raise ValueError(f"{domain} 的 full/no-drift 快照没有共同 sample_id")
    chosen = common[_balanced_indices(full.labels(domain)[common], limit, seed)]
    other = _indices_for_ids(no_drift, domain, full_ids[chosen])
    if not np.array_equal(full.labels(domain)[chosen], no_drift.labels(domain)[other]):
        raise ValueError(f"{domain} 的同名样本在 full/no-drift 中标签不一致")
    return chosen, other


def _groups(snapshot: Snapshot, stage: str, indices: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        f"{domain}_{modality}": snapshot.modality(domain, modality, stage)[indices[domain]]
        for domain in DOMAINS
        for modality in MODALITIES
    }


def _translated_groups(
    snapshot: Snapshot, indices: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for domain in DOMAINS:
        vis, infrared = snapshot.translated(domain)
        result[f"{domain}_vis"] = vis[indices[domain]]
        result[f"{domain}_ir"] = infrared[indices[domain]]
    return result


def _normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def _centroid_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(_normalize(left).mean(0) - _normalize(right).mean(0)))


def _class_separation(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    features = _normalize(features)
    classes = sorted(np.unique(labels).tolist())
    centroids = np.stack([features[labels == value].mean(0) for value in classes])
    intra = float(np.mean([
        np.linalg.norm(features[labels == value] - centroids[index], axis=1).mean()
        for index, value in enumerate(classes)
    ]))
    if len(classes) > 1:
        distances = pairwise_distances(centroids)
        inter = float(distances[np.triu_indices(len(classes), 1)].mean())
    else:
        inter = 0.0
    return {
        "intra_class_distance": intra,
        "inter_class_centroid_distance": inter,
        "separation_ratio": inter / max(intra, 1e-12),
    }


def _stage_metrics(groups: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "source_target_vis_centroid_distance": _centroid_distance(groups["source_vis"], groups["target_vis"]),
        "source_target_ir_centroid_distance": _centroid_distance(groups["source_ir"], groups["target_ir"]),
        "source_vis_ir_centroid_distance": _centroid_distance(groups["source_vis"], groups["source_ir"]),
        "target_vis_ir_centroid_distance": _centroid_distance(groups["target_vis"], groups["target_ir"]),
    }
    fused_by_domain = {}
    for domain in DOMAINS:
        vis = _normalize(groups[f"{domain}_vis"])
        infrared = _normalize(groups[f"{domain}_ir"])
        cosine = np.sum(vis * infrared, axis=1)
        result[f"{domain}_modality_relation"] = {
            "paired_cosine_mean": float(cosine.mean()),
            "paired_cosine_distance": float((1.0 - cosine).mean()),
        }
        fused = np.concatenate([vis, infrared], axis=1)
        fused_by_domain[domain] = _normalize(fused)
        result[f"{domain}_class_separability"] = _class_separation(fused, labels[domain])
    common_classes = sorted(
        set(np.unique(labels["source"]).tolist())
        & set(np.unique(labels["target"]).tolist())
    )
    per_class = {}
    for class_id in common_classes:
        source_center = fused_by_domain["source"][labels["source"] == class_id].mean(0)
        target_center = fused_by_domain["target"][labels["target"] == class_id].mean(0)
        per_class[str(class_id)] = float(np.linalg.norm(source_center - target_center))
    result["class_conditional_source_target_distance"] = {
        "per_class": per_class,
        "mean": float(np.mean(list(per_class.values()))) if per_class else None,
    }
    return result


def _numeric_delta(right: object, left: object) -> object:
    if isinstance(right, dict) and isinstance(left, dict):
        return {
            key: _numeric_delta(right[key], left[key])
            for key in sorted(right.keys() & left.keys())
        }
    if (
        isinstance(right, (int, float))
        and isinstance(left, (int, float))
        and not isinstance(right, bool)
        and not isinstance(left, bool)
    ):
        return float(right) - float(left)
    return None


def _tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    features = _normalize(features)
    if features.shape[1] > 50:
        features = PCA(
            n_components=min(50, features.shape[0] - 1, features.shape[1]),
            random_state=seed,
        ).fit_transform(features)
    kwargs = {
        "n_components": 2,
        "perplexity": min(float(perplexity), max(2.0, (len(features) - 1) / 3.0)),
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 1500
    else:
        kwargs["n_iter"] = 1500
    return TSNE(**kwargs).fit_transform(features).astype(np.float32)


def _build_comparisons(
    full: Snapshot, no_drift: Snapshot, limit: int, seed: int
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    base_indices = {
        domain: _balanced_indices(full.labels(domain), limit, seed + index)
        for index, domain in enumerate(DOMAINS)
    }
    labels = {domain: full.labels(domain)[base_indices[domain]] for domain in DOMAINS}
    ids = {domain: full.ids(domain)[base_indices[domain]] for domain in DOMAINS}
    tal_before = _groups(full, "pre", base_indices)
    tal_after = _groups(full, "post", base_indices)
    translated = _translated_groups(full, base_indices)

    drift_full_indices: Dict[str, np.ndarray] = {}
    drift_other_indices: Dict[str, np.ndarray] = {}
    for index, domain in enumerate(DOMAINS):
        drift_full_indices[domain], drift_other_indices[domain] = _matched_drift_indices(
            full, no_drift, domain, limit, seed + 100 + index
        )
    drift_labels = {
        domain: full.labels(domain)[drift_full_indices[domain]] for domain in DOMAINS
    }
    drift_ids = {
        domain: full.ids(domain)[drift_full_indices[domain]] for domain in DOMAINS
    }

    comparisons: Dict[str, Dict[str, object]] = {
        "tal": {
            "left_title": "TAL前：编码器输出",
            "right_title": "TAL后：对齐输出",
            "left": tal_before,
            "right": tal_after,
            "labels": labels,
            "ids": ids,
        },
        "dual_translation": {
            "left_title": "双向翻译前：TAL输出",
            "right_title": "双向翻译后",
            "left": tal_after,
            "right": translated,
            "labels": labels,
            "ids": ids,
        },
        "modality_drift": {
            "left_title": "无模态漂移约束",
            "right_title": "有模态漂移约束",
            "left": _translated_groups(no_drift, drift_other_indices),
            "right": _translated_groups(full, drift_full_indices),
            "labels": drift_labels,
            "ids": drift_ids,
        },
    }
    schema = {
        "full": full.schema(),
        "no_modality_drift": no_drift.schema(),
    }
    return comparisons, schema


def build_cache(
    full_path: Path,
    no_drift_path: Path,
    output_dir: Path,
    max_per_class: int,
    seed: int,
    perplexity: float,
    force: bool = False,
) -> None:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = (
        cache_dir / "features.npz",
        cache_dir / "tsne_coordinates.npz",
        cache_dir / "metadata.json",
        cache_dir / "high_dimensional_metrics.json",
    )
    if all(path.is_file() for path in required) and not force:
        print("缓存已经存在；跳过特征选择和 t-SNE。使用 --force-cache 可重建。")
        return

    full, no_drift = Snapshot(full_path), Snapshot(no_drift_path)
    comparisons, schema = _build_comparisons(full, no_drift, max_per_class, seed)
    features: Dict[str, np.ndarray] = {}
    coordinates: Dict[str, np.ndarray] = {}
    metrics: Dict[str, object] = {}
    metadata: Dict[str, object] = {
        "protocol": {
            "target_test_used": False,
            "checkpoint_source": "validation-selected feature_embeddings.npz",
            "ais_used": False,
            "tal": "encoder output versus per-modality TAL projection",
            "dual_translation": "TAL output versus translated output",
            "modality_drift": "matched no-drift versus full checkpoints",
            "tsne_panel_policy": (
                "left/right fitted independently with fixed settings; "
                "quantitative comparisons use high-dimensional metrics"
            ),
        },
        "snapshot_paths": {
            "full": str(full_path.resolve()),
            "no_modality_drift": str(no_drift_path.resolve()),
        },
        "snapshot_schema": schema,
        "comparisons": {},
    }
    sample_rows = []

    for comparison_index, (name, item) in enumerate(comparisons.items()):
        left = item["left"]
        right = item["right"]
        labels = item["labels"]
        metadata["comparisons"][name] = {
            "left_title": item["left_title"],
            "right_title": item["right_title"],
            "samples": {domain: int(len(labels[domain])) for domain in DOMAINS},
        }
        left_metrics = _stage_metrics(left, labels)
        right_metrics = _stage_metrics(right, labels)
        metrics[name] = {
            "left": left_metrics,
            "right": right_metrics,
            "right_minus_left": _numeric_delta(right_metrics, left_metrics),
        }

        # Fit each stage independently. TAL changes dimensionality (encoder
        # feature_dim -> projection_dim), so a joint t-SNE would manufacture a
        # common coordinate space that the network does not actually define.
        for side_name, groups in (("left", left), ("right", right)):
            ordered = []
            spans = []
            cursor = 0
            for group in GROUPS:
                value = np.asarray(groups[group], dtype=np.float32)
                key = f"{name}__{side_name}__{group}"
                features[key] = value
                ordered.append(value)
                spans.append((key, cursor, cursor + len(value)))
                cursor += len(value)
            dimensions = {value.shape[1] for value in ordered}
            if len(dimensions) != 1:
                raise ValueError(f"{name}/{side_name} 的四组特征维度不一致：{sorted(dimensions)}")
            embedding = _tsne(
                np.concatenate(ordered),
                seed + comparison_index * 10 + (0 if side_name == "left" else 1),
                perplexity,
            )
            for key, start, end in spans:
                coordinates[key] = embedding[start:end]

        # Class view: concatenate the paired modalities in high-dimensional
        # space first. Each stage again gets its own map; quantitative change
        # is reported by high_dimensional_metrics.json.
        for side_name, groups in (("left", left), ("right", right)):
            class_arrays = []
            class_spans = []
            cursor = 0
            for domain in DOMAINS:
                value = np.concatenate(
                    [groups[f"{domain}_vis"], groups[f"{domain}_ir"]], axis=1
                ).astype(np.float32)
                key = f"{name}__{side_name}__class__{domain}"
                features[key] = value
                class_arrays.append(value)
                class_spans.append((key, cursor, cursor + len(value)))
                cursor += len(value)
            class_embedding = _tsne(
                np.concatenate(class_arrays),
                seed + 1000 + comparison_index * 10 + (0 if side_name == "left" else 1),
                perplexity,
            )
            for key, start, end in class_spans:
                coordinates[key] = class_embedding[start:end]

        for domain in DOMAINS:
            label_key = f"{name}__labels__{domain}"
            features[label_key] = labels[domain]
            for sample_id, class_id in zip(item["ids"][domain], labels[domain]):
                sample_rows.append({
                    "comparison": name,
                    "domain": domain,
                    "sample_id": str(sample_id),
                    "class_id": int(class_id),
                })

    np.savez_compressed(cache_dir / "features.npz", **features)
    np.savez_compressed(cache_dir / "tsne_coordinates.npz", **coordinates)
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cache_dir / "high_dimensional_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (cache_dir / "samples.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("comparison", "domain", "sample_id", "class_id")
        )
        writer.writeheader()
        writer.writerows(sample_rows)


def _default_style() -> Dict[str, object]:
    return {
        "font_path": "",
        "font_candidates": [
            "SimSun",
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Droid Sans Fallback",
        ],
        "figure_size": [14.0, 6.2],
        "dpi": 220,
        "title_fontsize": 15,
        "axis_fontsize": 11,
        "legend_fontsize": 9,
        "point_size": 10,
        "point_alpha": 0.72,
        "legend_location": "best",
        "class_legend_columns": 2,
        "colors": COLORS,
        "group_labels": GROUP_LABELS,
        "class_cmap": "tab20",
    }


def _ensure_style(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_default_style(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _configure_chinese_font(style: Mapping[str, object]) -> str:
    explicit = str(style.get("font_path", "")).strip()
    if explicit:
        font_path = Path(explicit).expanduser()
        if not font_path.is_file():
            raise FileNotFoundError(f"指定的中文字体不存在：{font_path}")
        fm.fontManager.addfont(str(font_path))
        selected = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        installed = {font.name for font in fm.fontManager.ttflist}
        selected = next(
            (str(name) for name in style["font_candidates"] if str(name) in installed),
            "DejaVu Sans",
        )
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def _decorate_axis(axis, title: str, style: Mapping[str, object]) -> None:
    axis.set_title(title, fontsize=float(style["title_fontsize"]))
    axis.set_xlabel("t-SNE 维度1", fontsize=float(style["axis_fontsize"]))
    axis.set_ylabel("t-SNE 维度2", fontsize=float(style["axis_fontsize"]))
    axis.grid(alpha=0.12, linewidth=0.5)


def _plot_domain_modality(
    name: str,
    item: Mapping[str, object],
    coordinates: Mapping[str, np.ndarray],
    metrics: Mapping[str, object],
    style: Mapping[str, object],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=tuple(style["figure_size"]), constrained_layout=True
    )
    colors = style["colors"]
    labels = style["group_labels"]
    for axis, side, title in zip(
        axes,
        ("left", "right"),
        (str(item["left_title"]), str(item["right_title"])),
    ):
        for group in GROUPS:
            xy = coordinates[f"{name}__{side}__{group}"]
            axis.scatter(
                xy[:, 0],
                xy[:, 1],
                s=float(style["point_size"]),
                alpha=float(style["point_alpha"]),
                color=str(colors[group]),
                label=str(labels[group]),
                edgecolors="none",
            )
        side_metrics = metrics[side]
        metric_text = (
            f"VIS跨域距离={side_metrics['source_target_vis_centroid_distance']:.3f}  "
            f"IR跨域距离={side_metrics['source_target_ir_centroid_distance']:.3f}"
        )
        _decorate_axis(axis, f"{title}\n{metric_text}", style)
        axis.legend(
            loc=str(style["legend_location"]),
            fontsize=float(style["legend_fontsize"]),
            frameon=True,
        )
    figure.savefig(output, dpi=int(style["dpi"]), bbox_inches="tight")
    plt.close(figure)


def _plot_classes(
    name: str,
    item: Mapping[str, object],
    coordinates: Mapping[str, np.ndarray],
    features: Mapping[str, np.ndarray],
    metrics: Mapping[str, object],
    style: Mapping[str, object],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=tuple(style["figure_size"]), constrained_layout=True
    )
    cmap = plt.get_cmap(str(style["class_cmap"]))
    markers = {"source": "o", "target": "^"}
    for axis, side, title in zip(
        axes,
        ("left", "right"),
        (str(item["left_title"]), str(item["right_title"])),
    ):
        for domain in DOMAINS:
            xy = coordinates[f"{name}__{side}__class__{domain}"]
            labels = features[f"{name}__labels__{domain}"].astype(np.int64)
            for class_id in sorted(np.unique(labels).tolist()):
                selected = labels == class_id
                axis.scatter(
                    xy[selected, 0],
                    xy[selected, 1],
                    s=float(style["point_size"]),
                    alpha=float(style["point_alpha"]),
                    color=cmap(class_id % cmap.N),
                    marker=markers[domain],
                    label=f"{'源域' if domain == 'source' else '目标域'}-类别{class_id}",
                    edgecolors="none",
                )
        side_metrics = metrics[side]
        conditional = side_metrics["class_conditional_source_target_distance"]["mean"]
        conditional_text = "N/A" if conditional is None else f"{conditional:.3f}"
        source_ratio = side_metrics["source_class_separability"]["separation_ratio"]
        target_ratio = side_metrics["target_class_separability"]["separation_ratio"]
        metric_text = (
            f"类别条件跨域距离={conditional_text}  "
            f"类间/类内比(源/目标)={source_ratio:.2f}/{target_ratio:.2f}"
        )
        _decorate_axis(axis, f"{title}\n{metric_text}", style)
        handles, legend_labels = axis.get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        axis.legend(
            unique.values(),
            unique.keys(),
            loc=str(style["legend_location"]),
            fontsize=float(style["legend_fontsize"]),
            ncol=int(style["class_legend_columns"]),
            frameon=True,
        )
    figure.savefig(output, dpi=int(style["dpi"]), bbox_inches="tight")
    plt.close(figure)


def render(output_dir: Path, style_path: Optional[Path] = None) -> str:
    style_path = style_path or output_dir / "plot_style.json"
    _ensure_style(style_path)
    style = json.loads(style_path.read_text(encoding="utf-8"))
    selected_font = _configure_chinese_font(style)
    cache_dir = output_dir / "cache"
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    metric_data = json.loads(
        (cache_dir / "high_dimensional_metrics.json").read_text(encoding="utf-8")
    )
    feature_archive = np.load(cache_dir / "features.npz", allow_pickle=False)
    coordinate_archive = np.load(cache_dir / "tsne_coordinates.npz", allow_pickle=False)
    features = {key: feature_archive[key] for key in feature_archive.files}
    coordinates = {key: coordinate_archive[key] for key in coordinate_archive.files}
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, item in metadata["comparisons"].items():
        _plot_domain_modality(
            name,
            item,
            coordinates,
            metric_data[name],
            style,
            figure_dir / f"{name}_领域模态特征TSNE.png",
        )
        _plot_classes(
            name,
            item,
            coordinates,
            features,
            metric_data[name],
            style,
            figure_dir / f"{name}_类别特征TSNE.png",
        )
    return selected_font


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path)
    parser.add_argument("--no-drift-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--style", type=Path)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="只读取已有固定坐标重画；不会重新提取特征或运行 t-SNE",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查依赖和中文字体，不读取实验结果",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_only:
        style = _default_style()
        if args.style:
            _ensure_style(args.style)
            style = json.loads(args.style.read_text(encoding="utf-8"))
        selected = _configure_chinese_font(style)
        print(f"可视化依赖和中文字体检查通过：{selected}")
        return
    if args.output_dir is None:
        raise ValueError("必须提供 --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_style(args.style or args.output_dir / "plot_style.json")
    if not args.render_only:
        if args.full_run is None or args.no_drift_run is None:
            raise ValueError("构建缓存时必须同时提供 --full-run 和 --no-drift-run")
        build_cache(
            _resolve_snapshot(args.full_run),
            _resolve_snapshot(args.no_drift_run),
            args.output_dir,
            args.max_per_class,
            args.seed,
            args.perplexity,
            force=args.force_cache,
        )
    selected = render(args.output_dir, args.style)
    print(json.dumps({
        "status": "ok",
        "font": selected,
        "output_dir": str(args.output_dir.resolve()),
        "target_test_used": False,
        "cosmetic_redraw": (
            f"python scripts/visualize_weather_modules.py --render-only "
            f"--output-dir {args.output_dir}"
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
