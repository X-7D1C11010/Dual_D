"""Generate Chinese M4-SAR ablation tables and mechanism visualizations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

# Required Chinese-safe matplotlib defaults.
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
if "SimSun" in _available_fonts:
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Microsoft YaHei"]
elif "Microsoft YaHei" in _available_fonts:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]


_CHINESE_GLYPH_PROBE = "张量对齐模态类别漂移约束特征可视化"


def _supports_chinese_glyphs(path: Path) -> bool:
    """Return whether a font really contains the Chinese glyphs used by plots."""

    try:
        charmap = FT2Font(str(path)).get_charmap()
    except (OSError, RuntimeError, ValueError):
        return False
    return all(ord(character) in charmap for character in _CHINESE_GLYPH_PROBE)


def _configure_chinese_font(font_path: str = "") -> str:
    """Select a real CJK font by glyph coverage, preferring SimSun."""

    candidates = []
    if font_path:
        explicit = Path(font_path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"指定的中文字体文件不存在：{explicit}")
        candidates.append((explicit, None))
    candidates.extend(
        [
            (Path.home() / ".fonts" / "simsun.ttc", None),
            (Path.home() / ".local" / "share" / "fonts" / "simsun.ttc", None),
            (Path("/usr/share/fonts/truetype/msttcorefonts/simsun.ttf"), None),
        ]
    )
    preferred_families = (
        "SimSun",
        "宋体",
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
    )
    installed = list(font_manager.fontManager.ttflist)
    installed.sort(
        key=lambda entry: (
            preferred_families.index(entry.name)
            if entry.name in preferred_families
            else len(preferred_families),
            entry.name,
        )
    )
    # Retain FontManager's family name. A TTC collection may contain JP/SC/TC
    # faces at the same path; reconstructing a family only from the path often
    # selects the first (usually JP) face even when an SC face is installed.
    candidates.extend((Path(entry.fname), entry.name) for entry in installed)
    candidates.extend(
        [
            (Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"), None),
            (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), None),
            (Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"), None),
            (Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"), None),
        ]
    )
    # findSystemFonts also sees newly installed fonts when Matplotlib's cache is stale.
    candidates.extend((Path(path), None) for path in font_manager.findSystemFonts())

    visited = set()
    for candidate, family_hint in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        identity = (resolved, family_hint)
        if identity in visited or not resolved.is_file():
            continue
        visited.add(identity)
        if not _supports_chinese_glyphs(resolved):
            continue
        font_manager.fontManager.addfont(str(resolved))
        family = family_hint or font_manager.FontProperties(fname=str(resolved)).get_name()
        plt.rcParams["font.sans-serif"] = [family, "SimSun", "SimHei"]
        return f"{family} ({resolved})"
    raise RuntimeError(
        "服务器上未找到实际包含中文字形的字体。请安装 SimSun/SimHei/"
        "Noto CJK/文泉驿，或使用 --font-path 指定宋体 ttf/ttc 文件。"
    )


VARIANT_NAMES = {
    "full": "完整 Dual-D",
    "no_tal": "无张量对齐",
    "no_translation_stack": "无双向翻译模块",
    "no_module_c": "无类别感知反馈",
    "no_modality_drift": "无模态关系漂移约束",
}
CLASS_NAMES = ["桥梁", "港口", "油罐", "操场", "机场", "风力涡轮机"]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _discover_runs(experiment_dir: Path):
    runs = defaultdict(list)
    for path in sorted(experiment_dir.glob("*_seed*/result_summary.json")):
        name = path.parent.name
        variant, seed = name.rsplit("_seed", 1)
        runs[variant].append((int(seed), path.parent, _load_json(path)))
    return runs


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_overall_metrics(runs, output_dir: Path) -> None:
    variants = [name for name in VARIANT_NAMES if name in runs]
    metrics = [
        ("accuracy", "准确率"),
        ("precision_macro_present", "宏平均精确率"),
        ("recall_macro_present", "宏平均召回率"),
        ("f1_macro_present", "宏平均 F1"),
    ]
    x = np.arange(len(variants), dtype=np.float32)
    width = 0.19
    figure, axis = plt.subplots(figsize=(13, 6.5))
    for index, (key, label) in enumerate(metrics):
        means, errors = [], []
        for variant in variants:
            values = [
                float(item[2]["target_test"][key])
                for item in runs[variant]
                if item[2].get("target_test")
            ]
            means.append(float(np.mean(values)) if values else np.nan)
            errors.append(float(np.std(values)) if len(values) > 1 else 0.0)
        axis.bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=label,
        )
    axis.set_xticks(x, [VARIANT_NAMES[name] for name in variants], rotation=10)
    axis.set_ylabel("评价指标")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("M4-SAR 完整模型与消融实验结果（均值±标准差）")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    _save(figure, output_dir / "总体消融结果.png")


def plot_per_class_f1(runs, output_dir: Path) -> None:
    variants = [name for name in VARIANT_NAMES if name in runs]
    x = np.arange(len(CLASS_NAMES), dtype=np.float32)
    width = 0.8 / max(len(variants), 1)
    figure, axis = plt.subplots(figsize=(14, 6.5))
    for index, variant in enumerate(variants):
        values = [
            np.asarray(item[2]["target_test"]["per_class_f1"], dtype=np.float64)
            for item in runs[variant]
            if item[2].get("target_test")
        ]
        if not values:
            continue
        axis.bar(
            x + (index - (len(variants) - 1) / 2) * width,
            np.mean(values, axis=0),
            width,
            label=VARIANT_NAMES[variant],
        )
    axis.set_xticks(x, CLASS_NAMES)
    axis.set_ylabel("F1")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("各类别 F1 对比")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    _save(figure, output_dir / "各类别F1对比.png")


def _tsne(values: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    values = np.asarray(values, dtype=np.float32)
    if values.shape[1] > 50:
        values = PCA(n_components=50, random_state=seed).fit_transform(values)
    perplexity = min(30.0, max(5.0, (len(values) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(values)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def _fused_alignment_features(snapshot, domain: str, stage: str) -> np.ndarray:
    """Return the SAR/optical representation on one side of TAL."""

    return np.concatenate(
        [
            np.asarray(snapshot[f"{domain}_sar_{stage}"], dtype=np.float32),
            np.asarray(snapshot[f"{domain}_optical_{stage}"], dtype=np.float32),
        ],
        axis=1,
    )


def _class_conditional_centroid_distance(
    source: np.ndarray,
    target: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    """Mean cosine distance between same-class Source/Target prototypes."""

    distances = []
    for class_id in range(len(CLASS_NAMES)):
        source_mask = np.asarray(source_labels) == class_id
        target_mask = np.asarray(target_labels) == class_id
        if not np.any(source_mask) or not np.any(target_mask):
            continue
        source_center = np.mean(source[source_mask], axis=0)
        target_center = np.mean(target[target_mask], axis=0)
        similarity = float(
            np.dot(source_center, target_center)
            / (
                np.linalg.norm(source_center)
                * np.linalg.norm(target_center)
                + 1e-8
            )
        )
        distances.append(1.0 - similarity)
    return float(np.mean(distances)) if distances else float("nan")


def _class_conditional_domain_neighbor_purity(
    source: np.ndarray,
    target: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    neighbors: int = 10,
) -> float:
    """Measure local domain separability while controlling for class.

    A value near 0.5 means that a balanced Source/Target sample has equally
    many neighbours from either domain. Larger values mean stronger residual
    domain separation. The calculation is performed in the original feature
    space and therefore does not depend on a t-SNE layout.
    """

    purities = []
    for class_id in range(len(CLASS_NAMES)):
        source_class = source[np.asarray(source_labels) == class_id]
        target_class = target[np.asarray(target_labels) == class_id]
        sample_count = min(len(source_class), len(target_class))
        if sample_count < 2:
            continue
        values = _l2_normalize(
            np.concatenate(
                [source_class[:sample_count], target_class[:sample_count]], axis=0
            )
        )
        domains = np.concatenate(
            [np.zeros(sample_count, dtype=np.int64), np.ones(sample_count, dtype=np.int64)]
        )
        similarity = np.matmul(values, values.T)
        np.fill_diagonal(similarity, -np.inf)
        resolved_neighbors = min(max(int(neighbors), 1), len(values) - 1)
        indices = np.argpartition(
            similarity,
            kth=len(values) - resolved_neighbors,
            axis=1,
        )[:, -resolved_neighbors:]
        purity = np.mean(domains[indices] == domains[:, None], axis=1)
        purities.extend(purity.tolist())
    return float(np.mean(purities)) if purities else float("nan")


def _tensor_prototype_score(snapshot, stage: str) -> float:
    """Reproduce TAL's factorized class-prototype score on saved features."""

    source_labels = np.asarray(snapshot["source_labels"])
    target_labels = np.asarray(snapshot["target_labels"])
    class_scores = []
    for class_id in range(len(CLASS_NAMES)):
        modality_scores = []
        for modality in ("sar", "optical"):
            source = np.asarray(
                snapshot[f"source_{modality}_{stage}"], dtype=np.float32
            )
            target = np.asarray(
                snapshot[f"target_{modality}_{stage}"], dtype=np.float32
            )
            source_center = np.mean(source[source_labels == class_id], axis=0)
            target_center = np.mean(target[target_labels == class_id], axis=0)
            modality_scores.append(
                float(
                    np.dot(source_center, target_center)
                    / (
                        np.linalg.norm(source_center)
                        * np.linalg.norm(target_center)
                        + 1e-8
                    )
                )
            )
        class_scores.append(
            0.5 * (float(np.prod(modality_scores)) + float(np.mean(modality_scores)))
        )
    return float(np.mean(class_scores))


def _balanced_plot_indices(labels: np.ndarray, maximum: int = 600) -> np.ndarray:
    """Select a deterministic class-balanced subset for legible t-SNE plots."""

    labels = np.asarray(labels)
    per_class = max(int(maximum) // len(CLASS_NAMES), 1)
    selected = []
    for class_id in range(len(CLASS_NAMES)):
        selected.extend(np.flatnonzero(labels == class_id)[:per_class].tolist())
    return np.asarray(selected, dtype=np.int64)


def _class_separability_metrics(
    source: np.ndarray,
    target: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
):
    """Measure class compactness/separation in the original feature space."""

    values = _l2_normalize(np.concatenate([source, target], axis=0))
    labels = np.concatenate([source_labels, target_labels], axis=0)
    similarity = np.matmul(values, values.T)
    same_class = labels[:, None] == labels[None, :]
    diagonal = np.eye(len(labels), dtype=bool)
    same_class &= ~diagonal
    different_class = ~(same_class | diagonal)
    intra_class = float(np.mean(similarity[same_class]))
    inter_class = float(np.mean(similarity[different_class]))
    return {
        "intra_class_cosine_similarity": intra_class,
        "inter_class_cosine_similarity": inter_class,
        "class_separation_gap": intra_class - inter_class,
    }


def _tal_high_dimensional_metrics(snapshot):
    source_labels = np.asarray(snapshot["source_labels"])
    target_labels = np.asarray(snapshot["target_labels"])
    metrics = {}
    for stage in ("pre_alignment", "post_alignment"):
        stage_metrics = {"modalities": {}}
        for modality in ("sar", "optical"):
            source = np.asarray(
                snapshot[f"source_{modality}_{stage}"], dtype=np.float32
            )
            target = np.asarray(
                snapshot[f"target_{modality}_{stage}"], dtype=np.float32
            )
            stage_metrics["modalities"][modality] = {
                "class_conditional_source_target_centroid_distance": (
                    _class_conditional_centroid_distance(
                        source,
                        target,
                        source_labels,
                        target_labels,
                    )
                ),
                "same_domain_neighbor_ratio": (
                    _class_conditional_domain_neighbor_purity(
                        source,
                        target,
                        source_labels,
                        target_labels,
                    )
                ),
            }
        source_fused = _fused_alignment_features(snapshot, "source", stage)
        target_fused = _fused_alignment_features(snapshot, "target", stage)
        stage_metrics["fused"] = {
            "class_conditional_source_target_centroid_distance": (
                _class_conditional_centroid_distance(
                    source_fused,
                    target_fused,
                    source_labels,
                    target_labels,
                )
            ),
            "same_domain_neighbor_ratio": _class_conditional_domain_neighbor_purity(
                source_fused,
                target_fused,
                source_labels,
                target_labels,
            ),
            **_class_separability_metrics(
                source_fused,
                target_fused,
                source_labels,
                target_labels,
            ),
        }
        stage_metrics["tensor_prototype_score"] = _tensor_prototype_score(
            snapshot, stage
        )
        metrics[stage] = stage_metrics
    return metrics


def _four_group_modality_domain_scatter(
    axis,
    snapshot,
    stage: str,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    title: str,
):
    groups = [
        (
            "源域光学",
            _l2_normalize(
                np.asarray(
                    snapshot[f"source_optical_{stage}"], dtype=np.float32
                )[source_indices]
            ),
            "#e6862a",
            "o",
            0.55,
        ),
        (
            "源域 SAR",
            _l2_normalize(
                np.asarray(snapshot[f"source_sar_{stage}"], dtype=np.float32)[
                    source_indices
                ]
            ),
            "#2f6f9f",
            "o",
            0.55,
        ),
        (
            "目标域光学",
            _l2_normalize(
                np.asarray(
                    snapshot[f"target_optical_{stage}"], dtype=np.float32
                )[target_indices]
            ),
            "#c44e52",
            "x",
            0.72,
        ),
        (
            "目标域 SAR",
            _l2_normalize(
                np.asarray(snapshot[f"target_sar_{stage}"], dtype=np.float32)[
                    target_indices
                ]
            ),
            "#55a868",
            "x",
            0.72,
        ),
    ]
    counts = [len(values) for _, values, *_ in groups]
    embedding = _tsne(np.concatenate([values for _, values, *_ in groups], axis=0))
    start = 0
    for (label, _values, color, marker, alpha), count in zip(groups, counts):
        stop = start + count
        axis.scatter(
            embedding[start:stop, 0],
            embedding[start:stop, 1],
            s=12,
            alpha=alpha,
            c=color,
            marker=marker,
            label=label,
        )
        start = stop
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def _class_conditional_fused_scatter(
    axis,
    snapshot,
    stage: str,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    title: str,
):
    source = _l2_normalize(
        _fused_alignment_features(snapshot, "source", stage)[source_indices]
    )
    target = _l2_normalize(
        _fused_alignment_features(snapshot, "target", stage)[target_indices]
    )
    source_labels = np.asarray(snapshot["source_labels"])[source_indices]
    target_labels = np.asarray(snapshot["target_labels"])[target_indices]
    _domain_class_scatter(
        axis,
        source,
        target,
        source_labels,
        target_labels,
        title,
    )


def _domain_scatter(axis, source, target, title, diagnostics):
    values = np.concatenate([source, target], axis=0)
    embedding = _tsne(values)
    source_count = len(source)
    axis.scatter(
        embedding[:source_count, 0],
        embedding[:source_count, 1],
        s=11,
        alpha=0.55,
        c="#2f6f9f",
        marker="o",
        label="源域",
    )
    axis.scatter(
        embedding[source_count:, 0],
        embedding[source_count:, 1],
        s=13,
        alpha=0.65,
        c="#d95f8d",
        marker="x",
        label="目标域",
    )
    axis.set_title(title)
    axis.text(
        0.02,
        0.02,
        (
            f"类别条件质心距离={diagnostics['centroid_distance']:.4f}\n"
            f"同域近邻比例={diagnostics['neighbor_purity']:.3f}（0.5 越混合）"
        ),
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _domain_class_scatter(axis, source, target, source_labels, target_labels, title):
    values = np.concatenate([source, target], axis=0)
    embedding = _tsne(values)
    source_count = len(source)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    for class_id, class_name in enumerate(CLASS_NAMES):
        source_mask = np.asarray(source_labels) == class_id
        target_mask = np.asarray(target_labels) == class_id
        axis.scatter(
            embedding[:source_count][source_mask, 0],
            embedding[:source_count][source_mask, 1],
            s=10,
            alpha=0.55,
            c=[colors[class_id]],
            marker="o",
            label=f"{class_name}-源域",
        )
        axis.scatter(
            embedding[source_count:][target_mask, 0],
            embedding[source_count:][target_mask, 1],
            s=12,
            alpha=0.7,
            c=[colors[class_id]],
            marker="x",
            label=f"{class_name}-目标域",
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_tal_tsne(runs, output_dir: Path) -> None:
    if "full" not in runs:
        return
    snapshot_path = runs["full"][0][1] / "feature_embeddings.npz"
    if not snapshot_path.is_file():
        return
    snapshot = np.load(snapshot_path)
    required = [
        "source_sar_pre_alignment", "target_sar_pre_alignment",
        "source_sar_post_alignment", "target_sar_post_alignment",
        "source_optical_pre_alignment", "target_optical_pre_alignment",
        "source_optical_post_alignment", "target_optical_post_alignment",
    ]
    if any(key not in snapshot for key in required):
        return
    source_labels = np.asarray(snapshot["source_labels"])
    target_labels = np.asarray(snapshot["target_labels"])
    stage_data = {}
    for stage in ("pre_alignment", "post_alignment"):
        source = _fused_alignment_features(snapshot, "source", stage)
        target = _fused_alignment_features(snapshot, "target", stage)
        stage_data[stage] = {
            "source": source,
            "target": target,
            "centroid_distance": _class_conditional_centroid_distance(
                source,
                target,
                source_labels,
                target_labels,
            ),
            "neighbor_purity": _class_conditional_domain_neighbor_purity(
                source,
                target,
                source_labels,
                target_labels,
            ),
            "tensor_score": _tensor_prototype_score(snapshot, stage),
        }

    tal_metrics = _tal_high_dimensional_metrics(snapshot)
    source_indices = _balanced_plot_indices(source_labels)
    target_indices = _balanced_plot_indices(target_labels)

    # Requested four-group view: both modalities and both domains share one
    # t-SNE fit within each stage. Every modality-domain combination has a
    # distinct colour; marker shape redundantly encodes domain. Pre/post remain
    # separate because TAL changes dimensionality.
    modality_domain_figure, modality_domain_axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )
    for axis, stage, title in (
        (
            modality_domain_axes[0],
            "pre_alignment",
            "TAL 前：编码器输出的四组模态—领域特征",
        ),
        (
            modality_domain_axes[1],
            "post_alignment",
            "TAL 后：张量投影后的四组模态—领域特征",
        ),
    ):
        _four_group_modality_domain_scatter(
            axis,
            snapshot,
            stage,
            source_indices,
            target_indices,
            title,
        )
        sar_metrics = tal_metrics[stage]["modalities"]["sar"]
        optical_metrics = tal_metrics[stage]["modalities"]["optical"]
        axis.text(
            0.02,
            0.02,
            (
                "同类跨域质心距离（越小越对齐）\n"
                f"SAR={sar_metrics['class_conditional_source_target_centroid_distance']:.4f}，"
                f"光学={optical_metrics['class_conditional_source_target_centroid_distance']:.4f}"
            ),
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8"},
        )
    handles, labels = modality_domain_axes[0].get_legend_handles_labels()
    modality_domain_figure.legend(handles, labels, loc="lower center", ncol=4)
    modality_domain_figure.suptitle(
        "TAL 前后四组模态—领域特征分布（四组独立颜色，各面板独立 t-SNE）",
        fontsize=16,
    )
    modality_domain_figure.subplots_adjust(bottom=0.13)
    _save(
        modality_domain_figure,
        output_dir / "TAL前后四组模态领域TSNE.png",
    )

    # Requested class-conditioned view: colour encodes semantic class and the
    # marker encodes domain. Same-class Source/Target mixing and inter-class
    # separation can therefore be inspected simultaneously.
    class_figure, class_axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, stage, title in (
        (
            class_axes[0],
            "pre_alignment",
            "TAL 前：编码器融合类别特征",
        ),
        (
            class_axes[1],
            "post_alignment",
            "TAL 后：张量投影融合类别特征",
        ),
    ):
        _class_conditional_fused_scatter(
            axis,
            snapshot,
            stage,
            source_indices,
            target_indices,
            title,
        )
        fused_metrics = tal_metrics[stage]["fused"]
        axis.text(
            0.02,
            0.02,
            (
                f"同类跨域质心距离={fused_metrics['class_conditional_source_target_centroid_distance']:.4f}\n"
                f"类别可分间隔={fused_metrics['class_separation_gap']:.4f}"
            ),
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8"},
        )
    handles, labels = class_axes[0].get_legend_handles_labels()
    class_figure.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    class_figure.suptitle(
        "TAL 前后的类别条件跨域融合特征（颜色=类别，点形=领域）",
        fontsize=16,
    )
    class_figure.subplots_adjust(bottom=0.17)
    _save(
        class_figure,
        output_dir / "TAL前后类别条件跨域TSNE.png",
    )

    # High-dimensional metrics are reported independently of t-SNE so the
    # visual conclusion cannot be manufactured by a 2-D projection.
    metric_figure, metric_axes = plt.subplots(2, 2, figsize=(13, 9))
    stages = ("pre_alignment", "post_alignment")
    stage_names = ("TAL 前", "TAL 后")
    group_names = ("SAR", "光学", "融合")
    x = np.arange(len(group_names))
    width = 0.36
    for index, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        values = [
            tal_metrics[stage]["modalities"]["sar"][
                "class_conditional_source_target_centroid_distance"
            ],
            tal_metrics[stage]["modalities"]["optical"][
                "class_conditional_source_target_centroid_distance"
            ],
            tal_metrics[stage]["fused"][
                "class_conditional_source_target_centroid_distance"
            ],
        ]
        metric_axes[0, 0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=stage_name,
        )
        purity_values = [
            tal_metrics[stage]["modalities"]["sar"][
                "same_domain_neighbor_ratio"
            ],
            tal_metrics[stage]["modalities"]["optical"][
                "same_domain_neighbor_ratio"
            ],
            tal_metrics[stage]["fused"]["same_domain_neighbor_ratio"],
        ]
        metric_axes[0, 1].bar(
            x + (index - 0.5) * width,
            purity_values,
            width,
            label=stage_name,
        )
    metric_axes[0, 0].set_xticks(x, group_names)
    metric_axes[0, 0].set_ylabel("余弦距离")
    metric_axes[0, 0].set_title("同类别 Source/Target 原型距离（越低越好）")
    metric_axes[0, 0].legend()
    metric_axes[0, 0].grid(axis="y", alpha=0.25)
    metric_axes[0, 1].set_xticks(x, group_names)
    metric_axes[0, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    metric_axes[0, 1].set_ylabel("同域近邻比例")
    metric_axes[0, 1].set_title("类别条件领域可分性（越接近 0.5 越混合）")
    metric_axes[0, 1].legend()
    metric_axes[0, 1].grid(axis="y", alpha=0.25)

    class_metric_names = ("同类相似度", "异类相似度", "类别可分间隔")
    class_keys = (
        "intra_class_cosine_similarity",
        "inter_class_cosine_similarity",
        "class_separation_gap",
    )
    for index, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        metric_axes[1, 0].bar(
            x + (index - 0.5) * width,
            [tal_metrics[stage]["fused"][key] for key in class_keys],
            width,
            label=stage_name,
        )
    metric_axes[1, 0].set_xticks(x, class_metric_names)
    metric_axes[1, 0].set_ylabel("余弦相似度")
    metric_axes[1, 0].set_title("融合特征的类别结构")
    metric_axes[1, 0].legend()
    metric_axes[1, 0].grid(axis="y", alpha=0.25)

    tensor_scores = [
        tal_metrics[stage]["tensor_prototype_score"] for stage in stages
    ]
    metric_axes[1, 1].bar(stage_names, tensor_scores, color=["#7f7f7f", "#4c72b0"])
    metric_axes[1, 1].set_ylabel("相关分数")
    metric_axes[1, 1].set_title("张量类别原型相关（越高越好）")
    metric_axes[1, 1].grid(axis="y", alpha=0.25)
    metric_figure.suptitle("TAL 前后原始高维特征指标", fontsize=16)
    _save(metric_figure, output_dir / "TAL高维指标对比.png")

    metrics_payload = {
        "metric_notes": {
            "class_conditional_source_target_centroid_distance": "越低表示同类别跨域原型越接近",
            "same_domain_neighbor_ratio": "类别平衡条件下越接近0.5表示领域越混合",
            "class_separation_gap": "同类余弦相似度减异类余弦相似度，越高表示类别越可分",
            "tensor_prototype_score": "复现TAL类别原型张量相关分数，越高越好",
        },
        **tal_metrics,
    }
    with (output_dir / "TAL高维指标.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    # TAL changes the dimensionality (512-D concatenated encoder features to
    # 256-D projected features), so each panel necessarily has its own t-SNE
    # coordinate system. Only within-panel domain mixing is interpreted; the
    # original-space diagnostics make the before/after comparison quantitative.
    domain_figure, domain_axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, stage, title in (
        (domain_axes[0], "pre_alignment", "TAL 前：编码器融合特征"),
        (domain_axes[1], "post_alignment", "TAL 后：张量投影融合特征"),
    ):
        values = stage_data[stage]
        _domain_scatter(
            axis,
            values["source"],
            values["target"],
            title,
            values,
        )
        axis.text(
            0.02,
            0.13,
            f"张量类别原型相关={values['tensor_score']:.4f}",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
        )
    handles, labels = domain_axes[0].get_legend_handles_labels()
    domain_figure.legend(handles, labels, loc="lower center", ncol=2)
    domain_figure.suptitle(
        "TAL 前后的源域—目标域融合特征分布（各面板独立 t-SNE）",
        fontsize=16,
    )
    domain_figure.subplots_adjust(bottom=0.12)
    _save(domain_figure, output_dir / "TAL前后领域特征TSNE.png")

    # Retain the modality/class view as a supplementary diagnostic. It should
    # not be used to compare absolute positions across independently fitted
    # panels.
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    labels_source = snapshot["source_labels"]
    labels_target = snapshot["target_labels"]
    panels = [
        ("sar_pre_alignment", "SAR：张量模块前"),
        ("sar_post_alignment", "SAR：张量模块后"),
        ("optical_pre_alignment", "光学：张量模块前"),
        ("optical_post_alignment", "光学：张量模块后"),
    ]
    for axis, (key, title) in zip(axes.flat, panels):
        _domain_class_scatter(
            axis,
            snapshot[f"source_{key}"],
            snapshot[f"target_{key}"],
            labels_source,
            labels_target,
            title,
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    figure.suptitle(
        "张量对齐模块前后的同模态跨域特征分布（各面板独立 t-SNE）",
        fontsize=16,
    )
    figure.subplots_adjust(bottom=0.13)
    _save(figure, output_dir / "张量模块前后TSNE.png")


def _modality_relation_matrices(features: np.ndarray):
    """Return the same modality-wise sample relations used by training."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] % 2:
        raise ValueError("Expected an even-width [samples, features] array.")
    split = features.shape[1] // 2
    sar = _l2_normalize(features[:, :split])
    optical = _l2_normalize(features[:, split:])
    return np.matmul(sar, sar.T), np.matmul(optical, optical.T)


def _relation_statistics(
    original: np.ndarray,
    translated: np.ndarray,
    margin: float = 0.01,
):
    """Reproduce and decompose ``modality_relation_drift_loss`` in NumPy."""

    original_sar, original_optical = _modality_relation_matrices(original)
    translated_sar, translated_optical = _modality_relation_matrices(translated)
    before_squared = np.square(original_sar - original_optical)
    after_squared = np.square(translated_sar - translated_optical)
    before = float(np.mean(before_squared))
    after = float(np.mean(after_squared))
    signed_drift = after - before
    return {
        "before": before,
        "after": after,
        "signed_drift": signed_drift,
        "penalty": max(signed_drift - float(margin), 0.0),
        # Row means decompose the scalar MSE into a per-sample distribution for
        # boxplots without changing the loss definition.
        "per_sample_signed_drift": np.mean(
            after_squared - before_squared, axis=1
        ),
        "before_absolute": np.abs(original_sar - original_optical),
        "after_absolute": np.abs(translated_sar - translated_optical),
        "original_signatures": np.concatenate(
            [original_sar, original_optical], axis=1
        ),
        "translated_signatures": np.concatenate(
            [translated_sar, translated_optical], axis=1
        ),
    }


def _snapshot_relation_statistics(snapshot, margin: float = 0.01):
    directions = {}
    for direction, domain, translated_key in (
        ("source_to_target", "source", "source_target_like"),
        ("target_to_source", "target", "target_source_like"),
    ):
        if translated_key not in snapshot:
            continue
        directions[direction] = _relation_statistics(
            np.asarray(snapshot[f"{domain}_raw"], dtype=np.float32),
            np.asarray(snapshot[translated_key], dtype=np.float32),
            margin=margin,
        )
    return directions


def _relation_change(snapshot) -> np.ndarray:
    """Return signed per-sample drift under the actual training definition."""

    directions = _snapshot_relation_statistics(snapshot)
    values = [
        statistics["per_sample_signed_drift"]
        for statistics in directions.values()
    ]
    return np.concatenate(values) if values else np.empty(0)


def _class_scatter(axis, embedding, labels, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = np.asarray(labels) == class_id
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=11,
            alpha=0.65,
            c=[colors[class_id]],
            label=class_name,
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_drift_comparison(runs, output_dir: Path) -> None:
    if "full" not in runs or "no_modality_drift" not in runs:
        return
    snapshots = {}
    for variant in ("full", "no_modality_drift"):
        path = runs[variant][0][1] / "feature_embeddings.npz"
        if not path.is_file():
            return
        snapshots[variant] = np.load(path)
    relation_statistics = {
        variant: _snapshot_relation_statistics(snapshot)
        for variant, snapshot in snapshots.items()
    }

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    values = [_relation_change(snapshots[name]) for name in snapshots]
    axes[0].boxplot(
        values,
        labels=["完整模型", "无漂移约束"],
        showfliers=False,
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].axhline(
        0.01,
        color="#c44e52",
        linewidth=1.0,
        linestyle="--",
        label="训练 margin=0.01",
    )
    axes[0].set_ylabel("翻译后减翻译前的模态关系差异（有符号）")
    axes[0].set_title("真实训练定义下的样本关系漂移")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    groups = []
    before_values = []
    after_values = []
    for variant in ("full", "no_modality_drift"):
        for direction, direction_name in (
            ("source_to_target", "源→目标"),
            ("target_to_source", "目标→源"),
        ):
            statistics = relation_statistics[variant][direction]
            groups.append(f"{VARIANT_NAMES[variant]}\n{direction_name}")
            before_values.append(statistics["before"])
            after_values.append(statistics["after"])
    x = np.arange(len(groups))
    width = 0.36
    axes[1].bar(x - width / 2, before_values, width, label="翻译前")
    axes[1].bar(x + width / 2, after_values, width, label="翻译后")
    axes[1].set_xticks(x, groups, rotation=8)
    axes[1].set_ylabel("SAR/光学样本关系矩阵均方差")
    axes[1].set_title("翻译前后的模态关系差异")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    _save(figure, output_dir / "漂移约束对比.png")

    # Relation signatures use sample-to-sample similarities as coordinates,
    # matching the quantity protected by the loss. Before/after samples are
    # embedded jointly for each model and linked by a small set of trajectories.
    tsne_figure, tsne_axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, variant in zip(
        tsne_axes,
        ("full", "no_modality_drift"),
    ):
        statistics = relation_statistics[variant]["target_to_source"]
        before = statistics["original_signatures"]
        after = statistics["translated_signatures"]
        embedding = _tsne(np.concatenate([before, after], axis=0))
        count = len(before)
        for index in np.linspace(0, count - 1, min(80, count), dtype=int):
            axis.plot(
                [embedding[index, 0], embedding[count + index, 0]],
                [embedding[index, 1], embedding[count + index, 1]],
                color="0.55",
                linewidth=0.45,
                alpha=0.25,
            )
        axis.scatter(
            embedding[:count, 0],
            embedding[:count, 1],
            s=10,
            alpha=0.5,
            c="#2f6f9f",
            label="翻译前关系特征",
        )
        axis.scatter(
            embedding[count:, 0],
            embedding[count:, 1],
            s=11,
            alpha=0.55,
            c="#d95f8d",
            marker="x",
            label="翻译后关系特征",
        )
        axis.set_title(f"{VARIANT_NAMES[variant]}：目标→源")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, labels = tsne_axes[0].get_legend_handles_labels()
    tsne_figure.legend(handles, labels, loc="lower center", ncol=2)
    tsne_figure.suptitle(
        "漂移约束开启与关闭时的翻译前后模态关系特征",
        fontsize=16,
    )
    tsne_figure.subplots_adjust(bottom=0.12)
    _save(tsne_figure, output_dir / "漂移约束前后TSNE.png")

    # Heatmaps expose exactly where the SAR and optical sample-relation
    # matrices disagree. A class-balanced subset keeps the figure legible.
    heatmap_figure, heatmap_axes = plt.subplots(2, 3, figsize=(15, 9))
    heatmap_payload = []
    for variant in ("full", "no_modality_drift"):
        snapshot = snapshots[variant]
        labels = np.asarray(snapshot["target_labels"])
        selected = []
        per_class = 20
        for class_id in range(len(CLASS_NAMES)):
            selected.extend(np.flatnonzero(labels == class_id)[:per_class].tolist())
        selected = np.asarray(selected, dtype=np.int64)
        statistics = _relation_statistics(
            np.asarray(snapshot["target_raw"], dtype=np.float32)[selected],
            np.asarray(snapshot["target_source_like"], dtype=np.float32)[selected],
        )
        heatmap_payload.append(statistics)
    maximum = max(
        float(np.quantile(statistics[key], 0.98))
        for statistics in heatmap_payload
        for key in ("before_absolute", "after_absolute")
    )
    delta_maximum = max(
        float(
            np.quantile(
                np.abs(
                    statistics["after_absolute"]
                    - statistics["before_absolute"]
                ),
                0.98,
            )
        )
        for statistics in heatmap_payload
    )
    for row, (variant, statistics) in enumerate(
        zip(("full", "no_modality_drift"), heatmap_payload)
    ):
        delta = statistics["after_absolute"] - statistics["before_absolute"]
        images = [
            heatmap_axes[row, 0].imshow(
                statistics["before_absolute"],
                cmap="magma",
                vmin=0.0,
                vmax=maximum,
            ),
            heatmap_axes[row, 1].imshow(
                statistics["after_absolute"],
                cmap="magma",
                vmin=0.0,
                vmax=maximum,
            ),
            heatmap_axes[row, 2].imshow(
                delta,
                cmap="coolwarm",
                vmin=-delta_maximum,
                vmax=delta_maximum,
            ),
        ]
        for class_boundary in range(20, 120, 20):
            for axis in heatmap_axes[row]:
                axis.axhline(class_boundary - 0.5, color="white", linewidth=0.35)
                axis.axvline(class_boundary - 0.5, color="white", linewidth=0.35)
        heatmap_axes[row, 0].set_ylabel(VARIANT_NAMES[variant])
        for axis in heatmap_axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        heatmap_figure.colorbar(images[0], ax=heatmap_axes[row, :2], shrink=0.72)
        heatmap_figure.colorbar(images[2], ax=heatmap_axes[row, 2], shrink=0.72)
    for column, title in enumerate(
        ("翻译前关系差异", "翻译后关系差异", "翻译后 − 翻译前")
    ):
        heatmap_axes[0, column].set_title(title)
    heatmap_figure.suptitle(
        "目标→源翻译前后的 SAR/光学样本关系差异矩阵",
        fontsize=16,
    )
    _save(heatmap_figure, output_dir / "漂移约束模态关系矩阵.png")


def write_mechanism_diagnostics(runs, output_dir: Path) -> None:
    diagnostics = {}
    if "full" in runs:
        path = runs["full"][0][1] / "feature_embeddings.npz"
        if path.is_file():
            snapshot = np.load(path)
            source_labels = np.asarray(snapshot["source_labels"])
            target_labels = np.asarray(snapshot["target_labels"])
            for modality in ("sar", "optical"):
                for stage in ("pre_alignment", "post_alignment"):
                    source_key = f"source_{modality}_{stage}"
                    target_key = f"target_{modality}_{stage}"
                    if source_key in snapshot and target_key in snapshot:
                        source_values = np.asarray(snapshot[source_key], dtype=np.float32)
                        target_values = np.asarray(snapshot[target_key], dtype=np.float32)
                        source_values /= np.linalg.norm(
                            source_values, axis=1, keepdims=True
                        ) + 1e-8
                        target_values /= np.linalg.norm(
                            target_values, axis=1, keepdims=True
                        ) + 1e-8
                        source_center = np.mean(source_values, axis=0)
                        target_center = np.mean(target_values, axis=0)
                        diagnostics[f"{modality}_{stage}_centroid_distance"] = float(
                            np.linalg.norm(source_center - target_center)
                        )
            for stage in ("pre_alignment", "post_alignment"):
                source = _fused_alignment_features(snapshot, "source", stage)
                target = _fused_alignment_features(snapshot, "target", stage)
                diagnostics[
                    f"tal_{stage}_class_conditional_centroid_distance"
                ] = _class_conditional_centroid_distance(
                    source,
                    target,
                    source_labels,
                    target_labels,
                )
                diagnostics[
                    f"tal_{stage}_same_domain_neighbor_ratio"
                ] = _class_conditional_domain_neighbor_purity(
                    source,
                    target,
                    source_labels,
                    target_labels,
                )
                diagnostics[
                    f"tal_{stage}_tensor_prototype_score"
                ] = _tensor_prototype_score(snapshot, stage)
    for variant in ("full", "no_modality_drift"):
        if variant not in runs:
            continue
        path = runs[variant][0][1] / "feature_embeddings.npz"
        if path.is_file():
            snapshot = np.load(path)
            directions = _snapshot_relation_statistics(snapshot)
            for direction, statistics in directions.items():
                prefix = f"{variant}_{direction}"
                diagnostics[f"{prefix}_relation_before"] = statistics["before"]
                diagnostics[f"{prefix}_relation_after"] = statistics["after"]
                diagnostics[f"{prefix}_signed_drift"] = statistics["signed_drift"]
                diagnostics[f"{prefix}_penalty_margin_0_01"] = statistics["penalty"]
                diagnostics[f"{prefix}_per_sample_drift_std"] = float(
                    np.std(statistics["per_sample_signed_drift"])
                )
    with (output_dir / "mechanism_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--font-path", default="")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    selected_font = _configure_chinese_font(args.font_path)
    if args.check_only:
        from sklearn.manifold import TSNE  # noqa: F401

        print(f"可视化依赖和中文字体检查通过：{selected_font}")
        return
    if not args.experiment_dir:
        parser.error("--experiment-dir is required unless --check-only is used.")
    experiment_dir = Path(args.experiment_dir)
    runs = _discover_runs(experiment_dir)
    if not runs:
        raise SystemExit(f"No completed runs found in {experiment_dir}")
    output_dir = experiment_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_overall_metrics(runs, output_dir)
    plot_per_class_f1(runs, output_dir)
    plot_tal_tsne(runs, output_dir)
    plot_drift_comparison(runs, output_dir)
    write_mechanism_diagnostics(runs, output_dir)
    print(f"可视化已保存：{output_dir}")


if __name__ == "__main__":
    main()
