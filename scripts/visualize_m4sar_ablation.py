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

    candidates: list[Path] = []
    if font_path:
        explicit = Path(font_path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"指定的中文字体文件不存在：{explicit}")
        candidates.append(explicit)
    candidates.extend(
        [
            Path.home() / ".fonts" / "simsun.ttc",
            Path.home() / ".local" / "share" / "fonts" / "simsun.ttc",
            Path("/usr/share/fonts/truetype/msttcorefonts/simsun.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ]
    )
    preferred_families = (
        "SimSun",
        "宋体",
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
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
    candidates.extend(Path(entry.fname) for entry in installed)
    # findSystemFonts also sees newly installed fonts when Matplotlib's cache is stale.
    candidates.extend(Path(path) for path in font_manager.findSystemFonts())

    visited: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in visited or not resolved.is_file():
            continue
        visited.add(resolved)
        if not _supports_chinese_glyphs(resolved):
            continue
        font_manager.fontManager.addfont(str(resolved))
        family = font_manager.FontProperties(fname=str(resolved)).get_name()
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
    figure.suptitle("张量对齐模块前后的同模态跨域特征分布", fontsize=16)
    figure.subplots_adjust(bottom=0.13)
    _save(figure, output_dir / "张量模块前后TSNE.png")


def _relation_change(snapshot) -> np.ndarray:
    values = []
    for domain, translated_key in (
        ("source", "source_target_like"),
        ("target", "target_source_like"),
    ):
        if translated_key not in snapshot:
            continue
        raw = np.asarray(snapshot[f"{domain}_raw"], dtype=np.float32)
        translated = np.asarray(snapshot[translated_key], dtype=np.float32)
        split = raw.shape[1] // 2
        raw_cos = np.sum(raw[:, :split] * raw[:, split:], axis=1) / (
            np.linalg.norm(raw[:, :split], axis=1)
            * np.linalg.norm(raw[:, split:], axis=1)
            + 1e-8
        )
        translated_cos = np.sum(
            translated[:, :split] * translated[:, split:], axis=1
        ) / (
            np.linalg.norm(translated[:, :split], axis=1)
            * np.linalg.norm(translated[:, split:], axis=1)
            + 1e-8
        )
        values.append(np.abs(translated_cos - raw_cos))
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
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    values = [_relation_change(snapshots[name]) for name in snapshots]
    axes[0].boxplot(values, labels=["完整模型", "无漂移约束"], showfliers=False)
    axes[0].set_ylabel("翻译前后 SAR–光学余弦关系变化")
    axes[0].set_title("模态关系漂移量")
    axes[0].grid(axis="y", alpha=0.25)

    for variant, color in (("full", "#2f6f9f"), ("no_modality_drift", "#b85c38")):
        snapshot = snapshots[variant]
        before = np.asarray(snapshot["target_raw"], dtype=np.float32)
        after = np.asarray(snapshot["target_source_like"], dtype=np.float32)
        embedding = _tsne(np.concatenate([before, after], axis=0))
        count = len(before)
        displacement = np.linalg.norm(
            embedding[:count] - embedding[count:], axis=1
        )
        axes[1].hist(
            displacement,
            bins=30,
            alpha=0.55,
            color=color,
            label=VARIANT_NAMES[variant],
            density=True,
        )
    axes[1].set_xlabel("目标域样本翻译位移（联合 t-SNE 空间）")
    axes[1].set_ylabel("密度")
    axes[1].set_title("漂移约束对特征移动的影响")
    axes[1].legend()
    _save(figure, output_dir / "漂移约束对比.png")

    tsne_figure, tsne_axes = plt.subplots(2, 2, figsize=(13, 11))
    for row, variant in enumerate(("full", "no_modality_drift")):
        snapshot = snapshots[variant]
        before = np.asarray(snapshot["target_raw"], dtype=np.float32)
        after = np.asarray(snapshot["target_source_like"], dtype=np.float32)
        embedding = _tsne(np.concatenate([before, after], axis=0))
        count = len(before)
        labels = snapshot["target_labels"]
        _class_scatter(
            tsne_axes[row, 0],
            embedding[:count],
            labels,
            f"{VARIANT_NAMES[variant]}：翻译前",
        )
        _class_scatter(
            tsne_axes[row, 1],
            embedding[count:],
            labels,
            f"{VARIANT_NAMES[variant]}：翻译后",
        )
    handles, labels = tsne_axes[0, 0].get_legend_handles_labels()
    tsne_figure.legend(handles, labels, loc="lower center", ncol=6)
    tsne_figure.suptitle("漂移约束开启与关闭时的翻译前后特征分布", fontsize=16)
    tsne_figure.subplots_adjust(bottom=0.10)
    _save(tsne_figure, output_dir / "漂移约束前后TSNE.png")


def write_mechanism_diagnostics(runs, output_dir: Path) -> None:
    diagnostics = {}
    if "full" in runs:
        path = runs["full"][0][1] / "feature_embeddings.npz"
        if path.is_file():
            snapshot = np.load(path)
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
    for variant in ("full", "no_modality_drift"):
        if variant not in runs:
            continue
        path = runs[variant][0][1] / "feature_embeddings.npz"
        if path.is_file():
            values = _relation_change(np.load(path))
            if len(values):
                diagnostics[f"{variant}_relation_change_mean"] = float(np.mean(values))
                diagnostics[f"{variant}_relation_change_std"] = float(np.std(values))
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
