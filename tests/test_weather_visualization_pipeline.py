import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts import run_weather_visualization_suite as suite
from scripts import visualize_weather_modules as visual


def _snapshot(path: Path, legacy_names: bool = False, order=None) -> None:
    rng = np.random.default_rng(7)
    count = 12
    order = np.arange(count) if order is None else np.asarray(order)
    labels = np.tile(np.arange(3), 4)[order]
    ids = np.asarray([f"sample-{i}" for i in range(count)])[order]
    arrays = {}
    first_name, second_name = (("sar", "optical") if legacy_names else ("vis", "ir"))
    for domain_offset, domain in enumerate(visual.DOMAINS):
        arrays[f"{domain}_labels"] = labels
        arrays[f"{domain}_sample_ids"] = ids
        first = rng.normal(domain_offset, 1, (count, 6))[order]
        second = rng.normal(domain_offset + 0.3, 1, (count, 6))[order]
        arrays[f"{domain}_{first_name}_pre_alignment"] = first
        arrays[f"{domain}_{second_name}_pre_alignment"] = second
        arrays[f"{domain}_{first_name}_post_alignment"] = first[:, :4]
        arrays[f"{domain}_{second_name}_post_alignment"] = second[:, :4]
        translated = np.concatenate([first[:, :4] + 0.1, second[:, :4] - 0.1], axis=1)
        arrays[
            "source_target_like" if domain == "source" else "target_source_like"
        ] = translated
    np.savez_compressed(path, **arrays)


def test_legacy_weather_modality_names_follow_real_encoder_order(tmp_path):
    path = tmp_path / "legacy.npz"
    _snapshot(path, legacy_names=True)
    snapshot = visual.Snapshot(path)
    np.testing.assert_array_equal(
        snapshot.modality("source", "vis", "pre"),
        snapshot.data["source_sar_pre_alignment"],
    )
    np.testing.assert_array_equal(
        snapshot.modality("source", "ir", "pre"),
        snapshot.data["source_optical_pre_alignment"],
    )


def test_cache_matches_drift_samples_and_supports_tal_dimension_change(tmp_path, monkeypatch):
    full_path = tmp_path / "full.npz"
    no_drift_path = tmp_path / "no_drift.npz"
    _snapshot(full_path)
    _snapshot(no_drift_path, order=np.arange(12)[::-1])

    def fake_tsne(features, seed, perplexity):
        return np.column_stack([
            np.arange(len(features), dtype=np.float32),
            np.full(len(features), seed, dtype=np.float32),
        ])

    monkeypatch.setattr(visual, "_tsne", fake_tsne)
    output = tmp_path / "output"
    visual.build_cache(full_path, no_drift_path, output, 4, 42, 5.0)

    metadata = json.loads((output / "cache" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["protocol"]["target_test_used"] is False
    assert metadata["protocol"]["ais_used"] is False
    assert set(metadata["comparisons"]) == {"tal", "dual_translation", "modality_drift"}
    coordinates = np.load(output / "cache" / "tsne_coordinates.npz")
    assert "tal__left__source_vis" in coordinates
    assert "tal__right__source_vis" in coordinates
    assert "modality_drift__right__class__target" in coordinates


def test_training_commands_are_one_seed_no_ais_and_disable_drift_only():
    args = Namespace(
        python="python",
        config=Path("configs/train_dual_d_default.json"),
        weather_profile_config=Path("configs/module_c_weather_profiles_v13.json"),
        data_root=Path("/mnt/data/lixiang/data/Data"),
        source_weather="晴天",
        seed=42,
        feature_visualization_samples=1024,
        device="cuda",
        epochs=None,
        batch_size=None,
        num_workers=None,
    )
    full = suite._training_command(args, "黑天", "full", "full", Path("runs"))
    ablated = suite._training_command(
        args, "黑天", "no_modality_drift", "no_drift", Path("runs")
    )
    assert full[full.index("--iterations") + 1] == "1"
    assert full[full.index("--seed") + 1] == "42"
    assert "--no-use-ais" in full
    assert "--synchronize-modalities" in full
    assert "--no-evaluate-target-test" in full
    assert "--modality-drift-enabled" in full
    assert "--no-modality-drift-enabled" in ablated
    assert "--translation-enabled" in ablated
    assert "--module-c-enabled" in ablated
