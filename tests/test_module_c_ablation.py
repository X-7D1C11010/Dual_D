"""Regression tests for the Module-C ablation utility."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.ablate_module_c import (
    CONSTRAINTS,
    VARIANTS,
    aggregate_summaries,
    discover_summaries,
    make_variant_config,
    _load_matplotlib,
    _plot_feature_diagnostics,
    _plot_prototype_alignment_by_domain,
)


class ModuleCAblationTests(unittest.TestCase):
    def test_every_constraint_has_a_leave_one_out_variant(self) -> None:
        self.assertEqual(
            set(VARIANTS),
            {"full", "no_module_c", *(f"no_{name}" for name in CONSTRAINTS)},
        )
        base = {
            "loss_weights": {
                "classification": 0.65,
                "adv_primary": 0.035,
                "adv_auxiliary": 0.035,
                "cycle": 0.32,
                "identity": 0.15,
                "contrastive": 0.08,
                "prototype_contrastive": 0.09,
            }
        }
        for constraint in CONSTRAINTS:
            config = make_variant_config(base, f"no_{constraint}")
            weights = config["loss_weights"]
            key = {
                "paired_contrastive": "contrastive",
                "classification_feedback": "classification",
            }.get(constraint, constraint)
            self.assertEqual(weights[key], 0.0)
        no_module_c = make_variant_config(base, "no_module_c")["loss_weights"]
        for key in (
            "cycle",
            "identity",
            "contrastive",
            "prototype_contrastive",
            "classification",
        ):
            self.assertEqual(no_module_c[key], 0.0)

    def test_summary_uses_best_epoch_and_aggregates_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, value in enumerate((0.75, 0.85), start=1):
                run_dir = root / f"module_c_full_逆光_iter{index:02d}"
                run_dir.mkdir()
                (run_dir / "result_summary.json").write_text(
                    json.dumps({"target_domain": "逆光"}), encoding="utf-8"
                )
                with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "epoch",
                            "val_acc",
                            "val_precision_macro_present",
                            "val_recall_macro_present",
                            "val_f1_macro_present",
                            "train_full_acc",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow({"epoch": 1, "val_acc": value - 0.1, "val_precision_macro_present": value - 0.08, "val_recall_macro_present": value - 0.06, "val_f1_macro_present": value - 0.1, "train_full_acc": 0.8})
                    writer.writerow({"epoch": 2, "val_acc": value, "val_precision_macro_present": value - 0.02, "val_recall_macro_present": value - 0.01, "val_f1_macro_present": value, "train_full_acc": 0.9})
            summaries = discover_summaries(root, "module_c_*", "val_f1_macro_present")
            self.assertEqual(len(summaries), 2)
            self.assertTrue(all(item["best_epoch"] == 2 for item in summaries))
            aggregate = aggregate_summaries(summaries)
            self.assertEqual(len(aggregate), 1)
            self.assertAlmostEqual(aggregate[0]["best_val_f1_mean"], 0.8)
            self.assertAlmostEqual(aggregate[0]["best_val_precision_mean"], 0.78)
            self.assertAlmostEqual(aggregate[0]["best_val_recall_mean"], 0.79)
            self.assertAlmostEqual(aggregate[0]["best_val_f1_std"], 0.0707106781)

    def test_feature_plot_is_isolated_to_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "module_c_full_雨天_iter01"
            run_dir.mkdir()
            source = np.asarray(
                [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
                dtype=np.float32,
            )
            target = np.asarray(
                [[0.8, 0.2, 0.1], [0.7, 0.3, 0.0], [0.2, 0.8, 0.1], [0.3, 0.7, 0.0]],
                dtype=np.float32,
            )
            translated = np.asarray(
                [[0.95, 0.05, 0.0], [0.9, 0.1, 0.0], [0.05, 0.95, 0.0], [0.1, 0.9, 0.0]],
                dtype=np.float32,
            )
            labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
            np.savez_compressed(
                run_dir / "feature_embeddings.npz",
                source_raw=source,
                source_labels=labels,
                target_raw=target,
                target_source_like=translated,
                target_labels=labels,
            )
            summaries = [
                {
                    "run_dir": str(run_dir),
                    "run": run_dir.name,
                    "variant": "full",
                    "domain": "雨天",
                    "iteration": 1,
                    "seed": 42,
                }
            ]
            images, metrics = _plot_feature_diagnostics(
                summaries, root, _load_matplotlib()
            )
            self.assertEqual(len(images), 1)
            self.assertEqual(len(metrics), 1)
            self.assertGreater(metrics[0]["target_silhouette_gain"], 0.0)
            self.assertGreater(metrics[0]["prototype_margin_gain"], 0.0)
            self.assertTrue((root / images[0]).exists())
            prototype_images = _plot_prototype_alignment_by_domain(
                summaries, root, _load_matplotlib()
            )
            self.assertEqual(prototype_images, ["prototype_alignment_rain.png"])
            self.assertTrue((root / prototype_images[0]).exists())


if __name__ == "__main__":
    unittest.main()
