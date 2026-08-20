"""Regression tests for the Module-C ablation utility."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ablate_module_c import (
    CONSTRAINTS,
    VARIANTS,
    aggregate_summaries,
    discover_summaries,
    make_variant_config,
)


class ModuleCAblationTests(unittest.TestCase):
    def test_every_constraint_has_a_leave_one_out_variant(self) -> None:
        self.assertEqual(set(VARIANTS), {"full", *(f"no_{name}" for name in CONSTRAINTS)})
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
                        fieldnames=["epoch", "val_acc", "val_f1_macro_present", "train_full_acc"],
                    )
                    writer.writeheader()
                    writer.writerow({"epoch": 1, "val_acc": value - 0.1, "val_f1_macro_present": value - 0.1, "train_full_acc": 0.8})
                    writer.writerow({"epoch": 2, "val_acc": value, "val_f1_macro_present": value, "train_full_acc": 0.9})
            summaries = discover_summaries(root, "module_c_*", "val_f1_macro_present")
            self.assertEqual(len(summaries), 2)
            self.assertTrue(all(item["best_epoch"] == 2 for item in summaries))
            aggregate = aggregate_summaries(summaries)
            self.assertEqual(len(aggregate), 1)
            self.assertAlmostEqual(aggregate[0]["best_val_f1_mean"], 0.8)
            self.assertAlmostEqual(aggregate[0]["best_val_f1_std"], 0.05)


if __name__ == "__main__":
    unittest.main()
