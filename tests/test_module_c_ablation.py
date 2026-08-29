"""Regression tests for the Module-C ablation utility."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np

from scripts.ablate_module_c import (
    CONSTRAINTS,
    VARIANTS,
    analyse,
    aggregate_summaries,
    discover_summaries,
    make_variant_config,
    _load_matplotlib,
    _plot_constraint_feature_evidence,
    _plot_feature_diagnostics,
    _plot_constraint_feature_tsne,
    _plot_prototype_alignment_by_domain,
)
from scripts.run_all_experiments import (
    EXPERIMENT_VARIANTS,
    build_ablation_command,
    build_parser as build_all_experiments_parser,
)


class ModuleCAblationTests(unittest.TestCase):
    def test_one_command_launcher_contains_no_duplicate_or_extra_variants(self) -> None:
        self.assertEqual(
            EXPERIMENT_VARIANTS,
            [
                "full",
                "no_cycle",
                "no_identity",
                "no_paired_contrastive",
                "no_prototype_contrastive",
                "no_classification_feedback",
                "no_module_c",
            ],
        )
        self.assertEqual(set(EXPERIMENT_VARIANTS), set(VARIANTS))
        args = build_all_experiments_parser().parse_args(
            ["--source-root", "/data/clear", "--target-parent-root", "/data"]
        )
        self.assertEqual(args.target_domains, ["黑天", "逆光", "雨天"])
        self.assertTrue(
            str(args.weather_profile_config).endswith(
                "module_c_weather_profiles_v9.json"
            )
        )
        command = build_ablation_command(args)
        self.assertEqual(command.count("--run"), 1)
        self.assertIn("--no-pca-feature-view", command)
        self.assertIn("--no-tsne-feature-view", command)
        self.assertNotIn("--reference-runs-root", command)
        variant_start = command.index("--variants") + 1
        variant_end = command.index("--iterations")
        self.assertEqual(command[variant_start:variant_end], EXPERIMENT_VARIANTS)

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

    def test_variants_preserve_architecture_and_non_loss_settings(self) -> None:
        base = {
            "feature_dim": 384,
            "contrastive_temperature": 0.2,
            "generator": {"hidden_dim": 192, "num_layers": 2},
            "primary_discriminator": {"hidden_dims": [128, 64]},
            "auxiliary_discriminator": {"hidden_dims": [128, 64]},
            "loss_weights": {
                "classification": 0.65,
                "adv_primary": 0.025,
                "adv_auxiliary": 0.025,
                "cycle": 0.12,
                "identity": 0.03,
                "contrastive": 0.04,
                "prototype_contrastive": 0.06,
            },
        }
        expected_non_loss = {key: value for key, value in base.items() if key != "loss_weights"}
        for variant in VARIANTS:
            candidate = make_variant_config(base, variant)
            actual_non_loss = {
                key: value for key, value in candidate.items() if key != "loss_weights"
            }
            self.assertEqual(actual_non_loss, expected_non_loss)

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

    def test_summary_respects_trainer_selected_epoch_with_min_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "module_c_full_黑天_test"
            run_dir.mkdir()
            with (run_dir / "metrics_iter01.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "monitor_selection_score",
                        "val_acc",
                        "val_precision_macro_present",
                        "val_recall_macro_present",
                        "val_f1_macro_present",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 40,
                        "monitor_selection_score": 0.962567,
                        "val_acc": 0.95,
                        "val_precision_macro_present": 0.96,
                        "val_recall_macro_present": 0.96,
                        "val_f1_macro_present": 0.962567,
                    }
                )
                writer.writerow(
                    {
                        "epoch": 46,
                        "monitor_selection_score": 0.963426,
                        "val_acc": 0.96,
                        "val_precision_macro_present": 0.97,
                        "val_recall_macro_present": 0.97,
                        "val_f1_macro_present": 0.963426,
                    }
                )
            (run_dir / "result_summary_iter01.json").write_text(
                json.dumps({"best_metrics": {"epoch": 40}}), encoding="utf-8"
            )

            summaries = discover_summaries(root, "module_c_*", "val_f1_macro_present")
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["best_epoch"], 40)
            self.assertAlmostEqual(summaries[0]["best_val_f1"], 0.962567)

    def test_grouped_partial_iteration_is_not_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "module_c_full_逆光_test"
            run_dir.mkdir()
            for iteration in (1, 2):
                with (run_dir / f"metrics_iter{iteration:02d}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["epoch", "val_acc", "val_f1_macro_present"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"epoch": 1, "val_acc": 0.9, "val_f1_macro_present": 0.9}
                    )
            (run_dir / "result_summary_iter01.json").write_text(
                json.dumps({"best_metrics": {"epoch": 1}}), encoding="utf-8"
            )

            summaries = discover_summaries(root, "module_c_*", "val_f1_macro_present")
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["iteration"], 1)

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
                source_target_like=target,
                source_reconstruction=source,
                source_identity=source,
                source_labels=labels,
                target_raw=target,
                target_source_like=translated,
                target_reconstruction=target,
                target_identity=target,
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
            self.assertAlmostEqual(metrics[0]["cycle_cosine_error"], 0.0)
            self.assertAlmostEqual(metrics[0]["identity_cosine_error"], 0.0)
            self.assertTrue((root / images[0]).exists())
            prototype_images = _plot_prototype_alignment_by_domain(
                summaries, root, _load_matplotlib()
            )
            self.assertEqual(prototype_images, ["prototype_alignment_rain.png"])
            self.assertTrue((root / prototype_images[0]).exists())

    def test_constraint_feature_evidence_covers_every_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
                dtype=np.float32,
            )
            target = np.asarray(
                [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6]],
                dtype=np.float32,
            )
            aligned = np.asarray(
                [[0.95, 0.05], [0.9, 0.1], [0.05, 0.95], [0.1, 0.9]],
                dtype=np.float32,
            )
            degraded = np.asarray(
                [[0.45, 0.55], [0.5, 0.5], [0.55, 0.45], [0.5, 0.5]],
                dtype=np.float32,
            )
            labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
            good_logits = np.asarray(
                [[4.0, 0.0], [3.0, 0.2], [0.0, 4.0], [0.2, 3.0]],
                dtype=np.float32,
            )
            bad_logits = good_logits[:, ::-1].copy()
            sample_ids = np.asarray(["a|A", "b|B", "c|C", "d|D"])
            summaries = []
            for variant in ["full", *[f"no_{name}" for name in CONSTRAINTS]]:
                run_dir = root / f"module_c_{variant}_rain"
                run_dir.mkdir()
                is_cycle = variant == "no_cycle"
                is_identity = variant == "no_identity"
                is_alignment = variant in {
                    "no_paired_contrastive",
                    "no_prototype_contrastive",
                }
                is_feedback = variant == "no_classification_feedback"
                np.savez_compressed(
                    run_dir / "feature_embeddings.npz",
                    source_raw=source,
                    source_target_like=aligned,
                    source_reconstruction=degraded if is_cycle else source,
                    source_identity=degraded if is_identity else source,
                    source_raw_logits=good_logits,
                    source_target_like_logits=good_logits,
                    source_labels=labels,
                    source_sample_ids=sample_ids,
                    target_raw=target,
                    target_source_like=degraded if is_alignment else aligned,
                    target_reconstruction=degraded if is_cycle else target,
                    target_identity=degraded if is_identity else target,
                    target_raw_logits=good_logits,
                    target_source_like_logits=bad_logits if is_feedback else good_logits,
                    target_labels=labels,
                    target_sample_ids=sample_ids,
                )
                summaries.append(
                    {
                        "run_dir": str(run_dir),
                        "run": run_dir.name,
                        "variant": variant,
                        "domain": "雨天",
                        "iteration": 1,
                        "seed": 42,
                        "best_val_f1": 0.95 if variant == "full" else 0.85,
                    }
                )

            images, evidence = _plot_constraint_feature_evidence(
                summaries, root, _load_matplotlib()
            )
            self.assertEqual(len(images), len(CONSTRAINTS))
            self.assertEqual(len(evidence), len(CONSTRAINTS))
            self.assertEqual(
                {row["constraint"] for row in evidence}, set(CONSTRAINTS)
            )
            self.assertTrue(all(row["paired_target_samples"] == 4 for row in evidence))
            self.assertTrue(all(row["f1_gain"] > 0.0 for row in evidence))
            self.assertTrue(all((root / image).is_file() for image in images))

    def test_paired_tsne_feature_view_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
                dtype=np.float32,
            )
            translated = np.asarray(
                [[0.95, 0.05], [0.9, 0.1], [0.05, 0.95], [0.1, 0.9]],
                dtype=np.float32,
            )
            labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
            snapshots = {
                "source_raw": raw,
                "source_labels": labels,
                "source_sample_ids": np.asarray(["a", "b", "c", "d"]),
                "target_raw": raw,
                "target_source_like": translated,
                "source_reconstruction": translated,
                "source_identity": translated,
                "target_reconstruction": translated,
                "target_identity": translated,
                "target_labels": labels,
                "target_sample_ids": np.asarray(["a", "b", "c", "d"]),
            }
            summaries = []
            for variant in ("full", "no_cycle"):
                run_dir = root / f"module_c_{variant}_雨天_iter01"
                run_dir.mkdir()
                np.savez_compressed(run_dir / "feature_embeddings.npz", **snapshots)
                summary = {
                    "run_dir": str(run_dir),
                    "run": run_dir.name,
                    "variant": variant,
                    "domain": "雨天",
                    "iteration": 1,
                    "seed": 42,
                    "feature_embeddings": str(run_dir / "feature_embeddings.npz"),
                    "best_val_f1": 0.9,
                }
                summaries.append(summary)
            images = _plot_constraint_feature_tsne(
                summaries, root, _load_matplotlib()
            )
            self.assertEqual(len(images), 1)
            self.assertEqual(
                Path(images[0]).as_posix(),
                "constraint_feature_evidence/tsne/constraint_tsne_cycle_rain.png",
            )
            self.assertTrue((root / images[0]).is_file())

    def test_grouped_iteration_artifacts_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "module_c_full_雨天_20260825"
            run_dir.mkdir()
            for iteration, value in ((1, 0.80), (2, 0.90)):
                suffix = f"iter{iteration:02d}"
                (run_dir / f"resolved_config_{suffix}.json").write_text(
                    json.dumps({"args": {"seed": 41 + iteration}}), encoding="utf-8"
                )
                with (run_dir / f"metrics_{suffix}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
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
                    writer.writerow(
                        {
                            "epoch": 1,
                            "val_acc": value,
                            "val_precision_macro_present": value,
                            "val_recall_macro_present": value,
                            "val_f1_macro_present": value,
                            "train_full_acc": value + 0.02,
                        }
                    )
            summaries = discover_summaries(root, "module_c_*", "val_f1_macro_present")
            self.assertEqual(len(summaries), 2)
            self.assertEqual(sorted(row["iteration"] for row in summaries), [1, 2])
            self.assertTrue(all(row["domain"] == "雨天" for row in summaries))
            self.assertTrue(all("metrics_iter" in row["metrics_path"] for row in summaries))

    def test_analysis_merges_63_new_runs_with_21_frozen_fog_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary_root = root / "primary"
            reference_root = root / "reference"
            output_root = root / "analysis"
            primary_root.mkdir()
            reference_root.mkdir()

            def write_grouped_runs(
                runs_root: Path,
                domains: list[str],
                include_required_artifacts: bool,
            ) -> None:
                for variant in VARIANTS:
                    for domain in domains:
                        run_dir = runs_root / f"module_c_{variant}_{domain}_test"
                        run_dir.mkdir()
                        for iteration in range(1, 4):
                            suffix = f"iter{iteration:02d}"
                            with (run_dir / f"metrics_{suffix}.csv").open(
                                "w", encoding="utf-8", newline=""
                            ) as handle:
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
                                writer.writerow(
                                    {
                                        "epoch": 1,
                                        "val_acc": 0.9,
                                        "val_precision_macro_present": 0.9,
                                        "val_recall_macro_present": 0.9,
                                        "val_f1_macro_present": 0.9,
                                        "train_full_acc": 0.92,
                                    }
                                )
                            if include_required_artifacts:
                                (run_dir / f"feature_embeddings_{suffix}.npz").write_bytes(
                                    b"test"
                                )
                                (run_dir / f"result_summary_{suffix}.json").write_text(
                                    "{}", encoding="utf-8"
                                )

            primary_domains = ["黑天", "逆光", "雨天"]
            write_grouped_runs(primary_root, primary_domains, True)
            write_grouped_runs(reference_root, ["雾天"], False)
            (primary_root / "experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "variants": VARIANTS,
                        "domains": primary_domains,
                        "iterations": 3,
                        "expected_runs": 63,
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                runs_root=str(primary_root),
                analysis_output=str(output_root),
                analysis_glob="module_c_*",
                monitor_metric="val_f1_macro_present",
                pca_feature_view=False,
                reference_runs_root=[str(reference_root), str(reference_root)],
            )
            with (
                mock.patch(
                    "scripts.ablate_module_c._load_matplotlib",
                    side_effect=ImportError("test without matplotlib"),
                ),
                mock.patch("scripts.ablate_module_c._plot_overview_svg"),
                mock.patch("scripts.ablate_module_c._plot_effect_heatmap_svg"),
                mock.patch("scripts.ablate_module_c._plot_constraint_diagnostic_svg"),
                mock.patch("scripts.ablate_module_c._write_html_report"),
            ):
                payload = analyse(args)

            self.assertEqual(len(payload["runs"]), 84)
            self.assertEqual(len(payload["aggregate"]), 28)
            self.assertEqual(
                sum(row["domain"] == "雾天" for row in payload["runs"]), 21
            )
            self.assertEqual(
                sum(row["runs"] for row in payload["aggregate"]), 84
            )
            with (output_root / "ablation_runs.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 84)


if __name__ == "__main__":
    unittest.main()
