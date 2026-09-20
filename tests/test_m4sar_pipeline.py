"""Regression tests for the leakage-safe M4-SAR classification integration."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from dual_d.data import (
    M4SAR_CLASS_WEIGHTS,
    M4SAR_OPTICAL_MEAN,
    M4SAR_OPTICAL_STD,
    M4SAR_SAR_MEAN,
    M4SAR_SAR_STD,
    M4SARClassificationDataset,
    inverse_sqrt_class_weights,
    load_m4sar_manifest,
)
from dual_d.training.trainer import (
    IndependentDomainLoaders,
    build_classification_criterion,
    build_datasets,
    build_models,
    run_training,
)
from scripts.preflight_m4sar import preflight
from scripts.export_m4sar_class_examples import export_examples
from scripts.train_dual_d import build_parser, load_json_defaults


def _write_m4sar_fixture(root: Path, rows_per_split: int = 12) -> Path:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    height, width = 512, 512
    ramp = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    source_optical = np.stack((ramp, np.flip(ramp, axis=1), ramp // 2), axis=2)
    target_optical = np.stack((ramp // 2, ramp, np.flip(ramp, axis=1)), axis=2)
    Image.fromarray(source_optical, mode="RGB").save(image_dir / "source_optical.png")
    Image.fromarray(target_optical, mode="RGB").save(image_dir / "target_optical.png")
    Image.fromarray(ramp, mode="L").save(image_dir / "source_sar.png")
    Image.fromarray(np.flip(ramp, axis=1), mode="L").save(image_dir / "target_sar.png")

    manifest = root / "paired_manifest.csv.gz"
    fields = [
        "split",
        "class_id",
        "pair_id",
        "source_scene_id",
        "target_scene_id",
        "source_optical_path",
        "source_sar_path",
        "target_optical_path",
        "target_sar_path",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
    ]
    with gzip.open(manifest, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        row_index = 0
        for split in ("train", "val", "test"):
            for index in range(rows_per_split):
                source_id = row_index + 1
                writer.writerow(
                    {
                        "split": split,
                        "class_id": index % 6,
                        "pair_id": f"{split}-{index}",
                        "source_scene_id": source_id,
                        "target_scene_id": source_id + 56087,
                        "source_optical_path": "images/source_optical.png",
                        "source_sar_path": "images/source_sar.png",
                        "target_optical_path": "images/target_optical.png",
                        "target_sar_path": "images/target_sar.png",
                        "crop_left": 64,
                        "crop_top": 96,
                        "crop_right": 192,
                        "crop_bottom": 224,
                    }
                )
                row_index += 1
    return manifest


def _fixture_args(root: Path, manifest: Path):
    project_root = Path(__file__).resolve().parents[1]
    defaults = load_json_defaults(project_root / "configs" / "m4sar_classification.json")
    args = build_parser(defaults).parse_args([])
    args.m4sar_manifest = str(manifest)
    args.m4sar_data_root = str(root)
    args.dual_config = str(project_root / "configs" / "dual_d_m4sar.json")
    args.m4sar_input_size = 32
    args.batch_size = 4
    args.num_workers = 0
    args.persistent_workers = False
    args.device = "cpu"
    args.multi_gpu = False
    args.feature_dim = 16
    args.proj_dim = 8
    args.class_weights = list(M4SAR_CLASS_WEIGHTS)
    return args


class M4SARDatasetTests(unittest.TestCase):
    def test_export_separate_paired_optical_and_sar_examples(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            output = root / "examples"
            metadata = export_examples(
                manifest_path=manifest,
                data_root=root,
                output_dir=output,
                split="val",
                domain="target",
                input_size=32,
                seed=42,
            )
            self.assertEqual(len(metadata["examples"]), 6)
            self.assertEqual(len(list((output / "target" / "optical").glob("*.png"))), 6)
            self.assertEqual(len(list((output / "target" / "sar").glob("*.png"))), 6)
            self.assertTrue((output / "metadata.json").is_file())
            for example in metadata["examples"]:
                domain_files = example["domains"]["target"]
                with Image.open(domain_files["optical_png"]) as optical:
                    self.assertEqual(optical.mode, "RGB")
                    self.assertEqual(optical.size, (32, 32))
                with Image.open(domain_files["sar_png"]) as sar:
                    self.assertEqual(sar.mode, "L")
                    self.assertEqual(sar.size, (32, 32))
                self.assertIn(example["pair_id"], Path(domain_files["optical_png"]).name)
                self.assertIn(example["pair_id"], Path(domain_files["sar_png"]).name)

    def test_manifest_crop_shapes_domain_and_source_normalization(self) -> None:
        self.assertEqual(
            M4SAR_OPTICAL_MEAN,
            (0.32412607310812486, 0.3259470182913754, 0.3061976798789559),
        )
        self.assertEqual(
            M4SAR_OPTICAL_STD,
            (0.19963838987820892, 0.1975738079640792, 0.1994368095218188),
        )
        self.assertEqual(M4SAR_SAR_MEAN, (0.47364379746860025,))
        self.assertEqual(M4SAR_SAR_STD, (0.24904880500119383,))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            records = load_m4sar_manifest(manifest, "train", root)
            self.assertEqual(len(records), 12)
            self.assertEqual(records[0].crop_box, (64, 96, 192, 224))
            source = M4SARClassificationDataset(
                manifest, "train", "source", root, input_size=32, augment=False
            )
            target = source.with_domain("target", augment=False)
            source_sample = source[0]
            target_sample = target[0]
            self.assertEqual(tuple(source_sample["optical"].shape), (3, 32, 32))
            self.assertEqual(tuple(source_sample["sar"].shape), (1, 32, 32))
            self.assertEqual(source_sample["domain"], "source")
            self.assertEqual(target_sample["domain"], "target")
            self.assertEqual(source_sample["pair_id"], target_sample["pair_id"])
            optical_raw = (
                source_sample["optical"]
                * torch.tensor(M4SAR_OPTICAL_STD).view(3, 1, 1)
                + torch.tensor(M4SAR_OPTICAL_MEAN).view(3, 1, 1)
            )
            sar_raw = (
                source_sample["sar"] * M4SAR_SAR_STD[0] + M4SAR_SAR_MEAN[0]
            )
            self.assertTrue(bool(((optical_raw >= 0) & (optical_raw <= 1)).all()))
            self.assertTrue(bool(((sar_raw >= 0) & (sar_raw <= 1)).all()))

    def test_scene_boundary_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root, rows_per_split=1)
            rows = []
            with gzip.open(manifest, "rt", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["target_scene_id"] = "56089"
            with gzip.open(manifest, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "pair boundary mismatch"):
                load_m4sar_manifest(manifest, "train", root)

    def test_optical_and_sar_geometry_augmentation_is_synchronized(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            dataset = M4SARClassificationDataset(
                manifest, "train", "source", root, input_size=32, augment=True
            )
            torch.manual_seed(91)
            sample = dataset[0]
            optical_channel = (
                sample["optical"][0] * M4SAR_OPTICAL_STD[0]
                + M4SAR_OPTICAL_MEAN[0]
            )
            sar_channel = sample["sar"][0] * M4SAR_SAR_STD[0] + M4SAR_SAR_MEAN[0]
            self.assertTrue(torch.allclose(optical_channel, sar_channel, atol=1e-6))

    def test_modality_specific_radiometric_augmentation_is_training_only(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            augmented = M4SARClassificationDataset(
                manifest,
                "train",
                "source",
                root,
                input_size=32,
                augment=True,
                optical_jitter=0.10,
                sar_gain_jitter=0.08,
                sar_noise_std=0.01,
            )
            evaluation = M4SARClassificationDataset(
                manifest,
                "train",
                "source",
                root,
                input_size=32,
                augment=False,
                optical_jitter=0.10,
                sar_gain_jitter=0.08,
                sar_noise_std=0.01,
            )
            optical = torch.full((3, 32, 32), 0.5)
            sar = torch.full((1, 32, 32), 0.5)
            torch.manual_seed(23)
            augmented_optical, augmented_sar = augmented._photometric_augmentation(
                optical.clone(),
                sar.clone(),
            )
            evaluation_optical, evaluation_sar = evaluation._photometric_augmentation(
                optical.clone(),
                sar.clone(),
            )

            self.assertTrue(bool(torch.isfinite(augmented_optical).all()))
            self.assertTrue(bool(torch.isfinite(augmented_sar).all()))
            self.assertFalse(torch.allclose(augmented_optical, optical))
            self.assertFalse(torch.allclose(augmented_sar, sar))
            self.assertTrue(torch.equal(evaluation_optical, optical))
            self.assertTrue(torch.equal(evaluation_sar, sar))

    def test_inverse_sqrt_weights_have_mean_one(self) -> None:
        weights = inverse_sqrt_class_weights((105400, 6768, 88030, 38388, 783, 3554))
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, places=7)
        np.testing.assert_allclose(weights, M4SAR_CLASS_WEIGHTS, rtol=0, atol=1e-12)


class M4SARTrainingContractTests(unittest.TestCase):
    def test_train_domain_loaders_shuffle_independently(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            source, _, target, _, _, target_test, _ = build_datasets(args)
            self.assertIsNone(target_test)
            loaders = IndependentDomainLoaders(source, target, args)
            source_batch, target_batch = next(iter(loaders))
            self.assertNotEqual(list(source_batch["pair_id"]), list(target_batch["pair_id"]))

    def test_weighted_ce_uses_fixed_source_weights(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            source, *_ = build_datasets(args)
            criterion = build_classification_criterion(args, source, torch.device("cpu"), 6)
            self.assertTrue(torch.allclose(criterion.weight, torch.tensor(M4SAR_CLASS_WEIGHTS)))

    def test_m4sar_frontends_and_all_baseline_dimensions(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            expected_dims = {
                "dual_d": 16,
                "optical_only": 16,
                "sar_only": 16,
                "simple_concat": 32,
            }
            for mode, classifier_dim in expected_dims.items():
                args.model_mode = mode
                models = build_models(args, 6, torch.device("cpu"))
                self.assertEqual(models.net_vis.stem[0].in_channels, 3)
                self.assertEqual(models.net_ir.stem[0].in_channels, 1)
                first_linear = next(
                    module for module in models.classifier.modules()
                    if isinstance(module, torch.nn.Linear)
                )
                self.assertEqual(first_linear.in_features, classifier_dim)
                if mode != "dual_d":
                    self.assertFalse(any(p.requires_grad for p in models.tal.parameters()))
                    self.assertFalse(
                        any(p.requires_grad for p in models.dual_adapter.parameters())
                    )

    def test_plain_alignment_ablation_preserves_fused_dimension(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            args.model_mode = "dual_d"
            args.alignment_mode = "plain"
            models = build_models(args, 6, torch.device("cpu"))
            source = [torch.randn(4, 16), torch.randn(4, 16)]
            target = [torch.randn(4, 16), torch.randn(4, 16)]
            projected_source, projected_target, loss = models.tal(source, target)
            self.assertEqual([tuple(value.shape) for value in projected_source], [(4, 8), (4, 8)])
            self.assertEqual([tuple(value.shape) for value in projected_target], [(4, 8), (4, 8)])
            self.assertEqual(float(loss), 0.0)

    def test_no_translation_stack_freezes_adapter(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            args.model_mode = "dual_d"
            args.translation_enabled = False
            args.module_c_enabled = False
            args.modality_drift_enabled = False
            models = build_models(args, 6, torch.device("cpu"))
            self.assertFalse(any(p.requires_grad for p in models.dual_adapter.parameters()))
            self.assertTrue(any(p.requires_grad for p in models.tal.parameters()))

    def test_full_preflight_uses_no_target_test(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            report = preflight(args)
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["target_test_opened"])
            self.assertEqual(report["optical_shape"], [4, 3, 32, 32])
            self.assertEqual(report["sar_shape"], [4, 1, 32, 32])
            self.assertEqual(report["fused_shape"], [4, 16])
            self.assertNotEqual(
                report["source_batch_pair_ids"], report["target_batch_pair_ids"]
            )
            self.assertTrue(
                all(
                    status["present"] and status["finite"] and status["nonzero"]
                    for status in report["required_gradients"].values()
                )
            )

    def test_baseline_trainer_defers_target_test_until_after_selection(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_m4sar_fixture(root)
            args = _fixture_args(root, manifest)
            args.model_mode = "optical_only"
            args.output_dir = str(root / "runs")
            args.run_name = "m4sar_baseline_smoke"
            args.artifact_suffix = ""
            args.epochs = 1
            args.monitor_stability_window = 1
            args.checkpoint_selection_min_epoch = 1
            args.lr_scheduler_start_epoch = 1
            args.early_stopping_patience = 0
            args.train_eval_interval = 1
            args.raw_eval_interval = 1
            args.evaluate_target_test = True
            args.save_checkpoints = False
            summary = run_training(args)
            self.assertEqual(summary["source_test"]["total"], 12)
            self.assertEqual(summary["target_test"]["total"], 12)
            run_dir = root / "runs" / "m4sar_baseline_smoke"
            audit_path = run_dir / "data_audit.json"
            with audit_path.open("r", encoding="utf-8") as stream:
                audit = json.load(stream)
            self.assertFalse(audit["target_test_opened_during_selection"])

            with (run_dir / "training_component_status.json").open(
                "r", encoding="utf-8"
            ) as stream:
                component_status = json.load(stream)
            self.assertEqual(component_status["model_mode"], "optical_only")
            self.assertEqual(
                component_status["training_objective"],
                "source_only_classification_baseline",
            )
            self.assertFalse(component_status["tal_active"])
            self.assertFalse(component_status["translator_active"])

            with (run_dir / "metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                metric_row = next(csv.DictReader(stream))
            self.assertEqual(metric_row["train_acc_domain"], "source")
            self.assertEqual(metric_row["train_loss_tal"], "")
            self.assertEqual(metric_row["train_loss_dual_g"], "")
            self.assertEqual(metric_row["train_sampled_minus_full_acc"], "")
            self.assertEqual(
                metric_row["target_train_eval_acc"],
                metric_row["train_full_acc"],
            )
            self.assertIn("target_train_eval", summary["best_metrics"])

            train_log = (run_dir / "train.log").read_text(encoding="utf-8")
            self.assertIn("SOURCE-ONLY BASELINE ACTIVE", train_log)
            self.assertIn("tal n/a", train_log)
            self.assertIn("source_train_acc", train_log)
            self.assertIn("target_val_acc", train_log)

            with (run_dir / "per_class_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                per_class_rows = list(csv.DictReader(stream))
            self.assertEqual(len(per_class_rows), 30)
            self.assertEqual(
                {row["class_name"] for row in per_class_rows},
                {"bridge", "harbor", "oil_tank", "playground", "airport", "wind_turbine"},
            )
            for field in ("accuracy_ovr", "precision", "recall", "f1", "support"):
                self.assertIn(field, per_class_rows[0])


if __name__ == "__main__":
    unittest.main()
