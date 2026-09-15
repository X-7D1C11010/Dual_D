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
            audit_path = root / "runs" / "m4sar_baseline_smoke" / "data_audit.json"
            with audit_path.open("r", encoding="utf-8") as stream:
                audit = json.load(stream)
            self.assertFalse(audit["target_test_opened_during_selection"])


if __name__ == "__main__":
    unittest.main()
