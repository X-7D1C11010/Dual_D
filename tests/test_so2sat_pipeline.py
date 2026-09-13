"""Regression tests for the minimally invasive So2Sat migration path."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dual_d.config import DualDConfig, LossWeights
from dual_d.data import (
    S1_MEAN,
    S1_STD,
    S2_MEAN,
    S2_STD,
    So2SatLCZ42Dataset,
    stratified_split_indices,
)
from dual_d.integration_adapter import DualDTrainingAdapter
from dual_d.models import (
    Classifier,
    LabelSmoothingCrossEntropy,
    OpticalResNet20Encoder,
    SARResNet20Encoder,
    TensorBasedAlignmentStable,
)
from dual_d.training.trainer import run_training
from scripts.train_dual_d import build_parser, load_json_defaults, run_experiment_matrix
from scripts.preflight_so2sat import run_preflight


def _write_fixture(root: Path, name: str, samples: int = 34) -> tuple[Path, Path]:
    generator = np.random.default_rng(7)
    sar_mean = np.asarray(S1_MEAN, dtype=np.float32).reshape(1, 1, 1, 8)
    sar_std = np.asarray(S1_STD, dtype=np.float32).reshape(1, 1, 1, 8)
    optical_mean = np.asarray(S2_MEAN, dtype=np.float32).reshape(1, 1, 1, 10)
    optical_std = np.asarray(S2_STD, dtype=np.float32).reshape(1, 1, 1, 10)
    sar = sar_mean + sar_std * generator.normal(
        size=(samples, 32, 32, 8)
    ).astype(np.float32)
    optical = optical_mean + optical_std * generator.normal(
        size=(samples, 32, 32, 10)
    ).astype(np.float32)
    sar[0] = sar_mean
    optical[0] = optical_mean
    labels = np.zeros((samples, 17), dtype=np.float64)
    labels[np.arange(samples), np.arange(samples) % 17] = 1.0

    data_path = root / f"{name}.h5"
    geo_path = root / f"{name}_geo.h5"
    with h5py.File(data_path, "w") as data_file:
        data_file.create_dataset("sen1", data=sar)
        data_file.create_dataset("sen2", data=optical)
        data_file.create_dataset("label", data=labels)
    with h5py.File(geo_path, "w") as geo_file:
        geo_file.create_dataset(
            "city",
            data=np.asarray([b"testcity"] * samples, dtype="S12"),
        )
    return data_path, geo_path


class So2SatDatasetTests(unittest.TestCase):
    def test_shapes_labels_metadata_and_source_normalization(self) -> None:
        with TemporaryDirectory() as temporary:
            data_path, geo_path = _write_fixture(Path(temporary), "training")
            dataset = So2SatLCZ42Dataset(
                data_path,
                geo_path,
                domain_role="source",
                augment=False,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["sar"].shape), (8, 32, 32))
            self.assertEqual(tuple(sample["optical"].shape), (10, 32, 32))
            self.assertEqual(int(sample["label"]), 0)
            self.assertEqual(sample["city"], "testcity")
            self.assertEqual(sample["domain_role"], "source")
            self.assertTrue(torch.allclose(sample["sar"], torch.zeros_like(sample["sar"])))
            self.assertTrue(
                torch.allclose(sample["optical"], torch.zeros_like(sample["optical"]))
            )
            dataset.close()

    def test_stratified_adaptation_holdout_has_no_overlap_and_all_classes(self) -> None:
        labels = np.repeat(np.arange(17), 4)
        train_indices, val_indices = stratified_split_indices(labels, 0.25, seed=11)
        self.assertFalse(set(train_indices) & set(val_indices))
        self.assertEqual(set(labels[train_indices].tolist()), set(range(17)))
        self.assertEqual(set(labels[val_indices].tolist()), set(range(17)))

    def test_hdf5_handles_are_safe_with_dataloader_workers(self) -> None:
        with TemporaryDirectory() as temporary:
            data_path, geo_path = _write_fixture(Path(temporary), "training")
            dataset = So2SatLCZ42Dataset(
                data_path,
                geo_path,
                domain_role="source",
                augment=False,
            )
            loader = DataLoader(dataset, batch_size=4, num_workers=2)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["sar"].shape), (4, 8, 32, 32))
            self.assertEqual(tuple(batch["optical"].shape), (4, 10, 32, 32))


class So2SatModelTests(unittest.TestCase):
    def test_independent_encoder_shapes(self) -> None:
        sar_encoder = SARResNet20Encoder(output_dim=256)
        optical_encoder = OpticalResNet20Encoder(output_dim=256)
        sar_features = sar_encoder(torch.randn(2, 8, 32, 32))
        optical_features = optical_encoder(torch.randn(2, 10, 32, 32))
        self.assertEqual(tuple(sar_features.shape), (2, 256))
        self.assertEqual(tuple(optical_features.shape), (2, 256))
        self.assertIsNot(
            next(sar_encoder.parameters()),
            next(optical_encoder.parameters()),
        )

    def test_full_core_forward_and_backward_keeps_original_modules(self) -> None:
        batch_size = 4
        feature_dim = 32
        projection_dim = 16
        sar_encoder = SARResNet20Encoder(output_dim=feature_dim)
        optical_encoder = OpticalResNet20Encoder(output_dim=feature_dim)
        tal = TensorBasedAlignmentStable(
            input_dims=[feature_dim, feature_dim],
            output_dims=[projection_dim, projection_dim],
            num_modalities=2,
        )
        config = DualDConfig(
            feature_dim=2 * projection_dim,
            modality_dims=(projection_dim, projection_dim),
            loss_weights=LossWeights(modality_drift=0.02),
        )
        adapter = DualDTrainingAdapter(config)
        classifier = Classifier(2 * projection_dim, num_classes=17, dropout=0.0)
        criterion = LabelSmoothingCrossEntropy(eps=0.0)

        source_modalities = [
            sar_encoder(torch.randn(batch_size, 8, 32, 32)),
            optical_encoder(torch.randn(batch_size, 10, 32, 32)),
        ]
        target_modalities = [
            sar_encoder(torch.randn(batch_size, 8, 32, 32)),
            optical_encoder(torch.randn(batch_size, 10, 32, 32)),
        ]
        projected_source, projected_target, tal_loss = tal(
            source_modalities, target_modalities
        )
        source_features = torch.cat(projected_source, dim=1)
        target_features = torch.cat(projected_target, dim=1)
        outputs = adapter.forward_features(source_features, target_features)
        source_labels = torch.arange(batch_size) % 17
        target_labels = torch.arange(batch_size) % 17
        generator_loss, logs = adapter.compute_generator_loss(
            outputs,
            classifier=classifier,
            criterion_cls=criterion,
            source_labels=source_labels,
            target_labels=target_labels,
            num_classes=17,
            adversarial_scale=1.0,
            module_c_scale=1.0,
            modality_drift_scale=1.0,
        )
        classification_loss = criterion(classifier(source_features), source_labels)
        total = classification_loss + 0.3 * tal_loss + generator_loss
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIn("dual_d/modality_drift", logs)
        self.assertIsNotNone(sar_encoder.stem[0].weight.grad)
        self.assertIsNotNone(optical_encoder.stem[0].weight.grad)
        self.assertIsNotNone(tal.U_matrices[0].grad)
        self.assertIsNotNone(adapter.coordinator.translator.source_to_target.net[-1].weight.grad)

    def test_one_epoch_so2sat_trainer_keeps_testing_independent(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        defaults = load_json_defaults(project_root / "configs" / "so2sat_lcz42.json")
        args = build_parser(defaults).parse_args([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root, "training")
            _write_fixture(root, "validation")
            _write_fixture(root, "testing")
            args.dataset_root = str(root)
            args.source_root = str(root)
            args.target_root = str(root)
            args.dual_config = str(project_root / "configs" / "dual_d_so2sat.json")
            args.output_dir = str(root / "runs")
            args.run_name = "so2sat_smoke"
            args.artifact_suffix = ""
            args.epochs = 1
            args.batch_size = 17
            args.num_workers = 0
            args.min_steps_per_epoch = 1
            args.feature_dim = 32
            args.proj_dim = 8
            args.target_adapt_val_fraction = 0.5
            args.adversarial_warmup_epochs = 0
            args.adversarial_ramp_epochs = 0
            args.module_c_warmup_epochs = 0
            args.module_c_ramp_epochs = 0
            args.modality_drift_warmup_epochs = 0
            args.modality_drift_ramp_epochs = 0
            args.monitor_stability_window = 1
            args.checkpoint_selection_min_epoch = 1
            args.lr_scheduler_start_epoch = 1
            args.early_stopping_patience = 0
            args.train_eval_interval = 1
            args.raw_eval_interval = 1
            args.save_checkpoints = False
            args.save_feature_embeddings = False
            args.multi_gpu = False
            args.device = "cpu"
            summary = run_training(args)
            self.assertIsNotNone(summary["target_test"])
            self.assertEqual(summary["target_test"]["total"], 34)
            audit_path = root / "runs" / "so2sat_smoke" / "data_audit.json"
            self.assertTrue(audit_path.is_file())
            self.assertTrue(
                (root / "runs" / "so2sat_smoke" / "target_test_metrics.json").is_file()
            )

    def test_real_batch_preflight_contract_on_fixture(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        defaults = load_json_defaults(project_root / "configs" / "so2sat_lcz42.json")
        args = build_parser(defaults).parse_args([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root, "training")
            _write_fixture(root, "validation")
            _write_fixture(root, "testing")
            args.dataset_root = str(root)
            args.dual_config = str(project_root / "configs" / "dual_d_so2sat.json")
            args.batch_size = 17
            args.num_workers = 0
            args.device = "cpu"
            args.feature_dim = 32
            args.proj_dim = 8
            args.target_adapt_val_fraction = 0.5
            report = run_preflight(args)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["shape_trace"]["source_sar"], [17, 8, 32, 32])
            self.assertEqual(report["shape_trace"]["source_optical"], [17, 10, 32, 32])
            self.assertEqual(report["shape_trace"]["source_fused"], [17, 16])
            self.assertIn("dual_d/modality_drift", report["generator_logs"])

    def test_so2sat_smoke_can_skip_official_target_test(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        defaults = load_json_defaults(project_root / "configs" / "so2sat_lcz42.json")
        args = build_parser(defaults).parse_args([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root, "training")
            _write_fixture(root, "validation")
            _write_fixture(root, "testing")
            args.dataset_root = str(root)
            args.dual_config = str(project_root / "configs" / "dual_d_so2sat.json")
            args.output_dir = str(root / "runs")
            args.run_name = "so2sat_no_test_smoke"
            args.device = "cpu"
            args.epochs = 1
            args.batch_size = 17
            args.num_workers = 0
            args.feature_dim = 32
            args.proj_dim = 8
            args.target_adapt_val_fraction = 0.5
            args.checkpoint_selection_min_epoch = 1
            args.monitor_stability_window = 1
            args.evaluate_target_test = False
            batch_summary = run_experiment_matrix(args)
            summary = batch_summary["runs"][0]
            self.assertIsNone(summary["target_test"])
            self.assertEqual(
                batch_summary["domain_statistics"]["so2sat_target"]["test_runs"], 0
            )
            self.assertFalse(
                (root / "runs" / "so2sat_no_test_smoke" / "target_test_metrics.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
