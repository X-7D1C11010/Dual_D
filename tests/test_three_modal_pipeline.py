"""Regression tests for AIS integration and three-modality tensor alignment."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from dual_d.data.ais_signal import load_ais_signal
from dual_d.data.multimodal_dataset import MultiModalDomainDataset
from dual_d.losses import paired_contrastive_loss
from dual_d.models.backbones import AISFeatureExtractor
from dual_d.models.tensor_alignment import TensorBasedAlignmentStable
from dual_d.models.backbones import LabelSmoothingCrossEntropy
from dual_d.training.trainer import build_models, extract_fused_features
from scripts.train_dual_d import build_parser, resolve_experiments, run_experiment_matrix


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 12, 3), value, dtype=np.uint8)).save(path)


def _write_domain(root: Path, phases: list[str], offset: int) -> None:
    for phase_index, phase in enumerate(phases):
        for class_index, class_name in enumerate(("0", "1")):
            stem = f"sample_{class_name}"
            value = offset + phase_index * 30 + class_index * 60
            _write_rgb(root / phase / "可见光" / class_name / f"{stem}.png", value)
            _write_rgb(root / phase / "红外" / class_name / f"{stem}.png", value + 10)
            ais_path = root / phase / "AIS" / class_name / f"{stem}.npy"
            ais_path.parent.mkdir(parents=True, exist_ok=True)
            phase_axis = np.linspace(0.0, np.pi, 16, dtype=np.float32)
            np.save(
                ais_path,
                np.exp(1j * (phase_axis + class_index)).astype(np.complex64),
                allow_pickle=False,
            )


class ThreeModalPipelineTests(unittest.TestCase):
    def test_complex_ais_loader_and_encoder(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npy"
            signal = np.exp(1j * np.linspace(0.0, np.pi, 17)).astype(np.complex64)
            np.save(path, signal, allow_pickle=False)
            tensor = load_ais_signal(path, sequence_length=32, normalize=True)

        self.assertEqual(tuple(tensor.shape), (2, 32))
        self.assertTrue(bool(torch.isfinite(tensor).all()))
        encoder = AISFeatureExtractor("complex", sequence_length=32, output_dim=16)
        features = encoder(tensor.unsqueeze(0))
        self.assertEqual(tuple(features.shape), (1, 16))

    def test_dataset_returns_aligned_ais_tensor(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_rgb(root / "train" / "可见光" / "0" / "sample.png", 32)
            _write_rgb(root / "train" / "红外" / "0" / "sample.png", 96)
            ais_path = root / "train" / "AIS" / "0" / "sample.npy"
            ais_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(ais_path, np.stack([np.arange(8), np.arange(8) * 2], axis=0))

            dataset = MultiModalDomainDataset(
                root,
                phase="train",
                layout="modality_first",
                image_size=8,
                resize_size=10,
                train_augment=False,
                ais_sequence_length=16,
                ais_match="stem",
            )
            sample = dataset[0]

        self.assertEqual(tuple(sample["vis"].shape), (3, 8, 8))
        self.assertEqual(tuple(sample["ir"].shape), (3, 8, 8))
        self.assertEqual(tuple(sample["ais"].shape), (2, 16))
        self.assertTrue(sample["ais_path"].endswith("sample.npy"))

    def test_factorized_contraction_matches_explicit_tensor(self) -> None:
        torch.manual_seed(3)
        tal = TensorBasedAlignmentStable([3, 4, 2], [2, 3, 2], num_modalities=3)
        modalities = [torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 2)]

        for mode in range(3):
            explicit = tal.create_multimodal_tensor(modalities)
            for idx in range(3):
                if idx != mode:
                    explicit = tal.mode_n_product(explicit, tal.U_matrices[idx], idx)
            explicit = tal.tensor_contraction(explicit, mode)
            factorized = tal.factorized_tensor_contraction(
                modalities,
                tal.U_matrices,
                mode,
            )
            self.assertTrue(torch.allclose(explicit, factorized, atol=1e-5))

        source = [torch.randn(4, 3), torch.randn(4, 4), torch.randn(4, 2)]
        target = [torch.randn(4, 3), torch.randn(4, 4), torch.randn(4, 2)]
        projected_source, projected_target, loss = tal(source, target)
        self.assertEqual([tuple(item.shape) for item in projected_source], [(4, 2), (4, 3), (4, 2)])
        self.assertEqual([tuple(item.shape) for item in projected_target], [(4, 2), (4, 3), (4, 2)])
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_paired_contrastive_uses_separate_domain_labels(self) -> None:
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        candidates = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        anchor_labels = torch.tensor([10, 20])
        candidate_labels = torch.tensor([20, 10])
        correct = paired_contrastive_loss(
            anchors,
            candidates,
            labels=anchor_labels,
            positive_labels=candidate_labels,
            temperature=0.1,
        )
        incorrect = paired_contrastive_loss(
            anchors,
            candidates,
            labels=anchor_labels,
            positive_labels=anchor_labels,
            temperature=0.1,
        )
        self.assertLess(float(correct), float(incorrect))

    def test_four_domain_matrix_resolution(self) -> None:
        args = SimpleNamespace(
            target_root="",
            target_parent_root="/data/targets",
            target_domains=["黑天", "逆光", "雾天", "雨天"],
            target_ais_root="",
            target_ais_parent_root="/data/ais",
        )
        experiments = resolve_experiments(args)
        self.assertEqual(len(experiments), 4)
        self.assertEqual(experiments[0][0], "黑天")
        self.assertEqual(experiments[0][1], Path("/data/targets") / "黑天")
        self.assertEqual(experiments[0][2], str(Path("/data/ais") / "黑天"))

    def test_three_modal_forward_and_loss_backward(self) -> None:
        args = SimpleNamespace(
            dual_config=str(Path("configs") / "dual_d_default_config.json"),
            proj_dim=4,
            feature_dim=16,
            pretrained_visual=False,
            freeze_visual_backbone=False,
            ais_encoder="complex",
            ais_sequence_length=32,
            ais_dropout=0.0,
            classifier_dropout=0.0,
        )
        models = build_models(args, num_classes=2, device=torch.device("cpu"))
        source = {
            "vis": torch.randn(2, 3, 64, 64),
            "ir": torch.randn(2, 3, 64, 64),
            "ais": torch.randn(2, 2, 32),
        }
        target = {
            "vis": torch.randn(2, 3, 64, 64),
            "ir": torch.randn(2, 3, 64, 64),
            "ais": torch.randn(2, 2, 32),
        }
        source_labels = torch.tensor([0, 1])
        target_labels = torch.tensor([0, 1])
        feat_src, feat_tgt, loss_tal = extract_fused_features(
            models,
            source,
            target,
            torch.device("cpu"),
        )
        self.assertEqual(tuple(feat_src.shape), (2, 12))
        self.assertEqual(tuple(feat_tgt.shape), (2, 12))

        outputs = models.dual_adapter.forward_features(feat_src, feat_tgt)
        criterion = LabelSmoothingCrossEntropy(eps=0.1)
        loss_generator, _ = models.dual_adapter.compute_generator_loss(
            outputs,
            classifier=models.classifier,
            criterion_cls=criterion,
            source_labels=source_labels,
            target_labels=target_labels,
            num_classes=2,
            adversarial_scale=0.25,
        )
        loss_classification = criterion(models.classifier(feat_src), source_labels)
        loss = loss_tal + loss_generator + loss_classification
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(next(models.net_ais.parameters()).grad)

    def test_one_epoch_three_modal_training_smoke(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "晴天"
            target_root = root / "黑天"
            _write_domain(source_root, ["train"], 20)
            _write_domain(target_root, ["train", "val"], 120)
            args = build_parser({}).parse_args(
                [
                    "--source-root",
                    str(source_root),
                    "--target-root",
                    str(target_root),
                    "--output-dir",
                    str(root / "runs"),
                    "--run-name",
                    "smoke",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--image-size",
                    "32",
                    "--resize-size",
                    "36",
                    "--feature-dim",
                    "8",
                    "--proj-dim",
                    "2",
                    "--ais-sequence-length",
                    "16",
                    "--no-pretrained-visual",
                    "--no-freeze-visual-backbone",
                    "--no-data-audit-hashes",
                    "--early-stopping-patience",
                    "0",
                    "--adversarial-warmup-epochs",
                    "1",
                ]
            )
            batch_summary = run_experiment_matrix(args)
            summary = batch_summary["runs"][0]
            run_dir = Path(summary["run_dir"])
            self.assertTrue(Path(batch_summary["summary_path"]).is_file())
            self.assertTrue((run_dir / "checkpoints" / "last_model.pt").is_file())
            self.assertTrue((run_dir / "result_summary.json").is_file())
            checkpoint = torch.load(
                run_dir / "checkpoints" / "last_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertIn("net_ais", checkpoint)
            self.assertEqual(checkpoint["tal"]["U_matrices.2"].shape, torch.Size([8, 2]))


if __name__ == "__main__":
    unittest.main()
