"""Focused regression tests for the training-stability and data-audit changes."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from dual_d.collaborative_training import DualDiscriminatorCoordinator
from dual_d.config import DualDConfig, LossWeights
from dual_d.data.audit import audit_dataset_splits, data_audit_errors
from dual_d.data.multimodal_dataset import PairedImageTransform, SampleRecord
from dual_d.data.paired_sampler import PairedClassSampler
from dual_d.losses import class_prototype_contrastive_loss, paired_contrastive_loss
from dual_d.training.trainer import (
    _adversarial_scale,
    _module_c_scale,
    validate_cuda_architecture,
)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((12, 12, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


class TrainingSafetyTests(unittest.TestCase):
    @patch("dual_d.training.trainer.torch.cuda.get_arch_list", return_value=["sm_86", "sm_90"])
    @patch("dual_d.training.trainer.torch.cuda.get_device_capability", return_value=(8, 9))
    @patch("dual_d.training.trainer.torch.cuda.current_device", return_value=0)
    @patch("dual_d.training.trainer.torch.cuda.is_available", return_value=True)
    def test_cuda_architecture_accepts_same_major_forward_compatibility(
        self, _available, _current, _capability, _arches
    ) -> None:
        validate_cuda_architecture(torch.device("cuda"))

    @patch("dual_d.training.trainer.torch.cuda.get_device_name", return_value="RTX 5090")
    @patch("dual_d.training.trainer.torch.cuda.get_arch_list", return_value=["sm_86", "sm_90"])
    @patch("dual_d.training.trainer.torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("dual_d.training.trainer.torch.cuda.current_device", return_value=0)
    @patch("dual_d.training.trainer.torch.cuda.is_available", return_value=True)
    def test_cuda_architecture_rejects_missing_blackwell_target(
        self, _available, _current, _capability, _arches, _name
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "sm_12x"):
            validate_cuda_architecture(torch.device("cuda"))

    def test_paired_sampler_keeps_partial_batch_and_minimum_steps(self) -> None:
        class Dataset:
            def __init__(self, size: int) -> None:
                self.labels = [0] * size

            def __len__(self) -> int:
                return len(self.labels)

        source = Dataset(10)
        target = Dataset(5)
        natural = PairedClassSampler(source, target, batch_size=4, min_steps_per_epoch=1)
        resampled = PairedClassSampler(source, target, batch_size=4, min_steps_per_epoch=4)

        self.assertEqual(len(natural), 2)
        self.assertEqual(len(resampled), 4)

    def test_paired_transform_reuses_geometric_randomness(self) -> None:
        grid = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
        image = Image.fromarray(np.repeat(grid[..., None], 3, axis=2))
        transform = PairedImageTransform(True, image_size=10, resize_size=16)
        torch.manual_seed(7)
        vis, ir = transform(image, image)

        vis_denormalized = vis * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        vis_denormalized += torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ir_denormalized = ir * 0.5 + 0.5
        self.assertTrue(torch.allclose(vis_denormalized, ir_denormalized, atol=1e-6))

    def test_content_audit_detects_cross_split_duplicate(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train_vis = tmp_path / "train" / "1" / "vis.png"
            train_ir = tmp_path / "train" / "1" / "ir.png"
            val_vis = tmp_path / "val" / "1" / "vis-copy.png"
            val_ir = tmp_path / "val" / "1" / "ir-copy.png"
            for path in (train_vis, train_ir, val_vis, val_ir):
                _write_image(path, 127)

            train_dataset = SimpleNamespace(
                base_dir=tmp_path / "train",
                samples=[SampleRecord(train_vis, train_ir, "1")],
            )
            val_dataset = SimpleNamespace(
                base_dir=tmp_path / "val",
                samples=[SampleRecord(val_vis, val_ir, "1")],
            )
            audit = audit_dataset_splits(train_dataset, val_dataset, hash_contents=True)
            self.assertEqual(audit["content_overlap_vis_count"], 1)
            self.assertEqual(audit["content_overlap_ir_count"], 1)
            self.assertTrue(data_audit_errors(audit))

    def test_adversarial_warmup_and_ramp(self) -> None:
        args = SimpleNamespace(adversarial_warmup_epochs=5, adversarial_ramp_epochs=10)
        self.assertEqual(_adversarial_scale(args, 5), 0.0)
        self.assertEqual(_adversarial_scale(args, 10), 0.5)
        self.assertEqual(_adversarial_scale(args, 15), 1.0)
        self.assertEqual(_adversarial_scale(args, 100), 1.0)

    def test_module_c_warmup_and_ramp(self) -> None:
        args = SimpleNamespace(module_c_warmup_epochs=5, module_c_ramp_epochs=10)
        self.assertEqual(_module_c_scale(args, 5), 0.0)
        self.assertEqual(_module_c_scale(args, 10), 0.5)
        self.assertEqual(_module_c_scale(args, 15), 1.0)

    def test_multi_positive_contrast_is_not_penalized_by_positive_count(self) -> None:
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        candidates = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        anchor_labels = torch.tensor([0, 1])
        candidate_labels = torch.tensor([0, 0, 1, 1])
        loss = paired_contrastive_loss(
            anchors,
            candidates,
            labels=anchor_labels,
            positive_labels=candidate_labels,
            temperature=0.1,
        )
        self.assertLess(float(loss), 1e-3)

    def test_classification_feedback_updates_translator_not_classifier(self) -> None:
        config = DualDConfig(
            feature_dim=4,
            freeze_classifier_during_feedback=True,
            loss_weights=LossWeights(
                classification=1.0,
                adv_primary=0.0,
                adv_auxiliary=0.0,
                cycle=0.0,
                identity=0.0,
                contrastive=0.0,
                prototype_contrastive=0.0,
            ),
        )
        coordinator = DualDiscriminatorCoordinator(config)
        classifier = torch.nn.Linear(4, 2)
        source = torch.randn(4, 4)
        target = torch.randn(4, 4)
        source_labels = torch.tensor([0, 1, 0, 1])
        target_labels = torch.tensor([1, 0, 1, 0])

        outputs = coordinator(source, target)
        loss, _ = coordinator.compute_generator_loss(
            outputs,
            classifier=classifier,
            criterion_cls=torch.nn.CrossEntropyLoss(),
            source_labels=source_labels,
            target_labels=target_labels,
            num_classes=2,
        )
        loss.backward()

        self.assertTrue(all(parameter.grad is None for parameter in classifier.parameters()))
        translator_gradients = [
            parameter.grad for parameter in coordinator.generator_parameters()
        ]
        self.assertTrue(any(gradient is not None for gradient in translator_gradients))

    def test_class_prototype_contrastive_prefers_matching_class(self) -> None:
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prototype_mask = torch.tensor([True, True])
        labels = torch.tensor([0, 1])
        matching = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        swapped = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        matching_loss = class_prototype_contrastive_loss(
            matching,
            labels,
            prototypes,
            prototype_mask,
            temperature=0.1,
        )
        swapped_loss = class_prototype_contrastive_loss(
            swapped,
            labels,
            prototypes,
            prototype_mask,
            temperature=0.1,
        )

        self.assertLess(float(matching_loss), float(swapped_loss))


if __name__ == "__main__":
    unittest.main()
