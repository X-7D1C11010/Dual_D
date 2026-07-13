"""Focused regression tests for the training-stability and data-audit changes."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from dual_d.data.audit import audit_dataset_splits, data_audit_errors
from dual_d.data.multimodal_dataset import PairedImageTransform, SampleRecord
from dual_d.training.trainer import _adversarial_scale


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((12, 12, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


class TrainingSafetyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
