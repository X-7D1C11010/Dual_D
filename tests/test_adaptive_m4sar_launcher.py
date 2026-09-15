"""Tests for the memory-gated and epoch-adaptive M4-SAR launcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dual_d.training.trainer import (
    parse_adaptive_batch_plan,
    select_adaptive_batch_size,
)
from scripts.train_dual_d import build_parser, load_json_defaults
from scripts.wait_and_train_m4sar import query_gpu_free_memory, select_gpu


class AdaptiveBatchPlanTests(unittest.TestCase):
    def test_plan_parsing_and_selection(self) -> None:
        plan = parse_adaptive_batch_plan("26:128, 10:32,18:64")
        self.assertEqual(plan, ((10.0, 32), (18.0, 64), (26.0, 128)))
        self.assertEqual(select_adaptive_batch_size(plan, 10.0), 32)
        self.assertEqual(select_adaptive_batch_size(plan, 21.5), 64)
        self.assertEqual(select_adaptive_batch_size(plan, 29.0), 128)

    def test_non_monotonic_plan_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must increase"):
            parse_adaptive_batch_plan("10:64,18:32")

    def test_training_parser_accepts_adaptive_m4sar_options(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        defaults = load_json_defaults(project_root / "configs" / "m4sar_classification.json")
        args = build_parser(defaults).parse_args(
            [
                "--adaptive-batch-size",
                "--adaptive-batch-plan",
                "10:16,20:32",
            ]
        )
        self.assertTrue(args.adaptive_batch_size)
        self.assertEqual(args.adaptive_batch_plan, "10:16,20:32")


class GPULauncherTests(unittest.TestCase):
    @patch("scripts.wait_and_train_m4sar.subprocess.run")
    def test_nvidia_smi_rows_and_gpu_selection(self, run_mock) -> None:
        run_mock.return_value = SimpleNamespace(
            stdout="0, 1024, 32607\n1, 20480, 32607\n"
        )
        devices = query_gpu_free_memory()
        self.assertEqual(len(devices), 2)
        self.assertAlmostEqual(float(devices[1]["free_gb"]), 20.0)
        self.assertEqual(int(select_gpu(devices, None)["index"]), 1)
        self.assertEqual(int(select_gpu(devices, {0})["index"]), 0)


if __name__ == "__main__":
    unittest.main()
