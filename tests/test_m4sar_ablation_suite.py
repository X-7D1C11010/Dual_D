"""Contract tests for the one-command M4-SAR ablation matrix."""

import unittest

import numpy as np

from scripts.run_m4sar_ablation_suite import VARIANTS, build_parser
from scripts.visualize_m4sar_ablation import _relation_statistics


class M4SARAblationSuiteTests(unittest.TestCase):
    def test_matrix_contains_full_and_four_requested_ablations(self) -> None:
        self.assertEqual(
            list(VARIANTS),
            [
                "full",
                "no_tal",
                "no_translation_stack",
                "no_module_c",
                "no_modality_drift",
            ],
        )

    def test_translation_stack_ablation_disables_all_dependent_constraints(self) -> None:
        flags = VARIANTS["no_translation_stack"]
        self.assertIn("--no-translation-enabled", flags)
        self.assertIn("--no-module-c-enabled", flags)
        self.assertIn("--no-modality-drift-enabled", flags)

    def test_no_tal_keeps_translation_stack_active(self) -> None:
        flags = VARIANTS["no_tal"]
        self.assertEqual(flags[flags.index("--alignment-mode") + 1], "plain")
        self.assertIn("--translation-enabled", flags)
        self.assertIn("--module-c-enabled", flags)
        self.assertIn("--modality-drift-enabled", flags)

    def test_parallel_runner_accepts_two_exclusive_gpus(self) -> None:
        args = build_parser().parse_args(
            ["--parallel-runs", "2", "--gpu-ids", "0", "1", "--dry-run"]
        )
        self.assertEqual(args.parallel_runs, 2)
        self.assertEqual(args.gpu_ids, [0, 1])

    def test_visualized_relation_drift_matches_training_definition(self) -> None:
        sar = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=np.float32,
        )
        # An orthogonal modality basis has the same sample-relation matrix.
        optical = sar[:, ::-1]
        original = np.concatenate([sar, optical], axis=1)
        translated = np.concatenate([sar, optical[[1, 0, 3, 2]]], axis=1)

        statistics = _relation_statistics(original, translated, margin=0.01)

        self.assertAlmostEqual(statistics["before"], 0.0, places=6)
        self.assertGreater(statistics["after"], statistics["before"])
        self.assertAlmostEqual(
            statistics["signed_drift"],
            statistics["after"] - statistics["before"],
            places=7,
        )
        self.assertAlmostEqual(
            float(np.mean(statistics["per_sample_signed_drift"])),
            statistics["signed_drift"],
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
