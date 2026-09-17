"""Contract tests for the one-command M4-SAR ablation matrix."""

import unittest

from scripts.run_m4sar_ablation_suite import VARIANTS


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


if __name__ == "__main__":
    unittest.main()
