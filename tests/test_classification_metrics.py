"""Regression tests for global and per-class classification metrics."""

import unittest

import torch

from dual_d.training.metrics import classification_metrics


class ClassificationMetricTests(unittest.TestCase):
    def test_per_class_one_vs_rest_accuracy_precision_recall_and_f1(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        predictions = torch.tensor([0, 1, 1, 1])
        metrics = classification_metrics(predictions, labels, num_classes=2)

        self.assertEqual(metrics["per_class_correct"], [1.0, 2.0])
        self.assertEqual(metrics["per_class_support"], [2.0, 2.0])
        self.assertAlmostEqual(metrics["per_class_accuracy_ovr"][0], 0.75)
        self.assertAlmostEqual(metrics["per_class_accuracy_ovr"][1], 0.75)
        self.assertAlmostEqual(metrics["per_class_precision"][0], 1.0)
        self.assertAlmostEqual(metrics["per_class_recall"][0], 0.5)
        self.assertAlmostEqual(metrics["per_class_f1"][0], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
