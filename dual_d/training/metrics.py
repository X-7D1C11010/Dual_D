"""Metric utilities for standalone Dual_D experiments.

Module purpose:
    Compute classification metrics without relying on external project scripts.
    The implementation uses PyTorch tensors and plain Python values, avoiding a
    hard dependency on scikit-learn.

Public interface:
    - classification_metrics(predictions, labels, num_classes)
"""

from __future__ import annotations

from typing import Dict

import torch


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator with zero-division protection."""

    return numerator / denominator if denominator > 0 else 0.0


def classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, object]:
    """Compute accuracy, macro/micro precision, recall, F1, and confusion matrix.

    Args:
        predictions: Predicted class ids.
        labels: Ground-truth class ids.
        num_classes: Total number of classes.

    Returns:
        Dictionary containing scalar metrics and a nested-list confusion matrix.
    """

    predictions = predictions.detach().cpu().long().view(-1)
    labels = labels.detach().cpu().long().view(-1)
    num_classes = int(num_classes)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_label, pred_label in zip(labels, predictions):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion[true_label, pred_label] += 1

    total = int(confusion.sum().item())
    correct = int(torch.diagonal(confusion).sum().item())
    accuracy = _safe_divide(correct, total)

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    classes_present = []
    for class_idx in range(num_classes):
        tp = float(confusion[class_idx, class_idx].item())
        fp = float(confusion[:, class_idx].sum().item() - tp)
        fn = float(confusion[class_idx, :].sum().item() - tp)
        support = float(confusion[class_idx, :].sum().item())
        if support > 0:
            classes_present.append(class_idx)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    if classes_present:
        precision_macro_present = sum(per_class_precision[idx] for idx in classes_present) / len(classes_present)
        recall_macro_present = sum(per_class_recall[idx] for idx in classes_present) / len(classes_present)
        f1_macro_present = sum(per_class_f1[idx] for idx in classes_present) / len(classes_present)
    else:
        precision_macro_present = 0.0
        recall_macro_present = 0.0
        f1_macro_present = 0.0

    precision_macro = sum(per_class_precision) / num_classes if num_classes else 0.0
    recall_macro = sum(per_class_recall) / num_classes if num_classes else 0.0
    f1_macro = sum(per_class_f1) / num_classes if num_classes else 0.0

    tp_micro = float(torch.diagonal(confusion).sum().item())
    fp_micro = float(confusion.sum(dim=0).sum().item() - tp_micro)
    fn_micro = float(confusion.sum(dim=1).sum().item() - tp_micro)
    precision_micro = _safe_divide(tp_micro, tp_micro + fp_micro)
    recall_micro = _safe_divide(tp_micro, tp_micro + fn_micro)
    f1_micro = _safe_divide(
        2.0 * precision_micro * recall_micro,
        precision_micro + recall_micro,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_macro_present": precision_macro_present,
        "recall_macro_present": recall_macro_present,
        "f1_macro_present": f1_macro_present,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "classes_present": classes_present,
        "confusion_matrix": confusion.tolist(),
        "total": total,
        "correct": correct,
    }

