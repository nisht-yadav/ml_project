"""
Evaluation metrics for the binary tampering classifier.

Positive class is fixed project-wide: tampered = 1, authentic = 0.
This matters for precision/recall/F1, which are not symmetric - e.g.
"recall" here means "of all truly tampered images, how many did we
catch", which is the practically important number for a forensics tool.
"""

from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

POSITIVE_LABEL = 1  # tampered


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
    }


def compute_confusion(y_true: List[int], y_pred: List[int]):
    """Row/col order is fixed to [authentic(0), tampered(1)] so the
    confusion matrix is unambiguous regardless of what's present in a
    given batch."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def print_metrics(metrics: Dict[str, float]) -> None:
    print(f"  Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%  (of predicted-tampered, % actually tampered)")
    print(f"  Recall   : {metrics['recall'] * 100:.2f}%  (of actually-tampered, % correctly caught)")
    print(f"  F1-score : {metrics['f1'] * 100:.2f}%")
