
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


class ModelEvaluator:
    """Evaluate classification models using common metrics"""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        self.threshold = threshold

    def evaluate(self, y_true, y_proba) -> dict[str, float]:
        """Calculate classification metrics from positive-class probabilities"""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if len(y_true) != len(y_proba):
            raise ValueError("y_true and y_proba must have the same length")

        if len(y_true) == 0:
            raise ValueError("Evaluation data must not be empty")

        if not np.isfinite(y_proba).all():
            raise ValueError("Predicted probabilities must be finite")

        if ((y_proba < 0) | (y_proba > 1)).any():
            raise ValueError("Predicted probabilities must be between 0 and 1")

        y_pred = (y_proba >= self.threshold).astype(int)

        if len(np.unique(y_true)) != 2:
            raise ValueError("y_true must contain both binary classes")

        self.confusion_matrix_ = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix from the latest evaluation"""
        if not hasattr(self, "confusion_matrix_"):
            raise RuntimeError("Evaluate the model before requesting the confusion matrix")

        return self.confusion_matrix_
