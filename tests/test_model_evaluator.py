
import numpy as np
import pytest

from src.model_evaluator import ModelEvaluator


def test_evaluator_calculates_metrics_correctly():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.8, 0.7, 0.9])

    evaluator = ModelEvaluator(threshold=0.5)
    metrics = evaluator.evaluate(y_true, y_proba)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1

    np.testing.assert_array_equal(
        evaluator.get_confusion_matrix(),
        np.array([[1, 1], [0, 2]]),
    )


def test_evaluator_uses_custom_threshold():
    y_true = np.array([0, 1])
    y_proba = np.array([0.3, 0.4])

    evaluator = ModelEvaluator(threshold=0.2)
    metrics = evaluator.evaluate(y_true, y_proba)

    assert metrics["recall"] == pytest.approx(1.0)


def test_evaluator_rejects_invalid_probabilities():
    evaluator = ModelEvaluator()

    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluator.evaluate(
            np.array([0, 1]),
            np.array([0.2, 1.5]),
        )


def test_evaluator_requires_both_classes():
    evaluator = ModelEvaluator()

    with pytest.raises(ValueError, match="both binary classes"):
        evaluator.evaluate(
            np.array([0, 0]),
            np.array([0.1, 0.2]),
        )


def test_confusion_matrix_requires_evaluation():
    evaluator = ModelEvaluator()

    with pytest.raises(RuntimeError, match="Evaluate the model"):
        evaluator.get_confusion_matrix()
