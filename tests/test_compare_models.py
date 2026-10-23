
import numpy as np
import pandas as pd
import pytest

from scripts.compare_models import (
    build_models,
    evaluate_model,
    load_and_split_data,
)


def test_load_and_split_data_has_expected_sizes(monkeypatch):
    """Check the 64/16/20 split and leakage-feature removal."""
    rows = 1000
    df = pd.DataFrame({
        "age": np.arange(rows) % 60 + 18,
        "job": ["admin."] * rows,
        "duration": np.arange(rows) + 1,
        "y": ["yes", "no"] * (rows // 2),
    })

    class FakeDataLoader:
        def __init__(self, path):
            self.path = path

        def load_data(self):
            return df.copy()

    monkeypatch.setattr(
        "scripts.compare_models.DataLoader",
        FakeDataLoader,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        load_and_split_data()
    )

    assert len(X_train) == 640
    assert len(X_val) == 160
    assert len(X_test) == 200

    assert "duration" not in X_train.columns
    assert "y" not in X_train.columns
    assert set(y_train.unique()) == {0, 1}


def test_build_models_returns_both_models():
    X_train = pd.DataFrame({
        "age": [25, 35, 45, 55],
        "job": ["admin.", "technician", "admin.", "services"],
    })

    models = build_models(X_train)

    assert set(models) == {"XGBoost", "Logistic Regression"}


def test_evaluate_logistic_regression_uses_positive_class_probability():
    class FakeModel:
        def predict_proba(self, X):
            return np.array([
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.1, 0.9],
            ])

    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = np.array([0, 1, 0, 1])

    result = evaluate_model(
        "Logistic Regression",
        FakeModel(),
        X,
        y,
    )

    assert result["model"] == "Logistic Regression"
    assert result["roc_auc"] == pytest.approx(1.0)
