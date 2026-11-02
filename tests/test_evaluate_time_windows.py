import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_time_windows import (
    evaluate_time_windows,
    summarize_time_windows,
)
from src.feature_contract import PRE_CAMPAIGN_FEATURES


def synthetic_data(rows=300):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "age": rng.integers(20, 70, rows),
            "job": rng.choice(["admin", "technician"], rows),
            "marital": rng.choice(["single", "married"], rows),
            "education": rng.choice(["secondary", "tertiary"], rows),
            "default": rng.choice(["no", "yes"], rows),
            "balance": rng.integers(-100, 5000, rows),
            "housing": rng.choice(["no", "yes"], rows),
            "loan": rng.choice(["no", "yes"], rows),
            "pdays": rng.choice([-1, 30, 90], rows),
            "previous": rng.integers(0, 3, rows),
            "poutcome": rng.choice(["unknown", "success", "failure"], rows),
        },
        columns=PRE_CAMPAIGN_FEATURES,
    )
    # Alternate labels so every window contains both classes.
    y = pd.Series(np.arange(rows) % 3 == 0, dtype="int8")
    return X, y


def test_time_windows_are_chronological_and_never_train_on_validation_rows():
    X, y = synthetic_data()

    results = evaluate_time_windows(X, y, n_splits=3)

    assert sorted(results["model"].unique()) == ["Logistic Regression", "XGBoost"]
    for _, model_results in results.groupby("model"):
        model_results = model_results.sort_values("window")
        assert model_results["window"].tolist() == [1, 2, 3]
        # Expanding window: train ends exactly where validation starts.
        assert (model_results["train_rows"] == model_results["val_start"]).all()
        assert (
            model_results["val_start"].iloc[1:].to_numpy()
            == model_results["val_end"].iloc[:-1].to_numpy()
        ).all()
        assert model_results["val_end"].iloc[-1] == len(X)
        assert (
            model_results["support"]
            == model_results["val_end"] - model_results["val_start"]
        ).all()
        assert model_results["val_positive_rate"].between(0, 1).all()


def test_summary_reports_spread_per_model_and_metric():
    results = pd.DataFrame(
        {
            "model": ["XGBoost"] * 3,
            "window": [1, 2, 3],
            "pr_auc": [0.2, 0.4, 0.6],
            "roc_auc": [0.7, 0.7, 0.7],
        }
    )

    summary = summarize_time_windows(results, metrics=["pr_auc", "roc_auc"])

    row = summary.set_index(["model", "metric"]).loc[("XGBoost", "pr_auc")]
    assert row["windows"] == 3
    assert row["mean"] == pytest.approx(0.4)
    assert row["std"] == pytest.approx(0.2)
    assert row["min"] == pytest.approx(0.2)
    assert row["max"] == pytest.approx(0.6)
    assert (
        summary.set_index(["model", "metric"]).loc[("XGBoost", "roc_auc"), "std"] == 0
    )
