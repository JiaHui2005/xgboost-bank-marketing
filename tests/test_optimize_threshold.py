import numpy as np
import pandas as pd

import scripts.optimize_threshold as threshold_module


def test_evaluate_thresholds_reports_contact_rate_and_confusion_counts():
    results = threshold_module.evaluate_thresholds(
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.8, 0.4, 0.1]),
        [0.5],
    )

    row = results.iloc[0]
    assert row["predicted_positive"] == 2
    assert row["contact_rate"] == 0.5
    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["false_negative"] == 1


def test_evaluate_thresholds_uses_inclusive_threshold_and_handles_no_contacts():
    y_true = np.array([1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.4, 0.1])

    results = threshold_module.evaluate_thresholds(
        y_true,
        probabilities,
        [0.9, 1.0],
    )

    equal_threshold_row = results.iloc[0]
    assert equal_threshold_row["predicted_positive"] == 1
    assert equal_threshold_row["contact_rate"] == 0.25
    assert equal_threshold_row["true_positive"] == 1

    no_contact_row = results.iloc[1]
    assert no_contact_row["predicted_positive"] == 0
    assert no_contact_row["contact_rate"] == 0
    assert no_contact_row["true_positive"] == 0
    assert no_contact_row["false_positive"] == 0
    assert no_contact_row["false_negative"] == 2


def test_main_uses_four_value_validation_split_and_writes_reference_threshold(
    monkeypatch,
    tmp_path,
    capsys,
):
    X_train = pd.DataFrame({"age": [20, 30]})
    X_val = pd.DataFrame({"age": [40, 50]})
    y_train = pd.Series([0, 1])
    y_val = pd.Series([1, 0])

    class FakeModel:
        def fit(self, X, y):
            assert X is X_train
            assert y is y_train

        def predict_proba(self, X):
            assert X is X_val
            return np.array([0.7, 0.1])

    monkeypatch.setattr(
        threshold_module,
        "load_and_split_data",
        lambda: (X_train, X_val, y_train, y_val),
    )
    monkeypatch.setattr(
        threshold_module,
        "build_models",
        lambda X: {"XGBoost": FakeModel()},
    )
    output_path = tmp_path / "threshold_comparison.csv"
    monkeypatch.setattr(threshold_module, "OUTPUT_PATH", output_path)

    threshold_module.main()

    results = pd.read_csv(output_path)
    current = results.loc[results["threshold"].eq(0.20)].iloc[0]
    assert current["contact_rate"] == 0.5
    assert "Best threshold" not in capsys.readouterr().out
