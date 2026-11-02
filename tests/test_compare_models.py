import numpy as np
import pandas as pd
import pytest

from scripts.compare_models import (
    build_models,
    build_preprocessor,
    evaluate_model,
    main,
)


def test_build_models_returns_both_models():
    X_train = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "job": ["admin.", "technician", "admin.", "services"],
            "campaign": [1, 2, 1, 3],
            "contact": ["cellular"] * 4,
            "day": [1, 2, 3, 4],
            "month": ["jan"] * 4,
        }
    )

    models = build_models(X_train)

    assert set(models) == {"XGBoost", "Logistic Regression"}
    for model in models.values():
        preprocessor = (
            model.preprocessor
            if hasattr(model, "preprocessor")
            else model.named_steps["preprocessor"]
        )
        preprocessor = getattr(preprocessor, "transformer", preprocessor)
        selected_columns = [
            column for _, _, columns in preprocessor.transformers for column in columns
        ]
        assert not {"campaign", "contact", "day", "month"}.intersection(
            selected_columns
        )


@pytest.mark.parametrize("scale_numeric", [False, True])
def test_comparison_preprocessor_imputes_missing_values(scale_numeric):
    X_train = pd.DataFrame(
        {
            "age": [25.0, np.nan, 45.0],
            "job": ["admin.", None, "services"],
        }
    )

    transformed = build_preprocessor(
        X_train,
        scale_numeric=scale_numeric,
    ).fit_transform(X_train)
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert values.shape[0] == len(X_train)
    assert np.isfinite(values).all()


def test_evaluate_logistic_regression_uses_positive_class_probability():
    class FakeModel:
        def predict_proba(self, X):
            return np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.1, 0.9],
                ]
            )

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


def test_main_uses_validation_only_and_never_evaluates_test(
    monkeypatch,
    tmp_path,
):
    import scripts.compare_models as compare_module

    X_train = pd.DataFrame({"age": [30]})
    X_val = pd.DataFrame({"age": [40]})
    y_train = np.array([0])
    y_val = np.array([1])
    evaluated = []

    class FakeModel:
        def fit(self, X, y):
            assert X is X_train
            assert y is y_train

    monkeypatch.setattr(
        compare_module,
        "load_and_split_data",
        lambda: (X_train, X_val, y_train, y_val),
    )
    monkeypatch.setattr(
        compare_module,
        "build_models",
        lambda X: {"candidate": FakeModel()},
    )

    def fake_evaluate(name, model, X, y):
        evaluated.append((X, y))
        return {"model": name, "roc_auc": 0.5}

    monkeypatch.setattr(compare_module, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(compare_module, "OUTPUT_PATH", tmp_path / "comparison.csv")

    main()

    assert len(evaluated) == 1
    assert evaluated[0][0] is X_val
    assert evaluated[0][1] is y_val
    results = pd.read_csv(tmp_path / "comparison.csv")
    assert results["split"].tolist() == ["validation"]
