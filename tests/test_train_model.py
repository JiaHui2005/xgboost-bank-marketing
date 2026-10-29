import json

import numpy as np
import pandas as pd
import pytest

import scripts.train_model as train_module
from src.feature_contract import PRE_CAMPAIGN_FEATURES


def make_dataset(rows):
    data = {
        feature: np.arange(rows) for feature in PRE_CAMPAIGN_FEATURES
    }
    data["y"] = (["no", "yes"] * (rows // 2)) + (["no"] if rows % 2 else [])
    return pd.DataFrame(data)


def test_load_and_split_data_uses_ordered_80_20_split(monkeypatch):
    df = make_dataset(10)

    class FakeDataLoader:
        def __init__(self, path):
            self.path = path

        def load_data(self):
            return df.copy()

    monkeypatch.setattr(train_module, "DataLoader", FakeDataLoader)

    result = train_module.load_and_split_data()

    assert len(result) == 4
    X_train, X_val, y_train, y_val = result
    assert X_train["age"].tolist() == list(range(8))
    assert X_val["age"].tolist() == [8, 9]
    assert y_train.tolist() == [0, 1] * 4
    assert y_val.tolist() == [0, 1]


@pytest.mark.parametrize("rows", [0, 1])
def test_load_and_split_data_rejects_too_few_rows(monkeypatch, rows):
    df = make_dataset(rows)

    class FakeDataLoader:
        def __init__(self, path):
            self.path = path

        def load_data(self):
            return df.copy()

    monkeypatch.setattr(train_module, "DataLoader", FakeDataLoader)

    with pytest.raises(ValueError, match="at least two rows"):
        train_module.load_and_split_data()


def test_training_evaluates_validation_only_and_does_not_score_test(
    monkeypatch,
    tmp_path,
    capsys,
):
    X_train = pd.DataFrame({"age": [10, 20]})
    X_val = pd.DataFrame({"age": [30]})
    y_train = np.array([0, 1])
    y_val = np.array([1])
    predicted = []
    evaluated = []
    dumped = []

    monkeypatch.setattr(
        train_module,
        "load_and_split_data",
        lambda: (X_train, X_val, y_train, y_val),
    )

    class FakePreprocessor:
        def __init__(self, categorical_columns, numerical_columns):
            pass

    class FakeTrainer:
        def __init__(self, preprocessor, model_params):
            self.pipeline = object()

        def fit(self, X, y):
            assert X is X_train
            assert y is y_train

        def predict_proba(self, X):
            predicted.append(X)
            return np.array([0.8])

    class FakeEvaluator:
        def __init__(self, threshold):
            pass

        def evaluate(self, y, probabilities):
            evaluated.append((y, probabilities))
            return {"recall": 1.0}

        def get_confusion_matrix(self):
            return [[0, 0], [0, 1]]

    monkeypatch.setattr(train_module, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(train_module, "ModelTrainer", FakeTrainer)
    monkeypatch.setattr(train_module, "ModelEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        train_module.joblib,
        "dump",
        lambda pipeline, path: dumped.append((pipeline, path)),
    )
    monkeypatch.chdir(tmp_path)

    train_module.main()

    assert predicted == [X_val]
    assert evaluated == [(y_val, np.array([0.8]))]
    assert len(dumped) == 1
    metadata = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert metadata["validation_metrics"] == {"recall": 1.0}
    assert "Final test metrics" not in capsys.readouterr().out
