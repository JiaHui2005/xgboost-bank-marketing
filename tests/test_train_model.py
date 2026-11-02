import json
from datetime import datetime

import numpy as np
import pandas as pd

import scripts.train_model as train_module
from src.feature_contract import PRE_CAMPAIGN_FEATURES


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
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")

    train_module.main()

    assert predicted == [X_val]
    assert evaluated == [(y_val, np.array([0.8]))]
    assert len(dumped) == 1
    metadata = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert metadata["validation_metrics"] == {"recall": 1.0}
    assert metadata["artifact_version"] == 1
    assert metadata["model_type"] == "XGBClassifier"
    assert metadata["feature_names"] == PRE_CAMPAIGN_FEATURES
    assert metadata["dropped_features"] == [
        "duration",
        "campaign",
        "contact",
        "day",
        "month",
    ]
    assert metadata["threshold"] == 0.20
    assert metadata["target_mapping"] == {"no": 0, "yes": 1}
    assert metadata["python_version"]
    assert set(metadata["dependency_versions"]) == {
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
    }
    assert all(metadata["dependency_versions"].values())
    assert metadata["seed"] == 42
    trained_at = datetime.fromisoformat(metadata["trained_at_utc"])
    assert trained_at.utcoffset().total_seconds() == 0
    assert metadata["data_version"] == "UCI Bank Marketing bank-full.csv (ordered rows)"
    assert metadata["split"] == {
        "strategy": "ordered",
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
    }
    assert not any("test" in key.lower() for key in metadata)
    assert "Final test metrics" not in capsys.readouterr().out
