
import numpy as np
import pandas as pd
import pytest

from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "job": [
            "admin", "technician", "services", "admin",
            "technician", "services", "admin", "technician",
        ],
        "age": [25, 35, 45, 30, 40, 50, 28, 38],
    })

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    return X, y


def test_model_trainer_builds_pipeline(sample_data):
    X, _ = sample_data

    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )

    trainer = ModelTrainer(
        preprocessor=preprocessor,
        model_params={"n_estimators": 10, "max_depth": 2},
    )

    pipeline = trainer.build_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_model_trainer_fit_and_predict(sample_data):
    X, y = sample_data

    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )

    trainer = ModelTrainer(
        preprocessor=preprocessor,
        model_params={"n_estimators": 10, "max_depth": 2},
    )

    trainer.fit(X, y)

    predictions = trainer.predict(X)
    probabilities = trainer.predict_proba(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})
    assert len(probabilities) == len(y)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_model_trainer_predict_before_fit_raises_error(sample_data):
    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )

    trainer = ModelTrainer(preprocessor=preprocessor)

    with pytest.raises(RuntimeError, match="Model has not been trained yet"):
        trainer.predict(pd.DataFrame({"job": ["admin"], "age": [30]}))
