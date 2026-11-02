from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.model_trainer import ModelTrainer
from src.preprocessor import DataPreprocessor


def test_paths_are_absolute_and_rooted_at_the_repository():
    repo_root = Path(__file__).resolve().parents[1]

    assert config.ROOT_DIR == repo_root
    for path in (config.DATA_PATH, config.PROCESSED_DIR, config.MODELS_DIR):
        assert path.is_absolute()
        assert repo_root in path.parents


def test_training_settings_are_valid():
    assert 0 < config.THRESHOLD < 1
    assert 0 < config.TRAIN_FRACTION < 1
    assert isinstance(config.SEED, int)
    assert "random_state" not in config.XGB_PARAMS


def test_training_twice_with_the_configured_seed_gives_identical_predictions():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "job": rng.choice(["admin", "technician", "student"], 200),
            "age": rng.integers(20, 70, 200),
        }
    )
    y = pd.Series(rng.integers(0, 2, 200))

    def train_and_predict():
        trainer = ModelTrainer(
            preprocessor=DataPreprocessor(["job"], ["age"]),
            model_params=config.XGB_PARAMS,
        )
        trainer.fit(X, y)
        return trainer.predict_proba(X)

    np.testing.assert_array_equal(train_and_predict(), train_and_predict())
