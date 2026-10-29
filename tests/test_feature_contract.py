import numpy as np
import pandas as pd
import pytest

from scripts.train_model import load_and_split_data

EXPECTED_PRE_CAMPAIGN_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "pdays",
    "previous",
    "poutcome",
]


def test_pre_campaign_features_exclude_current_campaign_fields():
    try:
        from src.feature_contract import PRE_CAMPAIGN_FEATURES
    except ModuleNotFoundError:
        pytest.fail("PRE_CAMPAIGN_FEATURES contract is not defined")

    assert PRE_CAMPAIGN_FEATURES == [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "pdays",
        "previous",
        "poutcome",
    ]
    assert not {
        "duration",
        "campaign",
        "contact",
        "day",
        "month",
    }.intersection(PRE_CAMPAIGN_FEATURES)


def test_training_split_uses_only_pre_campaign_features(monkeypatch):
    rows = 100
    data = {
        feature: np.arange(rows) for feature in EXPECTED_PRE_CAMPAIGN_FEATURES
    }
    data.update(
        {
            "contact": ["cellular"] * rows,
            "day": np.arange(rows) % 31 + 1,
            "month": ["may"] * rows,
            "campaign": np.ones(rows),
            "duration": np.ones(rows),
            "y": ["yes", "no"] * (rows // 2),
        }
    )
    df = pd.DataFrame(data)

    class FakeDataLoader:
        def __init__(self, path):
            self.path = path

        def load_data(self):
            return df.copy()

    monkeypatch.setattr("scripts.train_model.DataLoader", FakeDataLoader)

    X_train, X_val, y_train, y_val = load_and_split_data()

    for split in (X_train, X_val):
        assert split.columns.tolist() == EXPECTED_PRE_CAMPAIGN_FEATURES
    assert len(X_train) == 80
    assert len(X_val) == 20
    assert X_train["age"].tolist() == list(range(80))
    assert X_val["age"].tolist() == list(range(80, 100))
    assert y_train.tolist() == [1, 0] * 40
    assert y_val.tolist() == [1, 0] * 10
