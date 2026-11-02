import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    check_training_data,
    encode_binary_target,
    load_and_split_data,
)
from src.feature_contract import PRE_CAMPAIGN_FEATURES


def test_encode_binary_target_maps_expected_labels():
    encoded = encode_binary_target(pd.Series(["no", "yes", "no"]))

    assert encoded.tolist() == [0, 1, 0]
    assert str(encoded.dtype) == "int8"


@pytest.mark.parametrize(
    "labels",
    [["no", None], ["yes", "maybe"], ["no", float("nan")], ["yes", pd.NA], ["no", ""]],
)
def test_encode_binary_target_rejects_missing_or_unknown_labels(labels):
    with pytest.raises(ValueError, match="missing or unsupported labels"):
        encode_binary_target(pd.Series(labels))


def valid_training_frame():
    return pd.DataFrame(
        {
            "age": [30, 45, 52, 30],
            "job": ["admin.", "technician", None, "admin."],
            "marital": ["single", "married", "married", "single"],
            "education": ["secondary", "tertiary", "unknown", "secondary"],
            "default": ["no", "no", "no", "no"],
            "balance": [100, -50, 2000, 100],
            "housing": ["yes", "no", "yes", "yes"],
            "loan": ["no", "no", "yes", "no"],
            "pdays": [-1, 90, -1, -1],
            "previous": [0, 2, 0, 0],
            "poutcome": ["unknown", "success", "unknown", "unknown"],
            "y": ["no", "yes", "no", "no"],
        }
    )


def test_check_training_data_reports_quality_summary():
    report = check_training_data(valid_training_frame())

    assert report["rows"] == 4
    assert report["positive_rate"] == 0.25
    assert report["duplicate_rows"] == 1
    assert report["missing_rate"] == {"job": 0.25}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.drop(columns=["balance"]), "missing required columns"),
        (lambda df: df.assign(age=["30", "x", "52", "30"]), "'age' must be numeric"),
        (lambda df: df.assign(age=[30, -5, 52, 30]), "'age' has values outside"),
        (lambda df: df.assign(pdays=[-1, -7, -1, -1]), "'pdays' has values outside"),
        (lambda df: df.assign(previous=[0, -1, 0, 0]), "'previous' has values outside"),
        (lambda df: df.assign(housing=None), "'housing' is entirely missing"),
        (lambda df: df.assign(y="no"), "both target classes"),
        (lambda df: df.assign(y=["no", "maybe", "no", "no"]), "unsupported labels"),
    ],
)
def test_check_training_data_rejects_invalid_data(change, message):
    with pytest.raises(ValueError, match=message):
        check_training_data(change(valid_training_frame()))


def test_load_and_split_data_uses_ordered_80_20_split(monkeypatch):
    """Check temporal order, 80/20 sizes, and leakage-feature removal."""
    rows = 1000
    df = pd.DataFrame(
        {
            "age": np.full(rows, 40),
            "job": ["admin."] * rows,
            "marital": ["single"] * rows,
            "education": ["secondary"] * rows,
            "default": ["no"] * rows,
            "balance": np.arange(rows),
            "housing": ["no"] * rows,
            "loan": ["no"] * rows,
            "pdays": np.full(rows, -1),
            "previous": np.zeros(rows),
            "poutcome": ["unknown"] * rows,
            "contact": ["cellular"] * rows,
            "day": np.arange(rows) % 31 + 1,
            "month": ["may"] * rows,
            "campaign": np.arange(rows) % 5 + 1,
            "duration": np.arange(rows) + 1,
            "y": ["yes", "no"] * (rows // 2),
        }
    )

    class FakeDataLoader:
        def __init__(self, path):
            self.path = path

        def load_data(self):
            return df.copy()

    monkeypatch.setattr(
        "src.data_loader.DataLoader",
        FakeDataLoader,
    )

    result = load_and_split_data()
    assert len(result) == 4
    X_train, X_val, y_train, y_val = result

    assert len(X_train) == 800
    assert len(X_val) == 200
    assert X_train["balance"].tolist() == list(range(800))
    assert X_val["balance"].tolist() == list(range(800, 1000))

    assert "duration" not in X_train.columns
    assert "y" not in X_train.columns
    assert not {"campaign", "contact", "day", "month"}.intersection(X_train.columns)
    assert set(y_train.unique()) == {0, 1}
    assert y_train.tolist() == [1, 0] * 400
    assert y_val.tolist() == [1, 0] * 100


@pytest.mark.parametrize("rows", [0, 1])
def test_load_and_split_data_rejects_too_few_rows(monkeypatch, rows):
    data = {feature: np.arange(rows) for feature in PRE_CAMPAIGN_FEATURES}
    data["y"] = ["no"] * rows
    df = pd.DataFrame(data)

    class FakeDataLoader:
        def __init__(self, path):
            self.path = path

        def load_data(self):
            return df.copy()

    monkeypatch.setattr("src.data_loader.DataLoader", FakeDataLoader)

    with pytest.raises(ValueError, match="at least two rows"):
        load_and_split_data()
