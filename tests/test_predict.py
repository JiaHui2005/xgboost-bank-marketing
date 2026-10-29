
import pandas as pd
import numpy as np
import pytest
import json
from unittest.mock import patch
from types import SimpleNamespace

import scripts.predict as predict_module
from scripts.predict import REQUIRED_COLUMNS, validate_input_data
from src.feature_contract import DROPPED_PRE_CAMPAIGN_FEATURES


def valid_metadata():
    return {
        "artifact_version": 1,
        "model_type": "XGBClassifier",
        "feature_names": REQUIRED_COLUMNS.copy(),
        "dropped_features": DROPPED_PRE_CAMPAIGN_FEATURES.copy(),
        "threshold": 0.2,
        "target_mapping": {"no": 0, "yes": 1},
        "python_version": "3.12.0",
        "dependency_versions": {
            "numpy": "2.0.0",
            "pandas": "2.0.0",
            "scikit-learn": "1.0.0",
            "xgboost": "2.0.0",
        },
        "seed": 42,
        "trained_at_utc": "2026-10-29T00:00:00+00:00",
        "data_version": "UCI Bank Marketing bank-full.csv (ordered rows)",
        "split": {
            "strategy": "ordered",
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
        },
        "validation_metrics": {"average_precision": 0.5},
    }


def pipeline_with_feature_columns(columns=REQUIRED_COLUMNS):
    numerical = [column for column in columns if column in predict_module.NUMERICAL_COLUMNS]
    categorical = [column for column in columns if column not in numerical]
    return SimpleNamespace(
        named_steps={
            "preprocessor": SimpleNamespace(
                numerical_columns=numerical,
                categorical_columns=categorical,
            )
        }
    )


def create_valid_dataframe():
    """Create a minimal valid input dataframe"""
    data = {
        column: [30]
        for column in REQUIRED_COLUMNS
    }

    data.update(
        {
            "job": ["technician"],
            "marital": ["single"],
            "education": ["secondary"],
            "default": ["no"],
            "housing": ["yes"],
            "loan": ["no"],
            "contact": ["cellular"],
            "month": ["may"],
            "poutcome": ["unknown"],
        }
    )

    return pd.DataFrame(data)


def test_validate_input_data_accepts_valid_dataframe():
    df = create_valid_dataframe()

    validate_input_data(df)


def test_validate_input_data_does_not_require_campaign_before_campaign():
    df = create_valid_dataframe()

    assert "campaign" not in REQUIRED_COLUMNS
    validate_input_data(df)


def test_prediction_uses_only_pre_campaign_features_and_ignores_excluded_columns(
    tmp_path,
    monkeypatch,
):
    df = create_valid_dataframe()
    df["contact"] = ["cellular"]
    df["day"] = [1]
    df["month"] = ["may"]
    df["campaign"] = [2]
    df["duration"] = [100]
    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, sep=";", index=False)
    monkeypatch.setattr(
        predict_module,
        "parse_args",
        lambda: type("Args", (), {"input": str(input_path)})(),
    )
    monkeypatch.setattr(predict_module, "ROOT_DIR", tmp_path)

    class FakePipeline:
        def predict_proba(self, features):
            assert features.columns.tolist() == REQUIRED_COLUMNS
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr(
        predict_module,
        "load_artifacts",
        lambda: (FakePipeline(), {"threshold": 0.2}),
    )

    predict_module.main()

    output = pd.read_csv(
        tmp_path / "data" / "processed" / "predictions.csv",
        sep=";",
    )
    assert output.loc[0, "prediction_label"] == "yes"
    assert {"contact", "day", "month", "campaign", "duration"}.issubset(
        output.columns
    )

    # The source CSV may contain this post-campaign field; it is ignored.
    df["campaign"] = [3]
    validate_input_data(df)


def test_validate_input_data_rejects_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    with pytest.raises(
        ValueError,
        match="Input CSV contains no data rows",
    ):
        validate_input_data(df)


def test_validate_input_data_rejects_missing_columns():
    df = create_valid_dataframe().drop(columns=["age", "balance"])

    with pytest.raises(
        ValueError,
        match="Input CSV is missing required columns",
    ):
        validate_input_data(df)


def test_validate_input_data_allows_missing_values_for_pipeline_imputation():
    df = create_valid_dataframe()
    df.loc[0, "age"] = None
    df.loc[0, "job"] = None

    validate_input_data(df)


def test_validate_input_data_allows_extra_columns():
    df = create_valid_dataframe()
    df["customer_id"] = ["C001"]

    validate_input_data(df)

    assert "customer_id" in df.columns


def test_validate_input_data_rejects_invalid_numeric_type():
    df = create_valid_dataframe()
    df["age"] = ["abc"]

    with pytest.raises(
        ValueError,
        match="Column 'age' must contain numerical values",
    ):
        validate_input_data(df)


def test_validate_input_data_rejects_infinite_values():
    df = create_valid_dataframe()
    df["balance"] = [float("inf")]

    with pytest.raises(
        ValueError,
        match="contains infinite values",
    ):
        validate_input_data(df)


def test_load_artifacts_rejects_missing_threshold(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"model_type": "XGBoost"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        predict_module,
        "MODEL_PATH",
        tmp_path / "fake_model.joblib",
    )
    monkeypatch.setattr(
        predict_module,
        "METADATA_PATH",
        metadata_path,
    )

    # The fake model path passes the existence check.
    (tmp_path / "fake_model.joblib").write_bytes(b"fake")

    with patch.object(predict_module.joblib, "load", return_value=object()):
        with pytest.raises(
            ValueError,
            match="Metadata is missing the 'threshold' field",
        ):
            predict_module.load_artifacts()


@pytest.mark.parametrize(
    "invalid_threshold",
    [-0.1, 1.5, float("nan"), float("inf"), "abc", None, True],
)
def test_load_artifacts_rejects_invalid_threshold(
    tmp_path,
    monkeypatch,
    invalid_threshold,
):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"threshold": invalid_threshold}),
        encoding="utf-8",
    )

    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(predict_module.joblib, "load", return_value=object()):
        with pytest.raises(
            ValueError,
            match="Metadata threshold must be a finite number",
        ):
            predict_module.load_artifacts()


def test_load_artifacts_rejects_invalid_json(tmp_path, monkeypatch):
    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"threshold": ',
        encoding="utf-8",
    )

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(predict_module.joblib, "load", return_value=object()):
        with pytest.raises(
            RuntimeError,
            match="Failed to load model artifacts",
        ):
            predict_module.load_artifacts()


def test_load_artifacts_rejects_non_object_metadata(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps([0.2, 0.5, 0.8]),
        encoding="utf-8",
    )

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(predict_module.joblib, "load", return_value=object()):
        with pytest.raises(
            ValueError,
            match="Metadata must be a JSON object",
        ):
            predict_module.load_artifacts()


def test_load_artifacts_rejects_model_trained_with_campaign(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(valid_metadata()), encoding="utf-8")

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(
        predict_module.joblib,
        "load",
        return_value=pipeline_with_feature_columns(["age", "campaign"]),
    ):
        with pytest.raises(RuntimeError, match="Retrain the model"):
            predict_module.load_artifacts()


def test_load_artifacts_rejects_metadata_for_model_using_campaign(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")
    metadata_path = tmp_path / "metadata.json"
    metadata = valid_metadata()
    metadata["dropped_features"] = ["duration"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(predict_module.joblib, "load", return_value=object()):
        with pytest.raises(RuntimeError, match="Retrain the model"):
            predict_module.load_artifacts()


def test_load_artifacts_rejects_metadata_missing_current_contact_exclusions(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")
    metadata_path = tmp_path / "metadata.json"
    metadata = valid_metadata()
    metadata["dropped_features"] = ["duration", "campaign"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(predict_module.joblib, "load", return_value=object()):
        with pytest.raises(RuntimeError, match="pre-campaign feature contract"):
            predict_module.load_artifacts()


@pytest.mark.parametrize(
    ("change", "expected_error"),
    [
        ("missing_version", ValueError),
        ("wrong_version", ValueError),
        ("boolean_version", ValueError),
        ("feature_order", RuntimeError),
        ("exclusions", RuntimeError),
        ("pipeline_schema", RuntimeError),
    ],
)
def test_load_artifacts_rejects_incompatible_versioned_contract(
    tmp_path,
    monkeypatch,
    change,
    expected_error,
):
    metadata = valid_metadata()
    pipeline = pipeline_with_feature_columns()
    if change == "missing_version":
        del metadata["artifact_version"]
    elif change == "wrong_version":
        metadata["artifact_version"] = 2
    elif change == "boolean_version":
        metadata["artifact_version"] = True
    elif change == "feature_order":
        metadata["feature_names"] = metadata["feature_names"][::-1]
    elif change == "exclusions":
        metadata["dropped_features"] = ["duration"]
    elif change == "pipeline_schema":
        pipeline = pipeline_with_feature_columns(REQUIRED_COLUMNS[:-1] + ["campaign"])

    model_path = tmp_path / "fake_model.joblib"
    model_path.write_bytes(b"fake")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "METADATA_PATH", metadata_path)

    with patch.object(predict_module.joblib, "load", return_value=pipeline):
        with pytest.raises(expected_error):
            predict_module.load_artifacts()
