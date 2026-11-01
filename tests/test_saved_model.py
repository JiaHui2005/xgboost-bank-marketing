import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.predict import load_artifacts
from src.feature_contract import PRE_CAMPAIGN_FEATURES


def test_saved_model_supports_pre_campaign_inference_with_missing_and_new_values():
    root = Path(__file__).resolve().parents[1]
    with (root / "models" / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    pipeline, loaded_metadata = load_artifacts()

    assert loaded_metadata == metadata
    assert metadata["artifact_version"] == 1
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
    assert metadata["seed"] == 42
    assert metadata["split"] == {
        "strategy": "ordered",
        "train_fraction": 0.8,
        "validation_fraction": 0.2,
    }
    assert metadata["validation_metrics"]
    assert "test_metrics" not in metadata
    assert set(pipeline.named_steps["preprocessor"].categorical_columns).union(
        pipeline.named_steps["preprocessor"].numerical_columns
    ) == set(PRE_CAMPAIGN_FEATURES)

    features = pd.DataFrame(
        [
            {
                "age": np.nan,
                "job": "category-not-seen-in-training",
                "marital": "single",
                "education": "secondary",
                "default": "no",
                "balance": 1000,
                "housing": "yes",
                "loan": "no",
                "pdays": -1,
                "previous": 0,
                "poutcome": "new-outcome",
            }
        ],
        columns=PRE_CAMPAIGN_FEATURES,
    )
    probabilities = pipeline.predict_proba(features)[:, 1]

    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()


def test_saved_model_predicts_from_csv_with_empty_fields_and_nullable_dtypes():
    from io import StringIO

    from scripts.predict import NUMERICAL_COLUMNS, validate_input_data

    row = {
        "age": "30", "job": "technician", "marital": "single",
        "education": "secondary", "default": "no", "balance": "100",
        "housing": "yes", "loan": "no", "pdays": "-1", "previous": "0",
        "poutcome": "unknown",
    }
    # Row 1: blank numeric and categorical fields; `job` is blank in every row.
    lines = [";".join(PRE_CAMPAIGN_FEATURES)]
    for blanks in ({"job", "age"}, {"job"}):
        lines.append(";".join("" if c in blanks else row[c] for c in PRE_CAMPAIGN_FEATURES))
    from_csv = pd.read_csv(StringIO("\n".join(lines)), sep=";")

    nullable = from_csv.copy()
    nullable["marital"] = pd.array([pd.NA, "single"], dtype="string")
    nullable["balance"] = pd.array([pd.NA, 100], dtype="Int64")

    pipeline, _ = load_artifacts()
    for df in (from_csv, nullable):
        validate_input_data(df)
        features = df[PRE_CAMPAIGN_FEATURES].copy()
        for column in NUMERICAL_COLUMNS:
            features[column] = pd.to_numeric(features[column])
        probabilities = pipeline.predict_proba(features)[:, 1]

        assert np.isfinite(probabilities).all()
        assert ((0 <= probabilities) & (probabilities <= 1)).all()
