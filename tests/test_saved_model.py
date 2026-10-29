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
