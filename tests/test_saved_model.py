
import json
import joblib
import numpy as np

from pathlib import Path

from scripts.train_model import load_and_split_data


def test_saved_pipeline_can_be_loaded_and_predict():
    root = Path(__file__).resolve().parents[1]

    model_path = root / "models" / "xgboost_pipeline.joblib"
    metadata_path = root / "models" / "metadata.json"

    assert model_path.exists()
    assert metadata_path.exists()

    pipeline = joblib.load(model_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    _, X_val, _, _, _, _ = load_and_split_data()

    probabilities = pipeline.predict_proba(X_val.iloc[:5])[:, 1]
    threshold = metadata["threshold"]
    predictions = (probabilities >= threshold).astype(int)

    assert probabilities.shape == (5,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(predictions).issubset({0, 1})
