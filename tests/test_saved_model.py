
import json
import pytest

from pathlib import Path

from scripts.predict import load_artifacts


def test_stale_saved_pipeline_is_rejected_for_pre_campaign_inference():
    root = Path(__file__).resolve().parents[1]

    model_path = root / "models" / "xgboost_pipeline.joblib"
    metadata_path = root / "models" / "metadata.json"

    assert model_path.exists()
    assert metadata_path.exists()

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert model_path.exists()
    assert metadata["threshold"] == 0.2

    with pytest.raises(RuntimeError, match="pre-campaign feature contract"):
        load_artifacts()
