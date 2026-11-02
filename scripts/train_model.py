import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import joblib

from src.config import MODELS_DIR, SEED, THRESHOLD, TRAIN_FRACTION, XGB_PARAMS
from src.data_loader import load_and_split_data
from src.feature_contract import (
    DROPPED_PRE_CAMPAIGN_FEATURES,
    PRE_CAMPAIGN_FEATURES,
)
from src.model_evaluator import ModelEvaluator
from src.model_trainer import ModelTrainer
from src.preprocessor import DataPreprocessor


def main():
    """Train on earlier rows and report validation metrics only."""
    X_train, X_val, y_train, y_val = load_and_split_data()

    categorical_columns = X_train.select_dtypes(
        include=["str", "object", "category"]
    ).columns.tolist()
    numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = DataPreprocessor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    trainer = ModelTrainer(
        preprocessor=preprocessor,
        model_params=XGB_PARAMS,
    )

    trainer.fit(X_train, y_train)

    print("Training completed!")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Numerical features: {len(numerical_columns)}")

    threshold = THRESHOLD
    evaluator = ModelEvaluator(threshold=threshold)

    y_val_proba = trainer.predict_proba(X_val)
    metrics = evaluator.evaluate(y_val, y_val_proba)

    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    print("\nConfusion matrix:")
    print(evaluator.get_confusion_matrix())

    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgboost_pipeline.joblib"
    metadata_path = models_dir / "metadata.json"

    joblib.dump(trainer.pipeline, model_path)

    metadata = {
        "artifact_version": 1,
        "model_type": "XGBClassifier",
        "feature_names": PRE_CAMPAIGN_FEATURES,
        "threshold": threshold,
        "target_mapping": {"no": 0, "yes": 1},
        "dropped_features": DROPPED_PRE_CAMPAIGN_FEATURES,
        "python_version": sys.version.split()[0],
        "dependency_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost")
        },
        "seed": SEED,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": "UCI Bank Marketing bank-full.csv (ordered rows)",
        "split": {
            "strategy": "ordered",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": round(1 - TRAIN_FRACTION, 2),
        },
        "validation_metrics": metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
