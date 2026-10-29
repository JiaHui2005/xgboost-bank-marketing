import json
from pathlib import Path

import joblib

from src.data_loader import DataLoader, encode_binary_target
from src.feature_contract import (
    DROPPED_PRE_CAMPAIGN_FEATURES,
    PRE_CAMPAIGN_FEATURES,
)
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

def load_and_split_data():
    """Load data and split it chronologically into train and validation sets."""
    loader = DataLoader("data/raw/bank-full.csv")
    df = loader.load_data()

    if len(df) < 2:
        raise ValueError("Dataset must contain at least two rows to split.")

    X = df[PRE_CAMPAIGN_FEATURES].copy()
    y = encode_binary_target(df["y"])
    split_index = int(len(df) * 0.8)

    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def main():
    """Train on earlier rows and report validation metrics only."""
    X_train, X_val, y_train, y_val = load_and_split_data()

    categorical_columns = X_train.select_dtypes(
        include=["str", "object", "category"]
    ).columns.tolist()
    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = DataPreprocessor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    trainer = ModelTrainer(
        preprocessor=preprocessor,
        model_params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    )

    trainer.fit(X_train, y_train)

    print("Training completed!")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Numerical features: {len(numerical_columns)}")

    evaluator = ModelEvaluator(threshold=0.20)

    y_val_proba = trainer.predict_proba(X_val)
    metrics = evaluator.evaluate(y_val, y_val_proba)

    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    print("\nConfusion matrix:")
    print(evaluator.get_confusion_matrix())

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgboost_pipeline.joblib"
    metadata_path = models_dir / "metadata.json"

    joblib.dump(trainer.pipeline, model_path)

    metadata = {
        "model_type": "XGBClassifier",
        "threshold": 0.20,
        "target_mapping": {"no": 0, "yes": 1},
        "dropped_features": DROPPED_PRE_CAMPAIGN_FEATURES,
        "validation_metrics": metrics,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
