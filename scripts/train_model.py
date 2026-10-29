import json
from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader, encode_binary_target
from src.feature_contract import (
    DROPPED_PRE_CAMPAIGN_FEATURES,
    PRE_CAMPAIGN_FEATURES,
)
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

def load_and_split_data():
    """Load data, remove leakage-prone feature, and split datasets"""
    loader = DataLoader("data/raw/bank-full.csv")
    df = loader.load_data()

    X = df[PRE_CAMPAIGN_FEATURES].copy()
    y = encode_binary_target(df["y"])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    categorical_columns = X_train.select_dtypes(
        include=["str", "object", "category"]
    ).columns.tolist()
    categorical_columns = [
        column for column in categorical_columns if column != "campaign"
    ]

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()
    numerical_columns = [
        column for column in numerical_columns if column != "campaign"
    ]

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

    saved_pipeline = joblib.load(model_path)
    y_test_proba = saved_pipeline.predict_proba(X_test)[:, 1]

    test_evaluator = ModelEvaluator(threshold=0.20)
    test_metrics = test_evaluator.evaluate(y_test, y_test_proba)

    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    print("\nTest confusion matrix:")
    print(test_evaluator.get_confusion_matrix())
