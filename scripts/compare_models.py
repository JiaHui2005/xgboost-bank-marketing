
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import DataLoader, encode_binary_target
from src.feature_contract import PRE_CAMPAIGN_FEATURES
from src.model_evaluator import ModelEvaluator
from src.model_trainer import ModelTrainer


RANDOM_STATE = 42
THRESHOLD = 0.20
OUTPUT_PATH = Path("data/processed/model_comparison.csv")


def load_and_split_data():
    """Load data and create reproducible stratified train/validation/test splits"""
    loader = DataLoader("data/raw/bank-full.csv")
    df = loader.load_data()

    X = df[PRE_CAMPAIGN_FEATURES].copy()
    y = encode_binary_target(df["y"])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train, scale_numeric=False):
    """Build a preprocessing transformer for categorical and numeric features"""
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    categorical_columns = [
        column for column in categorical_columns
        if column in PRE_CAMPAIGN_FEATURES
    ]

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()
    numerical_columns = [
        column for column in numerical_columns
        if column in PRE_CAMPAIGN_FEATURES
    ]

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numerical_steps.append(("scaler", StandardScaler()))
    numerical_transformer = Pipeline(steps=numerical_steps)

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                categorical_columns,
            ),
            (
                "numerical",
                numerical_transformer,
                numerical_columns,
            ),
        ]
    )


def build_models(X_train):
    """Build XGBoost and Logistic Regression models"""
    xgb_trainer = ModelTrainer(
        preprocessor=build_preprocessor(X_train),
        model_params={
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    )

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X_train, scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "XGBoost": xgb_trainer,
        "Logistic Regression": logistic_pipeline,
    }


def evaluate_model(name, model, X, y):
    """Evaluate a fitted model using the configured classification threshold"""
    probabilities = model.predict_proba(X)

    # ModelTrainer returns probabilities in the format expected by
    # ModelEvaluator; sklearn classifiers return an (n_samples, 2) matrix.
    if name != "XGBoost":
        probabilities = probabilities[:, 1]

    evaluator = ModelEvaluator(threshold=THRESHOLD)
    metrics = evaluator.evaluate(y, probabilities)

    return {
        "model": name,
        **metrics,
    }


def main():
    # The test split is intentionally discarded here. It is reserved for a
    # single final evaluation after model and threshold choices are locked.
    X_train, X_val, _, y_train, y_val, _ = load_and_split_data()
    models = build_models(X_train)

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        validation_result = evaluate_model(name, model, X_val, y_val)
        results.append({"split": "validation", **validation_result})

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nModel comparison:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
