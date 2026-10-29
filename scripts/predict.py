
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_contract import (
    DROPPED_PRE_CAMPAIGN_FEATURES,
    PRE_CAMPAIGN_FEATURES,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "xgboost_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"

REQUIRED_COLUMNS = PRE_CAMPAIGN_FEATURES

NUMERICAL_COLUMNS = [
    "age",
    "balance",
    "pdays",
    "previous",
]


def load_artifacts():
    """Load the trained pipeline and its metadata."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    try:
        pipeline = joblib.load(MODEL_PATH)

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to load model artifacts: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")

    if "threshold" not in metadata:
        raise ValueError("Metadata is missing the 'threshold' field.")

    threshold = metadata["threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Metadata threshold must be a finite number between 0 and 1."
        )

    required_metadata = {
        "artifact_version",
        "model_type",
        "feature_names",
        "dropped_features",
        "target_mapping",
        "python_version",
        "dependency_versions",
        "seed",
        "trained_at_utc",
        "data_version",
        "split",
        "validation_metrics",
    }
    missing_metadata = required_metadata - metadata.keys()
    if missing_metadata:
        raise ValueError(
            f"Metadata is missing required fields: {sorted(missing_metadata)}"
        )

    artifact_version = metadata["artifact_version"]
    if (
        isinstance(artifact_version, bool)
        or not isinstance(artifact_version, int)
        or artifact_version != 1
    ):
        raise ValueError("Unsupported or invalid artifact version.")

    if (
        metadata["model_type"] != "XGBClassifier"
        or metadata["feature_names"] != REQUIRED_COLUMNS
        or metadata["dropped_features"] != DROPPED_PRE_CAMPAIGN_FEATURES
        or metadata["target_mapping"] != {"no": 0, "yes": 1}
        or metadata["threshold"] != 0.20
        or metadata["seed"] != 42
    ):
        raise RuntimeError(
            "The saved model is incompatible with the pre-campaign feature "
            "contract or training settings. Retrain the model before running "
            "pre-campaign predictions."
        )

    if not isinstance(metadata["dependency_versions"], dict) or not all(
        isinstance(metadata["dependency_versions"].get(package), str)
        for package in ("numpy", "pandas", "scikit-learn", "xgboost")
    ):
        raise ValueError("Metadata contains invalid dependency versions.")

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model_features = (
            list(preprocessor.categorical_columns)
            + list(preprocessor.numerical_columns)
        )
    except (AttributeError, KeyError, TypeError):
        model_features = []

    if (
        len(model_features) != len(REQUIRED_COLUMNS)
        or len(set(model_features)) != len(model_features)
        or set(model_features) != set(REQUIRED_COLUMNS)
    ):
        raise RuntimeError(
            "The saved pipeline is incompatible with the pre-campaign feature "
            "contract. Retrain the model before running pre-campaign predictions."
        )

    return pipeline, metadata


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate input data before prediction."""
    if df.empty:
        raise ValueError("Input CSV contains no data rows.")

    if not df.columns.is_unique:
        raise ValueError("Input CSV contains duplicate column names.")

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}"
        )

    # Missing feature values are imputed by the fitted model pipeline.
    feature_df = df[REQUIRED_COLUMNS]

    # Validate numerical features while allowing NaN for pipeline imputation.
    for column in NUMERICAL_COLUMNS:
        try:
            numeric_values = pd.to_numeric(
                feature_df[column],
                errors="raise",
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column '{column}' must contain numerical values."
            ) from exc

        numeric_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric_array).any():
            raise ValueError(
                f"Column '{column}' contains infinite values."
            )

    # Validate categorical features.
    categorical_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in NUMERICAL_COLUMNS
    ]

    for column in categorical_columns:
        if feature_df[column].isna().all():
            continue
        if not (
            pd.api.types.is_object_dtype(feature_df[column])
            or isinstance(feature_df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(feature_df[column])
        ):
            raise ValueError(
                f"Column '{column}' must contain categorical text values."
            )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict bank marketing subscription probabilities."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )
    return parser.parse_args()


def main():
    """Load input data, predict probabilities, and save results."""
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    try:
        df = pd.read_csv(input_path, sep=";")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Input CSV is empty or contains no header: {input_path}"
        ) from exc
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Unable to read input CSV '{input_path}': {exc}"
        ) from exc

    validate_input_data(df)

    pipeline, metadata = load_artifacts()

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input loaded: {input_path}")
    print(f"Rows to predict: {len(df)}")
    print(f"Threshold: {metadata['threshold']}")

    # Use only the required model features.
    # Extra columns such as customer_id are preserved in results.
    feature_df = df[REQUIRED_COLUMNS].copy()

    # Convert numeric columns explicitly after validation.
    for column in NUMERICAL_COLUMNS:
        feature_df[column] = pd.to_numeric(
            feature_df[column],
            errors="raise",
        )

    try:
        probabilities = pipeline.predict_proba(feature_df)[:, 1]
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Prediction failed. Please check the input data: {exc}"
        ) from exc

    threshold = metadata["threshold"]
    predictions = (probabilities >= threshold).astype(int)

    results = df.copy()
    results["subscription_probability"] = probabilities
    results["prediction"] = predictions
    results["prediction_label"] = results["prediction"].map(
        {
            0: "no",
            1: "yes",
        }
    )

    output_path = (
        ROOT_DIR / "data" / "processed" / "predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        results.to_csv(output_path, sep=";", index=False)
    except OSError as exc:
        raise OSError(
            f"Failed to save predictions to '{output_path}': {exc}"
        ) from exc

    print("\nPrediction results:")
    print(
        results[
            ["subscription_probability", "prediction_label"]
        ].to_string(index=False)
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
