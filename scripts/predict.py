
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "xgboost_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"

REQUIRED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

NUMERICAL_COLUMNS = [
    "age",
    "balance",
    "day",
    "campaign",
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

    # Check missing values in the required model features.
    feature_df = df[REQUIRED_COLUMNS]
    missing_values = feature_df.columns[
        feature_df.isnull().any()
    ].tolist()

    if missing_values:
        raise ValueError(
            f"Input CSV contains missing values in columns: {missing_values}"
        )

    # Validate numerical features.
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

        if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Column '{column}' contains infinite or non-finite values."
            )

    # Validate categorical features.
    categorical_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in NUMERICAL_COLUMNS
    ]

    for column in categorical_columns:
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
