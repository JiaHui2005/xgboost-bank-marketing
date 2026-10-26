
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_models import build_models, load_and_split_data


THRESHOLD = 0.22
OUTPUT_DIR = Path("data/processed")

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "pdays",
    "previous",
]


def main():
    X_train, X_val, _, y_train, y_val, _ = load_and_split_data()

    model = build_models(X_train)["XGBoost"]

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)
    predictions = (probabilities >= THRESHOLD).astype(int)
    actual = y_val.to_numpy()

    error_types = np.select(
        [
            (actual == 1) & (predictions == 1),
            (actual == 0) & (predictions == 0),
            (actual == 0) & (predictions == 1),
        ],
        ["TP", "TN", "FP"],
        default="FN",
    )

    results = X_val.drop(columns=["campaign"], errors="ignore").copy()
    results["actual"] = actual
    results["probability"] = probabilities
    results["predicted"] = predictions
    results["error_type"] = error_types

    errors = results[results["error_type"].isin(["FP", "FN"])].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors.to_csv(
        OUTPUT_DIR / "prediction_errors_validation.csv",
        index=True,
        index_label="original_index",
    )

    categorical_rows = []

    for feature in CATEGORICAL_FEATURES:
        if feature not in errors.columns:
            continue

        grouped = (
            errors.groupby(["error_type", feature], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )

        grouped["feature"] = feature
        grouped = grouped.rename(columns={feature: "value"})

        grouped["percentage_within_error_type"] = (
            grouped["count"]
            / grouped.groupby("error_type")["count"].transform("sum")
            * 100
        )

        categorical_rows.append(
            grouped[
                [
                    "error_type",
                    "feature",
                    "value",
                    "count",
                    "percentage_within_error_type",
                ]
            ]
        )

    if categorical_rows:
        categorical_summary = pd.concat(
            categorical_rows, ignore_index=True
        )
        categorical_summary.to_csv(
            OUTPUT_DIR / "categorical_error_summary.csv",
            index=False,
        )

    available_numeric = [
        feature for feature in NUMERIC_FEATURES
        if feature in errors.columns
    ]

    numeric_summary = (
        errors.groupby("error_type")[available_numeric]
        .agg(["mean", "median"])
        .transpose()
        .reset_index()
    )

    numeric_summary.to_csv(
        OUTPUT_DIR / "numeric_error_summary.csv",
        index=False,
    )

    print(f"\nThreshold: {THRESHOLD:.2f}")
    print("\nValidation prediction counts:")
    print(results["error_type"].value_counts().to_string())

    print("\nMisclassified records:")
    print(f"False Positive: {(errors['error_type'] == 'FP').sum()}")
    print(f"False Negative: {(errors['error_type'] == 'FN').sum()}")

    print("\nFiles saved:")
    print(OUTPUT_DIR / "prediction_errors_validation.csv")
    print(OUTPUT_DIR / "categorical_error_summary.csv")
    print(OUTPUT_DIR / "numeric_error_summary.csv")


if __name__ == "__main__":
    main()
