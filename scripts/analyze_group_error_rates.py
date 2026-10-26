
from pathlib import Path

import pandas as pd

from scripts.compare_models import build_models, load_and_split_data


THRESHOLD = 0.22
OUTPUT_PATH = Path("data/processed/group_error_rates.csv")

CATEGORICAL_FEATURES = [
    "job",
    "education",
    "contact",
    "poutcome",
    "marital",
    "housing",
    "loan",
    "month",
]


def add_group_error_rates(summary: pd.DataFrame) -> pd.DataFrame:
    """Add explicitly denominated group error metrics to count summaries."""
    summary = summary.copy()
    actual_negative = summary["total"] - summary["actual_positive"]
    summary["actual_negative"] = actual_negative
    summary["false_positive_rate"] = (
        summary["false_positive"] / actual_negative.replace(0, pd.NA)
    )
    summary["false_positive_fraction_within_group"] = (
        summary["false_positive"] / summary["total"].replace(0, pd.NA)
    )
    summary["false_negative_rate_within_actual_positive"] = (
        summary["false_negative"]
        / summary["actual_positive"].replace(0, pd.NA)
    )
    summary["missed_positive_share"] = (
        summary["false_negative"]
        / summary["actual_positive"].replace(0, pd.NA)
    )
    return summary


def main():
    X_train, X_val, _, y_train, y_val, _ = load_and_split_data()

    model = build_models(X_train)["XGBoost"]
    print("Training XGBoost...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)
    predictions = (probabilities >= THRESHOLD).astype(int)

    evaluation = X_val.copy()
    evaluation["actual"] = y_val.to_numpy()
    evaluation["predicted"] = predictions
    evaluation["error_type"] = "TN"

    evaluation.loc[
        (evaluation["actual"] == 1)
        & (evaluation["predicted"] == 1),
        "error_type",
    ] = "TP"

    evaluation.loc[
        (evaluation["actual"] == 0)
        & (evaluation["predicted"] == 1),
        "error_type",
    ] = "FP"

    evaluation.loc[
        (evaluation["actual"] == 1)
        & (evaluation["predicted"] == 0),
        "error_type",
    ] = "FN"

    rows = []

    for feature in CATEGORICAL_FEATURES:
        summary = (
            evaluation.groupby(feature, dropna=False)
            .agg(
                total=("actual", "size"),
                actual_positive=("actual", "sum"),
                predicted_positive=("predicted", "sum"),
                false_positive=(
                    "error_type",
                    lambda s: (s == "FP").sum(),
                ),
                false_negative=(
                    "error_type",
                    lambda s: (s == "FN").sum(),
                ),
            )
            .reset_index()
            .rename(columns={feature: "value"})
        )

        summary = add_group_error_rates(summary)
        summary["feature"] = feature

        rows.append(summary)

    result = pd.concat(rows, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    for feature in ["job", "contact", "poutcome", "education"]:
        subset = result[result["feature"] == feature]

        print(f"\n{'=' * 65}")
        print(f"GROUP ERROR RATES: {feature}")
        print("=" * 65)

        print(
            subset.sort_values(
                "false_negative_rate_within_actual_positive",
                ascending=False,
                na_position="last",
            )[
                [
                    "value",
                    "total",
                    "actual_positive",
                    "actual_negative",
                    "false_negative",
                    "false_negative_rate_within_actual_positive",
                    "false_positive",
                    "false_positive_rate",
                    "false_positive_fraction_within_group",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
