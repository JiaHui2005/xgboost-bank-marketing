
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from scripts.compare_models import (
    build_models,
    load_and_split_data,
)


OUTPUT_PATH = Path("data/processed/threshold_comparison.csv")
CURRENT_THRESHOLD = 0.20


def evaluate_thresholds(y_true, probabilities, thresholds):
    """Evaluate classification metrics for each candidate threshold."""
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_true, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_true, predictions, zero_division=0
                ),
                "predicted_positive": int(predictions.sum()),
                "contact_rate": float(predictions.sum() / len(y_true)),
                "true_positive": int(
                    ((predictions == 1) & (y_true == 1)).sum()
                ),
                "false_positive": int(
                    ((predictions == 1) & (y_true == 0)).sum()
                ),
                "false_negative": int(
                    ((predictions == 0) & (y_true == 1)).sum()
                ),
            }
        )

    return pd.DataFrame(results)


def main():
    X_train, X_val, y_train, y_val = load_and_split_data()

    models = build_models(X_train)
    model = models["XGBoost"]

    print("Training XGBoost on the training split...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)

    if probabilities.ndim == 2:
        probabilities = probabilities[:, 1]

    thresholds = sorted(
        set(
            [CURRENT_THRESHOLD]
            + [round(value, 2) for value in np.arange(0.05, 0.81, 0.01)]
        )
    )

    results = evaluate_thresholds(y_val.to_numpy(), probabilities, thresholds)

    current_row = results.loc[
        (results["threshold"] - CURRENT_THRESHOLD).abs().lt(1e-9)
    ].iloc[0]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nCurrent threshold (validation):")
    print(current_row.to_string())

    print(f"\nValidation threshold frontier saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
