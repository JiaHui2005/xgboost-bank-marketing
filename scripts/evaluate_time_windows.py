from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from scripts.compare_models import build_models, evaluate_model
from src.data_loader import DataLoader, encode_binary_target
from src.feature_contract import PRE_CAMPAIGN_FEATURES


N_SPLITS = 5
OUTPUT_PATH = Path("data/processed/time_window_metrics.csv")
SUMMARY_PATH = Path("data/processed/time_window_summary.csv")
SUMMARY_METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fpr"]


def evaluate_time_windows(X, y, n_splits=N_SPLITS):
    """Train on earlier rows and validate on the next block, for each window."""
    results = []

    for window, (train_idx, val_idx) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(X), start=1
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in build_models(X_train).items():
            model.fit(X_train, y_train)
            results.append(
                {
                    "window": window,
                    "train_rows": len(train_idx),
                    "val_start": int(val_idx[0]),
                    "val_end": int(val_idx[-1]) + 1,
                    "val_positive_rate": float(y_val.mean()),
                    **evaluate_model(name, model, X_val, y_val),
                }
            )

    return pd.DataFrame(results)


def summarize_time_windows(results, metrics=SUMMARY_METRICS):
    """Report mean, standard deviation and range of each metric per model."""
    summary = (
        results.melt(id_vars=["model", "window"], value_vars=metrics, var_name="metric")
        .groupby(["model", "metric"], sort=False)["value"]
        .agg(windows="count", mean="mean", std="std", min="min", max="max")
        .reset_index()
    )
    return summary


def main():
    df = DataLoader("data/raw/bank-full.csv").load_data()
    X = df[PRE_CAMPAIGN_FEATURES].copy()
    y = encode_binary_target(df["y"])

    results = evaluate_time_windows(X, y)
    summary = summarize_time_windows(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    float_format = lambda value: f"{value:.4f}"
    print("Per-window validation metrics:")
    print(
        results[
            ["model", "window", "train_rows", "support", "val_positive_rate", *SUMMARY_METRICS]
        ].to_string(index=False, float_format=float_format)
    )
    print("\nSummary across windows:")
    print(summary.to_string(index=False, float_format=float_format))
    print(f"\nResults saved to: {OUTPUT_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
