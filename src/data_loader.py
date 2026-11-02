from pathlib import Path

import pandas as pd

from src.feature_contract import PRE_CAMPAIGN_FEATURES


def encode_binary_target(target: pd.Series) -> pd.Series:
    """Map Bank Marketing labels to 0/1 and reject missing or unknown labels."""
    encoded = target.map({"no": 0, "yes": 1})
    invalid_mask = encoded.isna()
    if invalid_mask.any():
        invalid_values = target.loc[invalid_mask].drop_duplicates().tolist()
        raise ValueError(
            "Target contains missing or unsupported labels; expected only "
            f"'no'/'yes'. Invalid values: {invalid_values}"
        )
    return encoded.astype("int8")


# Plausibility bounds; values outside indicate a corrupt or mislabelled file.
NUMERIC_RULES = {
    "age": (lambda s: s.between(0, 120), "0 to 120"),
    "balance": (lambda s: pd.Series(True, index=s.index), "any number"),
    "pdays": (lambda s: s.eq(-1) | s.ge(0), "-1 or >= 0"),
    "previous": (lambda s: s.ge(0), ">= 0"),
}


def check_training_data(df: pd.DataFrame) -> dict:
    """Reject unusable training data and summarise its quality."""
    missing_columns = sorted(set(PRE_CAMPAIGN_FEATURES + ["y"]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")

    features = df[PRE_CAMPAIGN_FEATURES]
    for column in PRE_CAMPAIGN_FEATURES:
        if features[column].isna().all():
            raise ValueError(f"Feature '{column}' is entirely missing.")

    for column, (is_valid, allowed) in NUMERIC_RULES.items():
        values = features[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"Feature '{column}' must be numeric.")
        present = values.dropna()
        if not is_valid(present).all():
            raise ValueError(f"Feature '{column}' has values outside {allowed}.")

    target = encode_binary_target(df["y"])
    if target.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    missing_rate = features.isna().mean()
    return {
        "rows": len(df),
        "positive_rate": float(target.mean()),
        "duplicate_rows": int(df[PRE_CAMPAIGN_FEATURES + ["y"]].duplicated().sum()),
        "missing_rate": {
            column: float(rate) for column, rate in missing_rate.items() if rate > 0
        },
    }


class DataLoader:
    """Load datasets from CSV file"""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load the Bank Marketing dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        return pd.read_csv(self.data_path, sep=";")
