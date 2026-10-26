from pathlib import Path

import pandas as pd


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


class DataLoader:
    """Load datasets from CSV file"""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load the Bank Marketing dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        return pd.read_csv(self.data_path, sep=";")
