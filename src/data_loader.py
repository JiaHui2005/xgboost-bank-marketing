from pathlib import Path

import pandas as pd

class DataLoader:
    """Load datasets from CSV file"""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load the Bank Marketing dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        return pd.read_csv(self.data_path, sep=";")