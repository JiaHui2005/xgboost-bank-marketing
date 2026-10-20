import pandas as pd

class DataAnalyzer:
    """Analyze the Bank Marketing dataset"""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def display_overview(self) -> None:
        """Display the dataset shape and column information"""
        print("Dataset overview")
        print("=" * 40)

        print(f"Number of rows: {self.dataframe.shape[0]}")
        print(f"Number of columns: {self.dataframe.shape[1]}")

        print("\nColunm information:")
        self.dataframe.info()

    def check_missing_values(self) -> pd.Series:
        """Return the number of missing values in each column"""
        missing_values = self.dataframe.isnull().sum()

        print("Missing Values")
        print("=" * 40)
        print(missing_values)

        return missing_values

    def analyze_target(self, target_column: str = "y") -> pd.DataFrame:
        """Analyze the distribution of the target variable"""
        target_counts = self.dataframe[target_column].value_counts()
        target_percentages = (
            self.dataframe[target_column].value_counts(normalize=True).mul(100).round(2)
        )

        summary = pd.DataFrame(
            {
                "count": target_counts,
                "percentage": target_percentages
            }
        )

        print("Target Distribution")
        print("=" * 40)
        print(summary)

        return summary

    def check_duplicates(self) -> int:
        """Count duplicate rows in dataset"""
        duplicate_count = self.dataframe.duplicated().sum()

        print("Duplicate Rows")
        print("=" * 40)
        print(f"Number of duplicate rows: {duplicate_count}")

        return int(duplicate_count)

    def analyze_unknown_values(self) -> pd.DataFrame:
        """Analyze unknown values in categorical columns."""

        categorical_columns = self.dataframe.select_dtypes(
            include=["str", "object", "category"]
        ).columns

        unknown_counts = (
            self.dataframe[categorical_columns]
            .eq("unknown")
            .sum()
        )

        unknown_counts = unknown_counts[unknown_counts > 0]

        summary = pd.DataFrame({
            "unknown_count": unknown_counts,
            "percentage": (
                unknown_counts / len(self.dataframe) * 100
            ).round(2),
        })

        print("Unknown Values")
        print("=" * 40)
        print(summary)

        return summary

    def analyze_categorical_column(self, column_name: str) -> pd.DataFrame:
        """Analyze the distribution of a categorical column."""

        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist.")

        value_counts = self.dataframe[column_name].value_counts(
            dropna=False
        )

        percentages = (
            self.dataframe[column_name]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
        )

        summary = pd.DataFrame({
            "count": value_counts,
            "percentage": percentages,
        })

        print(f"Distribution of '{column_name}'")
        print("=" * 40)
        print(summary)

        return summary