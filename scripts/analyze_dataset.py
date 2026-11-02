from src.config import DATA_PATH
from src.data_analyzer import DataAnalyzer
from src.data_loader import DataLoader


def main() -> None:
    """Print a descriptive overview of the raw dataset."""
    analyzer = DataAnalyzer(DataLoader(DATA_PATH).load_data())

    analyzer.display_overview()
    for step in (
        analyzer.check_missing_values,
        analyzer.check_duplicates,
        analyzer.analyze_target,
        analyzer.analyze_unknown_values,
    ):
        print()
        step()

    for column in ("job", "poutcome"):
        print()
        analyzer.analyze_categorical_column(column)


if __name__ == "__main__":
    main()
