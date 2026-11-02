from src.config import DATA_PATH
from src.data_analyzer import DataAnalyzer
from src.data_loader import DataLoader


def main() -> None:
    loader = DataLoader(DATA_PATH)
    dataframe = loader.load_data()

    analyzer = DataAnalyzer(dataframe)

    analyzer.analyze_categorical_column("job")
    analyzer.analyze_categorical_column("poutcome")


if __name__ == "__main__":
    main()