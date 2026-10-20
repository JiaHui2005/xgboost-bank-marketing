from src.data_loader import DataLoader
from src.data_analyzer import DataAnalyzer


def main() -> None:
    loader = DataLoader("data/raw/bank-full.csv")
    dataframe = loader.load_data()

    analyzer = DataAnalyzer(dataframe)

    analyzer.analyze_categorical_column("job")
    analyzer.analyze_categorical_column("poutcome")


if __name__ == "__main__":
    main()