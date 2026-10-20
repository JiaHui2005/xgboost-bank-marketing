import pandas as pd

DATA_PATH = "data/raw/bank-full.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["y"].value_counts())

print("\nTarget distribution (%):")
print(df["y"].value_counts(normalize=True).mul(100).round(2))