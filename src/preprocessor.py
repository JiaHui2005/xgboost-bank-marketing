from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess categorical and numerical features"""

    def __init__(self, categorical_columns: list[str], numerical_columns: list[str]) -> None:
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="__missing__",
                                ),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    self.categorical_columns
                ),
                (
                    "numerical",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                    self.numerical_columns
                )
            ]
        )

    def fit(self, X, y = None):
        """Learn preprocessing parameters from training data"""
        self.transformer.fit(X, y)

        return self

    def transform(self, X):
        """Transform features using the fitted transformer."""
        return self.transformer.transform(X)
