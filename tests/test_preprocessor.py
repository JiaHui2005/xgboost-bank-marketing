
import pandas as pd
import numpy as np

from src.preprocessor import DataPreprocessor


def test_preprocessor_transforms_categorical_and_numerical_data():
    X_train = pd.DataFrame({
        "job": ["admin", "technician", "admin"],
        "age": [30, 40, 50],
    })

    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )

    X_transformed = preprocessor.fit_transform(X_train)

    assert X_transformed.shape == (3, 3)


def test_preprocessor_handles_unknown_categories():
    X_train = pd.DataFrame({
        "job": ["admin", "technician"],
        "age": [30, 40],
    })

    X_test = pd.DataFrame({
        "job": ["student"],
        "age": [25],
    })

    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )

    preprocessor.fit(X_train)
    X_transformed = preprocessor.transform(X_test)

    assert X_transformed.shape == (1, 3)


def test_preprocessor_imputes_missing_categorical_and_numeric_values():
    X_train = pd.DataFrame(
        {
            "job": ["admin", None, "technician"],
            "age": [30.0, np.nan, 50.0],
        }
    )

    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )
    transformed = preprocessor.fit_transform(X_train)
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert values.shape[0] == len(X_train)
    assert np.isfinite(values).all()
