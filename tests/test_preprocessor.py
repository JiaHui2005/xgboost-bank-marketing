
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


def test_preprocessor_imputes_pd_na_in_object_and_nullable_string_columns():
    X_train = pd.DataFrame(
        {
            "job": pd.array(["admin", pd.NA, "technician"], dtype="string"),
            "marital": ["single", pd.NA, "married"],
            "age": pd.array([30, pd.NA, 50], dtype="Int64"),
        }
    )

    preprocessor = DataPreprocessor(
        categorical_columns=["job", "marital"],
        numerical_columns=["age"],
    )
    values = preprocessor.fit_transform(X_train)
    values = values.toarray() if hasattr(values, "toarray") else values

    assert values.shape[0] == len(X_train)
    assert np.isfinite(values.astype(float)).all()


def test_preprocessor_fits_when_a_column_is_entirely_missing_in_train():
    X_train = pd.DataFrame(
        {
            "job": [None, None, None],
            "age": [np.nan, np.nan, np.nan],
        }
    )

    preprocessor = DataPreprocessor(
        categorical_columns=["job"],
        numerical_columns=["age"],
    )
    preprocessor.fit(X_train)
    values = preprocessor.transform(pd.DataFrame({"job": ["admin"], "age": [np.nan]}))
    values = values.toarray() if hasattr(values, "toarray") else values

    # Only `__missing__` was seen, so `admin` encodes to zeros; all-missing numeric imputes 0.
    assert values.tolist() == [[0.0, 0.0]]
    missing_row = preprocessor.transform(pd.DataFrame({"job": [None], "age": [np.nan]}))
    missing_row = missing_row.toarray() if hasattr(missing_row, "toarray") else missing_row
    assert missing_row.tolist() == [[1.0, 0.0]]
