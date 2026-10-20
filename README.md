# XGBoost Bank Marketing

A machine learning project that predicts whether a customer will subscribe to a term deposit based on bank marketing campaign data. The project uses **XGBoost**, a preprocessing pipeline, threshold-based classification, and automated tests.

## 1. Project Overview

The goal is to build a reproducible machine learning pipeline for predicting customer subscription behavior.

The project covers:

- Dataset loading and exploratory data analysis.
- Data preprocessing for numerical and categorical features.
- Model training using XGBoost.
- Model evaluation and classification threshold selection.
- Model persistence and metadata management.
- Batch prediction through a command-line interface.
- Automated testing with pytest.

## 2. Dataset

This project uses the **Bank Marketing dataset** from the UCI Machine Learning Repository.

- **Dataset file:** `bank-full.csv`
- **Number of records:** 45,211
- **Number of original columns:** 17
- **Target variable:** `y`
- **Target values:** `yes` and `no`

The dataset contains customer demographic information, financial information, and details about marketing campaigns.

The target variable indicates whether a customer subscribed to a term deposit.

### Data leakage prevention

The `duration` feature is excluded from model training because it represents the duration of the last contact. This information is generally unavailable before a marketing call, so using it could introduce target leakage into a real-world prediction scenario.

### Dataset setup

Place the dataset at:

```text
data/raw/bank-full.csv
```

The original file uses semicolon (`;`) as its delimiter.

The dataset is not necessarily included in the repository. Obtain it from the official source and place it at the expected path.

## 3. Technology Stack

- **Python** — primary programming language
- **XGBoost** — gradient boosting classifier
- **pandas / NumPy** — data manipulation
- **scikit-learn** — preprocessing, pipelines, and evaluation utilities
- **joblib** — model serialization
- **pytest** — automated testing
- **uv** — dependency and environment management

## 4. Project Structure

```text
xgboost-bank-marketing/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── xgboost_pipeline.joblib
│   └── metadata.json
├── scripts/
│   ├── predict.py
│   └── ...
├── src/
│   └── ...
├── tests/
│   ├── test_model_evaluator.py
│   ├── test_model_trainer.py
│   ├── test_predict.py
│   ├── test_preprocessor.py
│   └── test_saved_model.py
├── pyproject.toml
├── uv.lock
└── README.md
```

The structure separates datasets, reusable machine learning components, executable scripts, saved artifacts, and tests.

## 5. Installation

### Prerequisites

- Python 3.12 or a compatible version supported by `pyproject.toml`
- `uv`

Install the project dependencies:

```bash
uv sync
```

Ensure that the dataset is available at `data/raw/bank-full.csv` before running the relevant data-processing or training scripts.

## 6. Machine Learning Pipeline

The project follows this general workflow:

```text
Raw Dataset
    |
    v
Data Loading and Validation
    |
    v
Feature Preprocessing
    |
    v
XGBoost Training
    |
    v
Model Evaluation
    |
    v
Save Pipeline and Metadata
    |
    v
Batch Prediction
```

The trained preprocessing and prediction pipeline is saved as a single artifact so that inference uses the same preprocessing steps as training.

The metadata file stores the classification threshold used by the prediction script.

## 7. Running Predictions

The project provides a command-line prediction script.

### Input requirements

The input CSV must use `;` as its delimiter and include the following 15 model features:

```text
age, job, marital, education, default,
balance, housing, loan, contact, day, month,
campaign, pdays, previous, poutcome
```

The input may contain additional columns. The prediction script validates the required features before running inference.

### Run the prediction script

From the project root:

```bash
uv run python scripts/predict.py --input data/raw/bank-full.csv
```

The script loads the saved model and metadata, calculates subscription probabilities, and generates binary predictions using the configured threshold.

### Output

Predictions are saved to:

```text
data/processed/predictions.csv
```

The output retains the original input columns and adds three fields:

| Column | Description |
|---|---|
| `subscription_probability` | Predicted probability of subscription |
| `prediction` | Binary prediction: `1` or `0` |
| `prediction_label` | Human-readable label: `yes` or `no` |

The output directory is created automatically if it does not exist.

### Classification threshold

The threshold is loaded from `models/metadata.json`.

The current configured threshold is **0.20**:

- Probability greater than or equal to `0.20` → `yes` (`1`)
- Probability below `0.20` → `no` (`0`)

The threshold is configurable through the saved metadata rather than hard-coded into the prediction logic.

## 8. Testing

Run the complete test suite:

```bash
uv run pytest -v
```

The tests cover model evaluation, model training, input validation, metadata validation, preprocessing, and loading the saved model.

At the latest verification, **28 tests passed**.

The prediction output was also checked on all 45,211 records:

- Output shape: `45,211 × 20`
- Missing values in the output: `0`
- Prediction labels were consistent with the configured threshold.
- Threshold mismatches: `0`

These checks verify pipeline execution and prediction consistency; they do not, by themselves, establish the model's predictive accuracy.

## 9. Model Artifacts

The prediction workflow expects the following files:

```text
models/
├── xgboost_pipeline.joblib
└── metadata.json
```

- `xgboost_pipeline.joblib` contains the trained machine learning pipeline.
- `metadata.json` contains inference configuration, including the classification threshold.

Both artifacts must be available before running batch prediction.

## 10. Future Improvements

Potential extensions include:

- Adding a reproducible training command.
- Comparing XGBoost against baseline models such as Logistic Regression.
- Tracking precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Evaluating threshold selection on a separate validation set.
- Adding a REST API for inference.
- Adding continuous integration to run tests automatically.

## License

Add the license applicable to this project before distributing it publicly.
