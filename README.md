# XGBoost Bank Marketing

A machine learning project that predicts whether a bank customer will subscribe to a term deposit based on marketing campaign data. The project implements an end-to-end pipeline using XGBoost, scikit-learn preprocessing, threshold-based classification, model persistence, batch inference, and automated testing.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [Training](#training)
- [Batch Prediction](#batch-prediction)
- [Testing](#testing)
- [Model Artifacts](#model-artifacts)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [License](#license)

## Project Overview

The goal is to develop a reproducible machine learning pipeline that predicts customer subscription behavior for bank marketing campaigns.

The project covers:

- Dataset loading and validation.
- Exploratory data analysis.
- Numerical and categorical feature preprocessing.
- XGBoost model training.
- Model evaluation using classification metrics.
- Classification threshold configuration.
- Model serialization and metadata management.
- Batch prediction through a command-line interface.
- Automated testing with pytest.

A scikit-learn pipeline combines preprocessing and model inference so that the same transformations are applied during training and prediction.

## Dataset

This project uses the **Bank Marketing dataset** from the UCI Machine Learning Repository.

| Property | Description |
|---|---|
| Dataset file | `bank-full.csv` |
| Number of records | 45,211 |
| Number of original columns | 17 |
| Target variable | `y` |
| Target values | `yes`, `no` |
| Input delimiter | Semicolon (`;`) |

The dataset contains customer demographic information, financial information, and details about marketing campaigns. The target variable indicates whether a customer subscribed to a term deposit.

### Data Leakage Prevention

The `duration` feature is excluded from model training because it represents the duration of the last contact with a customer.

This information is generally unavailable before a marketing call. Including it could introduce target leakage and make the model less representative of a real-world prediction scenario.

### Dataset Setup

Place the dataset at the following path:

```text
data/raw/bank-full.csv
```

The original CSV file uses a semicolon (`;`) as its delimiter.

The raw dataset is excluded from version control. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place the file at the expected path before running training or other scripts that require the full dataset.

## Technology Stack

- **Python 3.12+** — primary programming language, subject to the version constraints in `pyproject.toml`.
- **XGBoost** — gradient boosting classifier.
- **pandas and NumPy** — data manipulation and numerical computation.
- **scikit-learn** — preprocessing, data splitting, pipelines, and evaluation.
- **joblib** — model serialization.
- **pytest** — automated testing.
- **uv** — dependency and environment management.

## Project Structure

```text
xgboost-bank-marketing/
├── data/
│   ├── raw/                  # Local datasets; excluded from Git
│   └── processed/            # Processed data and prediction outputs
├── models/
│   ├── xgboost_pipeline.joblib
│   └── metadata.json
├── scripts/
│   ├── analyze_dataset.py
│   ├── predict.py
│   └── train_model.py
├── src/
│   ├── data_analyzer.py
│   ├── data_loader.py
│   ├── model_evaluator.py
│   ├── model_trainer.py
│   └── preprocessor.py
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

The project separates data handling, preprocessing, model training, evaluation, inference, and testing into dedicated modules.

## Installation

### Prerequisites

- Python version supported by `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/).

### Install Dependencies

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/JiaHui2005/xgboost-bank-marketing.git
cd xgboost-bank-marketing
```

Install the project dependencies:

```bash
uv sync
```

Before training, ensure that `data/raw/bank-full.csv` is available at the expected location.

## Machine Learning Pipeline

The training workflow follows these steps:

```text
Raw Dataset
    |
    v
Data Loading and Validation
    |
    v
Remove Leakage-Prone Feature
    |
    v
Stratified Train / Validation / Test Split
    |
    v
Feature Preprocessing
    |
    v
XGBoost Training
    |
    v
Validation Evaluation
    |
    v
Save Model Pipeline and Metadata
    |
    v
Final Test Evaluation
```

### Data Splitting

The dataset is divided into three subsets using stratified splitting with a fixed random seed:

| Subset | Percentage |
|---|---:|
| Training | 64% |
| Validation | 16% |
| Test | 20% |

Stratification helps preserve the target-class distribution across the subsets. The fixed random seed (`42`) makes the split reproducible for the same dataset and environment.

### Model Configuration

The current XGBoost configuration includes:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 4 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

The preprocessing pipeline handles numerical and categorical features before passing them to the classifier.

## Model Performance

The model was evaluated on separate validation and test sets. The reported results use a classification threshold of `0.20`.

### Evaluation Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.8734 | 0.8747 |
| Precision | 0.4656 | 0.4683 |
| Recall | 0.5591 | 0.5236 |
| F1-score | 0.5081 | 0.4944 |
| ROC-AUC | 0.8083 | 0.8097 |
| Average Precision (PR-AUC label in the implementation) | 0.4740 | 0.4710 |

The PR-AUC values above are produced by scikit-learn's `average_precision_score`, which calculates Average Precision.

### Test Confusion Matrix

At the configured threshold of `0.20`, the test confusion matrix is:

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 7,356 | 629 |
| Yes | 504 | 554 |

The model correctly identified 554 actual subscribers and 7,356 non-subscribers in the test set.

### Interpretation

- **ROC-AUC of 0.8097:** The model demonstrates useful discrimination between the two target classes.
- **Recall of 0.5236:** The model identified approximately 52.36% of actual subscribers in the test set.
- **Precision of 0.4683:** Approximately 46.83% of positive predictions corresponded to actual subscribers.
- **F1-score of 0.4944:** This reflects the balance between precision and recall at the configured threshold.

Accuracy should not be considered in isolation because the dataset contains more non-subscribers than subscribers.

The classification threshold affects precision, recall, F1-score, accuracy, and the confusion matrix. ROC-AUC and Average Precision are calculated from predicted probabilities rather than thresholded labels.

These results provide a baseline for further experiments with threshold selection, class imbalance handling, and alternative models.

## Training

To train the model, evaluate it, and save the trained pipeline and metadata, run the following command from the project root:

```bash
uv run python scripts/train_model.py
```

The script:

1. Loads and splits the dataset.
2. Removes the `duration` feature.
3. Fits the preprocessing and XGBoost pipeline.
4. Evaluates the model on the validation set.
5. Saves the trained pipeline and metadata.
6. Reloads the saved pipeline and evaluates it on the test set.

The training process generates or updates:

```text
models/
├── xgboost_pipeline.joblib
└── metadata.json
```

The script prints validation metrics, test metrics, and their respective confusion matrices.

## Batch Prediction

The project provides a command-line interface for generating predictions from a CSV file.

### Input Requirements

The input CSV must use a semicolon (`;`) delimiter and include the following 15 model features:

```text
age, job, marital, education, default,
balance, housing, loan, contact, day, month,
campaign, pdays, previous, poutcome
```

The `duration` feature is not required. Additional columns may be present in the input file, but the prediction pipeline uses the required model features.

### Run Predictions

From the project root, run:

```bash
uv run python scripts/predict.py --input data/raw/bank-full.csv
```

The script loads the saved model and metadata, calculates subscription probabilities, and generates binary predictions.

### Output

The output is saved to:

```text
data/processed/predictions.csv
```

The prediction file retains the original input columns and adds three fields:

| Column | Description |
|---|---|
| `subscription_probability` | Predicted probability that the customer subscribes |
| `prediction` | Binary prediction: `1` or `0` |
| `prediction_label` | Human-readable prediction: `yes` or `no` |

The output directory is created automatically if it does not exist.

### Classification Threshold

The prediction threshold is loaded from `models/metadata.json`.

The current configured threshold is **0.20**:

- Probability greater than or equal to `0.20` → `yes` (`1`).
- Probability below `0.20` → `no` (`0`).

The threshold is stored in metadata rather than hard-coded into the prediction logic.

## Testing

Run the complete test suite:

```bash
uv run pytest -v
```

The tests cover model evaluation, model training, input validation, metadata validation, preprocessing, and loading the saved model.

At the latest recorded verification, **28 tests passed**.

The batch prediction output was also checked across all 45,211 dataset records:

- Output shape: `45,211 × 20`.
- Missing values in the output: `0`.
- Prediction labels were consistent with the configured threshold.
- Threshold mismatches: `0`.

These checks verify pipeline execution and prediction consistency. They do not, by themselves, establish the model's predictive accuracy.

## Model Artifacts

The prediction workflow expects the following files:

```text
models/
├── xgboost_pipeline.joblib
└── metadata.json
```

- `xgboost_pipeline.joblib` contains the fitted preprocessing and prediction pipeline.
- `metadata.json` contains inference configuration, including the classification threshold and recorded validation metrics.

Both artifacts must be available before running batch prediction. Running the training script regenerates these artifacts.

## Limitations and Future Improvements

Potential extensions include:

- Selecting the classification threshold systematically using validation data and an explicit business objective.
- Comparing XGBoost against baseline models such as Logistic Regression.
- Evaluating class imbalance handling strategies.
- Tracking experiments and model configurations.
- Adding feature importance analysis and model interpretability.
- Adding a REST API for inference.
- Setting up continuous integration to run tests automatically.

## License

No license has been specified for this project yet. Add an appropriate license before granting others explicit permission to use, modify, or distribute the code.
