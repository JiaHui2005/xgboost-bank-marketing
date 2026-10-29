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
- [Baseline Comparison](#baseline-comparison)

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
| --- | --- |
| Dataset file | `bank-full.csv` |
| Number of records | 45,211 |
| Number of original columns | 17 |
| Target variable | `y` |
| Target values | `yes`, `no` |
| Input delimiter | Semicolon (`;`) |




The dataset contains customer demographic information, financial information, and details about marketing campaigns. The target variable indicates whether a customer subscribed to a term deposit.



### Data Leakage Prevention



The prediction point is before the current marketing campaign begins. The table distinguishes what the UCI dictionary says from operational availability: the dictionary describes fields, but does not prove when a production system snapshots or exposes them. Operational confirmation provided on 2026-10-29 states that `pdays`, `previous`, and `poutcome` are available before the current campaign; confirm profile-field freshness and availability in the operational source before deployment.

| Feature | UCI meaning / source | Available at pre-campaign prediction? | Missing / special values | Decision |
| --- | --- | --- | --- | --- |
| `age` | Client age | Assumed available; confirm source freshness | No missing attributes reported by UCI; numeric nulls use the training median | Keep |
| `job` | Client job type | Assumed available; confirm source freshness | Literal `unknown` retained as a category; nulls use `__missing__` | Keep |
| `marital` | Marital status (`divorced` includes divorced or widowed) | Assumed available; confirm source freshness | Literal `unknown` retained as a category; nulls use `__missing__` | Keep |
| `education` | Education level | Assumed available; confirm source freshness | Literal `unknown` retained as a category; nulls use `__missing__` | Keep |
| `default` | Whether the client has credit in default | Assumed available; confirm source freshness | Nulls use `__missing__`; `yes`/`no` are values | Keep |
| `balance` | Average yearly balance, in euros | Assumed available; confirm source freshness | Numeric nulls use the training median | Keep |
| `housing` | Whether the client has a housing loan | Assumed available; confirm source freshness | Nulls use `__missing__`; `yes`/`no` are values | Keep |
| `loan` | Whether the client has a personal loan | Assumed available; confirm source freshness | Nulls use `__missing__`; `yes`/`no` are values | Keep |
| `pdays` | Days since the client was last contacted in a previous campaign; `-1` means not previously contacted | UCI defines prior-campaign history; operationally confirmed available before the current campaign (2026-10-29) | `-1` is a meaningful sentinel, not missing; other numeric nulls use the training median | Keep |
| `previous` | Number of contacts before this campaign for this client | UCI defines prior-campaign history; operationally confirmed available before the current campaign (2026-10-29) | `0` means no prior contacts; numeric nulls use the training median | Keep |
| `poutcome` | Outcome of the previous marketing campaign | UCI defines prior-campaign history; operationally confirmed available before the current campaign (2026-10-29) | Literal `unknown` retained as a category; nulls use `__missing__` | Keep |

The UCI dictionary reports no missing attribute values. `unknown`, `-1`, and `0` are documented field values, not missing values. The preprocessing policy above describes handling for nulls if they occur in other input data.

The model excludes `duration` (known only after a call), `campaign` (counts contacts in the current campaign), and `contact`, `day`, and `month` (attributes of the current contact). These are not available at the stated prediction point.



### Dataset Setup



Place the dataset at the following path:



```text

data/raw/bank-full.csv

```



The original CSV file uses a semicolon (`;`) as its delimiter.



The raw dataset is excluded from version control. Download the dataset from the [UCI Machine Learning Repository](https\://archive.ics.uci.edu/dataset/222/bank+marketing) and place the file at the expected path before running training or other scripts that require the full dataset.



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

- [uv](https\://docs.astral.sh/uv/).



### Install Dependencies



Clone the repository and navigate to the project directory:



```bash

git clone https\://github.com/JiaHui2005/xgboost-bank-marketing.git

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

Chronological Train / Validation Split (80/20)

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

```



### Data Splitting



The dataset is split by its original UCI row order, without shuffling or stratification:



| Subset | Percentage |
| --- | --- |
| Training (first rows) | 80% |
| Validation (last rows) | 20% |




The source describes the rows as ordered by date. Validation is used for model development and selection, not as an independent final test estimate. No independent evaluation is currently available; that requires new campaign data not used for model or threshold selection.



### Model Configuration



The current XGBoost configuration includes:



| Parameter | Value |
| --- | --- |
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 4 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |




The preprocessing pipeline imputes numerical missing values with the training-split median and categorical missing values with a dedicated `__missing__` category before encoding. Imputation is fitted within each model pipeline, so validation rows do not influence the imputation statistics. Unknown categorical values remain supported by one-hot encoding. Target labels are validated as exactly `no` or `yes` before mapping to `0`/`1`.



## Model Performance

The training and comparison scripts preserve the UCI row order and use the first 80% of rows for training and the final 20% for validation. The dataset is documented as ordered by date; this split supports development on a later period, but it is validation—not an independent test or unbiased final estimate. The classification threshold currently used by the training workflow is `0.20`.

The README previously reported test-set metrics for multiple candidate models and used them to support a model comparison. That exposed the test results to model selection; those historical numbers are therefore withdrawn as an unbiased final estimate. No independent evaluation is currently available from this dataset. A final independent evaluation requires new campaign data that has not been used for model, hyperparameter, or threshold selection, evaluated after those choices are locked.

The campaign's contact capacity and the costs/value of false positives and false negatives have not been provided, so the operating threshold is not optimized or claimed to be business-optimal. Until those inputs are available, use validation Average Precision (AP) as a provisional model-ranking metric. At the existing fixed threshold of `0.20`, also report precision, recall, F1, FPR (`FP / (FP + TN)`), FNR (`FN / (FN + TP)`), confusion counts, and class support. These threshold-specific values describe tradeoffs; they do not determine the business-optimal threshold. ROC-AUC is a secondary ranking metric, and accuracy alone is insufficient for the imbalanced target.



## Baseline Comparison

XGBoost and Logistic Regression are compared on the same chronological 80/20 training/validation split, using the same target and fixed threshold. Categorical features are one-hot encoded; Logistic Regression additionally scales numerical features. Candidate ranking must be based on validation metrics only. No test metrics are produced by the comparison script.

### Reproduce the Comparison

Run the following command from the project root:

```bash
uv run python scripts/compare_models.py
```

The script trains both models, reports validation metrics only, and saves them to `data/processed/model_comparison.csv`. It does not create or score a test split.

The threshold is held fixed at `0.20` for this comparison. It should not be described as optimized unless a separate validation-based threshold-selection experiment has been performed.

## Training



To train the model, evaluate it, and save the trained pipeline and metadata, run the following command from the project root:



```bash

uv run python scripts/train_model.py

```



The script:



1\. Loads and splits the dataset.

2\. Uses the pre-campaign feature contract, excluding `duration`, `campaign`, `contact`, `day`, and `month`.

3\. Fits the preprocessing and XGBoost pipeline.

4\. Evaluates the model on the validation set.

5\. Saves the trained pipeline and metadata.




The training process generates or updates:



```text

models/

├── xgboost_pipeline.joblib

└── metadata.json

```



The script prints validation metrics and the validation confusion matrix only. Validation results are for development, not an independent final estimate; independent evaluation remains pending new campaign data.



## Batch Prediction



The project provides a command-line interface for generating predictions from a CSV file.



### Input Requirements



The input CSV must use a semicolon (`;`) delimiter and include the following 11 pre-campaign model features:



```text

age, job, marital, education, default,

balance, housing, loan,

pdays, previous, poutcome

```



`duration`, `campaign`, `contact`, `day`, and `month` are excluded and are not required. Additional columns may be present in the input file, but the prediction pipeline ignores them. Operations confirmed that `pdays`, `previous`, and `poutcome` are available before the current campaign starts; profile-field availability still requires confirmation.

Missing feature values are accepted and imputed by the fitted pipeline using statistics/categories learned during training. Infinite numeric values, non-numeric values in numeric columns, and non-text values in categorical columns are rejected.

The checked-in model artifact uses the versioned pre-campaign feature contract. The prediction command checks that metadata and the saved pipeline match this contract before inference; retrain the model after changing its feature contract or training settings.



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
| --- | --- |
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



The tests cover binary-label validation, model evaluation, missing-value imputation, input validation, metadata validation, preprocessing, and loading the saved model.



At the latest recorded verification, **53 tests passed**.



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


- Evaluating class imbalance handling strategies.

- Tracking experiments and model configurations.

- Adding feature importance analysis and model interpretability.

- Adding a REST API for inference.

- Setting up continuous integration to run tests automatically.



## License



No license has been specified for this project yet. Add an appropriate license before granting others explicit permission to use, modify, or distribute the code.
