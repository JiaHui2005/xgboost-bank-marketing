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



The prediction point is before the current marketing campaign begins. The model therefore excludes `duration` (known only after a call), `campaign` (counts current-campaign contacts), and `contact`, `day`, and `month` (attributes of the current contact in this dataset).



The historical features `pdays`, `previous`, and `poutcome` are retained only under the assumption that they contain information available before the current campaign. This timing must be confirmed against the operational data source before deployment.



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
| --- | --- |
| Training | 64% |
| Validation | 16% |
| Test | 20% |




Stratification helps preserve the target-class distribution across the subsets. The fixed random seed (`42`) makes the split reproducible for the same dataset and environment.



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




The preprocessing pipeline imputes numerical missing values with the training-split median and categorical missing values with a dedicated `__missing__` category before encoding. Imputation is fitted within each model pipeline, so validation/test rows do not influence the imputation statistics. Unknown categorical values remain supported by one-hot encoding. Target labels are validated as exactly `no` or `yes` before mapping to `0`/`1`.



## Model Performance

The project uses validation data for model and threshold decisions. The test split is reserved for one final evaluation after those decisions are locked. The classification threshold currently used by the training workflow is `0.20`.

The README previously reported test-set metrics for multiple candidate models and used them to support a model comparison. That exposed the test results to model selection; those historical numbers are therefore withdrawn as an unbiased final estimate. The checked-in split is no longer a pristine holdout. For a defensible final estimate, establish a fresh holdout (preferably a later time period) after model and threshold choices are fixed.

Choose metrics according to campaign costs: precision and recall describe different error tradeoffs; F1 balances them at a chosen threshold; ROC-AUC and Average Precision assess ranking across thresholds. Accuracy alone is insufficient for the imbalanced target. The fixed threshold should be selected using validation data and an explicit business objective, not the test set.



## Baseline Comparison

XGBoost and Logistic Regression are compared on the same stratified training and validation split, using the same target and fixed threshold. Categorical features are one-hot encoded; Logistic Regression additionally scales numerical features. Candidate ranking must be based on validation metrics only. No test metrics are produced by the comparison script.

### Reproduce the Comparison

Run the following command from the project root:

```bash
uv run python scripts/compare_models.py
```

The script trains both models, reports validation metrics only, and saves them to `data/processed/model_comparison.csv`. The test split is discarded without prediction or scoring.

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

6\. Reloads the saved pipeline and evaluates it on the test set.



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



The input CSV must use a semicolon (`;`) delimiter and include the following 11 pre-campaign model features:



```text

age, job, marital, education, default,

balance, housing, loan,

pdays, previous, poutcome

```



`duration`, `campaign`, `contact`, `day`, and `month` are excluded and are not required. Additional columns may be present in the input file, but the prediction pipeline ignores them. The historical features `pdays`, `previous`, and `poutcome` are valid only if they are available before the current campaign starts.

Missing feature values are accepted and imputed by the fitted pipeline using statistics/categories learned during training. Infinite numeric values, non-numeric values in numeric columns, and non-text values in categorical columns are rejected.

The checked-in model artifact predates this feature exclusion and must be regenerated with the training script before pre-campaign inference. The prediction command detects this stale artifact and fails with a retraining message rather than silently requiring unavailable information.



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
