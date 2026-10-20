
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessor import DataPreprocessor


class ModelTrainer:
    """Train a model using preprocessing and classification pipeline"""

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        model_params: dict | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.model_params = model_params or {}
        self.pipeline = None

    def build_pipeline(self) -> Pipeline:
        """Build the preprocessing and XGBoost pipeline"""
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **self.model_params,
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", model),
            ]
        )
        return self.pipeline

    def fit(self, X_train, y_train):
        """Train the pipeline"""
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict class labels"""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet")

        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet")

        return self.pipeline.predict_proba(X)[:, 1]
