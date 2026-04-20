"""Traditional ML baselines for CV mechanism classification.

Provides XGBoost and Random Forest classifiers that operate on hand-crafted
features extracted by :mod:`src.data.cv_features`.  These serve as
comparison baselines against the ResNet1D CNN.

Usage::

    from src.models.baseline_ml import BaselineClassifier
    from src.data.cv_features import extract_features_batch

    clf = BaselineClassifier(method="xgboost")
    clf.fit(X_train_features, y_train)
    accuracy = clf.score(X_test_features, y_test)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """Wrapper around sklearn / XGBoost classifiers for CV mechanism task.

    Args:
        method: ``"xgboost"``, ``"random_forest"``, or ``"logistic"``.
        params: Optional dict of hyperparameters passed to the underlying
            estimator.
    """

    def __init__(
        self,
        method: str = "xgboost",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.method = method
        self.params = params or {}
        self.model = self._build_model()
        self.classes_ = None

    def _build_model(self) -> Any:
        if self.method == "xgboost":
            try:
                from xgboost import XGBClassifier
                defaults = {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "eval_metric": "mlogloss",
                    "use_label_encoder": False,
                }
                defaults.update(self.params)
                return XGBClassifier(**defaults)
            except ImportError:
                logger.warning(
                    "xgboost not installed, falling back to random_forest"
                )
                self.method = "random_forest"
                return self._build_model()

        elif self.method == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            defaults = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 5,
                "n_jobs": -1,
            }
            defaults.update(self.params)
            return RandomForestClassifier(**defaults)

        elif self.method == "logistic":
            from sklearn.linear_model import LogisticRegression
            defaults = {
                "max_iter": 1000,
                "multi_class": "multinomial",
                "C": 1.0,
            }
            defaults.update(self.params)
            return LogisticRegression(**defaults)

        else:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                "Choose from: xgboost, random_forest, logistic"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        """Fit the classifier.

        Args:
            X: Feature matrix of shape ``(N, n_features)``.
            y: Labels of shape ``(N,)``.

        Returns:
            self
        """
        logger.info(
            "Fitting %s on %d samples with %d features",
            self.method, X.shape[0], X.shape[1],
        )
        # Replace NaN/Inf with 0 (safety for edge-case features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on the given data."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.score(X, y)

    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importance array if available."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
