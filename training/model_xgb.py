"""XGBoost models for VIX mean-reversion prediction.

Primary model: gradient-boosted trees on daily features.
Three separate models for three output heads:
  - p_revert classifier (Phase 2a)
  - p_spike_first classifier (Phase 2b)
  - expected_magnitude regressor (Phase 2c)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error,
)

logger = logging.getLogger(__name__)


@dataclass
class XGBHyperparams:
    """Hyperparameters for XGBoost models."""
    max_depth: int = 5
    n_estimators: int = 500
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    scale_pos_weight: float | None = None  # Auto-computed from class imbalance
    random_state: int = 42


@dataclass
class TrainResult:
    """Result from training a single model."""
    model: xgb.XGBClassifier | xgb.XGBRegressor
    metrics: dict
    feature_importance: pd.Series
    best_iteration: int


class VIXRevertClassifier:
    """XGBoost classifier for p_revert (Phase 2a).

    Predicts probability that VIX reverts >= 15% within the horizon window.
    """

    def __init__(self, params: XGBHyperparams | None = None):
        self.params = params or XGBHyperparams()
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainResult:
        """Train the p_revert classifier."""
        self.feature_names = list(X_train.columns)

        # Auto-compute scale_pos_weight if not set
        scale_pos_weight = self.params.scale_pos_weight
        if scale_pos_weight is None:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / max(n_pos, 1)
            logger.info(f"Auto scale_pos_weight: {scale_pos_weight:.2f} (pos={n_pos}, neg={n_neg})")

        self.model = xgb.XGBClassifier(
            max_depth=self.params.max_depth,
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            subsample=self.params.subsample,
            colsample_bytree=self.params.colsample_bytree,
            min_child_weight=self.params.min_child_weight,
            gamma=self.params.gamma,
            reg_alpha=self.params.reg_alpha,
            reg_lambda=self.params.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.params.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method="hist",
            early_stopping_rounds=self.params.early_stopping_rounds,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        }
        if len(np.unique(y_val)) > 1:
            metrics["auc"] = float(roc_auc_score(y_val, y_pred_proba))
        else:
            metrics["auc"] = float("nan")

        feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

        return TrainResult(
            model=self.model,
            metrics=metrics,
            feature_importance=feature_importance,
            best_iteration=self.model.best_iteration if hasattr(self.model, "best_iteration") else self.params.n_estimators,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return p_revert probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]


class VIXSpikeClassifier:
    """XGBoost classifier for p_spike_first (Phase 2b).

    Predicts probability that VIX rises >= 10% before reverting.
    This is the hardest and highest-value prediction.
    """

    def __init__(self, params: XGBHyperparams | None = None):
        self.params = params or XGBHyperparams()
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainResult:
        """Train the p_spike_first classifier."""
        self.feature_names = list(X_train.columns)

        scale_pos_weight = self.params.scale_pos_weight
        if scale_pos_weight is None:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / max(n_pos, 1)
            logger.info(f"Auto scale_pos_weight: {scale_pos_weight:.2f} (pos={n_pos}, neg={n_neg})")

        self.model = xgb.XGBClassifier(
            max_depth=self.params.max_depth,
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            subsample=self.params.subsample,
            colsample_bytree=self.params.colsample_bytree,
            min_child_weight=self.params.min_child_weight,
            gamma=self.params.gamma,
            reg_alpha=self.params.reg_alpha,
            reg_lambda=self.params.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.params.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method="hist",
            early_stopping_rounds=self.params.early_stopping_rounds,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        }
        if len(np.unique(y_val)) > 1:
            metrics["auc"] = float(roc_auc_score(y_val, y_pred_proba))
        else:
            metrics["auc"] = float("nan")

        feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

        return TrainResult(
            model=self.model,
            metrics=metrics,
            feature_importance=feature_importance,
            best_iteration=self.model.best_iteration if hasattr(self.model, "best_iteration") else self.params.n_estimators,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return p_spike_first probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]


class VIXMagnitudeRegressor:
    """XGBoost regressor for expected_magnitude (Phase 2c).

    Predicts expected reversion size in % over the horizon window.
    """

    def __init__(self, params: XGBHyperparams | None = None):
        self.params = params or XGBHyperparams()
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainResult:
        """Train the expected_magnitude regressor."""
        self.feature_names = list(X_train.columns)

        self.model = xgb.XGBRegressor(
            max_depth=self.params.max_depth,
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            subsample=self.params.subsample,
            colsample_bytree=self.params.colsample_bytree,
            min_child_weight=self.params.min_child_weight,
            gamma=self.params.gamma,
            reg_alpha=self.params.reg_alpha,
            reg_lambda=self.params.reg_lambda,
            random_state=self.params.random_state,
            eval_metric="rmse",
            tree_method="hist",
            early_stopping_rounds=self.params.early_stopping_rounds,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = self.model.predict(X_val)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "correlation": float(np.corrcoef(y_val, y_pred)[0, 1]) if len(y_val) > 1 else float("nan"),
        }

        feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)

        return TrainResult(
            model=self.model,
            metrics=metrics,
            feature_importance=feature_importance,
            best_iteration=self.model.best_iteration if hasattr(self.model, "best_iteration") else self.params.n_estimators,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return expected_magnitude predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)


class VIXEnsemble:
    """Ensemble of all three XGBoost heads.

    Provides a unified interface for training and prediction.
    """

    def __init__(self, params: XGBHyperparams | None = None):
        self.params = params or XGBHyperparams()
        self.revert_clf = VIXRevertClassifier(self.params)
        self.spike_clf = VIXSpikeClassifier(self.params)
        self.magnitude_reg = VIXMagnitudeRegressor(self.params)
        self.trained_heads: set[str] = set()

    def train_revert(self, X_train, y_train, X_val, y_val) -> TrainResult:
        """Phase 2a: Train p_revert head."""
        result = self.revert_clf.train(X_train, y_train, X_val, y_val)
        self.trained_heads.add("revert")
        return result

    def train_spike(self, X_train, y_train, X_val, y_val) -> TrainResult:
        """Phase 2b: Train p_spike_first head."""
        result = self.spike_clf.train(X_train, y_train, X_val, y_val)
        self.trained_heads.add("spike")
        return result

    def train_magnitude(self, X_train, y_train, X_val, y_val) -> TrainResult:
        """Phase 2c: Train expected_magnitude head."""
        result = self.magnitude_reg.train(X_train, y_train, X_val, y_val)
        self.trained_heads.add("magnitude")
        return result

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Run all trained heads and return predictions."""
        results = {}
        if "revert" in self.trained_heads:
            results["p_revert"] = self.revert_clf.predict(X)
        if "spike" in self.trained_heads:
            results["p_spike_first"] = self.spike_clf.predict(X)
        if "magnitude" in self.trained_heads:
            results["expected_magnitude"] = self.magnitude_reg.predict(X)
        return results

    def should_alert(
        self,
        predictions: dict[str, np.ndarray],
        vix_zscore: np.ndarray,
        p_revert_threshold: float = 0.7,
        p_spike_threshold: float = 0.3,
        zscore_threshold: float = 1.0,
    ) -> np.ndarray:
        """Determine which samples should trigger alerts.

        Alert when: p_revert > threshold AND p_spike_first < threshold AND vix_zscore > threshold
        """
        alert = np.ones(len(vix_zscore), dtype=bool)

        if "p_revert" in predictions:
            alert &= predictions["p_revert"] > p_revert_threshold
        if "p_spike_first" in predictions:
            alert &= predictions["p_spike_first"] < p_spike_threshold

        alert &= vix_zscore > zscore_threshold

        return alert
