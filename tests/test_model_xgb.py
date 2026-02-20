"""Tests for training/model_xgb.py."""

import numpy as np
import pandas as pd
import pytest
from training.model_xgb import (
    VIXEnsemble,
    VIXMagnitudeRegressor,
    VIXRevertClassifier,
    VIXSpikeClassifier,
    XGBHyperparams,
)


@pytest.fixture
def synthetic_data():
    """Create a small synthetic dataset for training tests."""
    np.random.seed(42)
    n_train, n_val = 200, 50
    n_features = 10

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    # Create labels correlated with first feature
    y_train_binary = pd.Series((X_train["feat_0"] > 0.3).astype(int))
    y_val_binary = pd.Series((X_val["feat_0"] > 0.3).astype(int))
    y_train_cont = pd.Series(X_train["feat_0"] * 10 + np.random.normal(0, 1, n_train))
    y_val_cont = pd.Series(X_val["feat_0"] * 10 + np.random.normal(0, 1, n_val))

    return X_train, X_val, y_train_binary, y_val_binary, y_train_cont, y_val_cont


@pytest.fixture
def fast_params():
    """Fast training params for tests."""
    return XGBHyperparams(
        n_estimators=20,
        max_depth=3,
        early_stopping_rounds=5,
    )


class TestVIXRevertClassifier:
    def test_train_and_predict(self, synthetic_data, fast_params):
        X_train, X_val, y_train, y_val, _, _ = synthetic_data
        clf = VIXRevertClassifier(fast_params)
        result = clf.train(X_train, y_train, X_val, y_val)

        assert result.model is not None
        assert "auc" in result.metrics
        assert "precision" in result.metrics
        assert result.feature_importance is not None

        preds = clf.predict(X_val)
        assert len(preds) == len(X_val)
        assert (preds >= 0).all() and (preds <= 1).all()


class TestVIXSpikeClassifier:
    def test_train_and_predict(self, synthetic_data, fast_params):
        X_train, X_val, y_train, y_val, _, _ = synthetic_data
        clf = VIXSpikeClassifier(fast_params)
        result = clf.train(X_train, y_train, X_val, y_val)
        preds = clf.predict(X_val)
        assert len(preds) == len(X_val)
        assert (preds >= 0).all() and (preds <= 1).all()


class TestVIXMagnitudeRegressor:
    def test_train_and_predict(self, synthetic_data, fast_params):
        X_train, X_val, _, _, y_train, y_val = synthetic_data
        reg = VIXMagnitudeRegressor(fast_params)
        result = reg.train(X_train, y_train, X_val, y_val)

        assert "rmse" in result.metrics
        assert "correlation" in result.metrics
        preds = reg.predict(X_val)
        assert len(preds) == len(X_val)


class TestVIXEnsemble:
    def test_phased_training(self, synthetic_data, fast_params):
        X_train, X_val, y_bin_train, y_bin_val, y_cont_train, y_cont_val = synthetic_data

        ensemble = VIXEnsemble(fast_params)

        # Phase 2a
        ensemble.train_revert(X_train, y_bin_train, X_val, y_bin_val)
        assert "revert" in ensemble.trained_heads

        # Phase 2b
        ensemble.train_spike(X_train, y_bin_train, X_val, y_bin_val)
        assert "spike" in ensemble.trained_heads

        # Phase 2c
        ensemble.train_magnitude(X_train, y_cont_train, X_val, y_cont_val)
        assert "magnitude" in ensemble.trained_heads

        # Predict all heads
        preds = ensemble.predict(X_val)
        assert "p_revert" in preds
        assert "p_spike_first" in preds
        assert "expected_magnitude" in preds

    def test_should_alert(self, synthetic_data, fast_params):
        X_train, X_val, y_bin_train, y_bin_val, _, _ = synthetic_data

        ensemble = VIXEnsemble(fast_params)
        ensemble.train_revert(X_train, y_bin_train, X_val, y_bin_val)
        ensemble.train_spike(X_train, y_bin_train, X_val, y_bin_val)

        preds = ensemble.predict(X_val)
        vix_zscore = np.array([2.0] * len(X_val))

        alerts = ensemble.should_alert(preds, vix_zscore)
        assert alerts.dtype == bool
        assert len(alerts) == len(X_val)

    def test_alert_respects_thresholds(self):
        ensemble = VIXEnsemble()
        preds = {
            "p_revert": np.array([0.8, 0.5, 0.9]),
            "p_spike_first": np.array([0.1, 0.1, 0.5]),
        }
        vix_zscore = np.array([2.0, 2.0, 2.0])

        alerts = ensemble.should_alert(preds, vix_zscore, p_revert_threshold=0.7, p_spike_threshold=0.3)
        assert alerts[0] == True   # High revert, low spike
        assert alerts[1] == False  # Low revert
        assert alerts[2] == False  # High spike

    def test_alert_requires_zscore(self):
        """Alert should not fire when vix_zscore is too low."""
        ensemble = VIXEnsemble()
        preds = {
            "p_revert": np.array([0.9]),
            "p_spike_first": np.array([0.1]),
        }
        low_zscore = np.array([0.5])
        alerts = ensemble.should_alert(preds, low_zscore, zscore_threshold=1.0)
        assert alerts[0] == False

    def test_alert_with_only_revert_head(self):
        """Alert should work even if only revert head is trained."""
        ensemble = VIXEnsemble()
        preds = {"p_revert": np.array([0.9, 0.5])}
        vix_zscore = np.array([2.0, 2.0])
        alerts = ensemble.should_alert(preds, vix_zscore, p_revert_threshold=0.7)
        assert alerts[0] == True
        assert alerts[1] == False


class TestVIXRevertClassifierDetails:
    def test_predict_without_train_raises(self):
        clf = VIXRevertClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(pd.DataFrame({"feat_0": [1.0]}))

    def test_feature_importance_ordered(self, synthetic_data, fast_params):
        X_train, X_val, y_train, y_val, _, _ = synthetic_data
        clf = VIXRevertClassifier(fast_params)
        result = clf.train(X_train, y_train, X_val, y_val)
        # Feature importance should be sorted descending
        assert list(result.feature_importance.values) == sorted(
            result.feature_importance.values, reverse=True
        )

    def test_auto_scale_pos_weight(self, synthetic_data, fast_params):
        """When scale_pos_weight is None, it should be auto-computed."""
        X_train, X_val, y_train, y_val, _, _ = synthetic_data
        assert fast_params.scale_pos_weight is None
        clf = VIXRevertClassifier(fast_params)
        result = clf.train(X_train, y_train, X_val, y_val)
        assert result.model is not None


class TestXGBHyperparams:
    def test_default_values(self):
        params = XGBHyperparams()
        assert params.max_depth == 5
        assert params.n_estimators == 500
        assert params.learning_rate == 0.05
        assert params.random_state == 42
        assert params.scale_pos_weight is None
