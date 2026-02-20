"""Tests for bot/inference.py."""

import numpy as np
import pytest
from bot.inference import MockInference, Prediction, BaseInference


class TestPredictionDataclass:
    def test_prediction_fields(self):
        pred = Prediction(p_revert=0.8, p_spike_first=0.2, expected_magnitude=15.0, model_version="v001")
        assert pred.p_revert == 0.8
        assert pred.p_spike_first == 0.2
        assert pred.expected_magnitude == 15.0
        assert pred.model_version == "v001"


class TestMockInference:
    def test_load(self):
        mock = MockInference()
        mock.load()  # Should not raise

    def test_predict_returns_prediction(self):
        mock = MockInference()
        mock.load()
        pred = mock.predict({"vix_spot": 25.0, "vix_zscore": 1.5})
        assert isinstance(pred, Prediction)
        assert 0 <= pred.p_revert <= 1
        assert 0 <= pred.p_spike_first <= 1
        assert pred.expected_magnitude >= 0

    def test_predict_higher_zscore_higher_revert(self):
        """Higher z-score should generally mean higher p_revert."""
        mock = MockInference()
        mock.load()

        # Run multiple predictions and check average
        low_reverts = [mock.predict({"vix_spot": 20, "vix_zscore": 0.5}).p_revert for _ in range(50)]
        high_reverts = [mock.predict({"vix_spot": 35, "vix_zscore": 3.0}).p_revert for _ in range(50)]

        assert np.mean(high_reverts) > np.mean(low_reverts)

    def test_model_version(self):
        mock = MockInference(version="test_v001")
        assert mock.model_version() == "test_v001"

    def test_default_version(self):
        mock = MockInference()
        assert mock.model_version() == "mock_v001"

    def test_predict_with_missing_keys(self):
        """MockInference should handle missing feature keys gracefully."""
        mock = MockInference()
        mock.load()
        pred = mock.predict({})  # Empty features dict
        assert isinstance(pred, Prediction)
        assert 0 <= pred.p_revert <= 1
        assert 0 <= pred.p_spike_first <= 1

    def test_predictions_are_clipped(self):
        """Predictions should be clipped to [0, 1] range."""
        mock = MockInference()
        mock.load()
        # Run many predictions to ensure all stay in bounds
        for _ in range(100):
            pred = mock.predict({"vix_spot": 50.0, "vix_zscore": 5.0})
            assert 0 <= pred.p_revert <= 1
            assert 0 <= pred.p_spike_first <= 1


class TestXGBoostInference:
    def test_load_trained_model(self):
        """Test loading the trained XGBoost model from models/ directory."""
        from pathlib import Path
        models_dir = Path("models")
        if not (models_dir / "model_manifest.json").exists():
            pytest.skip("No trained model available")

        from bot.inference import XGBoostInference
        inf = XGBoostInference(models_dir)
        inf.load()

        # Build a full feature vector
        features = {
            "vix_spot": 28.5, "vix_zscore": 2.0, "vix_velocity_1d": 0.05,
            "vix_velocity_3d": 0.12, "vix_velocity_5d": 0.18,
            "term_slope": -0.03, "term_slope_zscore": -1.5, "term_curvature": 0.01,
            "spy_drawdown": -0.04, "spy_velocity": -0.03,
            "rv_iv_spread": -5.0, "vix_percentile": 0.85,
            "days_elevated": 3, "vvix": 110, "vix9d_vix_ratio": 0.95,
            "skew": 130, "vix_futures_volume": 200000, "put_call_ratio": 0.8,
            "hy_spread_velocity": -0.02, "day_of_week": 2,
        }
        # Add lagged features
        for feat in ["vix_spot", "vix_zscore", "term_slope", "spy_drawdown", "vvix"]:
            for lag in [1, 3, 5, 10, 20]:
                features[f"{feat}_lag{lag}"] = 0.0

        pred = inf.predict(features)
        assert isinstance(pred, Prediction)
        assert 0 <= pred.p_revert <= 1
        assert 0 <= pred.p_spike_first <= 1

    def test_predict_without_load_raises(self):
        """Calling predict without load should raise RuntimeError."""
        from bot.inference import XGBoostInference
        inf = XGBoostInference("/nonexistent")
        with pytest.raises(RuntimeError, match="not loaded"):
            inf.predict({"vix_spot": 20.0})

    def test_load_missing_model_raises(self):
        """Loading from empty directory should raise FileNotFoundError."""
        import tempfile
        from bot.inference import XGBoostInference
        with tempfile.TemporaryDirectory() as tmpdir:
            inf = XGBoostInference(tmpdir)
            with pytest.raises(FileNotFoundError):
                inf.load()
