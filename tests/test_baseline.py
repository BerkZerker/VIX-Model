"""Tests for training/baseline.py."""

import numpy as np
import pandas as pd
import pytest
from training.baseline import RulesBaseline


@pytest.fixture
def baseline():
    return RulesBaseline()


@pytest.fixture
def signal_df():
    """DataFrame where all signal conditions are met."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame(
        {
            "vix_zscore": [2.0, 2.0, 2.0, 2.0, 2.0],
            "term_slope": [-0.05, -0.05, -0.05, -0.05, -0.05],
            "spy_velocity": [-0.03, -0.03, -0.03, -0.03, -0.03],
            "vvix": [100, 100, 100, 100, 100],
        },
        index=dates,
    )


class TestSignalGeneration:
    def test_all_conditions_met(self, baseline, signal_df):
        """Signal should fire when all three conditions are met."""
        result = baseline.generate_signals(signal_df)
        assert result["signal"].sum() == 5

    def test_output_columns(self, baseline, signal_df):
        """Output should have all expected columns."""
        result = baseline.generate_signals(signal_df)
        for col in ["signal", "spike_risk", "p_revert_proxy", "p_spike_proxy"]:
            assert col in result.columns

    def test_low_zscore_no_signal(self, baseline):
        """No signal when z-score is too low."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        df = pd.DataFrame(
            {
                "vix_zscore": [1.0, 1.0, 1.0],
                "term_slope": [-0.05, -0.05, -0.05],
                "spy_velocity": [-0.03, -0.03, -0.03],
                "vvix": [100, 100, 100],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["signal"].sum() == 0

    def test_no_backwardation_needs_strong_zscore(self, baseline):
        """Without backwardation, needs z-score > 2.0 for signal."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame(
            {
                "vix_zscore": [1.8, 2.2],
                "term_slope": [0.05, 0.05],  # Contango, no backwardation
                "spy_velocity": [-0.03, -0.03],
                "vvix": [100, 100],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["signal"].iloc[0] == 0  # z-score 1.8 + no backwardation
        assert result["signal"].iloc[1] == 1  # z-score 2.2 is enough

    def test_no_equity_selloff_no_signal(self, baseline):
        """No signal when SPY isn't selling off."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame(
            {
                "vix_zscore": [2.0, 2.0],
                "term_slope": [-0.05, -0.05],
                "spy_velocity": [0.02, 0.02],  # SPY going up
                "vvix": [100, 100],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["signal"].sum() == 0

    def test_one_condition_missing_no_signal(self, baseline):
        """Signal should not fire when one condition is missing."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        # zscore > 1.5 and selloff, but no backwardation and zscore < 2.0
        df = pd.DataFrame(
            {
                "vix_zscore": [1.6, 1.7, 1.8],
                "term_slope": [0.05, 0.05, 0.05],  # contango, NOT backwardation
                "spy_velocity": [-0.03, -0.03, -0.03],
                "vvix": [100, 100, 100],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["signal"].sum() == 0

    def test_backwardation_with_moderate_zscore(self, baseline):
        """Backwardation + zscore 1.5-2.0 + selloff should fire."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        df = pd.DataFrame(
            {
                "vix_zscore": [1.8, 1.7, 1.6],
                "term_slope": [-0.03, -0.03, -0.03],  # backwardation
                "spy_velocity": [-0.03, -0.03, -0.03],
                "vvix": [100, 100, 100],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["signal"].sum() == 3

    def test_missing_columns_fallback(self, baseline):
        """Should handle missing optional columns gracefully."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        df = pd.DataFrame(
            {"vix_zscore": [2.5, 2.5, 2.5]},
            index=dates,
        )
        # No term_slope, spy_velocity, vvix -> should still run with defaults
        result = baseline.generate_signals(df)
        assert len(result) == 3


class TestSpikeRisk:
    def test_high_vvix_flat_term_structure(self, baseline):
        """Spike risk should flag when VVIX > 120 and term structure flat."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame(
            {
                "vix_zscore": [2.0, 2.0],
                "term_slope": [0.005, 0.005],  # Nearly flat
                "spy_velocity": [-0.03, -0.03],
                "vvix": [130, 130],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["spike_risk"].iloc[0] == 1

    def test_low_vvix_no_spike_risk(self, baseline):
        """No spike risk when VVIX is low."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame(
            {
                "vix_zscore": [2.0, 2.0],
                "term_slope": [0.005, 0.005],
                "spy_velocity": [-0.03, -0.03],
                "vvix": [90, 90],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["spike_risk"].iloc[0] == 0

    def test_steep_term_no_spike_risk(self, baseline):
        """No spike risk when term structure is steep (not flat)."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        df = pd.DataFrame(
            {
                "vix_zscore": [2.0, 2.0],
                "term_slope": [0.05, -0.05],  # Not flat
                "spy_velocity": [-0.03, -0.03],
                "vvix": [130, 130],
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        assert result["spike_risk"].sum() == 0


class TestProxyProbabilities:
    def test_p_revert_proxy_range(self, baseline, signal_df):
        result = baseline.generate_signals(signal_df)
        assert (result["p_revert_proxy"] >= 0).all()
        assert (result["p_revert_proxy"] <= 1).all()

    def test_p_spike_proxy_range(self, baseline, signal_df):
        result = baseline.generate_signals(signal_df)
        assert (result["p_spike_proxy"] >= 0).all()
        assert (result["p_spike_proxy"] <= 1).all()

    def test_p_revert_higher_when_signal(self, baseline):
        """p_revert_proxy should be >= 0.55 when signal fires."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(
            {
                "vix_zscore": [2.5] * 5,
                "term_slope": [-0.05] * 5,
                "spy_velocity": [-0.03] * 5,
                "vvix": [100] * 5,
            },
            index=dates,
        )
        result = baseline.generate_signals(df)
        signal_mask = result["signal"] == 1
        assert signal_mask.any()
        assert (result.loc[signal_mask, "p_revert_proxy"] >= 0.55).all()


class TestCustomThresholds:
    def test_custom_zscore_threshold(self):
        """Custom threshold changes signal behavior."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        df = pd.DataFrame(
            {
                "vix_zscore": [1.0, 1.0, 1.0],
                "term_slope": [-0.03, -0.03, -0.03],
                "spy_velocity": [-0.03, -0.03, -0.03],
                "vvix": [100, 100, 100],
            },
            index=dates,
        )
        # Default threshold is 1.5 -> no signal
        default_bl = RulesBaseline()
        assert default_bl.generate_signals(df)["signal"].sum() == 0

        # Lower threshold -> signal fires
        custom_bl = RulesBaseline(zscore_threshold=0.5)
        assert custom_bl.generate_signals(df)["signal"].sum() == 3
