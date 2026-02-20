"""Tests for training/backtest.py."""

import numpy as np
import pandas as pd
import pytest
from training.backtest import (
    BacktestConfig,
    Instrument,
    Tier,
    VIXBacktester,
    black_scholes_call,
    black_scholes_put,
)


class TestBlackScholes:
    def test_put_atm(self):
        """ATM put should have positive value."""
        price = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price > 0
        assert price < 100  # Can't exceed strike

    def test_put_deep_itm(self):
        """Deep ITM put should be close to intrinsic value."""
        price = black_scholes_put(S=50, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price > 45  # At least close to K - S

    def test_put_deep_otm(self):
        """Deep OTM put should be near zero."""
        price = black_scholes_put(S=200, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price < 1

    def test_call_atm(self):
        """ATM call should have positive value."""
        price = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.3)
        assert price > 0

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.3
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        expected = S - K * np.exp(-r * T)
        assert abs((call - put) - expected) < 0.01

    def test_zero_time_put(self):
        """At expiry, put value = max(K-S, 0)."""
        assert black_scholes_put(80, 100, 0, 0.05, 0.3) == 20
        assert black_scholes_put(120, 100, 0, 0.05, 0.3) == 0


class TestBacktestConfig:
    def test_default_values(self):
        config = BacktestConfig()
        assert config.p_revert_threshold == 0.7
        assert config.p_spike_threshold == 0.3
        assert config.target_profit_pct == 0.50
        assert config.stop_loss_pct == -0.60


class TestVIXBacktester:
    @pytest.fixture
    def backtester(self):
        return VIXBacktester(BacktestConfig())

    @pytest.fixture
    def sample_data(self):
        """Create sample data with features and predictions."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            {
                "vix_spot": np.linspace(30, 20, 100),
                "vix_close": np.linspace(30, 20, 100),
                "vix_zscore": np.linspace(2.5, 0.5, 100),
            },
            index=dates,
        )
        predictions = pd.DataFrame(
            {
                "p_revert": [0.85] * 5 + [0.5] * 95,
                "p_spike_first": [0.1] * 5 + [0.5] * 95,
                "expected_magnitude": [20.0] * 5 + [5.0] * 95,
            },
            index=dates,
        )
        return df, predictions

    def test_signal_generation(self, backtester, sample_data):
        df, predictions = sample_data
        signals = backtester._generate_signals(df, predictions)
        # First few rows should generate signals (high p_revert, low p_spike, high zscore)
        assert len(signals) > 0

    def test_trade_tiers(self, backtester):
        """Test that trade setup maps VIX levels to correct tiers."""
        signal_high = {"date": pd.Timestamp("2020-01-01"), "p_revert": 0.9,
                       "p_spike_first": 0.1, "expected_magnitude": 25, "vix_zscore": 3.0, "vix_level": 35}
        signal_mod = {"date": pd.Timestamp("2020-01-01"), "p_revert": 0.8,
                      "p_spike_first": 0.2, "expected_magnitude": 15, "vix_zscore": 2.0, "vix_level": 25}

        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame({"vix_close": [25] * 100}, index=dates)

        setup_high = backtester._create_trade_setup(signal_high, df)
        setup_mod = backtester._create_trade_setup(signal_mod, df)

        assert setup_high.tier == Tier.MAJOR_SPIKE
        assert setup_high.instrument == Instrument.UVXY_PUT
        assert setup_mod.tier == Tier.MODERATE
        assert setup_mod.instrument == Instrument.VIX_PUT

    def test_metrics_computation(self, backtester):
        """Test metric computation with known trade results."""
        from training.backtest import TradeResult, TradeSetup

        setup = TradeSetup(
            date=pd.Timestamp("2020-01-01"), tier=Tier.MODERATE,
            instrument=Instrument.VIX_PUT, vix_level=25, strike=23,
            dte=45, entry_price=2.0, sizing="half", p_revert=0.8,
            p_spike_first=0.15, expected_magnitude=15,
        )

        trades = [
            TradeResult(setup=setup, exit_date=pd.Timestamp("2020-02-01"),
                        exit_price=3.0, exit_reason="target",
                        pnl_gross=1000, pnl_net=900, commission=13, spread_cost=50,
                        slippage=37, holding_days=22, max_adverse=-0.1),
            TradeResult(setup=setup, exit_date=pd.Timestamp("2020-03-01"),
                        exit_price=0.8, exit_reason="stop",
                        pnl_gross=-1200, pnl_net=-1300, commission=13, spread_cost=50,
                        slippage=37, holding_days=30, max_adverse=-0.6),
        ]

        metrics = backtester._compute_metrics(trades)
        assert metrics["total_trades"] == 2
        assert metrics["win_rate"] == 0.5
        assert metrics["avg_win"] == 900
        assert metrics["avg_loss"] == -1300

    def test_no_trades_returns_error(self, backtester):
        """No trades should return error metrics."""
        metrics = backtester._compute_metrics([])
        assert "error" in metrics

    def test_max_drawdown(self, backtester):
        """Test max drawdown computation."""
        pnls = [100, -200, 50, -100]
        dd = backtester._max_drawdown(pnls)
        # Cumulative: [100, -100, -50, -150]
        # Running max: [100, 100, 100, 100]
        # Drawdowns: [0, -200, -150, -250]
        assert dd == pytest.approx(-250.0)

    def test_equity_curve_empty_on_no_trades(self, backtester):
        eq = backtester._build_equity_curve([])
        assert len(eq) == 0

    def test_mild_tier_returns_none_for_low_vix(self, backtester):
        """VIX below 18 should return None trade setup."""
        signal = {
            "date": pd.Timestamp("2020-01-01"),
            "p_revert": 0.9, "p_spike_first": 0.1,
            "expected_magnitude": 10, "vix_zscore": 2.0, "vix_level": 15,
        }
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame({"vix_close": [15] * 100}, index=dates)
        setup = backtester._create_trade_setup(signal, df)
        assert setup is None

    def test_trade_setup_dte_by_tier(self, backtester):
        """Check DTE assignments per tier."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame({"vix_close": [25] * 100}, index=dates)

        signal_major = {
            "date": pd.Timestamp("2020-01-01"), "p_revert": 0.9,
            "p_spike_first": 0.1, "expected_magnitude": 25,
            "vix_zscore": 3.0, "vix_level": 35,
        }
        signal_moderate = {
            "date": pd.Timestamp("2020-01-01"), "p_revert": 0.8,
            "p_spike_first": 0.2, "expected_magnitude": 15,
            "vix_zscore": 2.0, "vix_level": 25,
        }
        signal_mild = {
            "date": pd.Timestamp("2020-01-01"), "p_revert": 0.75,
            "p_spike_first": 0.25, "expected_magnitude": 10,
            "vix_zscore": 1.5, "vix_level": 20,
        }

        setup_major = backtester._create_trade_setup(signal_major, df)
        setup_moderate = backtester._create_trade_setup(signal_moderate, df)
        setup_mild = backtester._create_trade_setup(signal_mild, df)

        assert setup_major.dte == 75
        assert setup_moderate.dte == 45
        assert setup_mild.dte == 30

    def test_exit_reasons_breakdown(self, backtester):
        """Test exit reason breakdown in metrics."""
        from training.backtest import TradeResult, TradeSetup

        setup = TradeSetup(
            date=pd.Timestamp("2020-01-01"), tier=Tier.MODERATE,
            instrument=Instrument.VIX_PUT, vix_level=25, strike=23,
            dte=45, entry_price=2.0, sizing="half", p_revert=0.8,
            p_spike_first=0.15, expected_magnitude=15,
        )

        trades = [
            TradeResult(setup=setup, exit_date=pd.Timestamp("2020-02-01"),
                        exit_price=3.0, exit_reason="target",
                        pnl_gross=1000, pnl_net=900, commission=13, spread_cost=50,
                        slippage=37, holding_days=22, max_adverse=-0.1),
            TradeResult(setup=setup, exit_date=pd.Timestamp("2020-03-01"),
                        exit_price=0.8, exit_reason="stop",
                        pnl_gross=-1200, pnl_net=-1300, commission=13, spread_cost=50,
                        slippage=37, holding_days=30, max_adverse=-0.6),
            TradeResult(setup=setup, exit_date=pd.Timestamp("2020-04-01"),
                        exit_price=1.5, exit_reason="expiry",
                        pnl_gross=-500, pnl_net=-600, commission=13, spread_cost=50,
                        slippage=37, holding_days=45, max_adverse=-0.3),
        ]

        metrics = backtester._compute_metrics(trades)
        assert metrics["exit_reasons"]["target"] == 1
        assert metrics["exit_reasons"]["stop"] == 1
        assert metrics["exit_reasons"]["expiry"] == 1


class TestBlackScholesEdgeCases:
    def test_zero_vol_put(self):
        """Zero vol put should return intrinsic value."""
        assert black_scholes_put(80, 100, 0.25, 0.05, 0) == pytest.approx(20.0)

    def test_known_value_approximation(self):
        """Test BS put against a known approximate value.
        S=100, K=100, T=1, r=0.05, sigma=0.2 -> put ~5.57
        """
        price = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert 4.0 < price < 7.0  # Rough range check
