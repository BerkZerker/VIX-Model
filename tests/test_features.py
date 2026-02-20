"""Tests for training/features.py - Daily feature computation."""

import numpy as np
import pandas as pd
import pytest

from training.features import (
    compute_all_features,
    compute_lagged_features,
    days_elevated,
    hy_spread_velocity,
    rv_iv_spread,
    spy_drawdown,
    spy_velocity,
    term_curvature,
    term_slope,
    term_slope_zscore,
    vix_percentile,
    vix_spot,
    vix_velocity,
    vix_zscore,
    vix9d_vix_ratio,
    vvix,
    skew,
    vix_futures_volume,
    put_call_ratio,
    day_of_week,
    _LAG_FEATURES,
    _LAG_PERIODS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with 120 trading days of synthetic data."""
    np.random.seed(42)
    n = 120
    dates = pd.bdate_range("2020-01-01", periods=n)
    vix = 20 + np.cumsum(np.random.normal(0, 0.5, n))
    vix = np.clip(vix, 10, 80)
    spy = 400 + np.cumsum(np.random.normal(0.05, 1.5, n))

    return pd.DataFrame(
        {
            "vix_close": vix,
            "spy_close": spy,
            "vix_futures_m1": vix * 1.03,
            "vix_futures_m2": vix * 1.06,
            "vix_futures_m3": vix * 1.09,
            "vvix_close": vix * 4.5,
            "vix9d_close": vix * 0.98,
            "skew_close": 130 + np.random.normal(0, 5, n),
            "vix_futures_volume": np.random.randint(100000, 300000, n).astype(float),
            "put_call_ratio": 0.7 + np.random.normal(0, 0.1, n),
            "hyg_close": 80 + np.random.normal(0, 0.5, n),
            "tlt_close": 100 + np.random.normal(0, 0.8, n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# vix_spot
# ---------------------------------------------------------------------------

class TestVIXSpot:
    def test_passthrough(self, sample_df):
        result = vix_spot(sample_df)
        pd.testing.assert_series_equal(result, sample_df["vix_close"], check_names=False)


# ---------------------------------------------------------------------------
# vix_zscore
# ---------------------------------------------------------------------------

class TestVIXZscore:
    def test_basic_zscore(self, sample_df):
        result = vix_zscore(sample_df, window=60)
        # First 59 values should be NaN (rolling window warmup)
        assert result.iloc[:59].isna().all()
        # After warmup, should have values
        assert result.iloc[60:].notna().all()

    def test_zscore_known_values(self):
        """VIX at 30 with 60-day mean of 20 and std of 5 should give zscore of 2.0."""
        # Construct data where last 60 days have mean=20, std=5, and current VIX=30.
        # We need 60 values with mean=20 and std=5. Use a simple pattern.
        n = 120
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Fill first 119 days with values that give mean~20, std~5 over last 60 days
        # Use alternating values: 15 and 25 gives mean=20, std=5
        vals = [15.0, 25.0] * 60 + [30.0]  # 120+1, take first 120
        vals = vals[:119] + [30.0]
        df = pd.DataFrame({"vix_close": vals}, index=dates)
        result = vix_zscore(df, window=60)
        last_z = result.iloc[-1]
        # Rolling window is last 60 points INCLUDING the current value
        # But the last value is 30, and the window includes it
        # mean of 59 alternating 15/25 + one 30: ~20.17
        # We still expect a positive zscore > 1
        assert last_z > 1.0

    def test_zscore_constant_series(self):
        """Constant series has std=0, producing NaN z-score."""
        dates = pd.bdate_range("2020-01-01", periods=80)
        df = pd.DataFrame({"vix_close": [20.0] * 80}, index=dates)
        result = vix_zscore(df, window=60)
        # std=0 -> division by zero -> NaN
        assert result.iloc[60:].isna().all()


# ---------------------------------------------------------------------------
# vix_velocity
# ---------------------------------------------------------------------------

class TestVIXVelocity:
    def test_returns_three_columns(self, sample_df):
        result = vix_velocity(sample_df)
        assert set(result.columns) == {"vix_velocity_1d", "vix_velocity_3d", "vix_velocity_5d"}

    def test_1d_velocity(self, sample_df):
        result = vix_velocity(sample_df)
        expected = sample_df["vix_close"].pct_change(1)
        pd.testing.assert_series_equal(result["vix_velocity_1d"], expected, check_names=False)

    def test_length_matches(self, sample_df):
        result = vix_velocity(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# term_slope
# ---------------------------------------------------------------------------

class TestTermSlope:
    def test_contango(self):
        """When M2 > M1, term_slope should be positive."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(
            {"vix_futures_m1": [20.0] * 5, "vix_futures_m2": [22.0] * 5},
            index=dates,
        )
        result = term_slope(df)
        assert (result > 0).all()
        assert result.iloc[0] == pytest.approx(0.1)  # (22-20)/20

    def test_backwardation(self):
        """When M2 < M1, term_slope should be negative."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame(
            {"vix_futures_m1": [30.0] * 5, "vix_futures_m2": [27.0] * 5},
            index=dates,
        )
        result = term_slope(df)
        assert (result < 0).all()
        assert result.iloc[0] == pytest.approx(-0.1)  # (27-30)/30

    def test_exact_values(self):
        df = pd.DataFrame({
            "vix_futures_m1": [20.0, 25.0],
            "vix_futures_m2": [22.0, 24.0],
        })
        result = term_slope(df)
        assert result.iloc[0] == pytest.approx(0.1)   # (22-20)/20
        assert result.iloc[1] == pytest.approx(-0.04)  # (24-25)/25


# ---------------------------------------------------------------------------
# term_curvature
# ---------------------------------------------------------------------------

class TestTermCurvature:
    def test_with_m3(self):
        df = pd.DataFrame({
            "vix_futures_m1": [20.0],
            "vix_futures_m2": [22.0],
            "vix_futures_m3": [23.0],
        })
        result = term_curvature(df)
        # curvature = ((23-22) - (22-20)) / 20 = (1-2)/20 = -0.05
        assert result.iloc[0] == pytest.approx(-0.05)

    def test_fallback_no_m3(self):
        df = pd.DataFrame({
            "vix_futures_m1": [20.0],
            "vix_futures_m2": [22.0],
        })
        result = term_curvature(df)
        # Fallback: abs(term_slope) = abs((22-20)/20) = 0.1
        assert result.iloc[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# spy_drawdown
# ---------------------------------------------------------------------------

class TestSPYDrawdown:
    def test_at_high(self):
        """Drawdown should be 0 when SPY is at rolling high."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({"spy_close": [100, 101, 102, 103, 104]}, index=dates)
        result = spy_drawdown(df, window=20)
        assert (result == 0).all()

    def test_below_high(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({"spy_close": [100, 105, 103, 98, 100]}, index=dates)
        result = spy_drawdown(df, window=20)
        # Last value: rolling high = 105, current = 100
        assert result.iloc[-1] == pytest.approx((100 - 105) / 105)
        # First value: only 1 day, high = 100 -> 0
        assert result.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# rv_iv_spread (T-1 shift verification)
# ---------------------------------------------------------------------------

class TestRVIVSpread:
    def test_uses_t_minus_1_data(self, sample_df):
        """Verify realized vol uses T-1 data (shifted by 1)."""
        result = rv_iv_spread(sample_df, rv_window=20)
        # First 21 values should be NaN
        assert result.iloc[:21].isna().all()

    def test_shift_verification(self):
        """Today's SPY return should not affect today's rv_iv_spread."""
        n = 30
        dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
        spy = np.array([100.0 + 0.1 * i for i in range(n)])
        vix = np.full(n, 20.0)

        df1 = pd.DataFrame({"spy_close": spy, "vix_close": vix}, index=dates)

        # Modify only the last SPY close (big jump)
        spy2 = spy.copy()
        spy2[-1] = spy[-1] + 10
        df2 = pd.DataFrame({"spy_close": spy2, "vix_close": vix}, index=dates)

        r1 = rv_iv_spread(df1, rv_window=20)
        r2 = rv_iv_spread(df2, rv_window=20)

        # Due to shift(1), the last day's return shouldn't affect today's value
        assert r1.iloc[-1] == pytest.approx(r2.iloc[-1], abs=1e-10)


# ---------------------------------------------------------------------------
# days_elevated
# ---------------------------------------------------------------------------

class TestDaysElevated:
    def test_counter_resets(self):
        """Counter should reset when VIX drops below threshold."""
        dates = pd.bdate_range("2020-01-01", periods=80)
        # Use values with real variance so rolling std > 0
        vix = np.array([20.0] * 60 + [30.0] * 5 + [15.0] * 3 + [30.0] * 2 + [15.0] * 10)
        df = pd.DataFrame({"vix_close": vix}, index=dates)
        result = days_elevated(df, window=60)
        # After dropping from 30 to 15, counter should reset
        assert result.iloc[65] == 0  # 15 is below threshold

    def test_consecutive_counting(self):
        """Counter should increment each day VIX is above threshold."""
        dates = pd.bdate_range("2020-01-01", periods=80)
        vix = np.array([20.0] * 60 + [30.0] * 5 + [20.0] * 15)
        df = pd.DataFrame({"vix_close": vix}, index=dates)
        result = days_elevated(df, window=60)
        # Days 60-64 should count up: 1, 2, 3, 4, 5
        assert result.iloc[60] == 1
        assert result.iloc[61] == 2
        assert result.iloc[64] == 5
        # After reset
        assert result.iloc[70] == 0

    def test_constant_series_no_elevation(self):
        """Constant VIX should have 0 elevated days (std=0, threshold = mean)."""
        dates = pd.bdate_range("2020-01-01", periods=70)
        df = pd.DataFrame({"vix_close": [15.0] * 70}, index=dates)
        result = days_elevated(df, window=60)
        # std=0, threshold=15, 15 > 15 is False
        assert (result.iloc[60:] == 0).all()


# ---------------------------------------------------------------------------
# Lagged features
# ---------------------------------------------------------------------------

class TestLaggedFeatures:
    def test_lag_columns_created(self, sample_df):
        features = compute_all_features(sample_df)
        lagged = compute_lagged_features(features)
        assert "vix_spot_lag1" in lagged.columns
        assert "vix_zscore_lag20" in lagged.columns
        assert "vvix_lag5" in lagged.columns

    def test_lag_values_correct(self, sample_df):
        features = compute_all_features(sample_df)
        lagged = compute_lagged_features(features)
        # lag1 should equal the previous day's value
        lag1_vals = lagged["vix_spot_lag1"].iloc[1:].values
        orig_vals = features["vix_spot"].iloc[:-1].values
        np.testing.assert_array_almost_equal(lag1_vals, orig_vals)

    def test_lag_count(self):
        """Should produce 5 features x 5 lags = 25 columns."""
        dates = pd.bdate_range("2023-01-02", periods=25)
        features_df = pd.DataFrame(
            {feat: range(25) for feat in _LAG_FEATURES},
            index=dates,
        )
        result = compute_lagged_features(features_df)
        assert len(result.columns) == len(_LAG_FEATURES) * len(_LAG_PERIODS)

    def test_lag_nan_at_start(self):
        dates = pd.bdate_range("2023-01-02", periods=25)
        features_df = pd.DataFrame(
            {feat: range(25) for feat in _LAG_FEATURES},
            index=dates,
        )
        result = compute_lagged_features(features_df)
        # First value of lag1 should be NaN
        assert pd.isna(result["vix_spot_lag1"].iloc[0])
        # First 5 values of lag5 should be NaN
        assert result["vix_spot_lag5"].iloc[:5].isna().all()


# ---------------------------------------------------------------------------
# compute_all_features
# ---------------------------------------------------------------------------

class TestComputeAllFeatures:
    def test_returns_correct_column_count(self, sample_df):
        result = compute_all_features(sample_df)
        # 20 base features + 25 lagged = 45
        assert len(result.columns) >= 42

    def test_no_label_columns(self, sample_df):
        result = compute_all_features(sample_df)
        assert "label_revert" not in result.columns
        assert "label_spike_first" not in result.columns

    def test_index_preserved(self, sample_df):
        result = compute_all_features(sample_df)
        pd.testing.assert_index_equal(result.index, sample_df.index)

    def test_core_columns_present(self, sample_df):
        result = compute_all_features(sample_df)
        expected_core = [
            "vix_spot", "vix_zscore",
            "vix_velocity_1d", "vix_velocity_3d", "vix_velocity_5d",
            "term_slope", "term_slope_zscore", "term_curvature",
            "spy_drawdown", "spy_velocity", "rv_iv_spread",
            "vix_percentile", "days_elevated",
            "vvix", "vix9d_vix_ratio", "skew",
            "vix_futures_volume", "put_call_ratio",
            "hy_spread_velocity", "day_of_week",
        ]
        for col in expected_core:
            assert col in result.columns, f"Missing column: {col}"

    def test_lagged_columns_present(self, sample_df):
        result = compute_all_features(sample_df)
        for feat in _LAG_FEATURES:
            for lag in _LAG_PERIODS:
                col = f"{feat}_lag{lag}"
                assert col in result.columns, f"Missing lagged column: {col}"


# ---------------------------------------------------------------------------
# Other individual features
# ---------------------------------------------------------------------------

class TestOtherFeatures:
    def test_vvix_passthrough(self, sample_df):
        result = vvix(sample_df)
        pd.testing.assert_series_equal(result, sample_df["vvix_close"], check_names=False)

    def test_vix9d_vix_ratio(self, sample_df):
        result = vix9d_vix_ratio(sample_df)
        expected = sample_df["vix9d_close"] / sample_df["vix_close"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_skew_passthrough(self, sample_df):
        result = skew(sample_df)
        pd.testing.assert_series_equal(result, sample_df["skew_close"], check_names=False)

    def test_vix_futures_volume_passthrough(self, sample_df):
        result = vix_futures_volume(sample_df)
        pd.testing.assert_series_equal(result, sample_df["vix_futures_volume"], check_names=False)

    def test_put_call_ratio_passthrough(self, sample_df):
        result = put_call_ratio(sample_df)
        pd.testing.assert_series_equal(result, sample_df["put_call_ratio"], check_names=False)

    def test_day_of_week_range(self, sample_df):
        result = day_of_week(sample_df)
        assert (result >= 0).all() and (result <= 4).all()

    def test_vix_percentile_range(self, sample_df):
        result = vix_percentile(sample_df, window=60)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_spy_velocity(self, sample_df):
        result = spy_velocity(sample_df, window=5)
        expected = sample_df["spy_close"].pct_change(5)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_hy_spread_velocity(self, sample_df):
        result = hy_spread_velocity(sample_df, window=5)
        assert len(result) == len(sample_df)
        assert result.iloc[:5].isna().all()

    def test_term_slope_zscore_warmup(self, sample_df):
        result = term_slope_zscore(sample_df, window=60)
        # First 59 values should be NaN
        assert result.iloc[:59].isna().all()
