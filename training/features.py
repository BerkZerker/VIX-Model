"""Daily feature computation for the VIX Alert Bot.

Each feature is a separate function that takes a DataFrame with raw data columns
and returns a Series or DataFrame. The compute_all_features() function orchestrates
all feature computations and returns the full ~42-feature DataFrame.

Expected input columns (from merged raw data):
    - vix_close: VIX daily close
    - spy_close: SPY daily close
    - vix_futures_m1: Front-month VIX futures settlement
    - vix_futures_m2: Second-month VIX futures settlement
    - vix_futures_m3: Third-month VIX futures settlement (optional)
    - vvix_close: CBOE VVIX index
    - vix9d_close: VIX9D index
    - skew_close: CBOE SKEW index
    - vix_futures_volume: Front-month VIX futures volume
    - put_call_ratio: CBOE equity put/call ratio
    - hyg_close: HYG ETF close
    - tlt_close: TLT ETF close
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def vix_spot(df: pd.DataFrame) -> pd.Series:
    """Standardized VIX spot level."""
    return df["vix_close"].copy()


def vix_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """VIX relative to its 60-day rolling mean/std."""
    rolling_mean = df["vix_close"].rolling(window, min_periods=window).mean()
    rolling_std = df["vix_close"].rolling(window, min_periods=window).std()
    return (df["vix_close"] - rolling_mean) / rolling_std


def vix_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Rate of change of VIX over 1, 3, and 5 days."""
    close = df["vix_close"]
    return pd.DataFrame({
        "vix_velocity_1d": close.pct_change(1),
        "vix_velocity_3d": close.pct_change(3),
        "vix_velocity_5d": close.pct_change(5),
    }, index=df.index)


def term_slope(df: pd.DataFrame) -> pd.Series:
    """Term structure slope: (Month2 - Month1) / Month1.

    Positive = contango, negative = backwardation.
    """
    return (df["vix_futures_m2"] - df["vix_futures_m1"]) / df["vix_futures_m1"]


def term_slope_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Term slope relative to its own 60-day rolling stats."""
    ts = term_slope(df)
    rolling_mean = ts.rolling(window, min_periods=window).mean()
    rolling_std = ts.rolling(window, min_periods=window).std()
    return (ts - rolling_mean) / rolling_std


def term_curvature(df: pd.DataFrame) -> pd.Series:
    """Shape of the full futures curve.

    Uses three months: curvature = (M3 - M2) - (M2 - M1), normalized by M1.
    If M3 is unavailable, falls back to a simpler measure of how far M2-M1
    deviates from linear extrapolation.
    """
    if "vix_futures_m3" in df.columns:
        m1 = df["vix_futures_m1"]
        m2 = df["vix_futures_m2"]
        m3 = df["vix_futures_m3"]
        return ((m3 - m2) - (m2 - m1)) / m1
    # Fallback: just use the term slope magnitude as a proxy
    return term_slope(df).abs()


def spy_drawdown(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """SPY percentage below its 20-day rolling high."""
    rolling_high = df["spy_close"].rolling(window, min_periods=1).max()
    return (df["spy_close"] - rolling_high) / rolling_high


def spy_velocity(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """SPY 5-day rate of change."""
    return df["spy_close"].pct_change(window)


def rv_iv_spread(df: pd.DataFrame, rv_window: int = 20) -> pd.Series:
    """20-day realized volatility minus VIX.

    CRITICAL: Uses T-1 data only to avoid look-ahead bias.
    Realized vol is computed from SPY log returns using data up to T-1,
    then compared against VIX at time T.
    """
    log_returns = np.log(df["spy_close"] / df["spy_close"].shift(1))
    # Realized vol from T-1 and earlier (shift by 1 so today's return is excluded)
    rv_daily = log_returns.shift(1).rolling(rv_window, min_periods=rv_window).std()
    # Annualize: multiply by sqrt(252)
    rv_annualized = rv_daily * np.sqrt(252) * 100  # in percentage points
    return rv_annualized - df["vix_close"]


def vix_percentile(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """VIX rank over trailing 252 trading days (percentile 0-1)."""
    return df["vix_close"].rolling(window, min_periods=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )


def days_elevated(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Count of consecutive days VIX has been above 60-day mean + 1 std.

    Uses a dynamic threshold based on rolling statistics rather than a fixed level.
    """
    rolling_mean = df["vix_close"].rolling(window, min_periods=window).mean()
    rolling_std = df["vix_close"].rolling(window, min_periods=window).std()
    threshold = rolling_mean + rolling_std
    above = (df["vix_close"] > threshold).astype(int)

    # Count consecutive days above threshold
    # Reset count whenever VIX drops below threshold
    counts = []
    current_count = 0
    for val in above:
        if val == 1:
            current_count += 1
        else:
            current_count = 0
        counts.append(current_count)
    return pd.Series(counts, index=df.index, name="days_elevated")


def vvix(df: pd.DataFrame) -> pd.Series:
    """CBOE VVIX index (vol-of-vol)."""
    return df["vvix_close"].copy()


def vix9d_vix_ratio(df: pd.DataFrame) -> pd.Series:
    """VIX9D / VIX ratio.

    Ratio < 1 signals near-term resolution of fear.
    """
    return df["vix9d_close"] / df["vix_close"]


def skew(df: pd.DataFrame) -> pd.Series:
    """CBOE SKEW index."""
    return df["skew_close"].copy()


def vix_futures_volume(df: pd.DataFrame) -> pd.Series:
    """Front-month VIX futures daily volume (raw; standardization done later)."""
    return df["vix_futures_volume"].copy()


def put_call_ratio(df: pd.DataFrame) -> pd.Series:
    """CBOE equity put/call ratio."""
    return df["put_call_ratio"].copy()


def hy_spread_velocity(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """5-day rate of change in HYG-TLT spread (credit stress proxy)."""
    spread = df["hyg_close"] - df["tlt_close"]
    return spread.pct_change(window)


def day_of_week(df: pd.DataFrame) -> pd.Series:
    """Day of week encoded as 0 (Monday) through 4 (Friday)."""
    return pd.Series(df.index.dayofweek, index=df.index, name="day_of_week")


# ---------------------------------------------------------------------------
# Lagged features
# ---------------------------------------------------------------------------

_LAG_FEATURES = ["vix_spot", "vix_zscore", "term_slope", "spy_drawdown", "vvix"]
_LAG_PERIODS = [1, 3, 5, 10, 20]


def compute_lagged_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged versions of key features at t-1, t-3, t-5, t-10, t-20."""
    lagged = {}
    for feat in _LAG_FEATURES:
        if feat not in features_df.columns:
            continue
        for lag in _LAG_PERIODS:
            lagged[f"{feat}_lag{lag}"] = features_df[feat].shift(lag)
    return pd.DataFrame(lagged, index=features_df.index)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all daily features from raw merged data.

    Parameters
    ----------
    df : pd.DataFrame
        Merged raw data with a DatetimeIndex and columns as documented
        at the module level.

    Returns
    -------
    pd.DataFrame
        ~42-column feature DataFrame indexed by date.
    """
    features = pd.DataFrame(index=df.index)

    # Core features
    features["vix_spot"] = vix_spot(df)
    features["vix_zscore"] = vix_zscore(df)

    vel = vix_velocity(df)
    features = features.join(vel)

    features["term_slope"] = term_slope(df)
    features["term_slope_zscore"] = term_slope_zscore(df)
    features["term_curvature"] = term_curvature(df)
    features["spy_drawdown"] = spy_drawdown(df)
    features["spy_velocity"] = spy_velocity(df)
    features["rv_iv_spread"] = rv_iv_spread(df)
    features["vix_percentile"] = vix_percentile(df)
    features["days_elevated"] = days_elevated(df)
    features["vvix"] = vvix(df)
    features["vix9d_vix_ratio"] = vix9d_vix_ratio(df)
    features["skew"] = skew(df)
    features["vix_futures_volume"] = vix_futures_volume(df)
    features["put_call_ratio"] = put_call_ratio(df)
    features["hy_spread_velocity"] = hy_spread_velocity(df)
    features["day_of_week"] = day_of_week(df)

    # Lagged features
    lagged = compute_lagged_features(features)
    features = features.join(lagged)

    return features
