"""Multi-resolution preprocessing for the Hierarchical CNN+GRU model.

Aggregates raw 5-minute OHLCV bars into hourly and daily bars, and builds
the three input tensors (intraday, short-term, regime) required by the model.

Expected input DataFrame schema (5-min bars):
    - datetime index (or 'datetime' column)
    - open, high, low, close, volume
    - Additional columns (e.g., vix_open, spy_close) are preserved where applicable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def aggregate_to_hourly(bars_5min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-minute OHLCV bars to hourly OHLCV bars.

    Parameters
    ----------
    bars_5min : pd.DataFrame
        DataFrame with a DatetimeIndex and columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Hourly OHLCV bars.
    """
    resampled = bars_5min.resample("1h")
    hourly = pd.DataFrame({
        "open": resampled["open"].first(),
        "high": resampled["high"].max(),
        "low": resampled["low"].min(),
        "close": resampled["close"].last(),
        "volume": resampled["volume"].sum(),
    })
    return hourly.dropna(subset=["open"])


def aggregate_to_daily(bars_5min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-minute OHLCV bars to daily OHLCV bars.

    Parameters
    ----------
    bars_5min : pd.DataFrame
        DataFrame with a DatetimeIndex and columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV bars.
    """
    resampled = bars_5min.resample("1D")
    daily = pd.DataFrame({
        "open": resampled["open"].first(),
        "high": resampled["high"].max(),
        "low": resampled["low"].min(),
        "close": resampled["close"].last(),
        "volume": resampled["volume"].sum(),
    })
    return daily.dropna(subset=["open"])


# Default lookback configuration matching the project spec
DEFAULT_LOOKBACK_CONFIG: dict[str, Any] = {
    "intraday": {
        "resolution": "5min",
        "bars": 156,        # ~2 trading days of 5-min bars (78 bars/day)
    },
    "short_term": {
        "resolution": "1h",
        "bars": 130,        # ~2 weeks of hourly bars (~6.5 hrs/day × 10 days)
    },
    "regime": {
        "resolution": "1D",
        "bars": 60,         # ~2-3 months of daily bars
    },
}


def build_multi_resolution_tensors(
    bars_5min: pd.DataFrame,
    lookback_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build three numpy arrays for the Hierarchical CNN+GRU model.

    Parameters
    ----------
    bars_5min : pd.DataFrame
        Raw 5-minute OHLCV bars with a DatetimeIndex. Must contain enough
        history to fill the regime lookback window.
    lookback_config : dict, optional
        Configuration for each resolution stream. Defaults to
        DEFAULT_LOOKBACK_CONFIG.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - intraday: shape (intraday_bars, n_features) -- raw 5-min bars
        - short_term: shape (short_term_bars, n_features) -- hourly aggregated
        - regime: shape (regime_bars, n_features) -- daily aggregated

    Raises
    ------
    ValueError
        If there is not enough data to fill the requested lookback windows.
    """
    if lookback_config is None:
        lookback_config = DEFAULT_LOOKBACK_CONFIG

    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    # --- Intraday stream: last N 5-min bars ---
    intraday_bars = lookback_config["intraday"]["bars"]
    if len(bars_5min) < intraday_bars:
        raise ValueError(
            f"Need at least {intraday_bars} 5-min bars for intraday stream, "
            f"got {len(bars_5min)}"
        )
    intraday_df = bars_5min[ohlcv_cols].iloc[-intraday_bars:]
    intraday = intraday_df.values.astype(np.float32)

    # --- Short-term stream: aggregate to hourly, take last N ---
    hourly = aggregate_to_hourly(bars_5min)
    short_term_bars = lookback_config["short_term"]["bars"]
    if len(hourly) < short_term_bars:
        raise ValueError(
            f"Need at least {short_term_bars} hourly bars for short-term stream, "
            f"got {len(hourly)}"
        )
    short_term_df = hourly[ohlcv_cols].iloc[-short_term_bars:]
    short_term = short_term_df.values.astype(np.float32)

    # --- Regime stream: aggregate to daily, take last N ---
    daily = aggregate_to_daily(bars_5min)
    regime_bars = lookback_config["regime"]["bars"]
    if len(daily) < regime_bars:
        raise ValueError(
            f"Need at least {regime_bars} daily bars for regime stream, "
            f"got {len(daily)}"
        )
    regime_df = daily[ohlcv_cols].iloc[-regime_bars:]
    regime = regime_df.values.astype(np.float32)

    return intraday, short_term, regime
