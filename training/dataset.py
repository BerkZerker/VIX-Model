"""PyTorch Dataset classes for VIX model training.

Provides:
  - VIXDailyDataset: For XGBoost (tabular features + labels)
  - VIXMultiResDataset: For Hierarchical CNN+GRU (three resolution tensors)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VIXDailyDataset(Dataset):
    """Dataset for daily tabular features (used with XGBoost and regime stream).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns and label columns.
    feature_cols : list[str]
        Feature column names.
    label_cols : list[str]
        Label column names (label_revert, label_spike_first, label_magnitude).
    eligible_only : bool
        If True, only include rows where all labels are non-NaN.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_cols: list[str] | None = None,
        eligible_only: bool = True,
    ):
        if label_cols is None:
            label_cols = ["label_revert", "label_spike_first", "label_magnitude"]

        if eligible_only:
            # Filter to rows where primary label exists
            mask = df[label_cols[0]].notna()
            df = df[mask].copy()

        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = {}
        for col in label_cols:
            if col in df.columns:
                self.labels[col] = df[col].values.astype(np.float32)

        self.dates = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        self.n_samples = len(df)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        item = {"features": torch.tensor(self.features[idx])}
        for key, vals in self.labels.items():
            item[key] = torch.tensor(vals[idx])
        return item


class VIXMultiResDataset(Dataset):
    """Dataset for multi-resolution inputs (Hierarchical CNN+GRU).

    For each sample, provides three input tensors at different resolutions:
      - intraday: (156, n_intraday_features) - 5-min bars, last 2 days
      - shortterm: (130, n_shortterm_features) - hourly bars, last 2 weeks
      - regime: (60, n_regime_features) - daily bars, last 2-3 months

    Parameters
    ----------
    bars_5min : pd.DataFrame
        5-minute bar data with columns: timestamp, open, high, low, close, volume, symbol.
    daily_features : pd.DataFrame
        Daily feature DataFrame (regime stream features).
    sample_dates : list[pd.Timestamp]
        Dates to generate samples for (typically dates where VIX z-score > 1.0).
    labels : pd.DataFrame, optional
        Labels for each sample date.
    intraday_seq_len : int
        Number of 5-min bars for intraday stream.
    shortterm_seq_len : int
        Number of hourly bars for short-term stream.
    regime_seq_len : int
        Number of daily bars for regime stream.
    """

    def __init__(
        self,
        bars_5min: pd.DataFrame | None,
        daily_features: pd.DataFrame,
        sample_dates: list,
        labels: pd.DataFrame | None = None,
        intraday_seq_len: int = 156,
        shortterm_seq_len: int = 130,
        regime_seq_len: int = 60,
    ):
        self.bars_5min = bars_5min
        self.daily_features = daily_features
        self.sample_dates = sample_dates
        self.labels = labels
        self.intraday_seq_len = intraday_seq_len
        self.shortterm_seq_len = shortterm_seq_len
        self.regime_seq_len = regime_seq_len

        # Precompute hourly bars if 5-min data available
        self.hourly_bars = None
        if bars_5min is not None and len(bars_5min) > 0:
            from training.aggregation import aggregate_to_hourly
            self.hourly_bars = aggregate_to_hourly(bars_5min)

        # Get feature column lists
        self._intraday_cols = self._get_intraday_feature_cols()
        self._shortterm_cols = self._get_shortterm_feature_cols()
        self._regime_cols = list(daily_features.columns)

    def _get_intraday_feature_cols(self) -> list[str]:
        """Feature columns for 5-min resolution."""
        if self.bars_5min is None:
            return ["open", "high", "low", "close", "volume"]
        return [c for c in self.bars_5min.columns
                if c not in ("timestamp", "symbol", "date")]

    def _get_shortterm_feature_cols(self) -> list[str]:
        """Feature columns for hourly resolution."""
        if self.hourly_bars is None:
            return ["open", "high", "low", "close", "volume", "range_pct", "momentum", "volatility"]
        return [c for c in self.hourly_bars.columns
                if c not in ("timestamp", "symbol", "date")]

    def __len__(self) -> int:
        return len(self.sample_dates)

    def __getitem__(self, idx: int) -> dict:
        date = self.sample_dates[idx]

        # Regime stream: last N daily bars up to and including this date
        regime_data = self._get_regime_tensor(date)

        # Short-term and intraday streams
        shortterm_data = self._get_shortterm_tensor(date)
        intraday_data = self._get_intraday_tensor(date)

        item = {
            "intraday": torch.tensor(intraday_data, dtype=torch.float32),
            "shortterm": torch.tensor(shortterm_data, dtype=torch.float32),
            "regime": torch.tensor(regime_data, dtype=torch.float32),
        }

        if self.labels is not None and date in self.labels.index:
            for col in self.labels.columns:
                val = self.labels.loc[date, col]
                item[col] = torch.tensor(float(val), dtype=torch.float32)

        return item

    def _get_regime_tensor(self, date) -> np.ndarray:
        """Get daily feature tensor for regime stream."""
        mask = self.daily_features.index <= date
        available = self.daily_features[mask].tail(self.regime_seq_len)

        data = available.values.astype(np.float32)
        n_features = len(self._regime_cols)

        # Pad if insufficient data
        if len(data) < self.regime_seq_len:
            pad = np.zeros((self.regime_seq_len - len(data), n_features), dtype=np.float32)
            data = np.vstack([pad, data])

        return data

    def _get_shortterm_tensor(self, date) -> np.ndarray:
        """Get hourly feature tensor for short-term stream."""
        n_features = len(self._shortterm_cols)

        if self.hourly_bars is None:
            # Return zeros if no intraday data available
            return np.zeros((self.shortterm_seq_len, n_features), dtype=np.float32)

        mask = self.hourly_bars.index <= date
        available = self.hourly_bars[mask].tail(self.shortterm_seq_len)
        data = available[self._shortterm_cols].values.astype(np.float32)

        if len(data) < self.shortterm_seq_len:
            pad = np.zeros((self.shortterm_seq_len - len(data), n_features), dtype=np.float32)
            data = np.vstack([pad, data])

        return data

    def _get_intraday_tensor(self, date) -> np.ndarray:
        """Get 5-min bar tensor for intraday stream."""
        n_features = len(self._intraday_cols)

        if self.bars_5min is None:
            return np.zeros((self.intraday_seq_len, n_features), dtype=np.float32)

        mask = self.bars_5min.index <= date
        available = self.bars_5min[mask].tail(self.intraday_seq_len)
        data = available[self._intraday_cols].values.astype(np.float32)

        if len(data) < self.intraday_seq_len:
            pad = np.zeros((self.intraday_seq_len - len(data), n_features), dtype=np.float32)
            data = np.vstack([pad, data])

        return data
