"""Real-time feature computation for the VIX Alert Bot.

Computes model features from live bars stored in SQLite.
Handles both daily (XGBoost) and multi-resolution (CNN+GRU) feature vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from bot.db import Database

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Feature output ready for model inference."""

    # Daily features for XGBoost (dict of feature_name -> value)
    daily_features: dict[str, float] = field(default_factory=dict)
    # Market snapshot values (for display in alerts)
    vix_spot: float = 0.0
    vix_zscore: float = 0.0
    vix_percentile: float = 0.0
    term_slope: float = 0.0
    term_slope_zscore: float = 0.0
    spy_price: float = 0.0
    spy_drawdown: float = 0.0
    vvix: float = 0.0
    vix9d: float = 0.0
    vix9d_vix_ratio: float = 0.0
    skew: float = 0.0
    days_elevated: int = 0
    futures: dict[str, float] = field(default_factory=dict)
    # Timestamp of computation
    computed_at: str = ""
    # Valid flag — false if insufficient data
    is_valid: bool = False


class FeaturePipeline:
    """Computes features from live bar data in the database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def compute(
        self,
        vvix: float | None = None,
        vix9d: float | None = None,
        skew: float | None = None,
        futures: dict[str, float] | None = None,
    ) -> FeatureVector:
        """Compute the full feature vector from current database state.

        Args:
            vvix: Current VVIX value (from poller snapshot).
            vix9d: Current VIX9D value.
            skew: Current SKEW value.
            futures: Dict of futures prices, e.g. {"VX_M1": 20.5, "VX_M2": 21.0, ...}.

        Returns:
            FeatureVector with all computed features.
        """
        fv = FeatureVector(
            computed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            vvix=vvix or 0.0,
            vix9d=vix9d or 0.0,
            skew=skew or 0.0,
            futures=futures or {},
        )

        # Get VIX bars (need ~60+ trading days for z-score)
        vix_bars = await self.db.get_latest_bars("VIX", limit=500)
        if len(vix_bars) < 10:
            logger.warning("Insufficient VIX bars for feature computation: %d", len(vix_bars))
            return fv

        # Bars come newest-first; reverse for chronological order
        vix_bars = list(reversed(vix_bars))
        vix_closes = np.array([b["close"] for b in vix_bars])

        # VIX spot
        fv.vix_spot = float(vix_closes[-1])

        # VIX z-score (60-day rolling)
        lookback = min(60, len(vix_closes) - 1)
        recent = vix_closes[-(lookback + 1) : -1]  # exclude current for mean/std
        if len(recent) >= 20:
            mean = float(np.mean(recent))
            std = float(np.std(recent))
            fv.vix_zscore = (fv.vix_spot - mean) / std if std > 0 else 0.0
        else:
            fv.vix_zscore = 0.0

        # VIX percentile (252-day)
        pct_lookback = min(252, len(vix_closes))
        pct_window = vix_closes[-pct_lookback:]
        fv.vix_percentile = float(np.mean(pct_window <= fv.vix_spot))

        # VIX velocity (1, 3, 5 day rate of change)
        vix_velocity_1d = _pct_change(vix_closes, 1)
        vix_velocity_3d = _pct_change(vix_closes, 3)
        vix_velocity_5d = _pct_change(vix_closes, 5)

        # SPY data
        spy_bars = await self.db.get_latest_bars("SPY", limit=100)
        spy_bars = list(reversed(spy_bars))
        if spy_bars:
            spy_closes = np.array([b["close"] for b in spy_bars])
            fv.spy_price = float(spy_closes[-1])
            # SPY drawdown from 20-day high
            dd_lookback = min(20, len(spy_closes))
            high_20d = float(np.max(spy_closes[-dd_lookback:]))
            fv.spy_drawdown = (fv.spy_price - high_20d) / high_20d if high_20d > 0 else 0.0
            spy_velocity_5d = _pct_change(spy_closes, 5)
        else:
            spy_velocity_5d = 0.0

        # Term structure features
        fut = futures or {}
        m1 = fut.get("VX_M1")
        m2 = fut.get("VX_M2")
        if m1 and m1 > 0:
            fv.term_slope = (m2 - m1) / m1 if m2 else 0.0
        # term_slope_zscore would need historical term slopes; approximate with 0 for now
        fv.term_slope_zscore = 0.0

        # VIX9D/VIX ratio
        if vix9d and fv.vix_spot > 0:
            fv.vix9d_vix_ratio = vix9d / fv.vix_spot

        # Days elevated (VIX above 60d mean + 1 std)
        if len(recent) >= 20:
            threshold = mean + std
            fv.days_elevated = 0
            for v in reversed(vix_closes):
                if v > threshold:
                    fv.days_elevated += 1
                else:
                    break

        # Realized vol - implied vol spread (use last 20 closes for realized vol)
        rv_lookback = min(20, len(vix_closes) - 1)
        if rv_lookback >= 5:
            log_returns = np.diff(np.log(vix_closes[-(rv_lookback + 1) :]))
            rv_20d = float(np.std(log_returns) * np.sqrt(252) * 100)
            rv_iv_spread = rv_20d - fv.vix_spot
        else:
            rv_iv_spread = 0.0

        # Build the daily features dict for XGBoost
        fv.daily_features = {
            "vix_spot": fv.vix_spot,
            "vix_zscore": fv.vix_zscore,
            "vix_velocity_1d": vix_velocity_1d,
            "vix_velocity_3d": vix_velocity_3d,
            "vix_velocity_5d": vix_velocity_5d,
            "term_slope": fv.term_slope,
            "term_slope_zscore": fv.term_slope_zscore,
            "spy_drawdown": fv.spy_drawdown,
            "spy_velocity_5d": spy_velocity_5d,
            "rv_iv_spread": rv_iv_spread,
            "vix_percentile": fv.vix_percentile,
            "days_elevated": float(fv.days_elevated),
            "vvix": fv.vvix,
            "vix9d_vix_ratio": fv.vix9d_vix_ratio,
            "skew": fv.skew,
            "day_of_week": float(datetime.now(timezone.utc).weekday()),
        }

        fv.is_valid = True
        return fv


def _pct_change(arr: np.ndarray, periods: int) -> float:
    """Compute percentage change over `periods` steps."""
    if len(arr) <= periods:
        return 0.0
    prev = arr[-(periods + 1)]
    curr = arr[-1]
    return float((curr - prev) / prev) if prev != 0 else 0.0
