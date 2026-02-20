"""Data staleness detection for the VIX Alert Bot.

Tracks the last successful quote per data point and alerts if critical
data becomes stale during market hours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Critical data points that must stay fresh during market hours
CRITICAL_SYMBOLS = {"VIX", "SPY", "VX_M1"}
# Data points that are high-priority but not inference-blocking
HIGH_PRIORITY_SYMBOLS = {"VVIX", "VIX9D", "SKEW"}


@dataclass
class StalenessEntry:
    symbol: str
    last_update: datetime
    is_stale: bool = False


@dataclass
class StalenessStatus:
    """Overall staleness report."""

    any_critical_stale: bool
    entries: dict[str, StalenessEntry]
    checked_at: datetime

    def summary(self) -> str:
        lines = [f"Staleness check at {self.checked_at.strftime('%H:%M:%S UTC')}"]
        for sym, entry in sorted(self.entries.items()):
            age = (self.checked_at - entry.last_update).total_seconds()
            status = "STALE" if entry.is_stale else "ok"
            lines.append(f"  {sym}: {status} (last update {age:.0f}s ago)")
        return "\n".join(lines)


class StalenessTracker:
    """Tracks the freshness of market data feeds."""

    def __init__(self, threshold_seconds: int = 300) -> None:
        self.threshold_seconds = threshold_seconds
        self._last_updates: dict[str, datetime] = {}

    def record_update(self, symbol: str, timestamp: datetime | None = None) -> None:
        """Record a successful data update for a symbol."""
        self._last_updates[symbol] = timestamp or datetime.now(timezone.utc)

    def record_updates(self, symbols: list[str], timestamp: datetime | None = None) -> None:
        """Record updates for multiple symbols at once."""
        ts = timestamp or datetime.now(timezone.utc)
        for sym in symbols:
            self._last_updates[sym] = ts

    def check(self) -> StalenessStatus:
        """Check all tracked symbols for staleness."""
        now = datetime.now(timezone.utc)
        entries: dict[str, StalenessEntry] = {}
        any_critical_stale = False

        all_symbols = CRITICAL_SYMBOLS | HIGH_PRIORITY_SYMBOLS
        for sym in all_symbols:
            if sym in self._last_updates:
                last = self._last_updates[sym]
                age = (now - last).total_seconds()
                is_stale = age > self.threshold_seconds
            else:
                # Never received data for this symbol
                last = datetime.min.replace(tzinfo=timezone.utc)
                is_stale = True

            entry = StalenessEntry(symbol=sym, last_update=last, is_stale=is_stale)
            entries[sym] = entry

            if is_stale and sym in CRITICAL_SYMBOLS:
                any_critical_stale = True

        return StalenessStatus(
            any_critical_stale=any_critical_stale,
            entries=entries,
            checked_at=now,
        )

    def get_last_update(self, symbol: str) -> datetime | None:
        """Get the last update time for a specific symbol."""
        return self._last_updates.get(symbol)

    def stale_symbols(self) -> list[str]:
        """Return list of currently stale critical symbols."""
        status = self.check()
        return [
            sym
            for sym, entry in status.entries.items()
            if entry.is_stale and sym in CRITICAL_SYMBOLS
        ]
