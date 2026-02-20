"""SQLite database layer for the VIX Alert Bot.

Uses aiosqlite for async operations. Auto-creates tables on first use.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bars_5min (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0,
    UNIQUE(timestamp, symbol)
);

CREATE INDEX IF NOT EXISTS idx_bars_5min_symbol_ts ON bars_5min(symbol, timestamp);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    p_revert REAL NOT NULL,
    p_spike_first REAL NOT NULL,
    expected_magnitude REAL NOT NULL,
    alert_sent INTEGER DEFAULT 0,
    tier TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);

CREATE TABLE IF NOT EXISTS daily_digest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    vix_close REAL,
    zscore REAL,
    p_revert REAL,
    p_spike_first REAL,
    term_slope REAL
);
"""


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open connection and ensure schema exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info("Database connected: %s", self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # ── Bar operations ──────────────────────────────────────────────

    async def insert_bar(
        self,
        timestamp: str,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        """Insert a 5-min bar, ignoring duplicates."""
        await self.db.execute(
            """INSERT OR IGNORE INTO bars_5min
               (timestamp, symbol, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, symbol, open_, high, low, close, volume),
        )
        await self.db.commit()

    async def insert_bars_batch(self, bars: list[dict[str, Any]]) -> None:
        """Insert multiple bars efficiently."""
        await self.db.executemany(
            """INSERT OR IGNORE INTO bars_5min
               (timestamp, symbol, open, high, low, close, volume)
               VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume)""",
            bars,
        )
        await self.db.commit()

    async def get_latest_bars(
        self, symbol: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get the most recent bars for a symbol."""
        cursor = await self.db.execute(
            """SELECT timestamp, symbol, open, high, low, close, volume
               FROM bars_5min
               WHERE symbol = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (symbol, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_bars_since(
        self, symbol: str, since: str
    ) -> list[dict[str, Any]]:
        """Get bars for a symbol since a given timestamp (inclusive)."""
        cursor = await self.db.execute(
            """SELECT timestamp, symbol, open, high, low, close, volume
               FROM bars_5min
               WHERE symbol = ? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (symbol, since),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_last_bar_timestamp(self, symbol: str) -> str | None:
        """Get the timestamp of the most recent bar for a symbol."""
        cursor = await self.db.execute(
            """SELECT timestamp FROM bars_5min
               WHERE symbol = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (symbol,),
        )
        row = await cursor.fetchone()
        return row["timestamp"] if row else None

    # ── Signal operations ───────────────────────────────────────────

    async def insert_signal(
        self,
        timestamp: str,
        model_version: str,
        p_revert: float,
        p_spike_first: float,
        expected_magnitude: float,
        alert_sent: bool = False,
        tier: str = "",
    ) -> int:
        """Insert a model signal, return the new row id."""
        cursor = await self.db.execute(
            """INSERT INTO signals
               (timestamp, model_version, p_revert, p_spike_first,
                expected_magnitude, alert_sent, tier)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                model_version,
                p_revert,
                p_spike_first,
                expected_magnitude,
                int(alert_sent),
                tier,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_signals_today(self) -> list[dict[str, Any]]:
        """Get all signals from today."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cursor = await self.db.execute(
            """SELECT * FROM signals
               WHERE timestamp LIKE ?
               ORDER BY timestamp DESC""",
            (f"{today}%",),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_last_alert_time(self, tier: str) -> str | None:
        """Get timestamp of last sent alert for a given tier."""
        cursor = await self.db.execute(
            """SELECT timestamp FROM signals
               WHERE alert_sent = 1 AND tier = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (tier,),
        )
        row = await cursor.fetchone()
        return row["timestamp"] if row else None

    # ── Daily digest operations ─────────────────────────────────────

    async def insert_daily_digest(
        self,
        dt: date,
        vix_close: float | None = None,
        zscore: float | None = None,
        p_revert: float | None = None,
        p_spike_first: float | None = None,
        term_slope: float | None = None,
    ) -> None:
        """Insert or replace daily digest entry."""
        await self.db.execute(
            """INSERT OR REPLACE INTO daily_digest
               (date, vix_close, zscore, p_revert, p_spike_first, term_slope)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (dt.isoformat(), vix_close, zscore, p_revert, p_spike_first, term_slope),
        )
        await self.db.commit()

    async def get_recent_digests(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get recent daily digest entries."""
        cursor = await self.db.execute(
            """SELECT * FROM daily_digest ORDER BY date DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
