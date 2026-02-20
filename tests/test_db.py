"""Tests for bot/db.py - Async SQLite database layer."""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from bot.db import Database, SCHEMA_SQL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_vix_bot.db"
        database = Database(db_path)
        await database.connect()
        yield database
        await database.close()


# ---------------------------------------------------------------------------
# Connection and schema
# ---------------------------------------------------------------------------

class TestConnection:
    async def test_connect_creates_tables(self, db):
        """Connect should create all required tables."""
        cursor = await db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        table_names = {row["name"] for row in rows}
        assert "bars_5min" in table_names
        assert "signals" in table_names
        assert "daily_digest" in table_names

    async def test_db_property_raises_before_connect(self):
        database = Database("/tmp/nonexistent.db")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = database.db

    async def test_close_sets_none(self, db):
        await db.close()
        assert db._db is None


# ---------------------------------------------------------------------------
# Bar operations
# ---------------------------------------------------------------------------

class TestBarOperations:
    async def test_insert_and_get_bar(self, db):
        await db.insert_bar(
            timestamp="2025-01-15 09:30:00",
            symbol="VIX",
            open_=20.0,
            high=21.0,
            low=19.5,
            close=20.5,
            volume=100000,
        )
        bars = await db.get_latest_bars("VIX", limit=10)
        assert len(bars) == 1
        assert bars[0]["symbol"] == "VIX"
        assert bars[0]["close"] == 20.5
        assert bars[0]["volume"] == 100000

    async def test_insert_duplicate_bar_ignored(self, db):
        """Inserting a duplicate (same timestamp + symbol) should be ignored."""
        for _ in range(3):
            await db.insert_bar(
                timestamp="2025-01-15 09:30:00",
                symbol="VIX",
                open_=20.0, high=21.0, low=19.5, close=20.5,
            )
        bars = await db.get_latest_bars("VIX", limit=10)
        assert len(bars) == 1

    async def test_insert_bars_batch(self, db):
        bars = [
            {
                "timestamp": f"2025-01-15 09:{30+i*5:02d}:00",
                "symbol": "VIX",
                "open": 20.0 + i,
                "high": 21.0 + i,
                "low": 19.5 + i,
                "close": 20.5 + i,
                "volume": 100000 + i * 1000,
            }
            for i in range(5)
        ]
        await db.insert_bars_batch(bars)
        result = await db.get_latest_bars("VIX", limit=10)
        assert len(result) == 5

    async def test_get_bars_since(self, db):
        for i in range(5):
            await db.insert_bar(
                timestamp=f"2025-01-1{5+i} 09:30:00",
                symbol="SPY",
                open_=400.0 + i, high=401.0, low=399.0, close=400.5 + i,
            )
        bars = await db.get_bars_since("SPY", "2025-01-17 00:00:00")
        assert len(bars) == 3  # 17, 18, 19

    async def test_get_last_bar_timestamp(self, db):
        await db.insert_bar(
            timestamp="2025-01-15 09:30:00",
            symbol="VIX",
            open_=20.0, high=21.0, low=19.5, close=20.5,
        )
        await db.insert_bar(
            timestamp="2025-01-15 09:35:00",
            symbol="VIX",
            open_=20.5, high=21.5, low=20.0, close=21.0,
        )
        last = await db.get_last_bar_timestamp("VIX")
        assert last == "2025-01-15 09:35:00"

    async def test_get_last_bar_timestamp_empty(self, db):
        last = await db.get_last_bar_timestamp("NONEXISTENT")
        assert last is None

    async def test_get_latest_bars_ordered(self, db):
        """Latest bars should be ordered newest first."""
        for i in range(3):
            await db.insert_bar(
                timestamp=f"2025-01-15 09:{30+i*5:02d}:00",
                symbol="VIX",
                open_=20.0, high=21.0, low=19.5, close=20.0 + i,
            )
        bars = await db.get_latest_bars("VIX", limit=10)
        assert bars[0]["timestamp"] > bars[-1]["timestamp"]

    async def test_bars_isolated_by_symbol(self, db):
        """Bars for different symbols should not interfere."""
        await db.insert_bar("2025-01-15 09:30:00", "VIX", 20.0, 21.0, 19.5, 20.5)
        await db.insert_bar("2025-01-15 09:30:00", "SPY", 400.0, 401.0, 399.0, 400.5)
        vix_bars = await db.get_latest_bars("VIX", limit=10)
        spy_bars = await db.get_latest_bars("SPY", limit=10)
        assert len(vix_bars) == 1
        assert len(spy_bars) == 1
        assert vix_bars[0]["close"] == 20.5
        assert spy_bars[0]["close"] == 400.5


# ---------------------------------------------------------------------------
# Signal operations
# ---------------------------------------------------------------------------

class TestSignalOperations:
    async def test_insert_signal(self, db):
        row_id = await db.insert_signal(
            timestamp="2025-01-15 16:00:00",
            model_version="v001",
            p_revert=0.85,
            p_spike_first=0.15,
            expected_magnitude=18.0,
            alert_sent=True,
            tier="major_spike",
        )
        assert row_id is not None
        assert row_id > 0

    async def test_get_last_alert_time(self, db):
        await db.insert_signal(
            timestamp="2025-01-15 10:00:00",
            model_version="v001",
            p_revert=0.8, p_spike_first=0.2, expected_magnitude=15.0,
            alert_sent=True, tier="moderate",
        )
        await db.insert_signal(
            timestamp="2025-01-15 14:00:00",
            model_version="v001",
            p_revert=0.9, p_spike_first=0.1, expected_magnitude=20.0,
            alert_sent=True, tier="moderate",
        )
        last = await db.get_last_alert_time("moderate")
        assert last == "2025-01-15 14:00:00"

    async def test_get_last_alert_time_no_alerts(self, db):
        last = await db.get_last_alert_time("major_spike")
        assert last is None


# ---------------------------------------------------------------------------
# Daily digest operations
# ---------------------------------------------------------------------------

class TestDailyDigest:
    async def test_insert_and_get_digest(self, db):
        await db.insert_daily_digest(
            dt=date(2025, 1, 15),
            vix_close=28.5,
            zscore=2.0,
            p_revert=0.85,
            p_spike_first=0.15,
            term_slope=-0.03,
        )
        digests = await db.get_recent_digests(limit=10)
        assert len(digests) == 1
        assert digests[0]["vix_close"] == 28.5
        assert digests[0]["zscore"] == 2.0

    async def test_digest_replace_on_duplicate_date(self, db):
        """Inserting same date should replace (INSERT OR REPLACE)."""
        await db.insert_daily_digest(dt=date(2025, 1, 15), vix_close=25.0)
        await db.insert_daily_digest(dt=date(2025, 1, 15), vix_close=28.0)
        digests = await db.get_recent_digests(limit=10)
        assert len(digests) == 1
        assert digests[0]["vix_close"] == 28.0

    async def test_recent_digests_ordered(self, db):
        """Recent digests should be ordered newest first."""
        for i in range(5):
            await db.insert_daily_digest(
                dt=date(2025, 1, 10 + i),
                vix_close=20.0 + i,
            )
        digests = await db.get_recent_digests(limit=10)
        assert digests[0]["date"] > digests[-1]["date"]

    async def test_digest_limit(self, db):
        for i in range(10):
            await db.insert_daily_digest(
                dt=date(2025, 1, 1 + i), vix_close=20.0,
            )
        digests = await db.get_recent_digests(limit=3)
        assert len(digests) == 3
