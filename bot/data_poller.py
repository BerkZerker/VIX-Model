"""IBKR data fetching and mock data generation.

IBKRPoller connects to Interactive Brokers via ib_insync.
MockPoller generates synthetic data for demo/testing.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """A single OHLCV bar."""

    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class MarketSnapshot:
    """Complete market data snapshot from a single poll cycle."""

    timestamp: str
    bars: list[BarData] = field(default_factory=list)
    futures: dict[str, float] = field(default_factory=dict)  # e.g. {"VXF6": 18.5, ...}
    vvix: float | None = None
    vix9d: float | None = None
    skew: float | None = None


class BasePoller(ABC):
    """Abstract base for data polling."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def poll(self) -> MarketSnapshot: ...

    @abstractmethod
    def is_connected(self) -> bool: ...


class IBKRPoller(BasePoller):
    """Live data poller using ib_insync to connect to IBKR Gateway/TWS."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4001, client_id: int = 1) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    async def connect(self) -> None:
        from ib_insync import IB

        self._ib = IB()
        await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
        logger.info("Connected to IBKR at %s:%d", self.host, self.port)

    async def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    async def poll(self) -> MarketSnapshot:
        """Fetch current market data from IBKR."""
        from ib_insync import Contract, Index

        if not self.is_connected():
            raise ConnectionError("Not connected to IBKR")

        ib = self._ib
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        bars: list[BarData] = []
        futures: dict[str, float] = {}

        # VIX spot
        vix_contract = Index("VIX", "CBOE", "USD")
        ib.qualifyContracts(vix_contract)
        ticker = ib.reqMktData(vix_contract)
        await asyncio.sleep(2)
        if ticker.last and not math.isnan(ticker.last):
            price = ticker.last
            bars.append(
                BarData(
                    timestamp=now, symbol="VIX", open=price, high=price,
                    low=price, close=price, volume=0,
                )
            )

        # SPY
        spy_contract = Contract(symbol="SPY", secType="STK", exchange="SMART", currency="USD")
        ib.qualifyContracts(spy_contract)
        spy_ticker = ib.reqMktData(spy_contract)
        await asyncio.sleep(2)
        if spy_ticker.last and not math.isnan(spy_ticker.last):
            price = spy_ticker.last
            bars.append(
                BarData(
                    timestamp=now, symbol="SPY", open=price, high=price,
                    low=price, close=price, volume=float(spy_ticker.volume or 0),
                )
            )

        # VIX Futures (front 4 months)
        from ib_insync import Future

        for i in range(1, 5):
            fut = Future("VIX", exchange="CFE", currency="USD")
            # ib_insync will resolve to nearest expiry when qualified
            try:
                qualified = ib.qualifyContracts(fut)
                if qualified:
                    ft = ib.reqMktData(qualified[0])
                    await asyncio.sleep(1)
                    if ft.last and not math.isnan(ft.last):
                        futures[f"VX_M{i}"] = ft.last
            except Exception:
                logger.warning("Failed to fetch VIX future month %d", i)

        # VVIX, VIX9D, SKEW
        vvix_val = await self._fetch_index(ib, "VVIX", "CBOE")
        vix9d_val = await self._fetch_index(ib, "VIX9D", "CBOE")
        skew_val = await self._fetch_index(ib, "SKEW", "CBOE")

        ib.cancelMktData(vix_contract)
        ib.cancelMktData(spy_contract)

        return MarketSnapshot(
            timestamp=now,
            bars=bars,
            futures=futures,
            vvix=vvix_val,
            vix9d=vix9d_val,
            skew=skew_val,
        )

    async def _fetch_index(self, ib, symbol: str, exchange: str) -> float | None:
        from ib_insync import Index

        try:
            contract = Index(symbol, exchange, "USD")
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract)
            await asyncio.sleep(1.5)
            val = ticker.last if ticker.last and not math.isnan(ticker.last) else None
            ib.cancelMktData(contract)
            return val
        except Exception:
            logger.warning("Failed to fetch index %s", symbol)
            return None


class MockPoller(BasePoller):
    """Generates synthetic market data for demo/testing mode."""

    def __init__(self) -> None:
        self._connected = False
        self._vix_base = 22.0
        self._spy_base = 510.0
        self._tick = 0

    async def connect(self) -> None:
        self._connected = True
        logger.info("MockPoller connected (synthetic data mode)")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("MockPoller disconnected")

    def is_connected(self) -> bool:
        return self._connected

    async def poll(self) -> MarketSnapshot:
        """Generate synthetic market data with realistic VIX dynamics."""
        self._tick += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Simulate mean-reverting VIX with occasional spikes
        drift = -0.02 * (self._vix_base - 18.0)  # mean-revert toward 18
        shock = random.gauss(0, 0.8)
        # Occasional spike
        if random.random() < 0.02:
            shock += random.uniform(3, 8)
        self._vix_base = max(10.0, self._vix_base + drift + shock)

        vix = round(self._vix_base, 2)
        vix_noise = lambda: round(random.uniform(-0.3, 0.3), 2)
        vix_bar = BarData(
            timestamp=now,
            symbol="VIX",
            open=round(vix + vix_noise(), 2),
            high=round(vix + abs(vix_noise()) + 0.2, 2),
            low=round(vix - abs(vix_noise()) - 0.2, 2),
            close=vix,
            volume=0,
        )

        # SPY inversely correlated with VIX
        spy_drift = 0.01 * (510.0 - self._spy_base)
        spy_shock = random.gauss(0, 1.5) - (shock * 0.3 if shock > 2 else 0)
        self._spy_base = max(400.0, self._spy_base + spy_drift + spy_shock)
        spy = round(self._spy_base, 2)
        spy_bar = BarData(
            timestamp=now,
            symbol="SPY",
            open=round(spy + random.uniform(-0.5, 0.5), 2),
            high=round(spy + abs(random.uniform(0, 1.0)), 2),
            low=round(spy - abs(random.uniform(0, 1.0)), 2),
            close=spy,
            volume=round(random.uniform(1e6, 5e6)),
        )

        # VIX futures in contango (normal) or backwardation (elevated VIX)
        if vix > 25:
            # Backwardation
            futures = {
                f"VX_M{i}": round(vix - i * random.uniform(0.5, 1.5), 2)
                for i in range(1, 5)
            }
        else:
            # Normal contango
            futures = {
                f"VX_M{i}": round(vix + i * random.uniform(0.3, 0.8), 2)
                for i in range(1, 5)
            }

        # Supplementary indices
        vvix = round(80 + (vix - 15) * 2.5 + random.gauss(0, 3), 1)
        vix9d = round(vix * random.uniform(0.88, 1.05), 2)
        skew = round(120 + random.gauss(0, 8), 1)

        return MarketSnapshot(
            timestamp=now,
            bars=[vix_bar, spy_bar],
            futures=futures,
            vvix=vvix,
            vix9d=vix9d,
            skew=skew,
        )
