"""Fetch intraday 5-minute bar data for VIX and SPY.

Supports two modes:
- IBKR mode: Uses ib_insync to fetch real 5-min bars (requires IBKR connection)
- Mock mode: Generates synthetic 5-min data for testing/development

Usage:
    python fetch_intraday.py              # IBKR mode (default, requires connection)
    python fetch_intraday.py --mock       # Mock mode (generates synthetic data)
    python fetch_intraday.py --mock --days 30  # Mock mode, 30 days of data
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
VIX_OUTPUT = RAW_DIR / "vix_intraday_5min.csv"
SPY_OUTPUT = RAW_DIR / "spy_intraday_5min.csv"

# Trading hours: 9:30 AM - 4:00 PM ET = 78 five-minute bars per day
BARS_PER_DAY = 78
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
BAR_INTERVAL_MINUTES = 5


def generate_mock_bars(
    symbol: str,
    base_price: float,
    volatility: float,
    num_days: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 5-min OHLCV bars for testing.

    Uses geometric Brownian motion with intraday patterns (higher vol at open/close).

    Args:
        symbol: Ticker symbol (for logging).
        base_price: Starting price level.
        volatility: Annualized volatility (e.g., 0.8 for VIX, 0.15 for SPY).
        num_days: Number of trading days to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume and DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d days of mock 5-min bars for %s", num_days, symbol)

    # Generate trading dates (skip weekends)
    end_date = datetime(2026, 2, 18)
    dates = []
    d = end_date - timedelta(days=int(num_days * 1.5))
    while len(dates) < num_days:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d += timedelta(days=1)
    dates = dates[-num_days:]

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = base_price
    dt_per_bar = volatility / np.sqrt(252 * BARS_PER_DAY)

    for date in dates:
        for bar_idx in range(BARS_PER_DAY):
            minutes_from_open = bar_idx * BAR_INTERVAL_MINUTES
            bar_time = datetime(
                date.year, date.month, date.day,
                MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
            ) + timedelta(minutes=minutes_from_open)
            timestamps.append(bar_time)

            # Intraday volatility pattern: higher at open and close
            intraday_factor = 1.0
            if bar_idx < 6:  # First 30 min
                intraday_factor = 1.8 - 0.13 * bar_idx
            elif bar_idx > BARS_PER_DAY - 7:  # Last 30 min
                intraday_factor = 1.2 + 0.1 * (bar_idx - (BARS_PER_DAY - 7))

            bar_vol = dt_per_bar * intraday_factor
            returns_in_bar = rng.normal(0, bar_vol, 5)  # 5 sub-steps per bar

            bar_open = price
            sub_prices = [bar_open]
            for r in returns_in_bar:
                sub_prices.append(sub_prices[-1] * np.exp(r))
            bar_close = sub_prices[-1]
            bar_high = max(sub_prices)
            bar_low = min(sub_prices)

            # Volume: higher at open/close, random component
            base_vol = 5000 if symbol == "^VIX" else 500000
            vol = int(base_vol * intraday_factor * rng.uniform(0.5, 1.5))

            opens.append(round(bar_open, 2))
            highs.append(round(bar_high, 2))
            lows.append(round(bar_low, 2))
            closes.append(round(bar_close, 2))
            volumes.append(vol)

            price = bar_close

            # Mean-revert VIX toward base price (VIX doesn't drift like stocks)
            if symbol == "^VIX":
                price = price + 0.001 * (base_price - price)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(timestamps, name="Datetime"),
    )

    logger.info(
        "  %s: %d bars (%s to %s)",
        symbol,
        len(df),
        df.index.min().strftime("%Y-%m-%d %H:%M"),
        df.index.max().strftime("%Y-%m-%d %H:%M"),
    )
    return df


def fetch_ibkr_bars(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical 5-min bars from IBKR via ib_insync.

    Requires an active IB Gateway or TWS connection.

    Args:
        symbol: Ticker symbol (^VIX or SPY).
        days: Number of days of history to request.

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex.
    """
    try:
        from ib_insync import IB, Contract, Index, Stock
    except ImportError:
        logger.error("ib_insync not installed. Install with: pip install ib_insync")
        sys.exit(1)

    ib = IB()
    try:
        ib.connect("127.0.0.1", 4001, clientId=1)
    except ConnectionRefusedError:
        logger.error(
            "Cannot connect to IB Gateway/TWS at 127.0.0.1:4001. "
            "Make sure IB Gateway or TWS is running with API enabled."
        )
        sys.exit(1)

    try:
        if symbol == "^VIX":
            contract = Index("VIX", "CBOE")
        elif symbol == "SPY":
            contract = Stock("SPY", "SMART", "USD")
        else:
            logger.error("Unknown symbol: %s", symbol)
            sys.exit(1)

        # IBKR limits: max ~1 year of 5-min bars, fetched in chunks
        all_bars = []
        end_dt = ""
        chunk_days = 5  # Fetch 5 days at a time to stay within limits

        remaining = days
        while remaining > 0:
            duration = f"{min(chunk_days, remaining)} D"
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting="5 mins",
                whatToShow="TRADES" if symbol == "SPY" else "MIDPOINT",
                useRTH=True,
            )
            if not bars:
                break

            all_bars = bars + all_bars
            end_dt = bars[0].date.strftime("%Y%m%d %H:%M:%S")
            remaining -= chunk_days
            logger.info("  Fetched %d bars, remaining ~%d days", len(bars), remaining)

        if not all_bars:
            logger.error("No bars returned from IBKR for %s", symbol)
            sys.exit(1)

        df = pd.DataFrame(
            {
                "Open": [b.open for b in all_bars],
                "High": [b.high for b in all_bars],
                "Low": [b.low for b in all_bars],
                "Close": [b.close for b in all_bars],
                "Volume": [b.volume for b in all_bars],
            },
            index=pd.DatetimeIndex([b.date for b in all_bars], name="Datetime"),
        )
        return df

    finally:
        ib.disconnect()


def main() -> None:
    """Fetch or generate intraday 5-min bar data."""
    parser = argparse.ArgumentParser(description="Fetch intraday 5-min bar data")
    parser.add_argument(
        "--mock", action="store_true", help="Generate synthetic data instead of fetching from IBKR"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Number of trading days (default: 90)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.mock:
        logger.info("Running in MOCK mode -- generating synthetic data")
        vix_df = generate_mock_bars("^VIX", base_price=18.0, volatility=0.80, num_days=args.days)
        spy_df = generate_mock_bars(
            "SPY", base_price=550.0, volatility=0.15, num_days=args.days, seed=123
        )
    else:
        logger.info("Running in IBKR mode -- connecting to IB Gateway")
        vix_df = fetch_ibkr_bars("^VIX", days=args.days)
        spy_df = fetch_ibkr_bars("SPY", days=args.days)

    vix_df.to_csv(VIX_OUTPUT)
    logger.info("Saved VIX intraday data to %s", VIX_OUTPUT)

    spy_df.to_csv(SPY_OUTPUT)
    logger.info("Saved SPY intraday data to %s", SPY_OUTPUT)


if __name__ == "__main__":
    main()
