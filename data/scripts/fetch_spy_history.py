"""Fetch historical SPY daily OHLCV data from Yahoo Finance.

Downloads SPY data from 1993-present and saves to data/raw/spy_daily.csv.
Idempotent: overwrites existing file with fresh data on each run.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
OUTPUT_FILE = RAW_DIR / "spy_daily.csv"


def main() -> None:
    """Download SPY daily OHLCV from Yahoo Finance and save to CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching SPY daily data from Yahoo Finance...")
    try:
        ticker = yf.Ticker("SPY")
        df: pd.DataFrame = ticker.history(period="max", interval="1d", auto_adjust=True)
    except Exception:
        logger.exception("Failed to download SPY data")
        sys.exit(1)

    if df.empty:
        logger.error("No data returned for SPY")
        sys.exit(1)

    drop_cols = [c for c in ("Dividends", "Stock Splits", "Capital Gains") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.index.name = "Date"
    df.index = df.index.tz_localize(None)

    logger.info(
        "Downloaded %d rows, date range: %s to %s",
        len(df),
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    )

    df.to_csv(OUTPUT_FILE)
    logger.info("Saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
