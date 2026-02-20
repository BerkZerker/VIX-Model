"""Fetch supplementary market data from Yahoo Finance and CBOE.

Downloads: VVIX (^VVIX), VIX9D (^VIX9D), SKEW (^SKEW), HYG, TLT.
Also attempts to fetch CBOE equity put/call ratio.
Saves each to data/raw/.
Idempotent: overwrites existing files on each run.
"""

import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

# Yahoo Finance tickers to fetch
YF_TICKERS = {
    "^VVIX": "vvix_daily.csv",
    "^VIX9D": "vix9d_daily.csv",
    "^SKEW": "skew_daily.csv",
    "HYG": "hyg_daily.csv",
    "TLT": "tlt_daily.csv",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_yf_ticker(symbol: str, filename: str) -> bool:
    """Fetch a single ticker from Yahoo Finance and save to CSV.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("Fetching %s from Yahoo Finance...", symbol)
    try:
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(period="max", interval="1d", auto_adjust=True)
    except Exception:
        logger.exception("Failed to download %s", symbol)
        return False

    if df.empty:
        logger.warning("No data returned for %s", symbol)
        return False

    drop_cols = [c for c in ("Dividends", "Stock Splits", "Capital Gains") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.index.name = "Date"
    df.index = df.index.tz_localize(None)

    out = RAW_DIR / filename
    df.to_csv(out)
    logger.info(
        "  %s: %d rows (%s to %s) -> %s",
        symbol,
        len(df),
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
        out,
    )
    return True


def fetch_cboe_put_call_ratio() -> bool:
    """Attempt to fetch CBOE equity put/call ratio from CBOE website.

    The CBOE publishes daily equity put/call ratio data. The URL format
    changes periodically. We try known endpoints.

    Returns:
        True if successful, False otherwise.
    """
    logger.info("Attempting to fetch CBOE equity put/call ratio...")

    urls_to_try = [
        "https://cdn.cboe.com/data/us/options/market_statistics/daily/equity_put_call_ratio.csv",
        "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_PCR.json",
    ]

    for url in urls_to_try:
        logger.info("  Trying: %s", url)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                logger.warning("  HTTP %d", resp.status_code)
                continue

            if url.endswith(".json"):
                data = resp.json()
                if "data" in data:
                    df = pd.DataFrame(data["data"])
                    if not df.empty:
                        out = RAW_DIR / "put_call_ratio.csv"
                        df.to_csv(out, index=False)
                        logger.info("  Put/call ratio: %d rows -> %s", len(df), out)
                        return True
            else:
                if len(resp.text) > 100:
                    df = pd.read_csv(StringIO(resp.text))
                    if not df.empty:
                        out = RAW_DIR / "put_call_ratio.csv"
                        df.to_csv(out, index=False)
                        logger.info("  Put/call ratio: %d rows -> %s", len(df), out)
                        return True
        except (requests.RequestException, ValueError) as e:
            logger.warning("  Failed: %s", e)

    logger.warning(
        "Could not fetch put/call ratio from CBOE. "
        "This data can be manually downloaded from cboe.com."
    )
    return False


def main() -> None:
    """Fetch all supplementary data sources."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0

    # Fetch Yahoo Finance tickers
    for symbol, filename in YF_TICKERS.items():
        if fetch_yf_ticker(symbol, filename):
            successes += 1
        else:
            failures += 1

    # Fetch CBOE put/call ratio
    if fetch_cboe_put_call_ratio():
        successes += 1
    else:
        failures += 1

    logger.info("Supplementary data fetch complete: %d succeeded, %d failed", successes, failures)

    if successes == 0:
        logger.error("All supplementary data fetches failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
