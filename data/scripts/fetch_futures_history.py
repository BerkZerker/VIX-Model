"""Fetch historical VIX futures data from multiple free sources.

Strategy:
1. Try Quandl/Nasdaq Data Link free CSVs for continuous contracts (CHRIS/CBOE_VX1..VX9)
2. Try CBOE VIX futures historical data page
3. Fall back to yfinance VIX futures tickers (limited availability)

Saves individual files to data/raw/vix_futures_*.csv.
Idempotent: overwrites existing files on each run.
"""

import logging
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

# Quandl/Nasdaq Data Link free CSV URLs (no API key required for some datasets)
QUANDL_BASE = "https://data.nasdaq.com/api/v3/datasets/CHRIS/CBOE_VX{month}.csv"

# CBOE VIX futures historical data
CBOE_FUTURES_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX+VXT/{filename}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def fetch_quandl_continuous(month: int) -> pd.DataFrame | None:
    """Fetch continuous VIX futures contract from Quandl free tier.

    Args:
        month: Contract month number (1-9).

    Returns:
        DataFrame or None if fetch fails.
    """
    url = QUANDL_BASE.format(month=month)
    logger.info("Trying Quandl VX%d: %s", month, url)
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 100:
            df = pd.read_csv(StringIO(resp.text))
            if not df.empty:
                logger.info("  VX%d: %d rows", month, len(df))
                return df
        logger.warning("  VX%d: HTTP %d or empty response", month, resp.status_code)
    except requests.RequestException as e:
        logger.warning("  VX%d: request failed: %s", month, e)
    return None


def fetch_cboe_futures() -> pd.DataFrame | None:
    """Try to fetch VIX futures data from CBOE's public data files.

    CBOE publishes historical settlement data. The exact URL format changes
    periodically. We try a few known patterns.
    """
    # Try the CBOE historical data page for VIX futures
    known_filenames = [
        "VX_History.csv",
        "VX+VXT_History.csv",
    ]

    for filename in known_filenames:
        url = CBOE_FUTURES_URL.format(filename=filename)
        logger.info("Trying CBOE futures URL: %s", url)
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 200:
                df = pd.read_csv(StringIO(resp.text))
                if not df.empty:
                    logger.info("  CBOE futures: %d rows from %s", len(df), filename)
                    return df
        except requests.RequestException as e:
            logger.warning("  CBOE %s failed: %s", filename, e)
        time.sleep(1)  # Rate limiting

    # Try the CBOE API endpoint for futures data
    cboe_api_url = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/VX.json"
    logger.info("Trying CBOE API: %s", cboe_api_url)
    try:
        resp = SESSION.get(cboe_api_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data:
                df = pd.DataFrame(data["data"])
                if not df.empty:
                    logger.info("  CBOE API: %d rows", len(df))
                    return df
    except (requests.RequestException, ValueError) as e:
        logger.warning("  CBOE API failed: %s", e)

    return None


def fetch_yfinance_futures() -> dict[str, pd.DataFrame]:
    """Try fetching VIX futures from yfinance (limited but sometimes available)."""
    import yfinance as yf

    results = {}
    # yfinance sometimes has VIX futures as VX=F or VXF24, VXG24, etc.
    tickers_to_try = ["VX=F"]

    for ticker_str in tickers_to_try:
        logger.info("Trying yfinance ticker: %s", ticker_str)
        try:
            ticker = yf.Ticker(ticker_str)
            df = ticker.history(period="max", interval="1d")
            if not df.empty:
                df.index = df.index.tz_localize(None)
                drop_cols = [
                    c
                    for c in ("Dividends", "Stock Splits", "Capital Gains")
                    if c in df.columns
                ]
                if drop_cols:
                    df = df.drop(columns=drop_cols)
                results[ticker_str] = df
                logger.info("  %s: %d rows", ticker_str, len(df))
        except Exception as e:
            logger.warning("  %s failed: %s", ticker_str, e)

    return results


def main() -> None:
    """Download VIX futures data from all available free sources."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    any_success = False

    # --- Source 1: Quandl continuous contracts ---
    logger.info("=== Trying Quandl continuous contracts ===")
    for month in range(1, 10):
        df = fetch_quandl_continuous(month)
        if df is not None:
            out = RAW_DIR / f"vix_futures_vx{month}_quandl.csv"
            df.to_csv(out, index=False)
            logger.info("Saved %s", out)
            any_success = True
        time.sleep(0.5)  # Rate limiting

    # --- Source 2: CBOE historical data ---
    logger.info("=== Trying CBOE historical data ===")
    cboe_df = fetch_cboe_futures()
    if cboe_df is not None:
        out = RAW_DIR / "vix_futures_cboe.csv"
        cboe_df.to_csv(out, index=False)
        logger.info("Saved %s", out)
        any_success = True

    # --- Source 3: yfinance fallback ---
    logger.info("=== Trying yfinance VIX futures ===")
    yf_results = fetch_yfinance_futures()
    for ticker_str, df in yf_results.items():
        safe_name = ticker_str.replace("=", "").replace("/", "_").lower()
        out = RAW_DIR / f"vix_futures_{safe_name}_yf.csv"
        df.index.name = "Date"
        df.to_csv(out)
        logger.info("Saved %s", out)
        any_success = True

    if any_success:
        logger.info("VIX futures data collection complete. Check data/raw/ for results.")
    else:
        logger.warning(
            "No VIX futures data could be fetched from any source. "
            "This is expected if free endpoints are rate-limited or unavailable. "
            "Consider manual download from CBOE or a paid Quandl subscription."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
