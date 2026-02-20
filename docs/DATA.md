# Data Pipeline

## Data Sources

| Source   | Ticker/File     | Description                    | Free?             |
| -------- | --------------- | ------------------------------ | ----------------- |
| yfinance | `^VIX`          | VIX daily OHLCV (1990-present) | Yes               |
| yfinance | `SPY`           | S&P 500 ETF daily OHLCV        | Yes               |
| yfinance | `^VVIX`         | CBOE VVIX (2007-present)       | Yes               |
| yfinance | `^VIX9D`        | 9-day VIX (2011-present)       | Yes               |
| yfinance | `^SKEW`         | CBOE SKEW index                | Yes               |
| yfinance | `HYG`           | High-yield corporate bond ETF  | Yes               |
| yfinance | `TLT`           | Long-term Treasury ETF         | Yes               |
| IBKR     | VIX/SPY         | 5-minute intraday bars (live)  | With account      |
| CBOE     | Various         | VIX futures settlement data    | Unreliable (403s) |
| Quandl   | CHRIS/CBOE_VX\* | VIX continuous futures         | Paid              |

### VIX Futures Data

Historical VIX futures data is the hardest to obtain for free. The pipeline handles this with a waterfall of fallbacks:

1. **Combined CSV** (`data/raw/vix_futures.csv`) — if you have it from a paid source
2. **Quandl format** (`data/raw/vx1_daily.csv`, `vx2_daily.csv`) — from Quandl subscription
3. **CBOE settlement data** — scraped from CBOE website (often returns 403)
4. **yfinance** — delisted VIX futures tickers (no longer available)
5. **Synthetic fallback** — generates a realistic term structure from VIX spot

The synthetic fallback uses a model based on observed VIX behavior:

- VIX > 30: backwardation (futures below spot, mean ~3% discount)
- VIX 20-30: mild contango or flat
- VIX < 20: contango (futures above spot, mean ~5% premium)

This produces realistic term slope and curvature features for training when real futures data is unavailable.

## Raw Data Files

After running all fetch scripts, `data/raw/` contains:

| File                    | Size    | Rows  | Date Range |
| ----------------------- | ------- | ----- | ---------- |
| `vix_daily.csv`         | ~780 KB | 9,100 | 1990-2026  |
| `spy_daily.csv`         | ~770 KB | 8,300 | 1993-2026  |
| `vvix_daily.csv`        | ~400 KB | 4,800 | 2007-2026  |
| `vix9d_daily.csv`       | ~325 KB | 3,800 | 2011-2026  |
| `skew_daily.csv`        | ~780 KB | 9,000 | 1990-2026  |
| `hyg_daily.csv`         | ~440 KB | 4,700 | 2007-2026  |
| `tlt_daily.csv`         | ~540 KB | 5,900 | 2002-2026  |
| `vix_intraday_5min.csv` | ~340 KB | 7,020 | Mock data  |
| `spy_intraday_5min.csv` | ~380 KB | 7,020 | Mock data  |

## Feature Engineering

Features are computed in `training/features.py`. All features are computed from data available at time T (no future data leakage).

### Rolling Windows

Most features use rolling windows that require warmup periods:

- `vix_zscore`: 60-day rolling mean and std
- `vix_percentile`: 252-day rolling rank
- `rv_iv_spread`: 20-day realized vol (uses T-1 close)
- `spy_drawdown`: 20-day rolling max

After warmup (first ~60 trading days are dropped due to NaN), the dataset starts from approximately 2011.

### Feature Alignment

The dataset is aligned to VIX9D availability (2011-present) since VIX9D is one of the supplementary features. Missing supplementary data (VVIX, SKEW, etc.) is forward-filled to handle gaps.

## Labels

Labels are generated based on VIX behavior in a forward-looking window after each eligible day:

### Eligibility

A day is eligible for labeling when `vix_zscore > 1.0` (VIX is at least 1 standard deviation above its 60-day mean).

### Tiered Horizon Windows

Different VIX levels get different time horizons for reversion:

| VIX Level | Horizon         | Rationale                      |
| --------- | --------------- | ------------------------------ |
| 18-25     | 15 trading days | Mild elevation reverts quickly |
| 25-35     | 30 trading days | Moderate spikes need more time |
| 35+       | 45 trading days | Major spikes can persist       |

### Label Definitions

| Label               | Type       | Definition                                         |
| ------------------- | ---------- | -------------------------------------------------- |
| `label_revert`      | Binary     | VIX drops >= 15% from current level within horizon |
| `label_spike_first` | Binary     | VIX rises >= 10% before any 15% drop occurs        |
| `label_magnitude`   | Continuous | Maximum % drop within horizon                      |

### Dataset Statistics (v001)

- **Total trading days**: 3,805 (2011-2026)
- **Eligible days**: 493 (13% of all days)
- **Revert rate**: 84.2% (415/493) — VIX almost always reverts
- **Spike-first rate**: 37.1% (183/493) — but sometimes gets worse first

**By tier:**

| Tier      | Days | Revert Rate | Spike-First Rate |
| --------- | ---- | ----------- | ---------------- |
| VIX 18-25 | 281  | 78%         | —                |
| VIX 25-35 | 149  | 91%         | —                |
| VIX 35+   | 63   | 98%         | —                |

## Building the Dataset

```bash
# Fetch all raw data first
uv run python data/scripts/fetch_vix_history.py
uv run python data/scripts/fetch_spy_history.py
uv run python data/scripts/fetch_supplementary.py

# Build the training dataset
uv run python data/scripts/build_dataset.py
```

Output: `data/processed/vix_dataset.parquet`

The parquet file contains all 45 features plus label and meta columns, indexed by date.

## Adding New Data Sources

1. Create a fetch script in `data/scripts/` following the pattern of existing scripts
2. Add the new data loading to `data/scripts/build_dataset.py`
3. If it produces a new feature, add the computation to `training/features.py`
4. Add tests in `tests/test_features.py`
5. Rebuild the dataset and retrain
