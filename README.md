# VIX Spike Alert Bot

A machine-learning-powered trading signal bot that detects VIX mean-reversion opportunities. Uses XGBoost trained on 15 years of data with walk-forward cross-validation, connected to Interactive Brokers for live data and Telegram for alerts.

## How It Works

When the VIX spikes, it almost always reverts. This bot detects those spikes and predicts:

1. **p_revert** — probability that VIX drops 15%+ within a tiered time window
2. **p_spike_first** — probability of a further 10%+ spike before reversion (risk filter)
3. **expected_magnitude** — predicted size of the reversion

When conditions are favorable (`p_revert > 0.7`, `p_spike_first < 0.3`, `vix_zscore > 1.0`), the bot sends a Telegram alert with a specific trade suggestion (instrument, strike, DTE, sizing).

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- macOS: `brew install libomp` (required by XGBoost)

### Installation

```bash
git clone <repo-url> && cd VIX-Model

# Install dependencies
uv venv
uv pip install -e ".[dev]"

# Verify installation
uv run pytest tests/ -v
```

### Run in Mock Mode (No Credentials Needed)

```bash
uv run python -m bot.main --mock
```

This runs the full pipeline with synthetic data — polling, feature computation, inference, and alert formatting — so you can see exactly how the bot behaves without any external connections.

## Configuration

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

### Environment Variables

| Variable                   | Default             | Description                             |
| -------------------------- | ------------------- | --------------------------------------- |
| `IBKR_HOST`                | `127.0.0.1`         | IB Gateway / TWS host                   |
| `IBKR_PORT`                | `4001`              | IB Gateway port (4001=live, 4002=paper) |
| `IBKR_CLIENT_ID`           | `1`                 | IBKR API client ID                      |
| `TELEGRAM_BOT_TOKEN`       | —                   | Telegram bot token from @BotFather      |
| `TELEGRAM_CHAT_ID`         | —                   | Your Telegram chat ID                   |
| `MODEL_VERSION`            | `v001`              | Which model version to load             |
| `ALERT_P_REVERT_THRESHOLD` | `0.7`               | Min p_revert to trigger alert           |
| `ALERT_P_SPIKE_THRESHOLD`  | `0.3`               | Max p_spike_first to trigger alert      |
| `ALERT_ZSCORE_THRESHOLD`   | `1.0`               | Min VIX z-score to trigger alert        |
| `ALERT_COOLDOWN_HOURS`     | `24`                | Hours between alerts for same tier      |
| `HEALTH_PORT`              | `8080`              | Port for health check endpoints         |
| `DATA_DIR`                 | `./data`            | Data directory path                     |
| `MODELS_DIR`               | `./models`          | Trained models directory                |
| `DB_PATH`                  | `./data/vix_bot.db` | SQLite database path                    |

### Setting Up Telegram

1. Message [@BotFather](https://t.me/BotFather) on Telegram and create a new bot
2. Copy the bot token to `TELEGRAM_BOT_TOKEN`
3. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat ID
4. Set `TELEGRAM_CHAT_ID`

### Setting Up Interactive Brokers

1. Install [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) or TWS
2. Enable API connections in Gateway settings (Configure > API > Settings)
3. Set `IBKR_HOST` and `IBKR_PORT` in `.env`
4. For paper trading, use port `4002`

## Usage

### Running the Bot

```bash
# Live mode (requires IBKR + Telegram credentials)
uv run python -m bot.main

# Mock mode (synthetic data, no credentials)
uv run python -m bot.main --mock

# With debug logging
uv run python -m bot.main --mock --log-level DEBUG

# Without health check server
uv run python -m bot.main --mock --no-health
```

The bot runs three scheduled jobs:

| Job              | Schedule                        | Description                                                   |
| ---------------- | ------------------------------- | ------------------------------------------------------------- |
| **Poll cycle**   | Every 5 min during market hours | Fetch data, compute features, run inference, send alerts      |
| **Daily digest** | 4:05 PM ET                      | End-of-day summary with VIX level, z-score, model predictions |
| **Heartbeat**    | Every hour                      | Keep-alive check during off-hours                             |

### Health Check

While the bot is running, you can monitor it:

```bash
# Basic health
curl http://localhost:8080/health

# Detailed status with per-symbol staleness
curl http://localhost:8080/status
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=bot --cov=training --cov-report=term-missing

# Specific test file
uv run pytest tests/test_features.py -v
```

## Data Pipeline

### Fetching Historical Data

```bash
# Download VIX daily data (yfinance)
uv run python data/scripts/fetch_vix_history.py

# Download SPY daily data
uv run python data/scripts/fetch_spy_history.py

# Download supplementary data (VVIX, VIX9D, SKEW, HYG, TLT)
uv run python data/scripts/fetch_supplementary.py

# Attempt VIX futures data (may require paid source)
uv run python data/scripts/fetch_futures_history.py

# Generate mock intraday 5-min bars
uv run python data/scripts/fetch_intraday.py
```

All raw data is saved to `data/raw/`.

### Building the Dataset

```bash
uv run python data/scripts/build_dataset.py
```

This merges raw data, computes 45 features, generates tiered labels, and saves to `data/processed/vix_dataset.parquet`. If VIX futures data is unavailable, it synthesizes a realistic term structure from VIX spot prices.

**Output summary:**

- 3,805 trading days (2011-2026)
- 45 features (20 base + 25 lagged)
- 493 eligible spike days (VIX z-score > 1.0)
- Tiered horizons: 15 days (VIX 18-25), 30 days (VIX 25-35), 45 days (VIX 35+)

## Training Pipeline

### Training Models

```bash
uv run python -m training.train \
    --dataset data/processed/vix_dataset.parquet \
    --output models \
    --min-train-years 5
```

Training uses **walk-forward cross-validation** (train on years 1-N, validate on year N+1) and proceeds in gated phases:

| Phase  | Target        | Method             | Gate Criteria                                 |
| ------ | ------------- | ------------------ | --------------------------------------------- |
| **2a** | p_revert      | XGBoost classifier | Must beat rules baseline in majority of folds |
| **2b** | p_spike_first | XGBoost classifier | AUC > 0.60, else falls back to rules          |
| **2c** | magnitude     | XGBoost regressor  | Only if 2a and 2b pass                        |
| **2d** | all heads     | CNN+GRU (optional) | Only if beats XGBoost by >2% AUC              |

### Current Model Performance (v001)

| Metric         | p_revert (Phase 2a) | p_spike_first (Phase 2b) |
| -------------- | ------------------- | ------------------------ |
| Mean AUC       | **0.706**           | 0.519 (below 0.60 gate)  |
| Mean F1        | **0.821**           | —                        |
| Mean Precision | 0.884               | —                        |
| Mean Recall    | 0.857               | —                        |
| Beats baseline | 9/10 folds          | —                        |
| Method used    | XGBoost             | Rules-based fallback     |

**Top predictive features:** `vix_percentile`, `vix_spot_lag20`, `vvix`, `vvix_lag10`, `term_slope_zscore`

### Retraining

To retrain after fetching new data:

```bash
# 1. Fetch latest data
uv run python data/scripts/fetch_vix_history.py
uv run python data/scripts/fetch_spy_history.py
uv run python data/scripts/fetch_supplementary.py

# 2. Rebuild dataset
uv run python data/scripts/build_dataset.py

# 3. Retrain models
uv run python -m training.train

# 4. Restart bot to pick up new model
uv run python -m bot.main
```

## Project Structure

```text
VIX-Model/
├── bot/                        # Live inference bot
│   ├── main.py                 # Entry point, scheduler, CLI
│   ├── config.py               # Configuration from .env
│   ├── data_poller.py          # IBKR + mock data polling
│   ├── db.py                   # SQLite async database
│   ├── feature_pipeline.py     # Real-time feature computation
│   ├── inference.py            # XGBoost + mock inference
│   ├── trade_suggester.py      # Tiered trade suggestions
│   ├── notifier.py             # Telegram + mock notifications
│   ├── staleness.py            # Data freshness tracking
│   └── health.py               # FastAPI health endpoints
│
├── training/                   # ML training pipeline
│   ├── train.py                # Walk-forward CV orchestrator
│   ├── model_xgb.py            # XGBoost model definitions
│   ├── model_hcg.py            # Hierarchical CNN+GRU (secondary)
│   ├── baseline.py             # Rules-based baseline
│   ├── features.py             # 45-feature computation
│   ├── aggregation.py          # Multi-resolution preprocessing
│   ├── scaler.py               # Feature scaling
│   ├── dataset.py              # PyTorch dataset classes
│   ├── backtest.py             # P&L backtester with Black-Scholes
│   └── export_onnx.py          # Model export utilities
│
├── data/
│   ├── scripts/                # Data fetching scripts
│   │   ├── build_dataset.py    # Merge raw data → training dataset
│   │   ├── fetch_vix_history.py
│   │   ├── fetch_spy_history.py
│   │   ├── fetch_futures_history.py
│   │   ├── fetch_supplementary.py
│   │   └── fetch_intraday.py
│   ├── raw/                    # Downloaded CSVs
│   └── processed/              # Parquet training dataset
│
├── models/                     # Trained model artifacts
│   ├── model_manifest.json     # Model metadata & feature list
│   ├── *.json                  # XGBoost model files
│   └── *.pkl                   # Fitted scalers
│
├── tests/                      # Test suite (170 tests)
│   ├── test_features.py        # Feature computation tests
│   ├── test_labeling.py        # Label generation tests
│   ├── test_baseline.py        # Rules baseline tests
│   ├── test_model_xgb.py       # XGBoost training tests
│   ├── test_backtest.py        # Backtester + Black-Scholes tests
│   ├── test_trade_suggester.py # Trade suggestion tests
│   ├── test_inference.py       # Inference pipeline tests
│   ├── test_db.py              # Database CRUD tests
│   └── test_config.py          # Configuration tests
│
├── pyproject.toml              # Dependencies and build config
├── .env.example                # Configuration template
└── PROJECT_SPEC.md             # Full project specification
```

## Features (45 total)

### Base Features (20)

| Feature                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `vix_spot`              | Current VIX level                              |
| `vix_zscore`            | VIX z-score (60-day rolling)                   |
| `vix_velocity_1d/3d/5d` | Rate of VIX change                             |
| `term_slope`            | VIX futures term structure slope               |
| `term_slope_zscore`     | Term slope z-score                             |
| `term_curvature`        | Term structure curvature (M3-M2 vs M2-M1)      |
| `spy_drawdown`          | SPY drawdown from 20-day high                  |
| `spy_velocity`          | SPY rate of change                             |
| `rv_iv_spread`          | Realized vs implied vol spread (uses T-1 data) |
| `vix_percentile`        | VIX percentile rank (252-day)                  |
| `days_elevated`         | Consecutive days with z-score > 1.0            |
| `vvix`                  | CBOE VVIX (volatility of VIX)                  |
| `vix9d_vix_ratio`       | VIX9D / VIX ratio                              |
| `skew`                  | CBOE SKEW index                                |
| `vix_futures_volume`    | VIX futures volume proxy                       |
| `put_call_ratio`        | Equity put/call ratio                          |
| `hy_spread_velocity`    | High-yield spread rate of change               |
| `day_of_week`           | Day of week (0=Monday)                         |

### Lagged Features (25)

Five key features (`vix_spot`, `vix_zscore`, `term_slope`, `spy_drawdown`, `vvix`) at lags t-1, t-3, t-5, t-10, t-20.

## Trade Signal Tiers

| VIX Level | Tier        | Instrument                | DTE   | Sizing |
| --------- | ----------- | ------------------------- | ----- | ------ |
| 35+       | Major Spike | UVXY puts                 | 60-90 | Full   |
| 25-35     | Moderate    | VIX puts / spreads        | 45-60 | Half   |
| 18-25     | Mild        | Call spreads / small puts | 30-45 | Small  |

## Architecture

### Data Flow

```text
Historical Data (yfinance)        Live Data (IBKR 5-min bars)
         │                                  │
         ▼                                  ▼
   build_dataset.py                   data_poller.py
         │                                  │
         ▼                                  ▼
  45 features + labels              feature_pipeline.py
         │                                  │
         ▼                                  ▼
  Walk-forward training              XGBoost inference
  (training/train.py)               (bot/inference.py)
         │                                  │
         ▼                                  ▼
  Exported model (.json)            Alert if thresholds met
  + scaler (.pkl)                   (bot/notifier.py)
                                            │
                                            ▼
                                    Telegram message with
                                    trade suggestion
```

### Design Decisions

- **XGBoost over neural nets**: With ~500 labeled samples, gradient-boosted trees outperform deep learning. The CNN+GRU architecture (`training/model_hcg.py`) is scaffolded for when more intraday data is available.
- **Walk-forward only**: No random train/test splits. Every evaluation fold only uses past data for training, preventing look-ahead bias.
- **Gated phases**: Each training phase must pass quality gates before proceeding. If p_spike_first can't be learned (AUC < 0.60), a rules-based proxy is used instead.
- **Feature leakage prevention**: `rv_iv_spread` uses T-1 data, scalers are fit only on training folds, and label/meta columns are explicitly excluded from features.
- **Synthetic futures fallback**: When real VIX futures data isn't available (free sources are unreliable), the pipeline synthesizes a realistic term structure from VIX spot using a contango/backwardation model.

## Backtesting

The backtester (`training/backtest.py`) simulates full P&L with:

- **Entry**: T+1 open price after signal
- **Exit**: Target profit (+50%), stop loss (-60%), or expiry
- **Pricing**: Black-Scholes option pricing (no historical options data available)
- **Costs**: $0.65/contract commission, 5-15% spread (by instrument), 2-5% slippage
- **Sensitivity analysis**: Execution delay sweep, threshold tuning, transaction cost doubling

```bash
# Run a backtest (after training)
uv run python -c "
from training.backtest import VIXBacktester, BacktestConfig
import pandas as pd

config = BacktestConfig()
bt = VIXBacktester(config)

df = pd.read_parquet('data/processed/vix_dataset.parquet')
# ... see training/backtest.py for full API
"
```

## Development

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Adding Features

1. Add computation to `training/features.py`
2. Add the feature name to `build_dataset.py` pipeline
3. Add tests in `tests/test_features.py`
4. Rebuild dataset and retrain

### Running Individual Components

```bash
# Test feature computation
uv run python -c "
from training.features import compute_all_features
import pandas as pd
df = pd.read_parquet('data/processed/vix_dataset.parquet')
print(df.columns.tolist())
print(df.describe())
"

# Test inference standalone
uv run python -c "
from bot.inference import MockInference
inf = MockInference()
inf.load()
pred = inf.predict({'vix_spot': 30, 'vix_zscore': 2.5})
print(f'p_revert={pred.p_revert:.3f}, p_spike={pred.p_spike_first:.3f}')
"
```

## License

Private / proprietary.
