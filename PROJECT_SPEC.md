# VIX Spike Alert Bot -- Project Specification

## Overview

A lightweight, always-on trading signal bot that monitors VIX conditions in real-time, runs a trained neural network model to evaluate trade opportunities, and sends push notifications to your phone when it identifies a high-confidence mean-reversion setup. The bot suggests specific trades (instrument, strike, expiry) but does not execute them automatically.

**Target hardware:** Raspberry Pi 5 (inference/monitoring) + NVIDIA 4070 (training only)

---

## Architecture

```text
┌──────────────────────────────────────────────────────┐
│                   Raspberry Pi 5                      │
│                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ Data Poller  │───>│ Feature Eng. │───>│  Model   │ │
│  │ (Scheduler)  │    │  Pipeline    │    │ (ONNX)   │ │
│  └─────────────┘    └──────────────┘    └────┬─────┘ │
│        │                                      │       │
│        │              ┌──────────────┐        │       │
│        │              │  Notifier    │<───────┘       │
│        │              │  (Telegram)  │                │
│        │              └──────────────┘                │
│        │              ┌──────────────┐                │
│        └─────────────>│  SQLite DB   │                │
│                       │  (data log)  │                │
│                       └──────────────┘                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│               Desktop (4070) -- Training Only         │
│                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ Historical   │───>│  Training    │───>│  Export  │ │
│  │ Data Scripts │    │  (PyTorch)   │    │  (ONNX)  │ │
│  └─────────────┘    └──────────────┘    └──────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Data Requirements

### Real-Time Data (for live monitoring)

| Data Point                           | Source   | Update Frequency                  | Priority |
| ------------------------------------ | -------- | --------------------------------- | -------- |
| VIX spot price                       | IBKR API | Every 1-2 min during market hours | Critical |
| VIX futures curve (front 4-6 months) | IBKR API | Every 5 min                       | Critical |
| SPY price + daily change             | IBKR API | Every 1-2 min                     | High     |
| UVXY/VXX price + options chain       | IBKR API | On signal trigger                 | High     |
| VIX options chain                    | IBKR API | On signal trigger                 | Medium   |

### Historical Data (for training)

| Data Point                         | Source                     | Coverage                                | Cost                                |
| ---------------------------------- | -------------------------- | --------------------------------------- | ----------------------------------- |
| VIX daily close                    | Yahoo Finance (`yfinance`) | 1990-present                            | Free                                |
| VIX futures historical settlements | CBOE website / Quandl      | 2004-present                            | Free (scraping) or ~$30/mo (Quandl) |
| SPY daily OHLCV                    | Yahoo Finance              | 1993-present                            | Free                                |
| VIX term structure (reconstructed) | CBOE + manual cleanup      | 2004-present                            | Free (labor-intensive)              |
| UVXY/VXX daily                     | Yahoo Finance              | 2009-present (VXX), 2011-present (UVXY) | Free                                |

### Historical Data Acquisition Strategy

The cleanest free path for VIX futures history:

1. **CBOE historical data** -- they publish daily settlement CSVs at `https://www.cboe.com/tradable_products/vix/vix_futures/`. Scrape or manually download monthly files going back to 2004.
2. **VIX Central** (`vixcentral.com`) -- has daily term structure snapshots. Can be scraped with care (respect rate limits).
3. **Quandl/Nasdaq Data Link** -- has `CHRIS/CBOE_VX1` through `CHRIS/CBOE_VX9` for continuous VIX futures contracts. Free tier covers some of this.
4. **yfinance** -- for everything else (VIX spot, SPY, VXX, UVXY).

Plan to spend a solid day or two cleaning and aligning this data. Futures roll dates, contract expirations, and ticker changes (VXX had a reverse split and reissue) all need handling.

---

## Model Design

### Input Features

The model receives a feature vector computed from raw market data:

| Feature          | Description                             | Rationale                                        |
| ---------------- | --------------------------------------- | ------------------------------------------------ |
| `vix_spot`       | Current VIX level                       | Primary signal                                   |
| `vix_zscore`     | VIX relative to 60-day rolling mean/std | Normalizes "how spiked" it is                    |
| `vix_velocity`   | Rate of change over 1, 3, 5 days        | Fast spikes revert differently than slow grinds  |
| `term_slope`     | (Month2 - Month1) / Month1              | Contango/backwardation indicator                 |
| `term_curvature` | Shape of the full futures curve         | Deeper backwardation = stronger reversion signal |
| `spy_drawdown`   | SPY % below 20-day high                 | Context for what's driving VIX                   |
| `spy_velocity`   | SPY rate of change (5-day)              | Fast crashes vs slow bleeds                      |
| `rv_iv_spread`   | 20-day realized vol minus VIX           | Implied vs realized divergence                   |
| `vix_percentile` | VIX rank over trailing 252 days         | Historical context                               |
| `days_elevated`  | Days VIX has been above 25              | Persistence of the spike                         |

### Model Architecture

A small LSTM or 1D-CNN is appropriate here. The dataset is small (roughly 5,000 trading days with complete features) and the signal is relatively simple.

## Recommended: 2-layer LSTM

```text
Input (10 features x 20-day lookback window)
    -> LSTM(hidden=64, layers=2, dropout=0.3)
    -> Linear(64, 32)
    -> ReLU
    -> Linear(32, 3)
    -> Output: [confidence, expected_reversion_magnitude, optimal_dte]
```

- **confidence**: 0-1 score, threshold at ~0.7 for alerts
- **expected_reversion_magnitude**: predicted VIX drop in points over next 30 days
- **optimal_dte**: suggested days-to-expiration for the trade (bucketed: 30/60/90)

Parameter count: roughly 50,000-100,000. Trains in seconds on the 4070. Inference in sub-millisecond on Pi 5.

### Training Approach

- **Labels**: For each historical day, compute forward 30-day VIX change. Label as positive if VIX dropped more than 20% in the next 30 days from a level above 25.
- **Walk-forward validation**: Train on years 1-N, validate on year N+1. Roll forward. Never peek at future data.
- **Class imbalance**: VIX spike events are rare. Use weighted loss or oversample spike periods.
- **Beware overfitting**: With only ~15-20 major spike events, the model should be kept small and heavily regularized. Consider an ensemble of simple models rather than one complex one.

---

## Technology Stack

### Training Environment (Desktop / 4070)

| Component     | Tool                       | Notes                                |
| ------------- | -------------------------- | ------------------------------------ |
| Language      | Python 3.11+               |                                      |
| ML Framework  | PyTorch                    | Training and model definition        |
| Data handling | pandas, numpy              | Feature engineering                  |
| Data fetching | yfinance, requests         | Historical data collection           |
| Export        | ONNX (`torch.onnx.export`) | For Pi deployment                    |
| Notebooks     | Jupyter                    | Exploratory analysis and backtesting |

### Inference/Monitoring Environment (Raspberry Pi 5)

| Component       | Tool                                         | Notes                                         |
| --------------- | -------------------------------------------- | --------------------------------------------- |
| Language        | Python 3.11+                                 |                                               |
| ML Inference    | ONNX Runtime (`onnxruntime`)                 | Lightweight, no PyTorch needed on Pi          |
| Broker API      | `ib_insync`                                  | Wraps IBKR TWS/Gateway API                    |
| Scheduling      | `APScheduler`                                | Cron-like job scheduling in Python            |
| Database        | SQLite                                       | Local logging of data points and signals      |
| Notifications   | Telegram Bot API (`python-telegram-bot`)     | Free, reliable, instant push                  |
| Process manager | `systemd`                                    | Auto-start on boot, restart on crash          |
| Monitoring      | Simple health-check endpoint (Flask/FastAPI) | Optional: hit from phone to verify it's alive |

### Why These Choices

**ONNX Runtime over PyTorch on Pi**: PyTorch CPU on ARM works but it's heavyweight (~500MB+). ONNX Runtime is ~30MB, faster inference, and purpose-built for deployment.

**ib_insync over raw IBKR API**: The raw TWS API is callback-hell in Python. `ib_insync` wraps it in a clean async interface. Night and day difference in developer experience.

**Telegram over SMS/Pushover**: Free, no per-message costs, supports rich formatting (you can send the trade details as a nicely formatted message), and has a dead-simple bot API. Creating a bot takes 2 minutes through BotFather.

**SQLite over Postgres/anything else**: You're logging maybe a few hundred rows per day on a single-user system. SQLite is zero-config, file-based, and perfect for this scale.

**APScheduler over cron**: Keeps everything in one Python process. Easier to manage state between polling intervals than separate cron-invoked scripts.

---

## Interactive Brokers Setup

### Account Requirements

1. Open an IBKR account (no minimum for cash account)
2. Subscribe to market data:
   - **CBOE One** -- real-time VIX index and VIX options (~$1/mo)
   - **CFE (CBOE Futures Exchange)** -- VIX futures real-time (~$4/mo)
   - **US Equity and Options** -- for UVXY/VXX options (~$1.50/mo if not waived)
3. Enable API access in account settings

Total market data cost: roughly $5-7/month.

### Connection Architecture

IBKR requires either **TWS (Trader Workstation)** or **IB Gateway** running as a middleman between your code and their servers. IB Gateway is the headless/lighter option -- ideal for the Pi.

However, IB Gateway is a Java application. Running it on a Pi is possible but annoying. Two practical options:

## Option A (recommended): Run IB Gateway on the Pi directly

- Install Java (OpenJDK ARM), download IB Gateway
- It's a bit heavy for a Pi but works. Uses ~200-400MB RAM.
- Everything runs on one device.

## Option B: Run IB Gateway on the desktop, connect from Pi over LAN

- Desktop runs IB Gateway 24/7 (or during market hours)
- Pi connects to it over your local network
- Cleaner separation but requires desktop to be on

For a "set it and forget it" setup, Option A keeps everything self-contained on the Pi.

### Auto-Restart Handling

IB Gateway requires daily restarts (IBKR forces disconnection around 11:45 PM ET). Use `ibc` (IB Controller) -- an open-source tool that handles automatic login and restart of IB Gateway. This is essential for unattended operation.

```text
Pi boot -> systemd starts IB Gateway (via ibc) -> systemd starts bot
```

---

## Notification Design

### Telegram Bot Message Format

When the model fires a signal, send a structured message:

```text
--- VIX ALERT: High Confidence Signal ---

Signal Strength: 0.83 / 1.00
Timestamp: 2026-02-19 14:32 CT

-- Market Snapshot --
VIX Spot: 34.2 (+41% above 60d mean)
VIX Futures M1: 31.8 (backwardation: -7.0%)
VIX Futures M2: 28.5
SPY: 482.30 (-6.2% from 20d high)

-- Suggested Trade --
Instrument: UVXY
Action: Buy Puts
Suggested Strike: $28 (current: $42.10)
Suggested Expiry: May 2026 (~90 DTE)
Model Expected VIX in 30d: ~24

-- Context --
Days VIX elevated (>25): 4
Historical win rate at this signal level: 78%
Avg return on similar setups: +35% on put value

---
```

### Alert Conditions

- **Primary trigger**: Model confidence > 0.7 AND VIX spot > 28
- **Cooldown**: No more than 1 alert per 24 hours (avoid spam during sustained spikes)
- **Daily digest**: Even without a signal, send a brief daily status at market close:
  - Current VIX, model confidence, term structure summary
  - Confirms the bot is alive and watching

---

## Project Structure

```text
vix-alert-bot/
├── README.md
├── requirements-train.txt          # Desktop/training dependencies
├── requirements-pi.txt             # Pi/inference dependencies
│
├── data/
│   ├── raw/                        # Raw downloaded CSVs
│   ├── processed/                  # Cleaned, aligned training data
│   └── scripts/
│       ├── fetch_vix_history.py    # Download historical VIX spot
│       ├── fetch_futures_history.py # Scrape/download futures data
│       ├── fetch_spy_history.py    # Download SPY data
│       └── build_dataset.py        # Merge, align, compute features
│
├── training/
│   ├── features.py                 # Feature engineering functions
│   ├── dataset.py                  # PyTorch Dataset class
│   ├── model.py                    # LSTM model definition
│   ├── train.py                    # Training loop with walk-forward CV
│   ├── backtest.py                 # Strategy backtester
│   ├── export_onnx.py             # Export trained model to ONNX
│   └── notebooks/
│       ├── eda.ipynb               # Exploratory data analysis
│       └── backtest_results.ipynb  # Visualization of backtest
│
├── bot/
│   ├── config.py                   # Configuration (API keys, thresholds)
│   ├── main.py                     # Entry point, scheduler setup
│   ├── data_poller.py              # IBKR data fetching via ib_insync
│   ├── feature_pipeline.py         # Real-time feature computation
│   ├── inference.py                # ONNX model loading and prediction
│   ├── notifier.py                 # Telegram bot integration
│   ├── trade_suggester.py          # Maps model output to trade params
│   ├── db.py                       # SQLite logging
│   └── health.py                   # Optional health-check endpoint
│
├── deploy/
│   ├── setup_pi.sh                 # Pi setup script (deps, dirs)
│   ├── vix-bot.service             # systemd unit file for the bot
│   ├── ib-gateway.service          # systemd unit file for IB Gateway
│   └── ibc-config.ini              # IB Controller configuration
│
└── models/
    └── vix_model.onnx              # Exported model file
```

---

## Implementation Phases

### Phase 1: Data Collection and Exploration (1-2 days)

- Write data fetching scripts for historical VIX spot, futures, SPY
- Clean and align the data into a unified DataFrame
- Exploratory analysis: visualize VIX spikes, term structure during events, reversion timelines
- Define labeling strategy for the model

### Phase 2: Feature Engineering and Model Training (2-3 days)

- Implement feature computation pipeline
- Build PyTorch Dataset and DataLoader
- Train LSTM with walk-forward cross-validation
- Iterate on features, hyperparameters, regularization
- Backtest the signal against historical data
- Export final model to ONNX

### Phase 3: Bot Infrastructure (1-2 days)

- Set up Telegram bot via BotFather
- Implement IBKR connection with `ib_insync`
- Build the data polling scheduler
- Wire up real-time feature pipeline to ONNX inference
- Implement notification formatting and sending
- Add SQLite logging

### Phase 4: Pi Deployment (1 day)

- Set up Pi with OS, Python, deps
- Install and configure IB Gateway + IBC
- Deploy bot as systemd service
- Test end-to-end: market data -> features -> model -> alert
- Set up daily digest messages

### Phase 5: Iteration and Monitoring (ongoing)

- Monitor model performance against real signals
- Retrain periodically as new data accumulates
- Tune alert thresholds based on live experience
- Consider adding features (credit spreads, put/call ratio, etc.)

---

## Estimated Costs

| Item                                    | Cost              | Frequency                       |
| --------------------------------------- | ----------------- | ------------------------------- |
| IBKR market data subscriptions          | ~$5-7             | Monthly                         |
| Raspberry Pi 5 (8GB) + case + PSU + SSD | ~$100-120         | One-time                        |
| Telegram bot                            | Free              | --                              |
| Historical data (if using free sources) | Free (time cost)  | One-time                        |
| Historical data (if using Quandl)       | ~$30              | Monthly (cancel after download) |
| IBKR account                            | Free (no minimum) | --                              |

**Total ongoing cost: ~$5-7/month.**

---

## Risk Considerations

- **Model overfitting**: With limited spike events, the model may memorize rather than generalize. Prioritize simplicity and regularization. A well-tuned threshold-based system with a few hand-picked features may outperform a complex model.
- **Data snooping**: Be rigorous about walk-forward validation. It's tempting to "optimize" on the full dataset.
- **IB Gateway reliability**: It disconnects daily and sometimes has maintenance windows. The bot needs graceful reconnection logic.
- **VIX spike severity**: The model might fire early in a spike that gets much worse. The notification should always include context (how far SPY has fallen, how long VIX has been elevated) so you can use judgment.
- **Regime changes**: VIX behavior during 2008, 2020, and 2022 looked quite different. A model trained primarily on recent data may not handle a 2008-style event well. Consider training on the full history despite the small sample.

---

## Potential Enhancements (Future)

- **Web dashboard**: Simple Flask/FastAPI app on the Pi showing current model state, historical signals, and P/L tracking on past alerts
- **Multiple models**: Ensemble of LSTM + gradient boosted trees (XGBoost) for more robust signals
- **Execution integration**: Semi-automated order placement through IBKR with one-tap Telegram confirmation
- **Additional instruments**: Extend to other mean-reverting signals (VVIX, skew, credit spreads)
- **SMS fallback**: Add Twilio SMS as backup notification channel in case Telegram is down
