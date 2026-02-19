# VIX Spike Alert Bot -- Project Specification

## Overview

A lightweight, always-on trading signal bot that monitors VIX conditions in real-time, runs a trained model to evaluate trade opportunities, and sends push notifications to your phone when it identifies a high-confidence mean-reversion setup. The bot targets the **full spectrum of VIX mean-reversion trades** -- not just extreme spikes (VIX 40+), but also moderate elevations (VIX 20-30) where shorter vol positions can profit from a reversion to the mid-teens. The bot suggests specific trades (instrument, strike, expiry) but does not execute them automatically.

**Target hardware:** Raspberry Pi 5 (inference/monitoring) + NVIDIA 4070 (training only)

**Core thesis:** VIX is mean-reverting at all elevated levels, not just during extreme spikes. By capturing moderate setups (e.g., VIX 22 reverting to 16) alongside major events, the bot sees 200-400+ tradeable setups over the historical dataset instead of ~15-20, enabling a genuinely trainable model with better signal diversity.

**Modeling philosophy:** Start with the simplest model that could work (gradient-boosted trees), prove edge in rigorous walk-forward backtesting, and only add complexity (LSTM) if the simpler model leaves clear performance on the table. Every model must beat a defined rules-based baseline to ship.

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
│  │ Data Scripts │    │  (XGBoost +  │    │  (ONNX)  │ │
│  │             │    │   PyTorch)   │    │          │ │
│  └─────────────┘    └──────────────┘    └──────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Data Requirements

### Real-Time Data (for live monitoring)

| Data Point                           | Source          | Update Frequency                  | Priority |
| ------------------------------------ | --------------- | --------------------------------- | -------- |
| VIX spot price                       | IBKR API        | Every 1-2 min during market hours | Critical |
| VIX futures curve (front 4-6 months) | IBKR API        | Every 5 min                       | Critical |
| SPY price + daily change             | IBKR API        | Every 1-2 min                     | High     |
| VVIX (vol-of-vol index)              | IBKR API        | Every 5 min                       | High     |
| VIX9D (9-day VIX)                    | IBKR API        | Every 5 min                       | High     |
| SKEW index                           | IBKR API        | Every 15 min                      | High     |
| UVXY/VXX price + options chain       | IBKR API        | On signal trigger                 | High     |
| VIX options chain                    | IBKR API        | On signal trigger                 | Medium   |
| VIX futures front-month volume       | IBKR API        | Every 5 min                       | Medium   |
| Put/call ratio (CBOE equity)         | IBKR API / CBOE | End of day                        | Medium   |
| HY credit spread (HYG-TLT proxy)     | IBKR API        | Every 15 min                      | Medium   |

### Historical Data (for training)

| Data Point                         | Source                     | Coverage                                | Cost                                |
| ---------------------------------- | -------------------------- | --------------------------------------- | ----------------------------------- |
| VIX daily close                    | Yahoo Finance (`yfinance`) | 1990-present                            | Free                                |
| VIX futures historical settlements | CBOE website / Quandl      | 2004-present                            | Free (scraping) or ~$30/mo (Quandl) |
| VIX futures volume (daily)         | CBOE / Quandl              | 2004-present                            | Free (scraping)                     |
| SPY daily OHLCV                    | Yahoo Finance              | 1993-present                            | Free                                |
| VIX term structure (reconstructed) | CBOE + manual cleanup      | 2004-present                            | Free (labor-intensive)              |
| UVXY/VXX daily                     | Yahoo Finance              | 2009-present (VXX), 2011-present (UVXY) | Free                                |
| VVIX daily close                   | Yahoo Finance / CBOE       | 2006-present                            | Free                                |
| VIX9D daily close                  | CBOE                       | 2011-present                            | Free                                |
| SKEW daily close                   | CBOE / Yahoo Finance       | 2011-present                            | Free                                |
| HYG daily (HY credit proxy)        | Yahoo Finance              | 2007-present                            | Free                                |
| CBOE equity put/call ratio         | CBOE website               | 2006-present                            | Free (scraping)                     |

### Historical Data Acquisition Strategy

The cleanest free path for VIX futures history:

1. **CBOE historical data** -- they publish daily settlement CSVs at `https://www.cboe.com/tradable_products/vix/vix_futures/`. Scrape or manually download monthly files going back to 2004.
2. **VIX Central** (`vixcentral.com`) -- has daily term structure snapshots. Can be scraped with care (respect rate limits).
3. **Quandl/Nasdaq Data Link** -- has `CHRIS/CBOE_VX1` through `CHRIS/CBOE_VX9` for continuous VIX futures contracts. Free tier covers some of this.
4. **yfinance** -- for everything else (VIX spot, SPY, VXX, UVXY).

Plan to spend **3-4 days** cleaning and aligning this data. Futures roll dates, contract expirations, and ticker changes (VXX had a reverse split and reissue) all need handling. CBOE futures data scraping and alignment is notoriously painful -- don't underestimate this step.

---

## Model Design

### Input Features

The model receives a feature vector computed from raw market data. All features are standardized before model input using a `StandardScaler` fit exclusively on the training window (never the full dataset) to avoid leakage. Z-score and percentile features are inherently normalized but are still passed through the scaler for consistency.

| Feature              | Description                                         | Rationale                                                               |
| -------------------- | --------------------------------------------------- | ----------------------------------------------------------------------- |
| `vix_spot`           | Current VIX level (standardized)                    | Primary signal                                                          |
| `vix_zscore`         | VIX relative to 60-day rolling mean/std             | Normalizes "how elevated" VIX is relative to recent regime              |
| `vix_velocity`       | Rate of change over 1, 3, 5 days                    | Fast spikes revert differently than slow grinds                         |
| `term_slope`         | (Month2 - Month1) / Month1                          | Contango/backwardation indicator                                        |
| `term_slope_zscore`  | `term_slope` relative to its own 60-day stats       | Distinguishes unusual term structure from normal contango/backwardation |
| `term_curvature`     | Shape of the full futures curve                     | Deeper backwardation = stronger reversion signal                        |
| `spy_drawdown`       | SPY % below 20-day high                             | Context for what's driving VIX                                          |
| `spy_velocity`       | SPY rate of change (5-day)                          | Fast crashes vs slow bleeds                                             |
| `rv_iv_spread`       | 20-day realized vol minus VIX (using T-1 data only) | Implied vs realized divergence. Use only settled data to avoid leakage  |
| `vix_percentile`     | VIX rank over trailing 252 days                     | Historical context                                                      |
| `days_elevated`      | Days VIX has been above its 60-day mean + 1 std     | Persistence of the elevation (dynamic threshold, not fixed)             |
| `vvix`               | CBOE VVIX index (vol-of-vol)                        | High VVIX at moderate VIX = market uncertain, spike risk higher         |
| `vix9d_vix_ratio`    | VIX9D / VIX                                         | Short-term vs medium-term fear; ratio < 1 signals near-term resolution  |
| `skew`               | CBOE SKEW index                                     | Tail risk pricing; elevated SKEW + moderate VIX = hidden risk           |
| `vix_futures_volume` | Front-month VIX futures daily volume (standardized) | Thin volume + backwardation is a different signal than heavy volume     |
| `put_call_ratio`     | CBOE equity put/call ratio                          | Sentiment indicator; extreme readings add context at moderate VIX       |
| `hy_spread_velocity` | 5-day rate of change in HYG-TLT spread              | Credit stress widening while VIX moderate = danger sign                 |
| `day_of_week`        | Encoded day of week (0-4)                           | VIX expiration week and options roll patterns have known effects        |

### Feature Normalization

All features are standardized using `StandardScaler` fit on the training window of each walk-forward fold. At inference time, the scaler is fit on a trailing 252-day window updated daily. The scaler parameters are saved alongside the model for deployment.

For the XGBoost model, normalization is less critical (tree models are scale-invariant) but is applied anyway for pipeline consistency when comparing against the LSTM.

### Model Architecture

With the broader scope (moderate + extreme VIX elevations), the training set grows to roughly 200-400 labeled setups across ~5,000 trading days. Given this sample size, the modeling strategy prioritizes simplicity and interpretability first.

#### Primary Model: XGBoost (Gradient-Boosted Trees)

XGBoost is the primary model. With 200-400 labeled samples, tree-based models consistently outperform neural networks on tabular data at this scale. They're also faster to iterate on, easier to interpret, and less prone to overfitting.

For the XGBoost model, temporal features are provided as explicit lagged columns rather than a sequence:

```text
Input: 17 current features + lagged features at t-1, t-3, t-5, t-10, t-20
       (for key features: vix_spot, vix_zscore, term_slope, spy_drawdown, vvix)
       Total: ~42 features

XGBoost Classifier (for p_revert, p_spike_first)
  - max_depth: 4-6
  - n_estimators: 200-500 (with early stopping)
  - learning_rate: 0.05-0.1
  - subsample: 0.8
  - colsample_bytree: 0.8
  - scale_pos_weight: tuned to class imbalance ratio

XGBoost Regressor (for expected_magnitude)
  - Same hyperparameter ranges
```

XGBoost exports to ONNX via `skl2onnx` or `onnxmltools`. Inference is sub-millisecond on Pi 5.

#### Secondary Model: 2-Layer LSTM (only if XGBoost leaves clear room for improvement)

The LSTM is trained only after XGBoost establishes a baseline. It must beat XGBoost by a meaningful margin (>2% AUC on walk-forward validation) to justify deployment complexity.

```text
Input (17 features x 20-day lookback window)
    -> LSTM(hidden=64, layers=2, dropout=0.3)
    -> Linear(64, 32)
    -> ReLU
    -> Linear(32, 3)
    -> Output: [p_revert, p_spike_first, expected_magnitude]
```

Parameter count: roughly 50,000-100,000. Trains in seconds on the 4070. Inference in sub-millisecond on Pi 5.

#### Ensemble Option

If both models show independent strengths (e.g., XGBoost better on moderate setups, LSTM better on spike detection), a simple average or stacked ensemble is worth testing. But this is Phase 2 optimization, not the starting point.

**Output heads (for both models):**

| Output               | Range      | Description                                                                 |
| -------------------- | ---------- | --------------------------------------------------------------------------- |
| `p_revert`           | 0-1        | Probability that VIX reverts >= threshold within the horizon window         |
| `p_spike_first`      | 0-1        | Probability that VIX rises >= 10% before reverting (adverse excursion risk) |
| `expected_magnitude` | continuous | Expected reversion size in % over the horizon window                        |

The key insight: **alert only when `p_revert` is high AND `p_spike_first` is low.** A setup where VIX will eventually revert but spikes another 30% first is one you'd get stopped out of or panic-close. The `p_spike_first` head is what separates "good entry" from "right idea, wrong timing."

**Alert logic:** `p_revert > 0.7 AND p_spike_first < 0.3 AND vix_zscore > 1.0`

DTE and strike selection are handled downstream by the trade suggester using heuristics (see Trade Suggestion Logic), not predicted by the model -- these depend more on options market conditions than on VIX direction.

### Rules-Based Baseline

Every model must beat this baseline in walk-forward testing to be considered useful. The baseline is defined upfront to prevent goalpost-shifting:

```text
Signal fires when ALL of:
  - vix_zscore > 1.5
  - term_slope < -0.02 (backwardation) OR vix_zscore > 2.0
  - spy_velocity (5-day) < -2% (equity selloff context)

Baseline p_spike_first proxy:
  - Flag as "elevated spike risk" when VVIX > 120 AND term structure flattening
```

If the trained model can't beat this on walk-forward precision/recall, use the rules-based system. A working rules-based bot that catches 60% of setups is more valuable than an overfit model that looks great in-sample.

### Training Approach

#### Labeling Strategy

For each historical trading day where VIX z-score > 1.0 (i.e., VIX is meaningfully elevated relative to its recent regime), compute labels using a **tiered horizon window** that adapts to VIX level:

| VIX Level | Horizon Window  | Rationale                                                    |
| --------- | --------------- | ------------------------------------------------------------ |
| 18-25     | 15 trading days | Moderate elevations revert quickly or not at all             |
| 25-35     | 30 trading days | Standard spike recovery timeline                             |
| 35+       | 45 trading days | Major spikes (2008, 2020, 2024) take longer to fully resolve |

Labels for each eligible day:

1. **`label_revert`** (binary): Did VIX drop >= 15% from today's close within the horizon window?
2. **`label_spike_first`** (binary): Did VIX rise >= 10% from today's close _before_ the reversion occurred (or within the horizon window if no reversion)?
3. **`label_magnitude`** (continuous): Maximum percentage drop from today's close within the horizon window.

This z-score-based entry threshold (instead of a fixed level like VIX > 25) is critical -- it captures both moderate setups (VIX 22 in a low-vol regime) and extreme spikes (VIX 40+) under the same framework. The 15% reversion threshold catches everything from VIX 22->18.7 to VIX 40->34.

**Negative labels matter as much as positive ones.** Days where VIX is elevated but _doesn't_ revert (or spikes further first) teach the model when _not_ to fire. These include:

- Early entries in a developing spike (VIX goes from 25 to 40 before reverting)
- Sustained high-vol regimes (2008 Q4, early 2020) where VIX stayed elevated for weeks
- Slow grinds higher that don't revert quickly

#### Training Details

- **Walk-forward validation**: Train on years 1-N, validate on year N+1. Roll forward. Never peek at future data. Minimum 5 walk-forward folds for statistical credibility.
- **Loss function (LSTM)**: Multi-task loss combining BCE for the two probability heads + MSE for magnitude, with tunable weights.
- **Loss function (XGBoost)**: Separate models per output head. `log_loss` for classifiers, `reg:squarederror` for magnitude.
- **Class balance**: With the broader scope, class imbalance is less severe (~200-400 positive labels out of ~5,000 days) but still present. Use `scale_pos_weight` (XGBoost) or focal loss (LSTM) rather than oversampling.
- **Regularization**: For XGBoost: max_depth cap, subsample, colsample_bytree, early stopping on validation loss. For LSTM: dropout (0.3), early stopping, weight decay. Both models are small enough that heavy regularization is appropriate.
- **Phased training approach**:
  - **Phase 2a**: Train `p_revert` head only (binary classifier) using XGBoost. Validate in walk-forward. Compare against the rules-based baseline. This is the sanity check -- if this doesn't beat the baseline, revisit features and labeling before adding complexity.
  - **Phase 2b**: Train `p_spike_first` head using XGBoost. This is the hard, high-value prediction. If walk-forward performance is not statistically meaningful (AUC < 0.60 across folds), fall back to the rules-based spike risk proxy (VVIX + term structure heuristic) and document why.
  - **Phase 2c**: Add `expected_magnitude` head only if 2a and 2b show solid walk-forward performance.
  - **Phase 2d**: Train LSTM on the same data. Compare head-to-head against XGBoost on all walk-forward folds. Only adopt LSTM if it beats XGBoost by >2% AUC consistently across folds. Consider ensemble if strengths are complementary.

---

## Backtesting

Backtesting is the gatekeeper between training and deployment. No model ships without passing a rigorous backtest that simulates realistic trading conditions.

### Backtest Requirements

The backtest must simulate the full signal-to-trade pipeline, not just "did VIX revert Y/N":

1. **Signal generation**: Run the model on each walk-forward validation fold. Record every signal that passes the alert threshold.
2. **Entry simulation**: Assume entry at T+1 open (next trading day) to account for the delay between alert and execution. For sensitivity analysis, also test T+0 close and T+1 close entries.
3. **Instrument selection**: Apply the tiered instrument selection heuristics to pick the suggested trade for each signal.
4. **P&L simulation**: For each suggested trade, compute:
   - Entry price using the mid-quote of the suggested option at entry time (or modeled via Black-Scholes if historical options data is unavailable)
   - Mark-to-market daily through expiry
   - Exit at expiry, or at a target profit (e.g., +50% on put value), or at a stop loss (e.g., -60% on put value), whichever comes first
5. **Transaction costs**: Apply realistic friction to every trade:
   - Commission: $0.65/contract (IBKR)
   - Bid-ask spread: assume 5% of mid for liquid VIX options, 8% for UVXY options, 15% for illiquid strikes/DTEs
   - Slippage from execution delay: model as an additional 2-5% adverse move on entry

### Backtest Metrics

Report these metrics separately for each tier (moderate / major spike) and in aggregate:

| Metric              | Description                                                    | Minimum Bar          |
| ------------------- | -------------------------------------------------------------- | -------------------- |
| Win rate            | % of trades that are profitable after costs                    | >55% aggregate       |
| Avg win / avg loss  | Ratio of average profitable trade to average losing trade      | >1.5                 |
| Max drawdown        | Worst peak-to-trough across the simulated P&L curve            | Report, no fixed bar |
| Profit factor       | Gross profits / gross losses                                   | >1.3                 |
| Signals per year    | Average number of alerts fired per year                        | 10-30 (sanity check) |
| Model vs baseline   | Performance lift over the rules-based baseline on same signals | >0% (must beat it)   |
| Sharpe (annualized) | Risk-adjusted return of the simulated strategy                 | >0.5                 |

### What-If Scenarios

The backtest should include sensitivity analysis for:

- **Execution delay**: How much does P&L degrade if entry is delayed 1 hour, 4 hours, or next day?
- **Threshold tuning**: Sweep `p_revert` from 0.6 to 0.9 and `p_spike_first` from 0.1 to 0.4. Plot precision/recall tradeoff and P&L impact.
- **Transaction cost sensitivity**: Double the assumed spread. Is the strategy still profitable?

---

## Technology Stack

### Training Environment (Desktop / 4070)

| Component     | Tool                               | Notes                                |
| ------------- | ---------------------------------- | ------------------------------------ |
| Language      | Python 3.11+                       |                                      |
| ML Framework  | XGBoost (primary), PyTorch (LSTM)  | XGBoost first, LSTM only if needed   |
| Data handling | pandas, numpy                      | Feature engineering                  |
| Data fetching | yfinance, requests                 | Historical data collection           |
| Export        | ONNX (`onnxmltools`, `torch.onnx`) | For Pi deployment                    |
| Backtesting   | Custom (see Backtesting section)   | Walk-forward P&L simulation          |
| Notebooks     | Jupyter                            | Exploratory analysis and backtesting |

### Inference/Monitoring Environment (Raspberry Pi 5)

| Component       | Tool                                         | Notes                                         |
| --------------- | -------------------------------------------- | --------------------------------------------- |
| Language        | Python 3.11+                                 |                                               |
| ML Inference    | ONNX Runtime (`onnxruntime`)                 | Lightweight, no PyTorch/XGBoost needed on Pi  |
| Broker API      | `ib_insync`                                  | Wraps IBKR TWS/Gateway API                    |
| Scheduling      | `APScheduler`                                | Cron-like job scheduling in Python            |
| Market calendar | `exchange_calendars`                         | NYSE/CBOE holiday and half-day awareness      |
| Database        | SQLite                                       | Local logging of data points and signals      |
| Notifications   | Telegram Bot API (`python-telegram-bot`)     | Free, reliable, instant push                  |
| Process manager | `systemd`                                    | Auto-start on boot, restart on crash          |
| Monitoring      | Simple health-check endpoint (Flask/FastAPI) | Optional: hit from phone to verify it's alive |

### Why These Choices

**XGBoost over LSTM as primary model**: With 200-400 labeled samples and ~42 features, gradient-boosted trees consistently outperform neural networks on tabular data at this scale. They're faster to train, easier to interpret (feature importance is built in), less prone to overfitting, and simpler to debug. The LSTM remains available as a secondary model if XGBoost leaves clear room for improvement.

**ONNX Runtime over PyTorch on Pi**: PyTorch CPU on ARM works but it's heavyweight (~500MB+). ONNX Runtime is ~30MB, faster inference, and purpose-built for deployment. Both XGBoost and PyTorch models export cleanly to ONNX.

**ib_insync over raw IBKR API**: The raw TWS API is callback-hell in Python. `ib_insync` wraps it in a clean async interface. Night and day difference in developer experience.

**Telegram over SMS/Pushover**: Free, no per-message costs, supports rich formatting (you can send the trade details as a nicely formatted message), and has a dead-simple bot API. Creating a bot takes 2 minutes through BotFather.

**SQLite over Postgres/anything else**: You're logging maybe a few hundred rows per day on a single-user system. SQLite is zero-config, file-based, and perfect for this scale.

**APScheduler over cron**: Keeps everything in one Python process. Easier to manage state between polling intervals than separate cron-invoked scripts.

**exchange_calendars for market awareness**: The bot must know when markets are open, closed, or on half-day schedules. Without this, you get stale data alerts on holidays and wasted cycles polling closed markets.

---

## Trade Suggestion Logic

The model predicts _whether_ to enter a trade. The `trade_suggester.py` module handles _what_ trade to suggest based on VIX level, term structure, and options market conditions. This is heuristic-driven, not model-predicted -- strike/DTE selection depends on live options pricing that the model doesn't see.

### Tiered Instrument Selection

| VIX Level | Regime             | Preferred Instrument                      | Rationale                                                                    |
| --------- | ------------------ | ----------------------------------------- | ---------------------------------------------------------------------------- |
| 30+       | Major spike        | UVXY puts                                 | High convexity, large expected move, premium is rich                         |
| 22-30     | Moderate elevation | VIX puts or short VIX call spreads        | More capital-efficient for moderate moves; UVXY less attractive at lower vol |
| 18-22     | Mild elevation     | Short VIX call spreads or small UVXY puts | Smallest position size; reversion less certain, risk/reward tighter          |

### Settlement and Exercise Notes

These matter for strike/DTE selection and must be surfaced in alerts:

- **VIX options**: European-style, cash-settled, AM settlement on expiration day. The settlement value (VRO) is calculated from opening prices of SPX options, not from the VIX spot close. This means VIX options can settle significantly different from where VIX spot closed the prior day.
- **UVXY options**: American-style, standard equity option settlement. Can be exercised early (relevant for deep ITM puts near expiry).
- **VIX futures**: Cash-settled at the same AM settlement value as VIX options. Front-month futures converge to this settlement, not to VIX spot.

The alert should note which settlement type applies for the suggested instrument.

### Position Sizing Guidance

The alert should suggest relative sizing based on signal strength and VIX level:

- **Full size**: VIX > 30, `p_revert` > 0.8, `p_spike_first` < 0.2
- **Half size**: VIX 22-30, `p_revert` > 0.7, `p_spike_first` < 0.3
- **Small/starter**: VIX 18-22, or any setup where `p_spike_first` > 0.2

These are suggestions, not rules -- the notification includes all the context for the user to decide. The alert should also include a **maximum suggested loss** (e.g., "risk no more than X% of portfolio on this setup") based on the sizing tier.

### DTE Selection Heuristics

- **VIX > 30 (backwardation)**: 60-90 DTE. Major spikes take time to fully revert; shorter expiries risk theta decay if the spike persists.
- **VIX 22-30 (contango or flat)**: 30-60 DTE. Moderate elevations revert faster; less time premium needed.
- **General rule**: Target an expiry where the front-month futures contract is closest to fair value relative to spot (i.e., where the term structure starts flattening out).

### Strike Selection Heuristics

- **UVXY puts**: Strike at roughly 60-70% of current price (targeting the expected decay).
- **VIX puts**: Strike near the front-month futures price (ATM relative to futures, not spot).
- **VIX call spreads**: Short strike near current spot, long strike 5-10 points higher as a cap.

### Liquidity Checks

Before suggesting a specific strike/DTE, the trade suggester should verify:

- **Bid-ask spread**: Skip strikes where the spread is >15% of mid. Illiquid options eat edge.
- **Open interest**: Prefer strikes with open interest >500 contracts.
- **If no liquid strike matches the heuristic**: Suggest the nearest liquid alternative and note the deviation in the alert.

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

However, IB Gateway is a Java application. Running it on a Pi is possible but resource-intensive. Two practical options:

## Option A (recommended): Run IB Gateway on the Pi directly

- Install Java (OpenJDK ARM), download IB Gateway
- Uses ~200-400MB RAM. Combined with Python + ONNX Runtime + SQLite, expect ~60-70% of the Pi's 8GB under normal operation. Monitor with a simple memory check in the health endpoint.
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

### Connection Resilience

The bot must handle IBKR connection issues gracefully:

- **Reconnection logic**: On disconnect, retry with exponential backoff (5s, 15s, 30s, 60s, then every 5 min). Log every disconnect/reconnect event.
- **Data staleness detection**: Track the timestamp of the last successful quote for each data point. If VIX spot is >5 minutes stale during market hours, mark data as stale and suppress model inference. Send a Telegram alert: "Data feed stale -- last VIX quote at HH:MM. Model inference paused."
- **Market calendar awareness**: Use `exchange_calendars` to know when NYSE/CBOE are open. Do not poll, run inference, or fire alerts outside market hours. Handle early closes (day before Thanksgiving, etc.).
- **Weekend/holiday handling**: Switch to a low-frequency heartbeat mode (one health check per hour). Resume normal polling 5 minutes before market open.

---

## Notification Design

### Telegram Bot Message Format

When the model fires a signal, send a structured message. The format adapts based on whether the setup is a major spike or a moderate elevation:

```text
--- VIX ALERT: High Confidence Signal ---
Model: vix_xgb_v003 (2026-01-15)

P(revert): 0.85    P(spike first): 0.12
Expected reversion: -22%
Sizing suggestion: FULL (max risk: 3% of portfolio)
Timestamp: 2026-02-19 14:32 CT

-- Market Snapshot --
VIX Spot: 34.2 (z-score: +2.4, 96th percentile)
VVIX: 118 (moderate)
VIX9D/VIX: 0.91 (near-term fear fading)
SKEW: 128 (moderate tail risk)
VIX Futures M1: 31.8 (backwardation: -7.0%)
VIX Futures M2: 28.5
Term slope z-score: -1.8 (unusual backwardation)
SPY: 482.30 (-6.2% from 20d high)
HY spread: stable

-- Suggested Trade --
Instrument: UVXY puts (major spike tier)
Settlement: American-style, equity settlement
Suggested Strike: $28 (current: $42.10)
Suggested Expiry: May 2026 (~90 DTE)
Bid-ask at suggested strike: $5.20 / $5.60 (7.4% spread)

-- Context --
Days VIX elevated: 4 (above 60d mean + 1 std)
Walk-forward win rate at this signal level: 78%
Avg return on similar setups: +35% on put value

---
```

```text
--- VIX ALERT: Moderate Setup ---
Model: vix_xgb_v003 (2026-01-15)

P(revert): 0.74    P(spike first): 0.25
Expected reversion: -16%
Sizing suggestion: HALF (max risk: 1.5% of portfolio)
Timestamp: 2026-03-10 10:15 CT

-- Market Snapshot --
VIX Spot: 22.8 (z-score: +1.5, 82nd percentile)
VVIX: 98 (low -- less uncertainty)
VIX9D/VIX: 0.96 (neutral)
SKEW: 135 (slightly elevated tail risk)
VIX Futures M1: 21.5 (contango: +3.2%)
VIX Futures M2: 20.8
Term slope z-score: +0.4 (normal contango)
SPY: 510.40 (-2.1% from 20d high)
HY spread: stable

-- Suggested Trade --
Instrument: VIX puts (moderate tier)
Settlement: European-style, AM cash settlement
Suggested Strike: 21 (near M1 futures)
Suggested Expiry: Apr 2026 (~45 DTE)
Bid-ask at suggested strike: $1.85 / $2.10 (12.7% spread)

-- Context --
Days VIX elevated: 2 (above 60d mean + 1 std)
Walk-forward win rate at this signal level: 65%
Avg return on similar setups: +18% on put value

---
```

**Important**: The "walk-forward win rate" and "avg return" figures in the notification must come exclusively from out-of-sample walk-forward validation folds. Never use in-sample or full-dataset statistics. If insufficient out-of-sample data exists for a given signal level, display "Insufficient data" rather than a misleading number.

### Alert Conditions

- **Primary trigger**: `p_revert > 0.7 AND p_spike_first < 0.3 AND vix_zscore > 1.0`
- **No fixed VIX floor**: The z-score threshold replaces a fixed level like "VIX > 28." This allows the bot to catch moderate setups in low-vol regimes (VIX 20 when the 60-day mean is 14) while ignoring VIX 20 when it's been at 25 for months.
- **Cooldown**: No more than 1 alert per 24 hours for the same tier. **Upgrade alerts bypass cooldown** -- if a previous alert was "moderate" and conditions worsen (VIX spikes higher, tier changes to "major spike"), send an updated alert immediately reclassifying the tier and adjusting the trade suggestion.
- **Daily digest**: Even without a signal, send a brief daily status at market close:
  - Current VIX, z-score, model outputs (`p_revert`, `p_spike_first`), term structure summary
  - Model version and last data timestamp (confirms the bot is alive and watching)
  - Any data staleness events that occurred during the day

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
│       ├── fetch_supplementary.py  # VIX9D, SKEW, VVIX, HYG, P/C ratio
│       └── build_dataset.py        # Merge, align, compute features
│
├── training/
│   ├── features.py                 # Feature engineering functions
│   ├── scaler.py                   # StandardScaler fitting and serialization
│   ├── dataset.py                  # PyTorch Dataset class (for LSTM)
│   ├── model_xgb.py               # XGBoost model definition and training
│   ├── model_lstm.py              # LSTM model definition (secondary)
│   ├── baseline.py                 # Rules-based baseline for comparison
│   ├── train.py                    # Training loop with walk-forward CV
│   ├── backtest.py                 # Full P&L backtester (see Backtesting section)
│   ├── export_onnx.py             # Export trained model to ONNX (versioned)
│   └── notebooks/
│       ├── eda.ipynb               # Exploratory data analysis
│       ├── backtest_results.ipynb  # Visualization of backtest
│       └── model_comparison.ipynb  # XGBoost vs LSTM vs baseline comparison
│
├── bot/
│   ├── config.py                   # Configuration (API keys, thresholds)
│   ├── main.py                     # Entry point, scheduler setup
│   ├── data_poller.py              # IBKR data fetching via ib_insync
│   ├── staleness.py                # Data staleness detection and alerting
│   ├── feature_pipeline.py         # Real-time feature computation
│   ├── inference.py                # ONNX model loading and prediction
│   ├── notifier.py                 # Telegram bot integration
│   ├── trade_suggester.py          # Maps model output to trade params
│   ├── db.py                       # SQLite logging
│   └── health.py                   # Health-check endpoint + memory monitoring
│
├── deploy/
│   ├── setup_pi.sh                 # Pi setup script (deps, dirs)
│   ├── vix-bot.service             # systemd unit file for the bot
│   ├── ib-gateway.service          # systemd unit file for IB Gateway
│   └── ibc-config.ini              # IB Controller configuration
│
└── models/
    ├── vix_xgb_v001_20260115.onnx  # Versioned model files
    ├── scaler_v001_20260115.pkl    # Corresponding scaler
    └── model_manifest.json         # Tracks deployed model version + metadata
```

### Model Versioning

Every trained model is saved with a version number and training date: `vix_{type}_v{NNN}_{YYYYMMDD}.onnx`. The `model_manifest.json` tracks:

- Currently deployed model version
- Training date and data range used
- Walk-forward validation metrics (AUC, precision, recall per fold)
- Which model type (XGBoost, LSTM, ensemble)
- Feature scaler version

The bot loads the model specified in the manifest. Model swaps are a manifest update + bot restart.

---

## Implementation Phases

### Phase 1: Data Collection and Exploration (3-4 days)

- Write data fetching scripts for historical VIX spot, futures, SPY, VVIX, VIX9D, SKEW, HYG, put/call ratio
- Clean and align the data into a unified DataFrame. Expect futures roll alignment and CBOE data format inconsistencies to be the biggest time sinks.
- Exploratory analysis: visualize the full spectrum of VIX elevations (not just extreme spikes), term structure during events, reversion timelines at different VIX levels
- Characterize the dataset: count how many labeled setups exist at different z-score thresholds and horizon windows
- Define and validate the labeling strategy (revert/spike-first/magnitude) with the tiered horizon windows
- Implement and validate the rules-based baseline. Record its walk-forward performance -- this is the bar every model must clear.

### Phase 2: Feature Engineering and Model Training (4-5 days)

- **Phase 2a**: Implement feature computation pipeline with standardization. Train XGBoost `p_revert` classifier with walk-forward CV. Compare against the rules-based baseline. If XGBoost doesn't beat the baseline, iterate on features and labeling before proceeding.
- **Phase 2b**: Train XGBoost `p_spike_first` classifier. This is the highest-value and hardest prediction -- expect iteration here. If walk-forward AUC < 0.60 across folds, fall back to the rules-based spike risk proxy (VVIX + term structure heuristic) and move on.
- **Phase 2c**: Add `expected_magnitude` regressor if 2a and 2b show solid walk-forward performance. Skip if not.
- **Phase 2d**: Train LSTM on the same data. Compare head-to-head against XGBoost. Only adopt if it beats XGBoost by >2% AUC consistently. Test ensemble if both show independent strengths.
- Export final model(s) to ONNX with versioned filenames.

### Phase 3: Backtesting (2-3 days)

- Implement the full P&L backtester (entry simulation, instrument selection, transaction costs, exit logic)
- Run backtest on all walk-forward validation folds
- Compute all required metrics (win rate, profit factor, Sharpe, max drawdown) separately for moderate and major spike tiers
- Run sensitivity analyses (execution delay, threshold tuning, cost sensitivity)
- **Gate**: If the backtest doesn't meet minimum bars (win rate >55%, profit factor >1.3, beats baseline), do not proceed to bot deployment. Go back to Phase 2 and iterate.

### Phase 4: Bot Infrastructure (2-3 days)

- Set up Telegram bot via BotFather
- Implement IBKR connection with `ib_insync` including reconnection logic and data staleness detection
- Build the data polling scheduler with market calendar awareness
- Wire up real-time feature pipeline to ONNX inference
- Implement tiered trade suggestion logic (instrument, sizing, DTE, strike heuristics, liquidity checks)
- Implement notification formatting (both major spike and moderate setup templates, model version, settlement notes)
- Add SQLite logging with model version tracking
- Implement health endpoint with memory monitoring

### Phase 5: Paper Trading (1-2 weeks)

Run the full pipeline against live market data without acting on alerts. This phase validates:

- **Data pipeline reliability**: Does the IBKR connection hold? How often does data go stale? Are reconnections smooth?
- **Feature pipeline correctness**: Do real-time features match what the model saw in training? Log real-time features alongside predictions and spot-check against historical computations.
- **Alert quality**: Are the alerts sensible? Do the suggested trades have reasonable strikes and DTEs? Are liquidity checks working?
- **Operational stability**: Does the bot stay up for days at a time? Memory leaks? Disk usage creeping?

Log every model inference (inputs, outputs, timestamp) to SQLite during this phase. Compare model predictions against what actually happens over the next 15-45 days. This builds the first live performance dataset.

**Do not skip this phase.** The gap between "works in backtest" and "works in production" is where most trading systems fail.

### Phase 6: Pi Deployment (1 day)

- Set up Pi with OS, Python, deps
- Install and configure IB Gateway + IBC
- Deploy bot as systemd service
- Test end-to-end: market data -> features -> model -> alert
- Set up daily digest messages
- Verify memory usage stays within safe bounds under normal operation
- Set up monitoring for disk space (SQLite will grow over time)

### Phase 7: Iteration and Monitoring (ongoing)

- Monitor model performance against real signals, tracking win rate separately for moderate vs. major setups
- Retrain when meaningful new data accumulates (a new VIX regime or several new spike events, not on a fixed schedule)
- Tune alert thresholds (`p_revert`, `p_spike_first`, z-score) based on live experience
- Track `p_spike_first` accuracy specifically -- this is the head most likely to need calibration
- Review the kill switch criteria monthly (see Risk Considerations)

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

- **Adverse excursion on moderate setups**: Shorting vol at VIX 22 is fundamentally riskier than at VIX 40. A move from 22 to 35 before reversion can wipe out a position even if the eventual direction is correct. The `p_spike_first` model head (or its rules-based fallback) is the primary defense here, but the notification should always surface this risk clearly. Position sizing (smaller at lower VIX) is the other key lever.
- **Model overfitting**: With ~200-400 labeled setups, overfitting is less catastrophic than with 15-20, but still a real risk. Keep the model simple (XGBoost with max_depth 4-6), use walk-forward validation exclusively, and always compare against the rules-based baseline. If the model doesn't beat the baseline, use the baseline.
- **Data snooping**: Be rigorous about walk-forward validation. It's tempting to "optimize" on the full dataset. The phased training approach (2a/2b/2c/2d) helps -- each head must independently prove value. The rules-based baseline, defined before training, prevents goalpost-shifting.
- **IB Gateway reliability**: It disconnects daily and sometimes has maintenance windows. The bot needs graceful reconnection logic, data staleness detection, and clear alerting when the pipeline is degraded (see Connection Resilience).
- **Regime changes**: VIX behavior during 2008, 2020, and 2022 looked quite different. The z-score-based approach helps here (it normalizes across regimes), but a model trained primarily on post-2010 low-vol environments may underestimate tail risk. Train on the full 2004-present history.
- **Moderate setups are noisier**: VIX at 22 sometimes just drifts back to 18, sometimes explodes to 40. The signal-to-noise ratio is inherently worse than at extreme levels. Expect lower win rates on moderate-tier alerts (~60-65%) vs. major spike alerts (~75-80%). The daily digest should track this over time.
- **Feature leakage with `rv_iv_spread`**: Realized vol is backward-looking, VIX is forward-looking. Always compute realized vol using only T-1 and earlier data to avoid including information not yet available at inference time.
- **Options liquidity in stressed markets**: During VIX spikes, bid-ask spreads on VIX and UVXY options can blow out. The trade suggester's liquidity checks are a first defense, but the alert should always include the current spread so the user can judge whether the trade is executable at a reasonable price.

### Kill Switch Criteria

Define these upfront. If any are triggered, pause the bot and investigate before continuing:

| Condition                                              | Action                                                                           |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Walk-forward win rate drops below 50% over 20+ signals | Pause alerts, retrain, re-validate                                               |
| 3 consecutive losing trades                            | Review recent signals manually, check for drift                                  |
| Model predictions cluster (always >0.8 or always <0.3) | Model may be stuck/overfit. Inspect feature pipeline                             |
| Data staleness >30 min during market hours             | Suppress all alerts until resolved                                               |
| Pi memory usage >85%                                   | Investigate, restart services, check for leaks                                   |
| New VIX regime (sustained >40 for 2+ weeks)            | Model likely out-of-distribution. Switch to rules-based baseline until retrained |

---

## Potential Enhancements (Future)

- **Web dashboard**: Simple Flask/FastAPI app on the Pi showing current model state, historical signals, and P&L tracking on past alerts (separate win rates for moderate vs. major setups)
- **LSTM or Transformer model**: If more labeled data accumulates over time (500+ setups), revisit sequence models that may better capture temporal dynamics
- **Ensemble**: Stack XGBoost + LSTM predictions if both show independent strengths. XGBoost may particularly help on moderate setups where tabular features dominate, LSTM on spike detection where sequence matters.
- **Execution integration**: Semi-automated order placement through IBKR with one-tap Telegram confirmation
- **Adaptive z-score windows**: Experiment with multiple lookback windows (30d, 60d, 120d) for the z-score calculation to capture different regime speeds
- **Intraday signals**: Move from end-of-day to intraday feature computation for faster entries on fast-moving spikes
- **SMS fallback**: Add Twilio SMS as backup notification channel in case Telegram is down
