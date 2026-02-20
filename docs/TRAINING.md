# Training Pipeline

## Overview

The training pipeline implements phased, walk-forward cross-validated model training. Every model evaluation uses strictly forward-looking data: train on years 1 through N, validate on year N+1.

## Walk-Forward Cross-Validation

```
Fold 0:  [2011-2015 train] → [2016 validate]
Fold 1:  [2011-2016 train] → [2017 validate]
Fold 2:  [2011-2017 train] → [2018 validate]
...
Fold 10: [2011-2025 train] → [2026 validate]
```

Minimum 5 years of training data before the first fold. This prevents look-ahead bias — the model never sees future data during training.

## Training Phases

### Phase 2a: p_revert Classifier

**Goal**: Predict probability that VIX drops >= 15% within the horizon window.

**Model**: XGBoost binary classifier with:
- `max_depth=5`, `n_estimators=500`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.8`
- `scale_pos_weight` auto-tuned to class imbalance
- Early stopping on validation AUC (50 rounds patience)

**Gate**: Must beat the rules-based baseline (`training/baseline.py`) in a majority of folds. If it fails, training stops — the model isn't good enough.

**Current results (v001)**: Mean AUC 0.706, Mean F1 0.821, beats baseline 9/10 folds. **PASS**.

### Phase 2b: p_spike_first Classifier

**Goal**: Predict probability of >= 10% VIX spike before the reversion occurs. This is a risk filter — high p_spike_first means "don't enter yet, it might get worse."

**Model**: Same XGBoost architecture as 2a.

**Gate**: AUC must exceed 0.60. If not, the system falls back to a rules-based proxy:
- Spike risk = 1 if VVIX > 120 AND |term_slope| < 0.01 (flat term structure)
- Otherwise spike risk = 0

**Current results (v001)**: Mean AUC 0.519. **FAIL** — using rules-based fallback. This is expected and anticipated in the project spec. VIX spike timing is inherently harder to predict than reversion.

### Phase 2c: Magnitude Regressor

**Goal**: Predict expected % drop magnitude.

**Model**: XGBoost regressor.

**Gate**: Only runs if both 2a and 2b pass their gates.

**Current results (v001)**: Skipped (2b used fallback).

### Phase 2d: Hierarchical CNN+GRU (Optional)

**Goal**: Multi-resolution temporal model using 5-min, hourly, and daily data streams.

**Status**: Architecture defined in `training/model_hcg.py`, but not yet trained. Requires sufficient intraday data from IBKR. Only deployed if it beats XGBoost by > 2% AUC consistently across folds.

## Running Training

```bash
# Full training pipeline
uv run python -m training.train \
    --dataset data/processed/vix_dataset.parquet \
    --output models \
    --min-train-years 5

# Output files:
# models/vix_xgb_revert_v001_YYYYMMDD.json   - p_revert model
# models/scaler_v001_YYYYMMDD.pkl             - fitted scaler
# models/scaler_v001_YYYYMMDD.json            - scaler metadata
# models/model_manifest.json                  - version + feature list
# models/training_results_YYYYMMDD_HHMMSS.json - full fold-by-fold results
```

## Rules-Based Baseline

The baseline (`training/baseline.py`) uses hand-crafted rules from volatility trading domain knowledge:

**Signal conditions** (all must be true):
1. `vix_zscore > 1.5` (elevated volatility)
2. `term_slope < -0.02` (backwardation) OR `vix_zscore > 2.0` (very elevated)
3. `spy_velocity < -0.02` (equity selloff confirming VIX spike)

**Spike risk proxy**:
- `vvix > 120` AND `|term_slope| < 0.01` (flat term structure = uncertainty)

Every trained model must beat this baseline. If it doesn't, the baseline is used instead.

## Feature Leakage Prevention

Several measures prevent data leakage:

1. **rv_iv_spread**: Uses T-1 (previous day) realized volatility data, never same-day
2. **Scaler**: Fit only on training fold data, applied to validation fold via `transform()`
3. **Meta columns excluded**: `date`, `vix_close`, `vix_level_tier`, `eligible`, `horizon` are never used as features
4. **Label columns excluded**: `label_revert`, `label_spike_first`, `label_magnitude`
5. **Walk-forward only**: No random splitting, no future data in any training fold

## Top Predictive Features

Consistently across folds, the most important features are:

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `vix_percentile` | Where current VIX sits in 1-year distribution |
| 2 | `vix_spot_lag20` | VIX level 20 trading days ago |
| 3 | `vvix` | Volatility of VIX — higher = more uncertain |
| 4 | `vvix_lag10` | VVIX 10 days ago |
| 5 | `term_slope_zscore` | How unusual the term structure is |

The model learns that VIX percentile rank and recent VVIX levels are the strongest predictors of mean reversion — which aligns with volatility trading intuition.

## Hyperparameter Tuning

Current hyperparameters (`training/model_xgb.py`):

```python
@dataclass
class XGBHyperparams:
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
```

These are conservative defaults. For systematic tuning, you could add Optuna or similar, but with ~500 samples, there's high risk of overfitting hyperparameters to the validation set.
