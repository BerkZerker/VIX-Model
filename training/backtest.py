"""Full P&L backtester for VIX mean-reversion strategy.

Simulates the complete signal-to-trade pipeline:
  1. Signal generation from model predictions
  2. Entry simulation (T+1 open, with sensitivity analysis)
  3. Instrument selection (tiered by VIX level)
  4. P&L via Black-Scholes modeled options pricing
  5. Transaction costs (commissions, spread, slippage)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ─── Black-Scholes Pricing ──────────────────────────────────────────────────

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price.

    Parameters
    ----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to expiry in years
    r : float - Risk-free rate
    sigma : float - Implied volatility
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ─── Trade Types ─────────────────────────────────────────────────────────────

class Instrument(Enum):
    UVXY_PUT = "UVXY put"
    VIX_PUT = "VIX put"
    VIX_CALL_SPREAD = "VIX call spread"


class Tier(Enum):
    MAJOR_SPIKE = "major_spike"    # VIX 30+
    MODERATE = "moderate"           # VIX 22-30
    MILD = "mild"                   # VIX 18-22


@dataclass
class TradeSetup:
    """A suggested trade from a model signal."""
    date: pd.Timestamp
    tier: Tier
    instrument: Instrument
    vix_level: float
    strike: float
    dte: int
    entry_price: float  # Option premium at entry
    sizing: str  # "full", "half", "small"
    p_revert: float
    p_spike_first: float
    expected_magnitude: float


@dataclass
class TradeResult:
    """Result of a simulated trade."""
    setup: TradeSetup
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str  # "target", "stop", "expiry"
    pnl_gross: float  # Before costs
    pnl_net: float    # After costs
    commission: float
    spread_cost: float
    slippage: float
    holding_days: int
    max_adverse: float  # Worst mark-to-market during trade


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Entry timing
    entry_delay: str = "t1_open"  # "t0_close", "t1_open", "t1_close"

    # Alert thresholds
    p_revert_threshold: float = 0.7
    p_spike_threshold: float = 0.3
    zscore_threshold: float = 1.0

    # Exit rules
    target_profit_pct: float = 0.50   # +50% on put value
    stop_loss_pct: float = -0.60      # -60% on put value

    # Transaction costs
    commission_per_contract: float = 0.65
    spread_pct_liquid: float = 0.05    # 5% for liquid VIX options
    spread_pct_uvxy: float = 0.08      # 8% for UVXY options
    spread_pct_illiquid: float = 0.15  # 15% for illiquid strikes
    slippage_pct: float = 0.03         # 3% adverse move on entry

    # BS model params
    risk_free_rate: float = 0.05
    vol_of_vol: float = 1.0  # VIX options implied vol (very high)

    # Position
    contracts_per_trade: int = 10


class VIXBacktester:
    """Full P&L backtester for VIX mean-reversion strategy.

    Parameters
    ----------
    config : BacktestConfig
        Backtesting configuration.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        vix_daily: pd.DataFrame | None = None,
    ) -> dict:
        """Run full backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataset with index as dates.
        predictions : pd.DataFrame
            Must have columns: p_revert, p_spike_first, expected_magnitude.
            Index aligned with df.
        vix_daily : pd.DataFrame, optional
            Full VIX daily OHLCV for mark-to-market. If None, uses df.

        Returns
        -------
        dict with trades, metrics, equity curve.
        """
        signals = self._generate_signals(df, predictions)
        logger.info(f"Generated {len(signals)} signals from {len(predictions)} predictions")

        trades = []
        for signal in signals:
            setup = self._create_trade_setup(signal, df)
            if setup is None:
                continue

            if vix_daily is not None:
                result = self._simulate_trade(setup, vix_daily)
            else:
                result = self._simulate_trade(setup, df)

            if result is not None:
                trades.append(result)

        logger.info(f"Simulated {len(trades)} trades")

        metrics = self._compute_metrics(trades)
        equity_curve = self._build_equity_curve(trades)

        return {
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "config": self.config,
        }

    def _generate_signals(
        self, df: pd.DataFrame, predictions: pd.DataFrame,
    ) -> list[dict]:
        """Identify rows that pass alert thresholds."""
        signals = []

        for date in predictions.index:
            if date not in df.index:
                continue

            row = predictions.loc[date]
            feat = df.loc[date]

            p_revert = row.get("p_revert", 0)
            p_spike = row.get("p_spike_first", 1)
            zscore = feat.get("vix_zscore", 0)

            if (p_revert > self.config.p_revert_threshold
                and p_spike < self.config.p_spike_threshold
                and zscore > self.config.zscore_threshold):
                signals.append({
                    "date": date,
                    "p_revert": p_revert,
                    "p_spike_first": p_spike,
                    "expected_magnitude": row.get("expected_magnitude", 0),
                    "vix_zscore": zscore,
                    "vix_level": feat.get("vix_spot", feat.get("vix_close", 20)),
                })

        return signals

    def _create_trade_setup(self, signal: dict, df: pd.DataFrame) -> TradeSetup | None:
        """Create a trade setup from a signal."""
        vix = signal["vix_level"]

        # Determine tier
        if vix >= 30:
            tier = Tier.MAJOR_SPIKE
            instrument = Instrument.UVXY_PUT
            dte = 75  # 60-90 DTE midpoint
            strike_pct = 0.65  # UVXY put at 65% of current
            sizing = "full" if signal["p_revert"] > 0.8 and signal["p_spike_first"] < 0.2 else "half"
        elif vix >= 22:
            tier = Tier.MODERATE
            instrument = Instrument.VIX_PUT
            dte = 45  # 30-60 DTE midpoint
            strike_pct = 0.92  # Near ATM relative to futures
            sizing = "half" if signal["p_revert"] > 0.7 and signal["p_spike_first"] < 0.3 else "small"
        elif vix >= 18:
            tier = Tier.MILD
            instrument = Instrument.VIX_CALL_SPREAD
            dte = 30
            strike_pct = 1.0
            sizing = "small"
        else:
            return None  # VIX too low

        # Compute option price via Black-Scholes
        T = dte / 252  # Trading days to years
        if instrument == Instrument.UVXY_PUT:
            # UVXY roughly tracks 1.5x VIX short-term futures
            uvxy_proxy = vix * 1.2  # Rough approximation
            strike = uvxy_proxy * strike_pct
            entry_price = black_scholes_put(uvxy_proxy, strike, T, self.config.risk_free_rate, self.config.vol_of_vol)
        elif instrument == Instrument.VIX_PUT:
            strike = vix * strike_pct
            entry_price = black_scholes_put(vix, strike, T, self.config.risk_free_rate, self.config.vol_of_vol)
        else:  # VIX_CALL_SPREAD
            short_strike = vix
            long_strike = vix + 7
            short_call = black_scholes_call(vix, short_strike, T, self.config.risk_free_rate, self.config.vol_of_vol)
            long_call = black_scholes_call(vix, long_strike, T, self.config.risk_free_rate, self.config.vol_of_vol)
            entry_price = short_call - long_call  # Credit received

        if entry_price <= 0.01:
            return None

        return TradeSetup(
            date=signal["date"],
            tier=tier,
            instrument=instrument,
            vix_level=vix,
            strike=strike if instrument != Instrument.VIX_CALL_SPREAD else short_strike,
            dte=dte,
            entry_price=entry_price,
            sizing=sizing,
            p_revert=signal["p_revert"],
            p_spike_first=signal["p_spike_first"],
            expected_magnitude=signal["expected_magnitude"],
        )

    def _simulate_trade(self, setup: TradeSetup, price_data: pd.DataFrame) -> TradeResult | None:
        """Simulate a single trade with daily mark-to-market."""
        entry_date = setup.date

        # Get forward price data
        future_mask = price_data.index > entry_date
        future_data = price_data[future_mask].head(setup.dte)

        if len(future_data) < 5:
            return None  # Not enough forward data

        # Transaction costs at entry
        spread_pct = (
            self.config.spread_pct_uvxy
            if setup.instrument == Instrument.UVXY_PUT
            else self.config.spread_pct_liquid
        )
        spread_cost = setup.entry_price * spread_pct * self.config.contracts_per_trade * 100
        slippage = setup.entry_price * self.config.slippage_pct * self.config.contracts_per_trade * 100
        commission = self.config.commission_per_contract * self.config.contracts_per_trade * 2  # Entry + exit

        # Mark-to-market through the trade
        max_adverse = 0
        exit_date = None
        exit_price = 0
        exit_reason = "expiry"

        vix_col = "vix_close" if "vix_close" in future_data.columns else "vix_spot"
        if vix_col not in future_data.columns:
            # Try to use the first numeric column as VIX proxy
            numeric_cols = future_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            vix_col = numeric_cols[0]

        for i, (date, row) in enumerate(future_data.iterrows()):
            remaining_dte = setup.dte - i - 1
            T_remaining = max(remaining_dte / 252, 0.001)

            current_vix = row[vix_col] if not pd.isna(row.get(vix_col)) else setup.vix_level

            # Price the option at current market
            if setup.instrument == Instrument.UVXY_PUT:
                uvxy_proxy = current_vix * 1.2
                current_price = black_scholes_put(
                    uvxy_proxy, setup.strike, T_remaining,
                    self.config.risk_free_rate, self.config.vol_of_vol,
                )
            elif setup.instrument == Instrument.VIX_PUT:
                current_price = black_scholes_put(
                    current_vix, setup.strike, T_remaining,
                    self.config.risk_free_rate, self.config.vol_of_vol,
                )
            else:  # Call spread
                short_call = black_scholes_call(
                    current_vix, setup.strike, T_remaining,
                    self.config.risk_free_rate, self.config.vol_of_vol,
                )
                long_call = black_scholes_call(
                    current_vix, setup.strike + 7, T_remaining,
                    self.config.risk_free_rate, self.config.vol_of_vol,
                )
                current_price = short_call - long_call

            pnl_pct = (current_price - setup.entry_price) / setup.entry_price
            max_adverse = min(max_adverse, pnl_pct)

            # Check exit conditions
            if pnl_pct >= self.config.target_profit_pct:
                exit_date = date
                exit_price = current_price
                exit_reason = "target"
                break
            elif pnl_pct <= self.config.stop_loss_pct:
                exit_date = date
                exit_price = current_price
                exit_reason = "stop"
                break

        if exit_date is None:
            exit_date = future_data.index[-1]
            # Price at expiry
            final_vix = future_data[vix_col].iloc[-1]
            if setup.instrument == Instrument.UVXY_PUT:
                exit_price = max(setup.strike - final_vix * 1.2, 0)
            elif setup.instrument == Instrument.VIX_PUT:
                exit_price = max(setup.strike - final_vix, 0)
            else:
                exit_price = max(setup.strike - final_vix, 0) - max(setup.strike + 7 - final_vix, 0)

        # Compute P&L
        pnl_per_contract = (exit_price - setup.entry_price) * 100  # Options are 100 multiplier
        pnl_gross = pnl_per_contract * self.config.contracts_per_trade
        total_costs = commission + spread_cost + slippage
        pnl_net = pnl_gross - total_costs

        holding_days = (exit_date - entry_date).days

        return TradeResult(
            setup=setup,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            holding_days=holding_days,
            max_adverse=max_adverse,
        )

    def _compute_metrics(self, trades: list[TradeResult]) -> dict:
        """Compute backtest metrics."""
        if not trades:
            return {"error": "No trades"}

        pnls = [t.pnl_net for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics = {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_pnl": np.mean(pnls),
            "total_pnl": np.sum(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "win_loss_ratio": abs(np.mean(wins) / np.mean(losses)) if wins and losses else float("inf"),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "max_drawdown": self._max_drawdown(pnls),
            "avg_holding_days": np.mean([t.holding_days for t in trades]),
            "total_costs": sum(t.commission + t.spread_cost + t.slippage for t in trades),
        }

        # Sharpe ratio (annualized)
        if len(pnls) > 1 and np.std(pnls) > 0:
            daily_returns = np.array(pnls) / (sum(abs(p) for p in pnls) / len(pnls))  # Normalized
            metrics["sharpe"] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252 / max(np.mean([t.holding_days for t in trades]), 1))
        else:
            metrics["sharpe"] = 0

        # Exit reason breakdown
        metrics["exit_reasons"] = {
            "target": sum(1 for t in trades if t.exit_reason == "target"),
            "stop": sum(1 for t in trades if t.exit_reason == "stop"),
            "expiry": sum(1 for t in trades if t.exit_reason == "expiry"),
        }

        # Per-tier metrics
        for tier in Tier:
            tier_trades = [t for t in trades if t.setup.tier == tier]
            if tier_trades:
                tier_pnls = [t.pnl_net for t in tier_trades]
                tier_wins = [p for p in tier_pnls if p > 0]
                metrics[f"{tier.value}_trades"] = len(tier_trades)
                metrics[f"{tier.value}_win_rate"] = len(tier_wins) / len(tier_trades)
                metrics[f"{tier.value}_avg_pnl"] = np.mean(tier_pnls)

        # Signals per year
        if trades:
            date_range = (trades[-1].setup.date - trades[0].setup.date).days
            years = max(date_range / 365.25, 0.5)
            metrics["signals_per_year"] = len(trades) / years

        return metrics

    def _max_drawdown(self, pnls: list[float]) -> float:
        """Compute maximum drawdown from a series of P&Ls."""
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    def _build_equity_curve(self, trades: list[TradeResult]) -> pd.DataFrame:
        """Build equity curve from trade results."""
        if not trades:
            return pd.DataFrame(columns=["date", "cumulative_pnl", "trade_count"])

        records = []
        cum_pnl = 0
        for i, trade in enumerate(trades):
            cum_pnl += trade.pnl_net
            records.append({
                "date": trade.exit_date,
                "cumulative_pnl": cum_pnl,
                "trade_count": i + 1,
                "tier": trade.setup.tier.value,
            })

        return pd.DataFrame(records)

    def run_sensitivity(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        vix_daily: pd.DataFrame | None = None,
    ) -> dict:
        """Run sensitivity analyses.

        Tests:
          - Execution delay: t0_close, t1_open, t1_close
          - Threshold tuning: p_revert 0.6-0.9, p_spike 0.1-0.4
          - Cost sensitivity: 2x transaction costs
        """
        results = {}

        # 1. Execution delay sensitivity
        for delay in ["t0_close", "t1_open", "t1_close"]:
            cfg = BacktestConfig(entry_delay=delay)
            bt = VIXBacktester(cfg)
            result = bt.run(df, predictions, vix_daily)
            results[f"delay_{delay}"] = result["metrics"]

        # 2. Threshold tuning
        threshold_results = []
        for p_revert in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            for p_spike in [0.1, 0.2, 0.3, 0.4]:
                cfg = BacktestConfig(
                    p_revert_threshold=p_revert,
                    p_spike_threshold=p_spike,
                )
                bt = VIXBacktester(cfg)
                result = bt.run(df, predictions, vix_daily)
                threshold_results.append({
                    "p_revert_threshold": p_revert,
                    "p_spike_threshold": p_spike,
                    **result["metrics"],
                })
        results["threshold_sweep"] = threshold_results

        # 3. Cost sensitivity (2x)
        cfg = BacktestConfig(
            spread_pct_liquid=0.10,
            spread_pct_uvxy=0.16,
            spread_pct_illiquid=0.30,
            slippage_pct=0.06,
        )
        bt = VIXBacktester(cfg)
        result = bt.run(df, predictions, vix_daily)
        results["double_costs"] = result["metrics"]

        return results


def run_walkforward_backtest(
    df: pd.DataFrame,
    fold_predictions: list[tuple[pd.Index, pd.DataFrame]],
    config: BacktestConfig | None = None,
) -> dict:
    """Run backtest across all walk-forward validation folds.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset.
    fold_predictions : list of (val_index, predictions_df) tuples
        Model predictions for each validation fold.
    config : BacktestConfig

    Returns
    -------
    Aggregate backtest results across all folds.
    """
    bt = VIXBacktester(config)
    all_trades = []

    for fold_idx, (val_idx, preds) in enumerate(fold_predictions):
        val_df = df.loc[val_idx]
        result = bt.run(val_df, preds)
        all_trades.extend(result["trades"])
        logger.info(f"Fold {fold_idx}: {len(result['trades'])} trades, metrics: {result['metrics']}")

    aggregate_metrics = bt._compute_metrics(all_trades)
    equity_curve = bt._build_equity_curve(all_trades)

    logger.info(f"Aggregate: {len(all_trades)} trades across all folds")
    logger.info(f"Win rate: {aggregate_metrics.get('win_rate', 0):.1%}")
    logger.info(f"Profit factor: {aggregate_metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Sharpe: {aggregate_metrics.get('sharpe', 0):.2f}")

    return {
        "trades": all_trades,
        "metrics": aggregate_metrics,
        "equity_curve": equity_curve,
    }
