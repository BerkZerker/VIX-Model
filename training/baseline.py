"""Rules-based baseline for VIX mean-reversion signals.

This baseline must be beaten by any trained model in walk-forward testing.
Defined upfront to prevent goalpost-shifting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BaselineSignal:
    """A baseline signal with its components."""

    date: pd.Timestamp
    fires: bool
    spike_risk: bool
    vix_zscore: float
    term_slope: float
    spy_velocity_5d: float
    vvix: float


class RulesBaseline:
    """Rules-based baseline for VIX mean-reversion signals.

    Signal fires when ALL of:
      - vix_zscore > 1.5
      - term_slope < -0.02 (backwardation) OR vix_zscore > 2.0
      - spy_velocity_5d < -2% (equity selloff context)

    Spike risk proxy:
      - VVIX > 120 AND term structure flattening (|term_slope| < 0.01)
    """

    def __init__(
        self,
        zscore_threshold: float = 1.5,
        zscore_strong: float = 2.0,
        backwardation_threshold: float = -0.02,
        spy_velocity_threshold: float = -0.02,
        vvix_spike_threshold: float = 120.0,
        term_flat_threshold: float = 0.01,
    ):
        self.zscore_threshold = zscore_threshold
        self.zscore_strong = zscore_strong
        self.backwardation_threshold = backwardation_threshold
        self.spy_velocity_threshold = spy_velocity_threshold
        self.vvix_spike_threshold = vvix_spike_threshold
        self.term_flat_threshold = term_flat_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate baseline signals for every row in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: vix_zscore, term_slope, spy_velocity, vvix.

        Returns
        -------
        pd.DataFrame with columns: signal, spike_risk, p_revert_proxy, p_spike_proxy
        """
        vix_zscore = df["vix_zscore"]
        term_slope = (
            df["term_slope"] if "term_slope" in df.columns else pd.Series(0, index=df.index)
        )
        spy_velocity = (
            df["spy_velocity"]
            if "spy_velocity" in df.columns
            else df.get("spy_velocity_5d", pd.Series(0, index=df.index))
        )
        vvix = df["vvix"] if "vvix" in df.columns else pd.Series(100, index=df.index)

        # Core signal conditions
        zscore_elevated = vix_zscore > self.zscore_threshold
        backwardation_or_strong = (term_slope < self.backwardation_threshold) | (
            vix_zscore > self.zscore_strong
        )
        equity_selloff = spy_velocity < self.spy_velocity_threshold

        signal = zscore_elevated & backwardation_or_strong & equity_selloff

        # Spike risk proxy
        vvix_high = vvix > self.vvix_spike_threshold
        term_flat = term_slope.abs() < self.term_flat_threshold
        spike_risk = vvix_high & term_flat

        # Convert to probability proxies for fair comparison with model outputs
        # p_revert_proxy: higher when signal fires, scaled by z-score strength
        p_revert_proxy = np.where(
            signal,
            np.clip(0.5 + 0.15 * (vix_zscore - self.zscore_threshold), 0.55, 0.95),
            np.clip(0.3 + 0.1 * vix_zscore, 0.0, 0.5),
        )

        # p_spike_proxy: higher when spike risk conditions met
        p_spike_proxy = np.where(
            spike_risk,
            np.clip(0.4 + 0.1 * (vvix - self.vvix_spike_threshold) / 20, 0.4, 0.8),
            np.clip(0.15 + 0.05 * np.maximum(0, vvix - 100) / 20, 0.1, 0.4),
        )

        result = pd.DataFrame(
            {
                "signal": signal.astype(int),
                "spike_risk": spike_risk.astype(int),
                "p_revert_proxy": p_revert_proxy,
                "p_spike_proxy": p_spike_proxy,
            },
            index=df.index,
        )

        return result

    def evaluate_walkforward(
        self,
        df: pd.DataFrame,
        label_col: str = "label_revert",
        folds: list[tuple[pd.Index, pd.Index]] | None = None,
    ) -> dict:
        """Evaluate baseline on walk-forward folds.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with features and labels.
        label_col : str
            Label column to evaluate against.
        folds : list of (train_idx, val_idx) tuples
            Walk-forward fold indices. If None, use year-based splits.

        Returns
        -------
        dict with per-fold and aggregate metrics.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        if folds is None:
            folds = self._make_year_folds(df)

        fold_metrics = []
        all_signals = []
        all_labels = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            val_df = df.loc[val_idx]
            if label_col not in val_df.columns:
                continue

            signals = self.generate_signals(val_df)
            y_true = val_df[label_col].values
            y_pred = signals["signal"].values
            p_revert = signals["p_revert_proxy"].values

            # Filter to eligible days only (where labels exist)
            mask = ~np.isnan(y_true)
            if mask.sum() == 0:
                continue

            y_true_m = y_true[mask].astype(int)
            y_pred_m = y_pred[mask]
            p_revert_m = p_revert[mask]

            metrics = {
                "fold": fold_idx,
                "n_samples": int(mask.sum()),
                "n_signals": int(y_pred_m.sum()),
                "precision": float(precision_score(y_true_m, y_pred_m, zero_division=0)),
                "recall": float(recall_score(y_true_m, y_pred_m, zero_division=0)),
                "f1": float(f1_score(y_true_m, y_pred_m, zero_division=0)),
            }

            if len(np.unique(y_true_m)) > 1:
                metrics["auc"] = float(roc_auc_score(y_true_m, p_revert_m))
            else:
                metrics["auc"] = float("nan")

            fold_metrics.append(metrics)
            all_signals.extend(y_pred_m.tolist())
            all_labels.extend(y_true_m.tolist())

        # Aggregate
        all_signals = np.array(all_signals)
        all_labels = np.array(all_labels)
        aggregate = {}
        if len(all_labels) > 0:
            aggregate = {
                "total_samples": len(all_labels),
                "total_signals": int(all_signals.sum()),
                "precision": float(precision_score(all_labels, all_signals, zero_division=0)),
                "recall": float(recall_score(all_labels, all_signals, zero_division=0)),
                "f1": float(f1_score(all_labels, all_signals, zero_division=0)),
            }
            if len(np.unique(all_labels)) > 1:
                aggregate["auc"] = float(
                    roc_auc_score(
                        all_labels,
                        np.array(
                            [
                                m["p_revert_proxy"]
                                for fold_df in [df.loc[val_idx] for _, val_idx in folds]
                                for m in [self.generate_signals(fold_df).iloc]
                            ]
                        )
                        if False
                        # Simpler: just use signal as proxy
                        else all_signals.astype(float),
                    )
                )

        return {
            "fold_metrics": fold_metrics,
            "aggregate": aggregate,
        }

    @staticmethod
    def _make_year_folds(df: pd.DataFrame) -> list[tuple[pd.Index, pd.Index]]:
        """Create year-based walk-forward folds."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'date' column")

        years = sorted(df.index.year.unique())
        if len(years) < 6:
            # Need at least 5 years training + 1 validation
            min_train = max(3, len(years) - 3)
        else:
            min_train = 5

        folds = []
        for i in range(min_train, len(years)):
            train_years = years[:i]
            val_year = years[i]
            train_idx = df.index[df.index.year.isin(train_years)]
            val_idx = df.index[df.index.year == val_year]
            if len(val_idx) > 0:
                folds.append((train_idx, val_idx))

        return folds
