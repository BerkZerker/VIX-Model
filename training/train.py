"""Walk-forward training pipeline for VIX models.

Implements phased training:
  Phase 2a: p_revert classifier
  Phase 2b: p_spike_first classifier
  Phase 2c: expected_magnitude regressor
  Phase 2d: Hierarchical CNN+GRU (if XGBoost leaves room)

All evaluation is walk-forward: train on years 1-N, validate on year N+1.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.model_xgb import (
    VIXEnsemble, VIXRevertClassifier, VIXSpikeClassifier,
    VIXMagnitudeRegressor, XGBHyperparams, TrainResult,
)
from training.baseline import RulesBaseline
from training.scaler import fit_scaler, transform, save_scaler

logger = logging.getLogger(__name__)

# Features that should never be used as model inputs
LABEL_COLUMNS = ["label_revert", "label_spike_first", "label_magnitude"]
META_COLUMNS = ["date", "vix_close", "vix_level_tier", "eligible", "horizon"]


def make_walkforward_folds(
    df: pd.DataFrame,
    min_train_years: int = 5,
    date_col: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward folds: train on years 1-N, validate on year N+1.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with DatetimeIndex or date column.
    min_train_years : int
        Minimum years of training data before first validation fold.
    date_col : str, optional
        Column name containing dates. If None, uses index.

    Returns
    -------
    List of (train_indices, val_indices) tuples using integer positional indices.
    """
    if date_col:
        dates = pd.to_datetime(df[date_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        raise ValueError("Need DatetimeIndex or date_col")

    years = sorted(dates.year.unique())

    if len(years) < min_train_years + 1:
        logger.warning(
            f"Only {len(years)} years available, need {min_train_years + 1} for walk-forward. "
            f"Reducing min_train_years to {len(years) - 1}"
        )
        min_train_years = max(2, len(years) - 1)

    folds = []
    for i in range(min_train_years, len(years)):
        train_years = set(years[:i])
        val_year = years[i]

        train_mask = dates.year.isin(train_years)
        val_mask = dates.year == val_year

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
            logger.info(
                f"Fold {len(folds)}: train years {min(train_years)}-{max(train_years)} "
                f"({len(train_idx)} samples), val year {val_year} ({len(val_idx)} samples)"
            )

    return folds


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns by excluding labels and meta columns."""
    exclude = set(LABEL_COLUMNS + META_COLUMNS)
    return [c for c in df.columns if c not in exclude and not c.startswith("label_")]


def train_phase_2a(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: XGBHyperparams | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Phase 2a: Train p_revert classifier with walk-forward CV.

    Returns
    -------
    dict with fold_results, aggregate_metrics, models, and comparison vs baseline.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2a: Training p_revert classifier")
    logger.info("=" * 60)

    feature_cols = get_feature_columns(df)
    label_col = "label_revert"

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    # Filter to eligible rows (where labels exist)
    eligible = df[df[label_col].notna()].copy()
    logger.info(f"Eligible samples: {len(eligible)} out of {len(df)} total")

    baseline = RulesBaseline()
    fold_results = []
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Map to eligible rows
        train_mask = eligible.index.isin(df.index[train_idx])
        val_mask = eligible.index.isin(df.index[val_idx])

        train_data = eligible[train_mask]
        val_data = eligible[val_mask]

        if len(train_data) < 20 or len(val_data) < 5:
            logger.warning(f"Fold {fold_idx}: insufficient data (train={len(train_data)}, val={len(val_data)}), skipping")
            continue

        # Scale features (fit only on training data)
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(train_data[feature_cols]),
            columns=feature_cols,
            index=train_data.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(val_data[feature_cols]),
            columns=feature_cols,
            index=val_data.index,
        )
        y_train = train_data[label_col].astype(int)
        y_val = val_data[label_col].astype(int)

        # Train XGBoost
        clf = VIXRevertClassifier(params)
        result = clf.train(X_train, y_train, X_val, y_val)

        # Baseline comparison
        baseline_signals = baseline.generate_signals(val_data)
        baseline_preds = baseline_signals["signal"].values

        from sklearn.metrics import precision_score, recall_score, f1_score
        baseline_metrics = {
            "precision": float(precision_score(y_val, baseline_preds, zero_division=0)),
            "recall": float(recall_score(y_val, baseline_preds, zero_division=0)),
            "f1": float(f1_score(y_val, baseline_preds, zero_division=0)),
        }

        fold_result = {
            "fold": fold_idx,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "val_positive_rate": float(y_val.mean()),
            "model_metrics": result.metrics,
            "baseline_metrics": baseline_metrics,
            "beats_baseline_auc": result.metrics.get("auc", 0) > baseline_metrics.get("auc", 0),
            "beats_baseline_f1": result.metrics["f1"] > baseline_metrics["f1"],
            "top_features": result.feature_importance.head(10).to_dict(),
        }
        fold_results.append(fold_result)
        models.append(clf)

        logger.info(
            f"Fold {fold_idx}: AUC={result.metrics.get('auc', 'N/A'):.4f}, "
            f"F1={result.metrics['f1']:.4f} (baseline F1={baseline_metrics['f1']:.4f})"
        )

    # Aggregate metrics
    if fold_results:
        aggregate = {
            "mean_auc": float(np.nanmean([r["model_metrics"].get("auc", np.nan) for r in fold_results])),
            "mean_f1": float(np.mean([r["model_metrics"]["f1"] for r in fold_results])),
            "mean_precision": float(np.mean([r["model_metrics"]["precision"] for r in fold_results])),
            "mean_recall": float(np.mean([r["model_metrics"]["recall"] for r in fold_results])),
            "baseline_mean_f1": float(np.mean([r["baseline_metrics"]["f1"] for r in fold_results])),
            "folds_beating_baseline": sum(r["beats_baseline_f1"] for r in fold_results),
            "total_folds": len(fold_results),
        }
    else:
        aggregate = {"error": "No valid folds"}

    logger.info(f"Phase 2a aggregate: {json.dumps(aggregate, indent=2)}")

    return {
        "phase": "2a_p_revert",
        "fold_results": fold_results,
        "aggregate": aggregate,
        "models": models,
    }


def train_phase_2b(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: XGBHyperparams | None = None,
) -> dict:
    """Phase 2b: Train p_spike_first classifier.

    The hardest and highest-value prediction. If walk-forward AUC < 0.60
    across folds, fall back to rules-based spike risk proxy.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2b: Training p_spike_first classifier")
    logger.info("=" * 60)

    feature_cols = get_feature_columns(df)
    label_col = "label_spike_first"

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    eligible = df[df[label_col].notna()].copy()
    logger.info(f"Eligible samples: {len(eligible)}")

    fold_results = []
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_mask = eligible.index.isin(df.index[train_idx])
        val_mask = eligible.index.isin(df.index[val_idx])

        train_data = eligible[train_mask]
        val_data = eligible[val_mask]

        if len(train_data) < 20 or len(val_data) < 5:
            continue

        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(train_data[feature_cols]),
            columns=feature_cols,
            index=train_data.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(val_data[feature_cols]),
            columns=feature_cols,
            index=val_data.index,
        )
        y_train = train_data[label_col].astype(int)
        y_val = val_data[label_col].astype(int)

        clf = VIXSpikeClassifier(params)
        result = clf.train(X_train, y_train, X_val, y_val)

        fold_result = {
            "fold": fold_idx,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "model_metrics": result.metrics,
            "top_features": result.feature_importance.head(10).to_dict(),
        }
        fold_results.append(fold_result)
        models.append(clf)

        logger.info(f"Fold {fold_idx}: AUC={result.metrics.get('auc', 'N/A'):.4f}")

    aggregate = {}
    if fold_results:
        mean_auc = float(np.nanmean([r["model_metrics"].get("auc", np.nan) for r in fold_results]))
        aggregate = {
            "mean_auc": mean_auc,
            "mean_f1": float(np.mean([r["model_metrics"]["f1"] for r in fold_results])),
            "sufficient_auc": mean_auc >= 0.60,
        }
        if not aggregate["sufficient_auc"]:
            logger.warning(
                f"p_spike_first mean AUC = {mean_auc:.4f} < 0.60. "
                f"Falling back to rules-based spike risk proxy."
            )

    return {
        "phase": "2b_p_spike_first",
        "fold_results": fold_results,
        "aggregate": aggregate,
        "models": models,
        "use_rules_fallback": not aggregate.get("sufficient_auc", False),
    }


def train_phase_2c(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: XGBHyperparams | None = None,
) -> dict:
    """Phase 2c: Train expected_magnitude regressor.

    Only run if Phase 2a and 2b show solid performance.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2c: Training expected_magnitude regressor")
    logger.info("=" * 60)

    feature_cols = get_feature_columns(df)
    label_col = "label_magnitude"

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    eligible = df[df[label_col].notna()].copy()
    logger.info(f"Eligible samples: {len(eligible)}")

    fold_results = []
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_mask = eligible.index.isin(df.index[train_idx])
        val_mask = eligible.index.isin(df.index[val_idx])

        train_data = eligible[train_mask]
        val_data = eligible[val_mask]

        if len(train_data) < 20 or len(val_data) < 5:
            continue

        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(train_data[feature_cols]),
            columns=feature_cols,
            index=train_data.index,
        )
        X_val = pd.DataFrame(
            scaler.transform(val_data[feature_cols]),
            columns=feature_cols,
            index=val_data.index,
        )
        y_train = train_data[label_col]
        y_val = val_data[label_col]

        reg = VIXMagnitudeRegressor(params)
        result = reg.train(X_train, y_train, X_val, y_val)

        fold_result = {
            "fold": fold_idx,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "model_metrics": result.metrics,
            "top_features": result.feature_importance.head(10).to_dict(),
        }
        fold_results.append(fold_result)
        models.append(reg)

        logger.info(
            f"Fold {fold_idx}: RMSE={result.metrics['rmse']:.4f}, "
            f"corr={result.metrics['correlation']:.4f}"
        )

    aggregate = {}
    if fold_results:
        aggregate = {
            "mean_rmse": float(np.mean([r["model_metrics"]["rmse"] for r in fold_results])),
            "mean_mae": float(np.mean([r["model_metrics"]["mae"] for r in fold_results])),
            "mean_correlation": float(np.nanmean([r["model_metrics"]["correlation"] for r in fold_results])),
        }

    return {
        "phase": "2c_magnitude",
        "fold_results": fold_results,
        "aggregate": aggregate,
        "models": models,
    }


def run_full_training(
    dataset_path: str | Path,
    output_dir: str | Path = "models",
    min_train_years: int = 5,
    params: XGBHyperparams | None = None,
) -> dict:
    """Run the full phased training pipeline.

    Parameters
    ----------
    dataset_path : path to processed dataset (parquet or csv)
    output_dir : directory to save models
    min_train_years : minimum training years before first fold
    params : XGBoost hyperparameters

    Returns
    -------
    dict with results from all phases
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_path = Path(dataset_path)
    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path, parse_dates=["date"])

    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("date")

    logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    # Create walk-forward folds
    folds = make_walkforward_folds(df, min_train_years=min_train_years)
    logger.info(f"Created {len(folds)} walk-forward folds")

    results = {"timestamp": datetime.now().isoformat(), "n_folds": len(folds)}

    # Phase 2a: p_revert
    results_2a = train_phase_2a(df, folds, params, output_dir)
    results["phase_2a"] = {k: v for k, v in results_2a.items() if k != "models"}

    # Gate: Phase 2a must beat baseline
    if results_2a["aggregate"].get("folds_beating_baseline", 0) < len(folds) // 2:
        logger.warning("Phase 2a: Model does NOT beat baseline in majority of folds. Iterating on features.")
        results["gate_2a"] = "FAIL"
    else:
        results["gate_2a"] = "PASS"

    # Phase 2b: p_spike_first
    results_2b = train_phase_2b(df, folds, params)
    results["phase_2b"] = {k: v for k, v in results_2b.items() if k != "models"}

    if results_2b.get("use_rules_fallback"):
        results["spike_method"] = "rules_fallback"
        logger.info("Using rules-based spike risk proxy (model AUC < 0.60)")
    else:
        results["spike_method"] = "model"

    # Phase 2c: magnitude (only if 2a and 2b look good)
    if results["gate_2a"] == "PASS" and not results_2b.get("use_rules_fallback", True):
        results_2c = train_phase_2c(df, folds, params)
        results["phase_2c"] = {k: v for k, v in results_2c.items() if k != "models"}
    else:
        logger.info("Skipping Phase 2c (magnitude) - prerequisites not met")
        results["phase_2c"] = {"skipped": True, "reason": "Phase 2a/2b prerequisites not met"}

    # Save results
    results_path = output_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training results saved to {results_path}")

    # Train final models on full data (up to last fold's validation year) for export
    if results_2a.get("models"):
        _export_best_models(results_2a, results_2b, results.get("phase_2c_models"), df, folds, output_dir)

    return results


def _export_best_models(
    results_2a: dict,
    results_2b: dict,
    results_2c: dict | None,
    df: pd.DataFrame,
    folds: list,
    output_dir: Path,
):
    """Export the best models from the last fold for deployment."""
    from training.export_onnx import export_xgb_to_onnx

    # Use models from the last fold (most data)
    if results_2a["models"]:
        model_2a = results_2a["models"][-1]
        feature_cols = get_feature_columns(df)

        export_path = output_dir / f"vix_xgb_revert_v001_{datetime.now().strftime('%Y%m%d')}.onnx"
        export_xgb_to_onnx(model_2a.model, feature_cols, export_path, "revert")
        logger.info(f"Exported p_revert model to {export_path}")

    if results_2b.get("models") and not results_2b.get("use_rules_fallback"):
        model_2b = results_2b["models"][-1]
        export_path = output_dir / f"vix_xgb_spike_v001_{datetime.now().strftime('%Y%m%d')}.onnx"
        export_xgb_to_onnx(model_2b.model, feature_cols, export_path, "spike")
        logger.info(f"Exported p_spike_first model to {export_path}")

    # Save scaler from last fold
    last_train_idx = folds[-1][0]
    eligible = df[df["label_revert"].notna()]
    train_data = eligible[eligible.index.isin(df.index[last_train_idx])]
    feature_cols = get_feature_columns(df)

    scaler = fit_scaler(train_data[feature_cols])
    scaler_path = output_dir / f"scaler_v001_{datetime.now().strftime('%Y%m%d')}.pkl"
    save_scaler(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Save model manifest
    manifest = {
        "version": "v001",
        "training_date": datetime.now().isoformat(),
        "model_type": "xgboost",
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "data_range": f"{df.index.min()} to {df.index.max()}",
    }
    manifest_path = output_dir / "model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Train VIX models with walk-forward CV")
    parser.add_argument("--dataset", default="data/processed/vix_dataset.parquet")
    parser.add_argument("--output", default="models")
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()

    results = run_full_training(args.dataset, args.output, args.min_train_years)
    print(f"\nTraining complete. Gate 2a: {results['gate_2a']}, Spike method: {results['spike_method']}")
