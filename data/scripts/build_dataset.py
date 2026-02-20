"""Build the unified VIX dataset from raw CSV files.

This script:
1. Loads all raw CSVs from data/raw/
2. Merges and aligns by date (forward-fill then drop remaining NaN)
3. Computes all daily features via training/features.py
4. Generates labels using tiered horizon windows
5. Saves processed dataset to data/processed/vix_dataset.parquet
6. Prints summary statistics

Usage:
    uv run python data/scripts/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so we can import training modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.features import compute_all_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "vix_dataset.parquet"

# Expected raw CSV filenames (produced by Phase 1 fetch scripts)
RAW_FILES = {
    "vix": "vix_daily.csv",
    "spy": "spy_daily.csv",
    "vix_futures": "vix_futures.csv",
    "vvix": "vvix_daily.csv",
    "vix9d": "vix9d_daily.csv",
    "skew": "skew_daily.csv",
    "hyg": "hyg_daily.csv",
    "tlt": "tlt_daily.csv",
    "put_call": "put_call_ratio.csv",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _parse_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a proper DatetimeIndex named 'date'."""
    # Try common date column names
    for col in ["date", "Date", "DATE", "datetime", "Datetime"]:
        if col in df.columns:
            df = df.set_index(col)
            break

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index.name = "date"
    df = df.sort_index()
    # Drop duplicate dates, keeping the last entry
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_vix(path: Path) -> pd.DataFrame:
    """Load VIX daily data."""
    df = pd.read_csv(path)
    df = _parse_date_index(df)
    # Standardize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "close" in lower or "adj" in lower:
            col_map[col] = "vix_close"
        elif "high" in lower:
            col_map[col] = "vix_high"
        elif "low" in lower:
            col_map[col] = "vix_low"
        elif "open" in lower:
            col_map[col] = "vix_open"
    df = df.rename(columns=col_map)
    if "vix_close" not in df.columns:
        # Fallback: take the first numeric column
        numeric = df.select_dtypes(include="number").columns
        if len(numeric) > 0:
            df = df.rename(columns={numeric[0]: "vix_close"})
    return df[["vix_close"]].dropna()


def load_spy(path: Path) -> pd.DataFrame:
    """Load SPY daily data."""
    df = pd.read_csv(path)
    df = _parse_date_index(df)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "close" in lower or "adj" in lower:
            col_map[col] = "spy_close"
    df = df.rename(columns=col_map)
    if "spy_close" not in df.columns:
        numeric = df.select_dtypes(include="number").columns
        if len(numeric) > 0:
            df = df.rename(columns={numeric[0]: "spy_close"})
    return df[["spy_close"]].dropna()


def load_vix_futures_single(path: Path) -> pd.DataFrame:
    """Load a single VIX futures file that already has M1/M2/M3 columns."""
    df = pd.read_csv(path)
    df = _parse_date_index(df)

    result = pd.DataFrame(index=df.index)
    col_lower_map = {c: c.lower() for c in df.columns}

    for col, lower in col_lower_map.items():
        if any(k in lower for k in ["m1", "month1", "front", "f1"]) and "volume" not in lower:
            result["vix_futures_m1"] = pd.to_numeric(df[col], errors="coerce")
        elif any(k in lower for k in ["m2", "month2", "second", "f2"]):
            result["vix_futures_m2"] = pd.to_numeric(df[col], errors="coerce")
        elif any(k in lower for k in ["m3", "month3", "third", "f3"]):
            result["vix_futures_m3"] = pd.to_numeric(df[col], errors="coerce")
        elif "volume" in lower or "vol" == lower:
            result["vix_futures_volume"] = pd.to_numeric(df[col], errors="coerce")

    if "vix_futures_volume" not in result.columns:
        result["vix_futures_volume"] = 0

    return result.dropna(subset=["vix_futures_m1", "vix_futures_m2"], how="all")


def _load_quandl_continuous(path: Path) -> pd.DataFrame:
    """Load a Quandl continuous contract CSV (has Settle/Close column)."""
    df = pd.read_csv(path)
    df = _parse_date_index(df)
    # Quandl format: Date, Open, High, Low, Settle (or Close), Volume, ...
    for col in df.columns:
        lower = col.lower()
        if "settle" in lower or "close" in lower or "last" in lower:
            return pd.DataFrame({
                "settle": pd.to_numeric(df[col], errors="coerce"),
            }, index=df.index).dropna()
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) > 0:
        return pd.DataFrame({"settle": df[numeric[0]]}, index=df.index).dropna()
    return pd.DataFrame()


def load_vix_futures(raw_dir: Path) -> pd.DataFrame:
    """Load VIX futures term structure from available raw files.

    Handles multiple file patterns from the fetch scripts:
    - vix_futures.csv (single combined file)
    - vix_futures_vx{N}_quandl.csv (Quandl continuous contracts)
    - vix_futures_cboe.csv (CBOE settlement data)
    - vix_futures_*_yf.csv (yfinance data)
    """
    # Try 1: Single combined file
    combined_path = raw_dir / "vix_futures.csv"
    if combined_path.exists():
        print("  Loading combined vix_futures.csv")
        return load_vix_futures_single(combined_path)

    # Try 2: Build term structure from Quandl continuous contract files
    quandl_files = sorted(raw_dir.glob("vix_futures_vx*_quandl.csv"))
    if len(quandl_files) >= 2:
        print(f"  Building term structure from {len(quandl_files)} Quandl files")
        contracts = {}
        for f in quandl_files:
            # Extract month number from filename like vix_futures_vx1_quandl.csv
            name = f.stem
            for i in range(1, 10):
                if f"vx{i}" in name:
                    df = _load_quandl_continuous(f)
                    if not df.empty:
                        contracts[i] = df["settle"]
                    break

        if len(contracts) >= 2:
            result = pd.DataFrame()
            if 1 in contracts:
                result["vix_futures_m1"] = contracts[1]
            if 2 in contracts:
                result["vix_futures_m2"] = contracts[2]
            if 3 in contracts:
                result["vix_futures_m3"] = contracts[3]
            result["vix_futures_volume"] = 0  # Quandl may not have volume
            # Try to add volume from the VX1 file
            for f in quandl_files:
                if "vx1" in f.stem:
                    vx1 = pd.read_csv(f)
                    vx1 = _parse_date_index(vx1)
                    for col in vx1.columns:
                        if "volume" in col.lower():
                            result["vix_futures_volume"] = pd.to_numeric(
                                vx1[col], errors="coerce"
                            ).reindex(result.index, fill_value=0)
                    break
            result = result.dropna(subset=["vix_futures_m1", "vix_futures_m2"])
            if not result.empty:
                print(f"  Term structure: {len(result)} rows")
                return result

    # Try 3: CBOE futures settlement data
    cboe_path = raw_dir / "vix_futures_cboe.csv"
    if cboe_path.exists():
        print("  Loading CBOE futures data")
        return load_vix_futures_single(cboe_path)

    # Try 4: yfinance data (usually just front month)
    yf_files = sorted(raw_dir.glob("vix_futures_*_yf.csv"))
    if yf_files:
        print(f"  Loading yfinance futures data from {yf_files[0].name}")
        df = pd.read_csv(yf_files[0])
        df = _parse_date_index(df)
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            lower = col.lower()
            if "close" in lower:
                result["vix_futures_m1"] = pd.to_numeric(df[col], errors="coerce")
                break
        if "vix_futures_m1" not in result.columns:
            numeric = df.select_dtypes(include="number").columns
            if len(numeric) > 0:
                result["vix_futures_m1"] = df[numeric[0]]
        # Without M2 from yfinance, estimate M2 as M1 * 1.05 (typical contango)
        if "vix_futures_m1" in result.columns:
            result["vix_futures_m2"] = result["vix_futures_m1"] * 1.05
            result["vix_futures_volume"] = 0
            for col in df.columns:
                if "volume" in col.lower():
                    result["vix_futures_volume"] = pd.to_numeric(df[col], errors="coerce")
            result = result.dropna(subset=["vix_futures_m1"])
            if not result.empty:
                print(f"  yfinance futures: {len(result)} rows (M2 estimated)")
                return result

    # Fallback: synthesize term structure from VIX spot data
    # VIX futures typically trade at a premium to spot (contango).
    # During spikes (VIX > 30), futures go to backwardation.
    # This is an approximation but captures the key dynamics.
    vix_path = raw_dir / "vix_daily.csv"
    if vix_path.exists():
        print("  WARNING: No VIX futures data available. Synthesizing term structure from VIX spot.")
        print("  (This is approximate. Real futures data will improve model quality.)")
        vix_df = pd.read_csv(vix_path)
        vix_df = _parse_date_index(vix_df)
        # Find VIX close column
        vix_close = None
        for col in vix_df.columns:
            if "close" in col.lower() or "adj" in col.lower():
                vix_close = pd.to_numeric(vix_df[col], errors="coerce")
                break
        if vix_close is None:
            numeric = vix_df.select_dtypes(include="number").columns
            if len(numeric) > 0:
                vix_close = vix_df[numeric[0]]

        if vix_close is not None:
            # Model contango/backwardation based on VIX level
            # When VIX < 20: strong contango (futures premium ~5-8%)
            # When VIX 20-30: mild contango (~2-4%)
            # When VIX > 30: backwardation (futures discount ~2-5%)
            rng = np.random.default_rng(42)  # Reproducible
            n = len(vix_close)
            contango_m1 = np.where(
                vix_close > 30,
                rng.normal(-0.03, 0.01, n),   # backwardation
                np.where(
                    vix_close > 20,
                    rng.normal(0.03, 0.01, n),  # mild contango
                    rng.normal(0.06, 0.02, n),  # strong contango
                ),
            )
            contango_m2 = contango_m1 * 1.5  # M2 further out on the curve
            result = pd.DataFrame({
                "vix_futures_m1": vix_close * (1 + contango_m1),
                "vix_futures_m2": vix_close * (1 + contango_m2),
                "vix_futures_volume": 150000,  # Typical volume placeholder
            }, index=vix_close.index)
            result = result.dropna()
            print(f"  Synthesized futures: {len(result)} rows")
            return result

    raise FileNotFoundError(
        "No VIX futures data found and cannot synthesize. Expected one of:\n"
        "  - data/raw/vix_futures.csv\n"
        "  - data/raw/vix_futures_vx{1,2,...}_quandl.csv\n"
        "  - data/raw/vix_futures_cboe.csv\n"
        "  - data/raw/vix_futures_*_yf.csv\n"
        "  - data/raw/vix_daily.csv (for synthesis fallback)"
    )


def load_single_series(path: Path, col_name: str) -> pd.DataFrame:
    """Load a single-series CSV (VVIX, VIX9D, SKEW, etc.)."""
    df = pd.read_csv(path)
    df = _parse_date_index(df)
    # Take the close or first numeric column
    for col in df.columns:
        lower = col.lower()
        if "close" in lower or "adj" in lower:
            return pd.DataFrame({col_name: pd.to_numeric(df[col], errors="coerce")},
                                index=df.index).dropna()
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) > 0:
        return pd.DataFrame({col_name: pd.to_numeric(df[numeric[0]], errors="coerce")},
                            index=df.index).dropna()
    raise ValueError(f"No numeric columns found in {path}")


def load_put_call_ratio(path: Path) -> pd.DataFrame:
    """Load CBOE equity put/call ratio data."""
    df = pd.read_csv(path)
    df = _parse_date_index(df)
    for col in df.columns:
        lower = col.lower()
        if "ratio" in lower or "p/c" in lower or "put" in lower:
            return pd.DataFrame(
                {"put_call_ratio": pd.to_numeric(df[col], errors="coerce")},
                index=df.index,
            ).dropna()
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) > 0:
        return pd.DataFrame(
            {"put_call_ratio": pd.to_numeric(df[numeric[0]], errors="coerce")},
            index=df.index,
        ).dropna()
    raise ValueError(f"No numeric columns found in {path}")


# ---------------------------------------------------------------------------
# Merge raw data
# ---------------------------------------------------------------------------

def load_and_merge_raw(raw_dir: Path) -> pd.DataFrame:
    """Load all raw CSVs and merge into a single DataFrame aligned by date.

    Missing data is forward-filled then remaining NaN rows are dropped.
    """
    frames: dict[str, pd.DataFrame] = {}

    # VIX (required)
    vix_path = raw_dir / RAW_FILES["vix"]
    if not vix_path.exists():
        raise FileNotFoundError(f"VIX data not found at {vix_path}")
    frames["vix"] = load_vix(vix_path)

    # SPY (required)
    spy_path = raw_dir / RAW_FILES["spy"]
    if not spy_path.exists():
        raise FileNotFoundError(f"SPY data not found at {spy_path}")
    frames["spy"] = load_spy(spy_path)

    # VIX Futures (required -- tries multiple file patterns)
    frames["futures"] = load_vix_futures(raw_dir)

    # Optional supplementary data
    optional_loaders = {
        "vvix": ("vvix", "vvix_close"),
        "vix9d": ("vix9d", "vix9d_close"),
        "skew": ("skew", "skew_close"),
        "hyg": ("hyg", "hyg_close"),
        "tlt": ("tlt", "tlt_close"),
    }
    for key, (raw_key, col_name) in optional_loaders.items():
        fpath = raw_dir / RAW_FILES[raw_key]
        if fpath.exists():
            try:
                frames[key] = load_single_series(fpath, col_name)
                print(f"  Loaded {key}: {len(frames[key])} rows")
            except Exception as e:
                print(f"  Warning: Could not load {key}: {e}")
        else:
            print(f"  Warning: {fpath.name} not found, will use placeholder")

    # Put/call ratio
    pc_path = raw_dir / RAW_FILES["put_call"]
    if pc_path.exists():
        try:
            frames["put_call"] = load_put_call_ratio(pc_path)
            print(f"  Loaded put_call: {len(frames['put_call'])} rows")
        except Exception as e:
            print(f"  Warning: Could not load put/call ratio: {e}")

    # Merge all frames on the date index
    merged = frames["vix"]
    for key, df in frames.items():
        if key == "vix":
            continue
        merged = merged.join(df, how="outer")

    # Sort by date
    merged = merged.sort_index()

    # Forward-fill missing data (accounts for different date ranges)
    merged = merged.ffill()

    # Fill remaining missing supplementary data with reasonable defaults
    # so the feature pipeline doesn't break
    defaults = {
        "vvix_close": merged.get("vix_close", pd.Series(dtype=float)) * 4.5 if "vix_close" in merged else 90.0,
        "vix9d_close": merged.get("vix_close", pd.Series(dtype=float)) if "vix_close" in merged else 15.0,
        "skew_close": 130.0,
        "hyg_close": 80.0,
        "tlt_close": 100.0,
        "put_call_ratio": 0.7,
        "vix_futures_volume": 0,
    }
    for col, default in defaults.items():
        if col not in merged.columns:
            if isinstance(default, pd.Series):
                merged[col] = default
            else:
                merged[col] = default

    # Drop rows where critical columns are still NaN
    critical = ["vix_close", "spy_close", "vix_futures_m1", "vix_futures_m2"]
    merged = merged.dropna(subset=critical)

    print(f"\n  Merged dataset: {len(merged)} rows, "
          f"date range {merged.index.min().date()} to {merged.index.max().date()}")
    return merged


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate training labels using tiered horizon windows.

    Eligible days: vix_zscore > 1.0
    Horizon depends on VIX level:
        VIX 18-25: 15 trading days
        VIX 25-35: 30 trading days
        VIX 35+:   45 trading days

    Labels:
        label_revert (binary):       VIX drops >= 15% within horizon
        label_spike_first (binary):  VIX rises >= 10% before reversion
        label_magnitude (continuous): max % drop within horizon

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'vix_close' and 'vix_zscore' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: eligible, horizon, label_revert,
        label_spike_first, label_magnitude. Non-eligible rows have NaN labels.
    """
    vix = df["vix_close"].values
    zscore = df["vix_zscore"].values
    n = len(df)

    eligible = np.full(n, False)
    horizon = np.full(n, np.nan)
    label_revert = np.full(n, np.nan)
    label_spike_first = np.full(n, np.nan)
    label_magnitude = np.full(n, np.nan)

    for i in range(n):
        # Check eligibility
        if np.isnan(zscore[i]) or zscore[i] <= 1.0:
            continue
        if np.isnan(vix[i]) or vix[i] < 18.0:
            continue

        eligible[i] = True

        # Determine horizon based on VIX level
        if vix[i] < 25:
            h = 15
        elif vix[i] < 35:
            h = 30
        else:
            h = 45
        horizon[i] = h

        # Look forward up to horizon trading days
        end = min(i + h + 1, n)
        if i + 1 >= n:
            # Can't compute labels for the very last row
            label_revert[i] = 0
            label_spike_first[i] = 0
            label_magnitude[i] = 0.0
            continue

        future_vix = vix[i + 1: end]
        if len(future_vix) == 0:
            label_revert[i] = 0
            label_spike_first[i] = 0
            label_magnitude[i] = 0.0
            continue

        current = vix[i]
        pct_changes = (future_vix - current) / current

        # label_magnitude: max percentage drop (negative = drop, so we look for min)
        min_change = np.nanmin(pct_changes)
        label_magnitude[i] = -min_change  # store as positive % drop

        # label_revert: did VIX drop >= 15% within horizon?
        revert_threshold = -0.15
        reverted = pct_changes <= revert_threshold
        label_revert[i] = 1 if np.any(reverted) else 0

        # label_spike_first: did VIX rise >= 10% before the reversion?
        spike_threshold = 0.10
        if np.any(reverted):
            first_revert_idx = np.argmax(reverted)
            # Check if there was a >= 10% rise before the first reversion
            pre_revert = pct_changes[:first_revert_idx]
            if len(pre_revert) > 0 and np.any(pre_revert >= spike_threshold):
                label_spike_first[i] = 1
            else:
                label_spike_first[i] = 0
        else:
            # No reversion -- check if VIX spiked at all within horizon
            if np.any(pct_changes >= spike_threshold):
                label_spike_first[i] = 1
            else:
                label_spike_first[i] = 0

    labels = pd.DataFrame({
        "eligible": eligible,
        "horizon": horizon,
        "label_revert": label_revert,
        "label_spike_first": label_spike_first,
        "label_magnitude": label_magnitude,
    }, index=df.index)

    return labels


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Run the full dataset build pipeline."""
    print("=" * 60)
    print("VIX Dataset Builder")
    print("=" * 60)

    # Step 1: Load and merge raw data
    print("\n[1/4] Loading raw data...")
    merged = load_and_merge_raw(RAW_DIR)

    # Step 2: Compute features
    print("\n[2/4] Computing features...")
    features = compute_all_features(merged)
    print(f"  Computed {len(features.columns)} features")

    # Step 3: Generate labels (needs vix_zscore from features)
    print("\n[3/4] Generating labels...")
    label_input = merged[["vix_close"]].copy()
    label_input["vix_zscore"] = features["vix_zscore"]
    labels = generate_labels(label_input)

    # Step 4: Combine and save
    print("\n[4/4] Combining and saving...")
    dataset = features.join(labels)

    # Drop rows with NaN features (due to rolling window warmup)
    feature_cols = features.columns.tolist()
    before = len(dataset)
    dataset = dataset.dropna(subset=feature_cols)
    dropped = before - len(dataset)
    if dropped > 0:
        print(f"  Dropped {dropped} rows due to rolling window warmup")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PATH, index=True)
    print(f"  Saved to {OUTPUT_PATH}")

    # Summary stats
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total trading days:    {len(dataset)}")
    print(f"  Date range:            {dataset.index.min().date()} to {dataset.index.max().date()}")
    print(f"  Feature columns:       {len(feature_cols)}")

    eligible_mask = dataset["eligible"] == True  # noqa: E712
    n_eligible = eligible_mask.sum()
    print(f"\n  Eligible days (zscore > 1.0): {n_eligible} "
          f"({100 * n_eligible / len(dataset):.1f}%)")

    if n_eligible > 0:
        eligible_data = dataset[eligible_mask]
        n_revert = (eligible_data["label_revert"] == 1).sum()
        n_spike = (eligible_data["label_spike_first"] == 1).sum()
        avg_mag = eligible_data["label_magnitude"].mean()

        print(f"  label_revert = 1:      {n_revert} ({100 * n_revert / n_eligible:.1f}%)")
        print(f"  label_spike_first = 1: {n_spike} ({100 * n_spike / n_eligible:.1f}%)")
        print(f"  label_magnitude avg:   {avg_mag:.1f}%")

        # Breakdown by VIX tier
        print("\n  Breakdown by VIX tier:")
        for tier_name, lo, hi in [("18-25", 18, 25), ("25-35", 25, 35), ("35+", 35, 200)]:
            mask = (eligible_data["vix_spot"] >= lo) & (eligible_data["vix_spot"] < hi)
            n_tier = mask.sum()
            if n_tier > 0:
                n_rev = (eligible_data.loc[mask, "label_revert"] == 1).sum()
                print(f"    VIX {tier_name}: {n_tier} days, "
                      f"revert rate {100 * n_rev / n_tier:.0f}%")

    print("\n" + "=" * 60)
    return dataset


if __name__ == "__main__":
    build_dataset()
