"""StandardScaler wrapper for the VIX Alert Bot.

Provides fit/transform/save/load utilities with versioning.
CRITICAL: The scaler must only be fit on training data, never on validation or test data.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training data only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training features. Must contain only numeric columns.

    Returns
    -------
    StandardScaler
        Fitted scaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(train_df)
    return scaler


def transform(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Features to transform. Columns must match those used during fit.
    scaler : StandardScaler
        A previously fitted scaler.

    Returns
    -------
    pd.DataFrame
        Scaled features with the same index and column names.
    """
    scaled_values = scaler.transform(df)
    return pd.DataFrame(scaled_values, index=df.index, columns=df.columns)


def save_scaler(scaler: StandardScaler, path: str | Path, version: str | None = None) -> Path:
    """Serialize a fitted scaler with version metadata.

    The scaler is saved as a pickle file alongside a JSON metadata file
    that records the version, timestamp, and feature names.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler to save.
    path : str or Path
        Output file path (e.g., 'models/scaler_v001.pkl').
    version : str, optional
        Version string. If not provided, extracted from the filename.

    Returns
    -------
    Path
        The path where the scaler was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the scaler
    with open(path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata alongside
    meta_path = path.with_suffix(".json")
    feature_names = (
        scaler.feature_names_in_.tolist()
        if hasattr(scaler, "feature_names_in_")
        else []
    )
    metadata = {
        "version": version or path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_features": int(scaler.n_features_in_),
        "feature_names": feature_names,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return path


def load_scaler(path: str | Path) -> StandardScaler:
    """Deserialize a scaler from disk.

    Parameters
    ----------
    path : str or Path
        Path to the pickle file.

    Returns
    -------
    StandardScaler
        The loaded scaler instance.
    """
    path = Path(path)
    with open(path, "rb") as f:
        scaler = pickle.load(f)  # noqa: S301
    return scaler
