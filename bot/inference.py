"""ONNX model inference for the VIX Alert Bot.

Loads an ONNX model and runs predictions.
MockInference returns synthetic predictions when no model is available.
"""

from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Model output."""

    p_revert: float
    p_spike_first: float
    expected_magnitude: float
    model_version: str


class BaseInference(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, features: dict[str, float]) -> Prediction: ...

    @abstractmethod
    def model_version(self) -> str: ...


class ModelInference(BaseInference):
    """ONNX Runtime model inference."""

    def __init__(self, models_dir: str | Path, version: str = "v001") -> None:
        self.models_dir = Path(models_dir)
        self.version = version
        self._session = None
        self._manifest: dict = {}
        self._feature_names: list[str] = []

    def load(self) -> None:
        """Load the ONNX model specified in model_manifest.json."""
        import onnxruntime as ort

        manifest_path = self.models_dir / "model_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Model manifest not found at {manifest_path}. "
                "Run training and export first, or use mock mode."
            )

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        model_file = self._manifest.get("model_file", "")
        model_path = self.models_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._feature_names = self._manifest.get("feature_names", [])
        self._session = ort.InferenceSession(str(model_path))
        logger.info(
            "Loaded ONNX model: %s (version: %s)",
            model_file,
            self._manifest.get("version", "unknown"),
        )

    def predict(self, features: dict[str, float]) -> Prediction:
        """Run inference on a feature dict."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build feature array in the expected order
        if self._feature_names:
            feat_array = np.array(
                [[features.get(name, 0.0) for name in self._feature_names]],
                dtype=np.float32,
            )
        else:
            feat_array = np.array(
                [list(features.values())], dtype=np.float32
            )

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: feat_array})

        # Parse outputs — layout depends on how the model was exported.
        # Convention: outputs[0] = [p_revert, p_spike_first, expected_magnitude]
        # or separate outputs for each head.
        if len(outputs) == 1 and outputs[0].shape[-1] >= 3:
            vals = outputs[0].flatten()
            p_revert = float(vals[0])
            p_spike_first = float(vals[1])
            expected_magnitude = float(vals[2])
        elif len(outputs) >= 3:
            p_revert = float(outputs[0].flatten()[0])
            p_spike_first = float(outputs[1].flatten()[0])
            expected_magnitude = float(outputs[2].flatten()[0])
        else:
            # Fallback: treat first output as p_revert
            p_revert = float(outputs[0].flatten()[0])
            p_spike_first = float(outputs[1].flatten()[0]) if len(outputs) > 1 else 0.5
            expected_magnitude = float(outputs[2].flatten()[0]) if len(outputs) > 2 else 0.0

        return Prediction(
            p_revert=np.clip(p_revert, 0, 1),
            p_spike_first=np.clip(p_spike_first, 0, 1),
            expected_magnitude=expected_magnitude,
            model_version=self._manifest.get("version", self.version),
        )

    def model_version(self) -> str:
        return self._manifest.get("version", self.version)


class XGBoostInference(BaseInference):
    """Native XGBoost model inference (fallback when ONNX export unavailable).

    Loads p_revert model from JSON, uses rules-based p_spike_first proxy.
    """

    def __init__(self, models_dir: str | Path, version: str = "v001") -> None:
        self.models_dir = Path(models_dir)
        self.version = version
        self._model = None
        self._manifest: dict = {}
        self._feature_names: list[str] = []
        self._scaler = None

    def load(self) -> None:
        """Load XGBoost model and scaler."""
        import xgboost as xgb

        manifest_path = self.models_dir / "model_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest = json.load(f)
            self._feature_names = self._manifest.get("feature_columns", [])

        # Find the model file (JSON or ONNX)
        model_files = sorted(self.models_dir.glob("vix_xgb_revert_*.json"))
        if not model_files:
            model_files = sorted(self.models_dir.glob("vix_xgb_revert_*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No XGBoost model found in {self.models_dir}")

        model_path = model_files[-1]  # Latest version
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_path))
        logger.info("Loaded XGBoost model: %s", model_path.name)

        # Load scaler
        scaler_files = sorted(self.models_dir.glob("scaler_*.pkl"))
        if scaler_files:
            import pickle
            with open(scaler_files[-1], "rb") as f:
                self._scaler = pickle.load(f)  # noqa: S301
            logger.info("Loaded scaler: %s", scaler_files[-1].name)

    def predict(self, features: dict[str, float]) -> Prediction:
        """Run inference on a feature dict."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build feature array
        if self._feature_names:
            feat_values = [features.get(name, 0.0) for name in self._feature_names]
        else:
            feat_values = list(features.values())

        feat_array = np.array([feat_values], dtype=np.float32)

        # Scale if scaler available
        if self._scaler is not None:
            feat_array = self._scaler.transform(feat_array)

        # Predict p_revert
        p_revert = float(self._model.predict_proba(feat_array)[0, 1])

        # Rules-based p_spike_first proxy (since model AUC < 0.60)
        vvix = features.get("vvix", 90)
        term_slope = features.get("term_slope", 0)
        p_spike_first = 0.15
        if vvix > 120 and abs(term_slope) < 0.01:
            p_spike_first = 0.5 + (vvix - 120) / 100
        elif vvix > 100:
            p_spike_first = 0.2 + (vvix - 100) / 200

        # Simple magnitude estimate
        vix_zscore = features.get("vix_zscore", 0)
        expected_magnitude = max(0, vix_zscore * 8.0)

        return Prediction(
            p_revert=np.clip(p_revert, 0, 1),
            p_spike_first=np.clip(p_spike_first, 0, 1),
            expected_magnitude=round(expected_magnitude, 1),
            model_version=self._manifest.get("version", self.version),
        )

    def model_version(self) -> str:
        return self._manifest.get("version", self.version)


class MockInference(BaseInference):
    """Returns synthetic predictions for demo/testing mode.

    Predictions are loosely conditioned on VIX level to produce
    realistic-looking outputs.
    """

    def __init__(self, version: str = "mock_v001") -> None:
        self.version = version

    def load(self) -> None:
        logger.info("MockInference loaded (synthetic predictions)")

    def predict(self, features: dict[str, float]) -> Prediction:
        vix = features.get("vix_spot", 18.0)
        zscore = features.get("vix_zscore", 0.0)

        # Higher VIX / z-score -> higher p_revert, lower p_spike_first
        base_revert = min(0.95, 0.3 + zscore * 0.2 + random.gauss(0, 0.05))
        base_spike = max(0.05, 0.5 - zscore * 0.15 + random.gauss(0, 0.05))
        magnitude = max(0.0, (vix - 15) * 0.8 + random.gauss(0, 2))

        return Prediction(
            p_revert=round(np.clip(base_revert, 0, 1), 3),
            p_spike_first=round(np.clip(base_spike, 0, 1), 3),
            expected_magnitude=round(magnitude, 1),
            model_version=self.version,
        )

    def model_version(self) -> str:
        return self.version
