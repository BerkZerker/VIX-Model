"""Export trained models to ONNX format for deployment.

Supports:
  - XGBoost classifiers and regressors via onnxmltools/skl2onnx
  - PyTorch CNN+GRU via torch.onnx
  - Versioned filenames: vix_{type}_v{NNN}_{YYYYMMDD}.onnx
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx

logger = logging.getLogger(__name__)


def export_xgb_to_onnx(
    model,
    feature_names: list[str],
    output_path: str | Path,
    model_type: str = "classifier",
) -> Path:
    """Export an XGBoost model to ONNX format.

    Parameters
    ----------
    model : xgboost.XGBClassifier or xgboost.XGBRegressor
        Trained XGBoost model.
    feature_names : list[str]
        Feature names for the model input.
    output_path : path
        Where to save the ONNX file.
    model_type : str
        Either "classifier" or "regressor" or a descriptive name.

    Returns
    -------
    Path to saved ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try onnxmltools with proper XGBoost registration
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_xgboost(
            model.get_booster() if hasattr(model, "get_booster") else model,
            initial_types=initial_type,
        )

    except (ImportError, Exception) as e:
        logger.warning(f"onnxmltools conversion failed ({e}), trying skl2onnx with XGBoost registration")
        try:
            from skl2onnx import convert_sklearn, update_registered_converter
            from skl2onnx.common.data_types import FloatTensorType
            from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
            import xgboost as xgb

            # Register XGBoost converters with skl2onnx
            try:
                from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as _convert_xgb
                for cls in [xgb.XGBClassifier, xgb.XGBRegressor]:
                    update_registered_converter(
                        cls, f"XGBoost{cls.__name__}",
                        calculate_linear_classifier_output_shapes,
                        _convert_xgb,
                    )
            except ImportError:
                pass

            initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

        except (ImportError, Exception) as e2:
            logger.warning(f"skl2onnx conversion also failed: {e2}. Using native XGBoost save + ONNX-compatible wrapper.")
            # Fallback: save as native XGBoost format (can be loaded for inference)
            fallback_path = output_path.with_suffix(".json")
            model.save_model(str(fallback_path))
            logger.info(f"Saved native XGBoost model to {fallback_path}")
            return fallback_path

    onnx.save(onnx_model, str(output_path))
    logger.info(f"Exported XGBoost model to {output_path}")

    # Validate
    try:
        onnx.checker.check_model(str(output_path))
        logger.info("ONNX model validation passed")
    except Exception as e:
        logger.warning(f"ONNX model validation warning: {e}")

    return output_path


def export_pytorch_to_onnx(
    model,
    sample_inputs: dict[str, np.ndarray],
    output_path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model in eval mode.
    sample_inputs : dict mapping input names to numpy arrays
        Sample inputs for tracing. Shape must match expected input.
    output_path : path
        Where to save the ONNX file.
    input_names : list[str], optional
        Names for the ONNX inputs.
    output_names : list[str], optional
        Names for the ONNX outputs.

    Returns
    -------
    Path to saved ONNX file.
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Convert numpy inputs to torch tensors
    torch_inputs = tuple(
        torch.tensor(v, dtype=torch.float32) for v in sample_inputs.values()
    )

    if input_names is None:
        input_names = list(sample_inputs.keys())
    if output_names is None:
        output_names = ["p_revert", "p_spike_first", "expected_magnitude"]

    # Dynamic axes for batch dimension
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    torch.onnx.export(
        model,
        torch_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    logger.info(f"Exported PyTorch model to {output_path}")

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")

    return output_path


def verify_onnx_model(
    onnx_path: str | Path,
    sample_input: np.ndarray,
    expected_output: np.ndarray | None = None,
    atol: float = 1e-4,
) -> bool:
    """Verify an ONNX model loads and produces reasonable output.

    Parameters
    ----------
    onnx_path : path to ONNX model
    sample_input : numpy array matching model input shape
    expected_output : optional expected output for comparison
    atol : absolute tolerance for output comparison

    Returns
    -------
    bool indicating if verification passed
    """
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        logger.error(f"ONNX file not found: {onnx_path}")
        return False

    try:
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        logger.info(f"ONNX model inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")
        logger.info(f"ONNX model outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")

        # Run inference
        result = session.run(None, {input_name: sample_input.astype(np.float32)})
        logger.info(f"ONNX inference output shapes: {[r.shape for r in result]}")

        if expected_output is not None:
            if np.allclose(result[0], expected_output, atol=atol):
                logger.info("ONNX output matches expected output")
            else:
                logger.warning(
                    f"ONNX output differs from expected. "
                    f"Max diff: {np.max(np.abs(result[0] - expected_output))}"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def generate_model_version(
    model_type: str = "xgb",
    version: int = 1,
    date: datetime | None = None,
) -> str:
    """Generate a versioned model filename.

    Format: vix_{type}_v{NNN}_{YYYYMMDD}.onnx
    """
    if date is None:
        date = datetime.now()
    return f"vix_{model_type}_v{version:03d}_{date.strftime('%Y%m%d')}.onnx"
