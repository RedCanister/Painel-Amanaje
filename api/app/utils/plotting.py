"""
utils/plotting.py

Reusable plotting utilities for model training, evaluation, and forecasting.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - only used when matplotlib is unavailable.
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - only used when numpy is unavailable.
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

from .io import ensure_dir
from .logging import get_logger
from .mlflow_utils import log_artifact

logger = get_logger("plotting")


def _require_plot_dependencies() -> None:
    if not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError("matplotlib is required for plotting utilities.")
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("numpy is required for plotting utilities.")


def _save_and_log_plot(
    fig: Any,
    filename: str,
    *,
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
    artifact_path: str = "plots",
) -> str:
    """
    Save a matplotlib figure and optionally log it to MLflow.
    """

    _require_plot_dependencies()
    plot_dir = ensure_dir(output_dir)
    file_path = plot_dir / filename
    fig.savefig(file_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    logger.info("Saved plot to '%s'.", file_path)
    if log_to_mlflow:
        try:
            log_artifact(str(file_path), artifact_path=artifact_path)
        except Exception:
            logger.exception("Could not log plot '%s' to MLflow.", file_path)

    return str(file_path)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actuals",
    *,
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
) -> str:
    """
    Plot predicted values against actual values for regression models.
    """

    _require_plot_dependencies()
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of observations.")

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(true_values, predicted_values, color="royalblue", alpha=0.7, label="Predictions")
    min_value = min(true_values.min(), predicted_values.min())
    max_value = max(true_values.max(), predicted_values.max())
    axis.plot([min_value, max_value], [min_value, max_value], "r--", lw=2, label="Perfect fit")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)

    return _save_and_log_plot(
        fig,
        "predictions_vs_actuals.png",
        output_dir=output_dir,
        log_to_mlflow=log_to_mlflow,
    )


def plot_time_series_forecast(
    timestamps: Iterable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual",
    *,
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
) -> str:
    """
    Plot actual and predicted values over time.
    """

    _require_plot_dependencies()
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    timestamp_values = list(timestamps)

    if len(timestamp_values) != len(true_values) or len(true_values) != len(predicted_values):
        raise ValueError("timestamps, y_true, and y_pred must have the same length.")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(timestamp_values, true_values, label="Actual", color="black", linewidth=2)
    axis.plot(timestamp_values, predicted_values, label="Forecast", color="deepskyblue", linewidth=2)
    axis.fill_between(timestamp_values, predicted_values, true_values, color="lightblue", alpha=0.3)
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.legend()
    axis.grid(alpha=0.3)

    return _save_and_log_plot(
        fig,
        "forecast_plot.png",
        output_dir=output_dir,
        log_to_mlflow=log_to_mlflow,
    )


def plot_loss_curve(
    history: Mapping[str, List[float]],
    title: str = "Training and Validation Loss",
    *,
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
) -> str:
    """
    Plot one or more loss curves from a training history dictionary.
    """

    _require_plot_dependencies()
    if not history:
        raise ValueError("history must contain at least one metric series.")

    fig, axis = plt.subplots(figsize=(8, 5))
    for key, values in history.items():
        axis.plot(values, label=key)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.3)

    return _save_and_log_plot(
        fig,
        "loss_curve.png",
        output_dir=output_dir,
        log_to_mlflow=log_to_mlflow,
    )


def plot_metric_comparison(
    metrics: Mapping[str, float],
    title: str = "Model Metrics Comparison",
    *,
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
) -> str:
    """
    Plot metrics as a horizontal bar chart.
    """

    _require_plot_dependencies()
    if not metrics:
        raise ValueError("metrics must contain at least one value.")

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.barh(metric_names, metric_values, color="cornflowerblue", alpha=0.85)
    axis.set_xlabel("Value")
    axis.set_title(title)
    for index, value in enumerate(metric_values):
        axis.text(value, index, f" {value:.3f}", va="center", fontsize=9)
    axis.grid(axis="x", alpha=0.3)

    return _save_and_log_plot(
        fig,
        "metric_comparison.png",
        output_dir=output_dir,
        log_to_mlflow=log_to_mlflow,
    )
