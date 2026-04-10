"""
utils/metrics.py

Centralized metric utilities for evaluating ML models.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - only used when numpy is unavailable.
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )

    SKLEARN_METRICS_AVAILABLE = True
except Exception:  # pragma: no cover - only used when scikit-learn is unavailable.
    accuracy_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    mean_absolute_error = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    r2_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    SKLEARN_METRICS_AVAILABLE = False

from .logging import get_logger
from .mlflow_utils import log_metrics

logger = get_logger("metrics")

LOWER_IS_BETTER_TOKENS = {
    "loss",
    "error",
    "rmse",
    "mae",
    "mse",
    "mape",
    "latency",
    "duration",
}


def _require_metric_dependencies() -> None:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("numpy is required for metric computation.")
    if not SKLEARN_METRICS_AVAILABLE:
        raise RuntimeError("scikit-learn is required for metric computation.")


def is_higher_better_metric(metric_name: str) -> bool:
    """
    Infer whether a metric should improve when its value increases.
    """

    normalized_name = metric_name.lower()
    return not any(token in normalized_name for token in LOWER_IS_BETTER_TOKENS)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """

    _require_metric_dependencies()
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(true_values, predicted_values))),
        "mse": float(mean_squared_error(true_values, predicted_values)),
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "mape": float(
            np.mean(
                np.abs((true_values - predicted_values) / np.maximum(np.abs(true_values), 1e-8))
            )
            * 100
        ),
        "r2": float(r2_score(true_values, predicted_values)),
    }
    logger.info("Computed regression metrics: %s", metrics)
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """

    _require_metric_dependencies()
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "precision": float(
            precision_score(true_values, predicted_values, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(true_values, predicted_values, average=average, zero_division=0)
        ),
        "f1": float(f1_score(true_values, predicted_values, average=average, zero_division=0)),
    }
    logger.info("Computed classification metrics: %s", metrics)
    return metrics


def evaluate_and_log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
    step: Optional[int] = None,
    prefix: str = "",
    log_to_mlflow: bool = True,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute evaluation metrics and optionally log them to MLflow.
    """

    normalized_task_type = task_type.lower()
    if normalized_task_type == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
    elif normalized_task_type == "classification":
        metrics = compute_classification_metrics(y_true, y_pred, average=average)
    else:
        raise ValueError("task_type must be 'regression' or 'classification'.")

    if prefix:
        metrics = {f"{prefix}{key}": value for key, value in metrics.items()}

    if log_to_mlflow:
        try:
            log_metrics(metrics, step=step)
        except Exception:
            logger.exception("Could not log metrics to MLflow.")

    return metrics


def pretty_print_metrics(metrics: Mapping[str, Any]) -> None:
    """
    Log a clean, aligned view of metric values.
    """

    logger.info("Evaluation summary:")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)):
            logger.info("  %-20s %.6f", key, float(value))
        else:
            logger.info("  %-20s %s", key, value)
