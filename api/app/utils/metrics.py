"""
utils/metrics.py

Centralized metric utilities for evaluating ML models.

Features:
- Common regression & classification metrics (RMSE, MAE, MAPE, R2, Accuracy, F1)
- Unified login to MLflow
- Integration with project-wide logger
- Safe handling for Airflow / FastAPI/ Prometheus contexts
"""

# TODO - Consider hardware and system usage as metrics as well

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from typing import Dict, Any, Optional

from app.utils.logging import get_logger
from app.utils.mlflow_utils import log_metrics

logger = get_logger("metrics")

# Regression 

def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes core regression metrics.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }
    logger.info(f"📊 Regression metrics computed: {metrics}")
    return metrics


# Classification

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Computes core classification metrics
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_divison=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division = 0)),
    }
    logger.info(f"📊 Classification metrics computed: {metrics}")
    return metrics

# Combined

def evaluate_and_log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
    step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes and logs metrics to Mlflow based on task type.
    """

    if task_type == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        metrics = compute_classification_metrics(y_true, y_pred)
    else:
        raise ValueError("task_type must be 'regression' or 'classification'. ")

    try:
        log_metrics(metrics, step=step)
    except Exception as e:
        logger.warning(f"⚠️ Could not log metrics to MLflow: {e}")

    return metrics


# Utility

def pretty_print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Prints metrics in a clean, aligned format.
    """
    logger.info("🧮 Evaluation Summary:")
    for k, v in metrics.items():
        logger.info(f" {k:<10}: {v:.6f}")


