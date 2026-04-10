"""
utils/monitoring.py

Utilities for continuous model and data monitoring.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - only used when numpy is unavailable.
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - only used when pandas is unavailable.
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

try:
    from scipy.stats import ks_2samp

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - only used when scipy is unavailable.
    ks_2samp = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

from .logging import get_logger
from .metrics import is_higher_better_metric
from .mlflow_utils import log_metrics
from .prometheus_utils import create_registry, define_gauge, push_metrics, sanitize_metric_name

logger = get_logger("monitoring")


def _require_monitoring_dependencies() -> None:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("numpy is required for monitoring utilities.")
    if not PANDAS_AVAILABLE or pd is None:
        raise RuntimeError("pandas is required for monitoring utilities.")
    if not SCIPY_AVAILABLE or ks_2samp is None:
        raise RuntimeError("scipy is required for drift detection utilities.")


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict[str, bool]:
    """
    Detect numeric feature drift using the Kolmogorov-Smirnov test.
    """

    _require_monitoring_dependencies()
    drift_results: Dict[str, bool] = {}
    shared_columns = [column for column in reference_data.columns if column in current_data.columns]

    logger.info("Checking drift on %d shared columns.", len(shared_columns))
    for column in shared_columns:
        reference_series = reference_data[column].dropna()
        current_series = current_data[column].dropna()

        if not pd.api.types.is_numeric_dtype(reference_series) or not pd.api.types.is_numeric_dtype(current_series):
            logger.debug("Skipping non-numeric column '%s' during drift detection.", column)
            continue

        if reference_series.empty or current_series.empty:
            logger.debug("Skipping column '%s' because one of the samples is empty.", column)
            continue

        statistic, p_value = ks_2samp(reference_series, current_series)
        drift_detected = bool(p_value < threshold)
        drift_results[column] = drift_detected

        if drift_detected:
            logger.warning("Drift detected in feature '%s' (p=%.6f, stat=%.6f).", column, p_value, statistic)
        else:
            logger.info("No drift detected in feature '%s' (p=%.6f).", column, p_value)

    return drift_results


def detect_prediction_drift(
    reference_preds: np.ndarray,
    current_preds: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[bool, float]:
    """
    Detect drift between prediction distributions using the KS test.
    """

    _require_monitoring_dependencies()
    reference_values = np.asarray(reference_preds)
    current_values = np.asarray(current_preds)
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("Prediction arrays must not be empty.")

    _, p_value = ks_2samp(reference_values, current_values)
    drift_detected = bool(p_value < threshold)

    if drift_detected:
        logger.warning("Prediction drift detected (p=%.6f).", p_value)
    else:
        logger.info("No prediction drift detected (p=%.6f).", p_value)

    return drift_detected, float(p_value)


def track_performance(
    current_metrics: Mapping[str, float],
    previous_metrics: Mapping[str, float],
    tolerance: float = 0.05,
    higher_is_better: Optional[Mapping[str, bool]] = None,
) -> Dict[str, bool]:
    """
    Compare current and previous metrics and flag degraded metrics.
    """

    degradation: Dict[str, bool] = {}
    logger.info("Checking performance degradation across %d metrics.", len(current_metrics))

    for metric_name, current_value in current_metrics.items():
        previous_value = previous_metrics.get(metric_name)
        if previous_value is None:
            continue

        prefers_higher = (
            higher_is_better[metric_name]
            if higher_is_better and metric_name in higher_is_better
            else is_higher_better_metric(metric_name)
        )
        if prefers_higher:
            degraded = current_value < previous_value * (1 - tolerance)
        else:
            degraded = current_value > previous_value * (1 + tolerance)

        degradation[metric_name] = degraded
        if degraded:
            logger.warning(
                "Performance degraded for '%s': previous=%.6f current=%.6f.",
                metric_name,
                previous_value,
                current_value,
            )
        else:
            logger.info(
                "Performance stable for '%s': previous=%.6f current=%.6f.",
                metric_name,
                previous_value,
                current_value,
            )

    return degradation


def push_drift_metrics_to_prometheus(
    drift_results: Mapping[str, bool],
    job_name: str = "model_monitoring",
    gateway: str = "http://localhost:9091",
    metric_prefix: str = "model_drift",
) -> None:
    """
    Push feature drift signals to Prometheus as gauge values.
    """

    registry = create_registry()
    for feature_name, drifted in drift_results.items():
        metric_name = sanitize_metric_name(f"{metric_prefix}_{feature_name}")
        gauge = define_gauge(metric_name, f"Drift detected for feature '{feature_name}'", registry)
        gauge.set(int(drifted))

    push_metrics(registry, job_name=job_name, gateway=gateway)
    logger.info("Pushed drift metrics to Prometheus.")


def log_monitoring_results_to_mlflow(
    drift_results: Mapping[str, bool],
    degradation_results: Mapping[str, bool],
    prefix: str = "monitoring_",
) -> None:
    """
    Log drift and degradation signals to MLflow as numeric metrics.
    """

    drift_metrics = {f"{prefix}drift_{key}": int(value) for key, value in drift_results.items()}
    degradation_metrics = {
        f"{prefix}degraded_{key}": int(value) for key, value in degradation_results.items()
    }
    combined_metrics = {**drift_metrics, **degradation_metrics}
    log_metrics(combined_metrics)
    logger.info("Logged monitoring results to MLflow: %s", combined_metrics)


def summarize_monitoring_results(
    drift_results: Mapping[str, bool],
    degradation_results: Mapping[str, bool],
) -> Dict[str, int]:
    """
    Produce a compact monitoring summary for dashboarding and retraining logic.
    """

    summary = {
        "drifted_features": int(sum(bool(value) for value in drift_results.values())),
        "degraded_metrics": int(sum(bool(value) for value in degradation_results.values())),
    }
    logger.info("Monitoring summary: %s", summary)
    return summary
