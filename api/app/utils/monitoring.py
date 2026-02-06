"""
utils/monitoring.py

Utilities for continuous model and data monitoring.

Features:
- Data drift detection via statistical tests
- Prediction drift and performance degradation tracking
- Integration with MLflow and Prometheus
- Alert logging and optional threshold-based warnings
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import ks_2samp

from utils.logging import get_logger
from utils.mlflow_utils import log_metrics
from utils.prometheus_utils import create_registry, define_gauge, push_metrics

logger = get_logger("monitoring")

# Statistical Drift Detection

def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: float = 0.05,
) -> Dict[str, bool]:
    """
    Detects feature-level drift using Kolmogorov-Smirnov test.
    Returns a dictionary {feature: drift_detected}.
    """
    
    drift_results = {}
    shared_columns = [col for col in reference_data.columns if col in current_data]

    logger.info(f"🔍 Checking drift on {len(shared_columns)} shared_features...")

    for col in shared_columns:
        try:
            stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            drift = p_value < threshold
            drift_results[col] = drift
            
            if drift:
                logger.warning(f"⚠️ Drift detected in feature '{col}' (p={p_value:.4f})")
            else:
                logger.debug(f"✅ No drift in '{col}' (p={p_value:.4f})")
        except Exception as e:
            logger.error(f"❌ Failed to compute drift for '{col}': {e}")

    return drift_results


# Prediction drift detection

def detect_prediction_drift(
    reference_preds: np.ndarray,
    current_preds: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[bool, float]:
    """
    Detects drift in prediction distributions.
    Returns (drift_detected, p_value).
    """
    try:
        stat, p_value = ks_2samp(reference_preds, current_preds)
        drift = p_value < threshold

        if drift:
            logger.warning(f"⚠️ Prediction drift detected (p={p_value:.4f})")
        else:
            logger.info(f"✅ No prediction drift (p={p_value:.4f})")
        return drift, p_value
    except Exception as e:
        logger.error(f"❌ Failed to compute prediction drifT: {e}")
        return False, np.nan


# Performance Monitoring

def track_performance(
    current_metrics: Dict[str, float],
    previous_metrics: Dict[str, float],
    tolerance: float = 0.05
) -> Dict[str, bool]:
    """
    Compares current vs previous metrics and flags degradation.
    Returns {metric_name: degraded}.
    """
    degradation = {}
    logger.info("📊 Checking performance degradation...")

    for metric, current_value in current_metrics.items():
        prev_value = previous_metrics.get(metric)
        if prev_value is None:
            continue

        degraded = current_value > prev_value * (1 + tolerance)
        degradation[metric] = degraded

        if degraded:
            logger.warning(f"📉 Metric '{metric} degraded: {prev_value:.4f} → {current_value:.4f}")
        else:
            logger.info(f"✅ Metric '{metric}' stable: {current_value:.4f}")

    return degradation

# Prometheus Integration

def push_drift_metrics_to_prometheus(
    drift_results: Dict[str, bool],
    job_name: str = "model_monitoring",
    gateway: str = "http://localhost:9091"
) -> None:
    """
    Pushes data drift metrics (1 for drift detected, 0 for stable to Prometheus.
    """
    registry = create_registry()

    for feature, drifted in drift_results.items():
        g = define_gauge(f"drift_{feature}", f"Drift detected for feature '{feature}'", registry)
        g.set(1 if drifted else 0)

    push_metrics(registry, job_name=job_name, gateway=gateway)
    logger.info("📤 Drift metrics pushed to Prometheus.")


# MLflow Integration

def log_monitoring_results_to_mlflow(
    drift_results: Dict[str, bool],
    degradation_results: Dict[str, bool],
    prefix: str = "monitoring_"
) -> None:
    """
    Logs drift and degradation results to MLflow as metrics.
    """
    drift_metrics = {f"{prefix}drift_{k}": int(v) for k, v in drift_results.items()}
    degrade_metrics = {f"{prefix}degraded_{k}": int(v) for k,v in degradation_results.items()}
    combined = {**drift_metrics, **degrade_metrics}

    log_metrics(combined)
    logger.info(f"🧾 Monitoring results logged to MLflow: {combined}")

