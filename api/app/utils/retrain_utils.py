"""
utils/retrain_utils.py

Automatic retraining utilities triggered by model monitoring.

Features:
- Monitors drift or degradation signals and triggers retraining
- Supports Airflow DAG triggers or FastAPI endpoints
- Logs retraining runs in MLflow
- Pushes retraining events and results to Prometheus
"""

import time
import json
from typing import Dict, Any, Optional

from utils.logging import get_logger
from utils.mlflow_utils import start_run, log_params, log_metrics, end_run
from utils.prometheus_utils import create_registry, define_counter, push_metrics

logger = get_logger("retrain_utils")

# Core Retraining Trigger

def should_retrain(
    drift_results: Dict[str, bool],
    degradation_results: Dict[str, bool],
    drift_threshold: int = 1,
    degradation_threshold: int = 1,
) -> bool:
    """
    Determines whether retrainig should be triggered based on monitoring results.

    Args:
    drift_results: dict of features -> bool (True if drift detected)
    degradation_results: dict of metrics -> bool (True if degraded)
    drift_threshold: number of drifted features to trigger retraining
    degradation_threshold: number of degraded metrics to trigger retrainig

    Returns:
        bool: True if retraining should occur
    """

    drift_count = sum(drift_results.values())
    degradation_count = sum(degradation_results.values())

    logger.info(
        f"🔎 Drift detected in {drift_count} features",
        f"performance degraded in {degradation_count} metrics."
    )

    if drift_count >= drift_threshold or degradation_count >= degradation_threshold:
        logger.warning("⚠️ Retraining condition met.")
        return True
    
    logger.info("✅ Model stable, no retraining needed.")

    return False


# Retraining Workflow

def trigger_retraining(
    train_func,
    train_params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "model_retraining",
    job_name: str = "retrain_job",
    gateway: str = "http://localhost:9091"
) -> Dict[str, Any]:
    """
    Executes a retraining workflow, logging in MLflow and Prometheus.

    Args:
    train_func: callable retraining function (e.g., train_model)
    train_params: params dict for training
    experiment_name: MLflow experiment name
    job_name: Prometheus job name
    gateway: Prometheus Pushgateway URL

    Returns:
        dict: retraining summary (status, metrics, duration)
    """

    start_time = time.time()
    registry = create_registry()
    retrain_counter = define_counter("model_retrain_total", "Number of model retraining runs", registry)

    start_run(experiment_name, "auto_retrain_run")
    log_params(train_params or {})
    retrain_counter.inc()

    try:
        logger.info("🚀 Starting automatic model retraining...")
        result = train_func(**(train_params or {}))

        duration = time.time() - start_time
        log_metrics({"retrain_duration_sec": duration})
        log_metrics(result.get("metrics", {}))
        end_run("FINISHED")

        push_metrics(registry, job_name=job_name, gateway=gateway)
        logger.info(f"✅ Retraining completed in {duration:.2f}s.")
        return {"status": "success", "duration": duration, "metrics": result.get("metrics", {})}

    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        end_run("FAILED")
        push_metrics(registry, job_name=job_name, gateway=gateway)
        return {"status": "failed", "error": str(e)}
    
# Full Monitoring + Retraining Cycle

def monitor_and_retrain(
    drift_results: Dict[str, bool],
    degradation_results: Dict[str, bool],
    train_func,
    train_params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "model_retraining",
) -> Dict[str, Any]:
    """ 
    Checks for drift/degradation and triggers retraining if needed.

    Returns retraining summary.
    """

    if should_retrain(drift_results, degradation_results):
        summary = trigger_retraining(
            train_func=train_func,
            train_params=train_params,
            experiment_name=experiment_name
        )
        logger.info("📈 Retraining triggered succesfully.")
        return summary
    else:
        logger.info("🧩 No retraining action taken.")
        return {"status": "stable"}

    