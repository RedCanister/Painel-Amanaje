"""
utils/mlflow_utils.py

Utility functions and decorators to simplify MLflow experiment tracking.

Features:
- Auto-handling of MLflow run lifecycle (start/stop)
- Safe logging of parameters, metrics, and artifacts
- Integration with project-wide logger
- Decorator `@track_experiment` for automatic experiment tracking
- Compatibility with Airflow, FastAPI, and local training scripts
"""

import os
import mlflow
from functools import wraps
from typing import Any, Dict, Optional, Callable
from datetime import datetime

from utils.serialization import safe_log_params, to_json
from utils.logging import get_logger


# Default logger

logger = get_logger("mlflow_utils")

def start_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> mlflow.ActiveRun:
    """
    Starts a new MLflow run inside a named experiment.
    If experiment doesn't exist, it is created automatically.
    """
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    
    logger.info(f"🚀 Started MLflow run: {run.info.run_name} ({run.info.run_id})")
    return run

def end_run(status: str = "FINISHED") -> None:
    """
    Safely ends the active MLflow run.
    """

    try:
        mlflow.end_run(status=status)
        logger.info(f"🏁 MLflow run ended with status: {status}")
    except Exception as e:
        logger.warning(f"⚠️ Could not end MLflow run: {e}")


# Logging

def log_params(params: Any) -> None:
    """
    Logs a dictionary or object parameters to MLflow.
    """
    try:
        if not mlflow.active_run():
            logger.warning("⚠️ No active MLflow run found. Skipping param logging.")
            return
        safe_params = safe_log_params(params)
        mlflow.log_params(safe_params)
        logger.info(f"🧩 Logged {len(safe_params)} parameters to Mlflow.")

    except Exception as e:
        logger.error(f"Failed to log params: {e}")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Logs metrics to MLflow.
    """
    try:
        if not mlflow.active_run():
            logger.warning("⚠️ No active MLflow run found. Skipping metric logging.")
            return
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"📈 Logged metrics: {metrics}")
    except Exception as e:
        logger.error(f"failed to log metrics: {e}")


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    """
    Logs a single file as an artifact to Mlflow.
    """
    try:
        if not mlflow.active_run():
            logger.warning("⚠️ No active MLflow run found. Skipping artifact logging.")
            return
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path=artifact_path)
            logger.info(f"🗂️ Logged artifact: {path}")
        else:
            logger.warning(f"⚠️ Artifact path does not exist: {path}")
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}")
        
def log_json(data: Any, filename: str = "data_snapshot.json") -> None:
    """
    Converts data to JSON and logs it as as MLflow artifact.
    """
    try:
        if not mlflow.active_run():
            logger.warning("⚠️ No active MLflow run found. Skipping JSON logging.")
            return
        json_str = to_json(data)
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        mlflow.log_artifact(temp_path)
        logger.info(f"🧾 Logged JSON artifact: {filename}")
    except Exception as e:
        logger.error(f"Failed to log JSON: {e}")


# Decorator for experiment tracking

def track_experiment(
    experiment_name: str,
    run_name: Optional[str] = None,
    params: Optional[Any] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable:
    """
    Decorator that wraps a function with MLflow experiment tracking.
    Automatically starts and ends runs, and logs parameters.

    Example:
    @track_experiment("stock_forecasting", run_name="LSTM_v1")
    def train_model(config):
        ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            run = start_run(experiment_name, run_name, tags)
            try:
                if params:
                    log_params(params)
                result = func(*args, **kwargs)
                end_run("FINISHED")
                return result
            except Exception as e:
                logger.error(f"❌ Experiment failed: {e}")
                end_run("FAILED")
                raise e
        return wrapper

    return decorator

    

