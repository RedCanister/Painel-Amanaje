"""
utils/retrain_utils.py

Automatic retraining utilities triggered by monitoring signals.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Optional

from .logging import get_logger
from .mlflow_utils import end_run, log_json, log_metrics, log_params, start_run
from .training import TrainingResult

try:
    from .prometheus_utils import create_registry, define_counter, define_histogram, push_metrics

    PROMETHEUS_HELPERS_AVAILABLE = True
except Exception:  # pragma: no cover - only used when Prometheus helpers are unavailable.
    create_registry = None  # type: ignore[assignment]
    define_counter = None  # type: ignore[assignment]
    define_histogram = None  # type: ignore[assignment]
    push_metrics = None  # type: ignore[assignment]
    PROMETHEUS_HELPERS_AVAILABLE = False

logger = get_logger("retrain_utils")


def should_retrain(
    drift_results: Mapping[str, bool],
    degradation_results: Mapping[str, bool],
    drift_threshold: int = 1,
    degradation_threshold: int = 1,
) -> bool:
    """
    Determine whether monitoring signals justify retraining.
    """

    drift_count = int(sum(bool(value) for value in drift_results.values()))
    degradation_count = int(sum(bool(value) for value in degradation_results.values()))

    logger.info(
        "Retraining check: drift_count=%d degradation_count=%d.",
        drift_count,
        degradation_count,
    )

    should_trigger = drift_count >= drift_threshold or degradation_count >= degradation_threshold
    if should_trigger:
        logger.warning("Retraining condition met.")
    else:
        logger.info("Model is stable. Retraining is not required.")
    return should_trigger


def _normalize_retraining_result(result: Any) -> tuple[Dict[str, float], Any]:
    """
    Extract metrics from common retraining return shapes.
    """

    if isinstance(result, TrainingResult):
        return dict(result.metrics), result.to_dict()

    if isinstance(result, Mapping):
        raw_metrics = result.get("metrics", {})
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        return dict(metrics), dict(result)

    return {}, result


def trigger_retraining(
    train_func: Callable[..., Any],
    train_params: Optional[Mapping[str, Any]] = None,
    experiment_name: str = "model_retraining",
    run_name: str = "auto_retrain_run",
    job_name: str = "retrain_job",
    gateway: str = "http://localhost:9091",
    push_metrics_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Execute a retraining workflow and track it in MLflow and Prometheus.
    """

    parameters = dict(train_params or {})
    registry = None
    retrain_counter = None
    retrain_duration = None

    if push_metrics_enabled and PROMETHEUS_HELPERS_AVAILABLE:
        try:
            registry = create_registry()
            retrain_counter = define_counter(
                "model_retrain_total",
                "Number of model retraining runs",
                registry,
            )
            retrain_duration = define_histogram(
                "model_retrain_duration_seconds",
                "Duration of retraining runs in seconds",
                registry,
            )
        except Exception:
            logger.warning("Prometheus metrics are unavailable for retraining.", exc_info=True)
            registry = None
            retrain_counter = None
            retrain_duration = None

    start_run(experiment_name, run_name=run_name)
    log_params(parameters)
    if retrain_counter is not None:
        retrain_counter.inc()
    start_time = time.perf_counter()

    try:
        result = train_func(**parameters)
        duration_sec = time.perf_counter() - start_time
        if retrain_duration is not None:
            retrain_duration.observe(duration_sec)

        result_metrics, result_payload = _normalize_retraining_result(result)
        combined_metrics = {"retrain_duration_sec": duration_sec, **result_metrics}
        log_metrics(combined_metrics)
        log_json(result_payload, filename="retraining_summary.json", artifact_path="retraining")

        end_run("FINISHED")
        if registry is not None and push_metrics is not None:
            push_metrics(registry, job_name=job_name, gateway=gateway)
        logger.info("Retraining completed successfully in %.3f seconds.", duration_sec)
        return {
            "status": "success",
            "duration": duration_sec,
            "metrics": result_metrics,
            "result": result_payload,
        }
    except Exception as exc:
        end_run("FAILED")
        if registry is not None and push_metrics is not None:
            push_metrics(registry, job_name=job_name, gateway=gateway)
        logger.exception("Retraining failed.")
        return {"status": "failed", "error": str(exc)}


def monitor_and_retrain(
    drift_results: Mapping[str, bool],
    degradation_results: Mapping[str, bool],
    train_func: Callable[..., Any],
    train_params: Optional[Mapping[str, Any]] = None,
    experiment_name: str = "model_retraining",
    run_name: str = "auto_retrain_run",
    job_name: str = "retrain_job",
    gateway: str = "http://localhost:9091",
    push_metrics_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Check monitoring signals and trigger retraining when thresholds are exceeded.
    """

    if not should_retrain(drift_results, degradation_results):
        logger.info("No retraining action taken.")
        return {"status": "stable"}

    return trigger_retraining(
        train_func=train_func,
        train_params=train_params,
        experiment_name=experiment_name,
        run_name=run_name,
        job_name=job_name,
        gateway=gateway,
        push_metrics_enabled=push_metrics_enabled,
    )
