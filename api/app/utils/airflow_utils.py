"""
utils/airflow_utils.py

Helper utilities and decorators for Airflow DAGs and tasks.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Optional

from .logging import get_logger
from .mlflow_utils import end_run, is_active_run, log_metrics, log_params, start_run
from .prometheus_utils import create_registry, define_histogram, push_metrics, sanitize_metric_name

logger = get_logger("airflow_utils")


def mlflow_task(
    experiment_name: str,
    run_prefix: str = "task_",
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
) -> Callable:
    """
    Decorate an Airflow task so it always runs inside an MLflow run.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            task_id = func.__name__
            run_name = f"{run_prefix}{task_id}"
            nested_run = is_active_run()

            start_run(
                experiment_name=experiment_name,
                run_name=run_name,
                tags=dict(tags or {}),
                nested=nested_run,
            )
            start_time = time.perf_counter()

            try:
                if params:
                    log_params(params)

                logger.info("Starting Airflow task '%s' under MLflow run '%s'.", task_id, run_name)
                result = func(*args, **kwargs)
                duration_sec = time.perf_counter() - start_time
                log_metrics({"task_duration_sec": duration_sec})
                end_run("FINISHED")
                logger.info("Task '%s' completed in %.3f seconds.", task_id, duration_sec)
                return result
            except Exception:
                end_run("FAILED")
                logger.exception("Task '%s' failed.", task_id)
                raise

        return wrapper

    return decorator


def timed_task(
    job_name: str = "airflow_task",
    gateway: str = "http://localhost:9091",
) -> Callable:
    """
    Decorate an Airflow task to measure runtime and push it to Prometheus.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = create_registry()
            histogram = define_histogram(
                name=sanitize_metric_name(f"{func.__name__}_runtime_seconds"),
                description=f"Execution time for Airflow task '{func.__name__}'",
                registry=registry,
            )
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                runtime_sec = time.perf_counter() - start_time
                histogram.observe(runtime_sec)
                push_metrics(registry, job_name=job_name, gateway=gateway)
                logger.info(
                    "Pushed runtime metric for task '%s' (%.3f seconds).",
                    func.__name__,
                    runtime_sec,
                )
                return result
            except Exception:
                logger.exception("Timed Airflow task '%s' failed.", func.__name__)
                raise

        return wrapper

    return decorator


def log_task_event(task_id: str, event: str, extra_info: Optional[Mapping[str, Any]] = None) -> None:
    """
    Log an event emitted by an Airflow task.
    """

    if extra_info:
        logger.info("Task event | task_id=%s event=%s details=%s", task_id, event, dict(extra_info))
    else:
        logger.info("Task event | task_id=%s event=%s", task_id, event)


def summarize_dag_run(dag_id: str, total_tasks: int, failed_tasks: int) -> None:
    """
    Log a concise summary for a DAG run.
    """

    if total_tasks <= 0:
        logger.warning("DAG '%s' reported zero tasks.", dag_id)
        return

    success_rate = (total_tasks - failed_tasks) / total_tasks * 100
    logger.info(
        "DAG '%s' completed with %.1f%% success (%d/%d failed).",
        dag_id,
        success_rate,
        failed_tasks,
        total_tasks,
    )
