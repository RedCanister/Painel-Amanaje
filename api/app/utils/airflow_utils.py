"""
utils/airflow_utils.py

Helper utilities and decorators for Airflow DAGs and tasks.

Features:
- Automatic MLflow tracking for Airflow tasks
- Execution time and error logging
- Prometheus metric integration (Pushgateway)
- Consistent logger initialization for DAGs
"""

import time 
from functools import wraps
from typing import Any, Callable, Dict, Optional

from app.utils.logging import get_logger
from app.utils.mlflow_utils import start_run, end_run, log_params, log_metrics
from app.utils.prometheus_utils import create_registry, define_histogram, push_metrics

logger = get_logger("airflow_utils")

# Decorators

def mlflow_task(
    experiment_name: str,
    run_prefix: str = "task_",
    params: Optional[Dict[str, Any]] = None,
):
    """
    Airflow task decorator that automatically:
    - starts and ends an MLflow run
    - logs parameters and metrics
    - captures runtime duration
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_id = func.__name__
            run_name = f"{run_prefix}{task_id}"

            run = start_run(experiment_name, run_name)
            start_time = time.time()

            try:
                if params:
                    log_params(params)
                
                logger.info(f"🚀 Starting Airflow task '{task_id}' under MLflow run '{run_name}'")
                result = func(*args, **kwargs)

                duration = time.time() - start_time
                log_metrics({"task_duration_sec": duration})
                end_run("FINISHED")

                logger.info(f"✅ Task '{task_id}' completed in {duration:.2f}s")
                return result

            except Exception as e:
                logger.error(f"❌ Task '{task_id}' failed: {e}")
                end_run("FAILED")
                raise result
        
        return wrapper
    
    return decorator


def timed_task(
    job_name: str = "airflow_task",
    gateway: str = "http://localhost:9091",
):
    """
    Airflow task decorator that measures and pushes execution time to Prometheus.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry = create_registry()
            hist = define_histogram(
                name=f"{func.__name__}_runtime_seconds",
                description=f"Execution time for Airflow task '{func__name__}'",
                registry=registry,
            )

            logger.info(f"⏱️ Timing Airflow task '{func.__name__}' ...")
            start_time =  time.time()

            try:
                result = func(*args, **kwargs)
                runtime = time.time() - start_time
                hist.observe(runtime)
                push_metrics(registry, job_name=job_name, gateway=gateway)
                logger.info(f"📤 Task '{func.__name__}' runtime {runtime:.2f}s pushed to Prometheus.")
                return result
            except Exception as e:
                logger.error(f"❌ Task '{func.__name__}' failed: {e}")
                raise e
        
        return wrapper
    
    return decorator


# Utility Functions

def log_task_event(task_id:str, event: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Logs an event from an Airflow task (for observability).
    """
    msg = f"📡 Task event - ID: {task_id}, Event: {event}"
    if extra_info:
        msg += f", Details: {extra_info}"
        logger.info(msg)


def summarize_dag_run(dag_id: str, total_tasks: int, failed_tasks: int) -> None:
    """
    Logs a concise DAG run summary.
    """
    success_rate = (total_tasks - failed_tasks) / total_tasks * 100
    logger.info(f"📊 DAG '{dag_id}' completed - {success_rate:.1f}% success ({failed_tasks}/{total_tasks} failed)")
            
