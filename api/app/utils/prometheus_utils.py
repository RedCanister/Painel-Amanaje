"""
utils/prometheus_utils.py

Utilities for integratin Prometheus metrics with the project.

Features:
- Define and expose Prometheus metrics
- Push metrics to Prometheus Pushgateway (for Airflow, MLflow jobs)
- Integration with FastAPI and batch workflows
"""

import time
import socket
from typing import Dict, Optional
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway, start_http_server
from app.utils.logging import get_logger

logger = get_logger("prometheus_utils")

# Metric Registry

def create_registry() -> CollectorRegistry:
    """
    Creates a new Prometheus metric registry.
    """
    
    registry = CollectorRegistry()
    logger.info("🧩 Created new Prometheus registry.")
    return registry

# Metric Definitions

def define_gauge(name: str, description:str, registry: CollectorRegistry) -> Gauge:
    """
    Defines a Prometheus Gauge metric.
    """
    gauge = Gauge(name, description, registry=registry)
    logger.info(f"📈 Created gauge metric: {name}")
    return gauge

def define_counter(name: str, description: str, registry: CollectorRegistry) -> Counter:
    """
    Defines a Prometheus Counter metric.
    """
    counter = Counter(name, description, registry=registry)
    logger.info(f"🔢 Created counter metric: {name}")
    return counter

def define_histogram(
    name: str, description:str, registry: CollectorRegistry, buckets: Optional[list] = None
) -> Histogram:
    """
    Defines a Prometheus Histogram metric (e.g., latency).
    """
    histogram = Histogram(name, description, registry=registry, buckets=buckets or [0.1, 0.5, 1, 2, 5, 10])
    logger.info(f"⏱️ Created histogram metric: {name}")
    return histogram


# Metric Push Helpers

def push_metrics(
    registry: CollectorRegistry,
    job_name: str,
    gateway: str = "http://localhost:9091",
    grouping_key: Optional[Dict[str, str]] = None,
) -> None:
    """
    Pushes all metrics in the given registry to a Prometheus Pushgateway.
    """
    try:
        push_to_gateway(gateway, job=job_name, registry=registry, grouping_key= grouping_key or {})
        logger.info(f"🚀 Pushed metrics for job '{job_name}' to {gateway}")
    except Exception as e:
        logger.error(f"❌ Failed to push metrics to Prometheus gateway: {e}")


# FastAPI Endpoint Helper

def start_prometheus_server(port: int = 8001) -> None:
    """
    Starts a lightweight Prometheus metrics HTTP server (for FastAPI or background scripts).
    """
    start_http_server(port)
    hostname = socket.gethostname()
    logger.info(f"🌐 Prometheus metrics server started at http://{hostname}:{port}/metrics")


# Predefined Metrics

def create_training_metrics(registry: CollectorRegistry) -> Dict[str, Any]:
    """
    Predefines standard training metrics (loss, accuracy, runtime).
    """
    metrics = {
        "train_loss": define_gauge("train_loss", "Current training loss value", registry),
        "val_loss": define_gauge("val_loss", "Current validation loss value", registry),
        "epoch_runtime": define_gauge("epoch_runtime_seconds", "Training epoch runtime", registry),
        "epoch_counter": define_gauge("epoch_counter", "Number of completed training epochs", registry)
    }
    logger.info("🏋️ Initialized default training metrics.")
    return metrics

def update_and_push_training_metrics(
    metrics: Dict[str, Any],
    train_loss: float,
    val_loss: float,
    runtime_sec: float,
    gateway: str = "http://localhost:9091",
    job_name: str = "training_job",
) -> None:
    """
    Updates and pushes standard training metrics to Prometheus
    """
    metrics["train_loss"].set(train_loss)
    metrics["val_loss"].set(val_loss)
    metrics["epoch_runtime"].observe(runtime_sec)
    metrics["epoch_counter"].inc()

    push_metrics(metrics["train_loss"]._registry, job_name=job_name, gateway=gateway)
    logger.info(f"📤 Training metrics pushed to Prometheus ({gateway})")

