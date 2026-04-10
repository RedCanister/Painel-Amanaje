"""
utils/prometheus_utils.py

Utilities for integrating Prometheus metrics with the project.
"""

from __future__ import annotations

import re
import socket
from typing import Any, Dict, Mapping, Optional, Sequence

from .logging import get_logger

logger = get_logger("prometheus_utils")

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        push_to_gateway,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - only used when prometheus_client is unavailable.
    CollectorRegistry = Any  # type: ignore[assignment]
    Counter = Any  # type: ignore[assignment]
    Gauge = Any  # type: ignore[assignment]
    Histogram = Any  # type: ignore[assignment]
    push_to_gateway = None  # type: ignore[assignment]
    start_http_server = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False

_INVALID_METRIC_CHARS = re.compile(r"[^a-zA-Z0-9_:]")


def _require_prometheus() -> None:
    if not PROMETHEUS_AVAILABLE:
        raise RuntimeError("prometheus_client is not available in the current environment.")


def sanitize_metric_name(name: str) -> str:
    """
    Convert arbitrary strings into Prometheus-safe metric names.
    """

    normalized = _INVALID_METRIC_CHARS.sub("_", name.strip())
    if not normalized:
        normalized = "metric"
    if normalized[0].isdigit():
        normalized = f"metric_{normalized}"
    return normalized.lower()


def create_registry() -> CollectorRegistry:
    """
    Create a new Prometheus metric registry.
    """

    _require_prometheus()
    registry = CollectorRegistry()
    logger.info("Created a Prometheus registry.")
    return registry


def define_gauge(
    name: str,
    description: str,
    registry: CollectorRegistry,
    labelnames: Optional[Sequence[str]] = None,
) -> Gauge:
    """
    Define a Prometheus gauge metric.
    """

    _require_prometheus()
    metric = Gauge(
        sanitize_metric_name(name),
        description,
        labelnames=tuple(labelnames or ()),
        registry=registry,
    )
    logger.info("Created gauge metric '%s'.", name)
    return metric


def define_counter(
    name: str,
    description: str,
    registry: CollectorRegistry,
    labelnames: Optional[Sequence[str]] = None,
) -> Counter:
    """
    Define a Prometheus counter metric.
    """

    _require_prometheus()
    metric = Counter(
        sanitize_metric_name(name),
        description,
        labelnames=tuple(labelnames or ()),
        registry=registry,
    )
    logger.info("Created counter metric '%s'.", name)
    return metric


def define_histogram(
    name: str,
    description: str,
    registry: CollectorRegistry,
    buckets: Optional[Sequence[float]] = None,
    labelnames: Optional[Sequence[str]] = None,
) -> Histogram:
    """
    Define a Prometheus histogram metric.
    """

    _require_prometheus()
    metric = Histogram(
        sanitize_metric_name(name),
        description,
        labelnames=tuple(labelnames or ()),
        registry=registry,
        buckets=tuple(buckets or (0.1, 0.5, 1, 2, 5, 10)),
    )
    logger.info("Created histogram metric '%s'.", name)
    return metric


def push_metrics(
    registry: CollectorRegistry,
    job_name: str,
    gateway: str = "http://localhost:9091",
    grouping_key: Optional[Mapping[str, str]] = None,
) -> None:
    """
    Push all metrics from a registry to a Prometheus Pushgateway.
    """

    _require_prometheus()
    push_to_gateway(gateway, job=sanitize_metric_name(job_name), registry=registry, grouping_key=dict(grouping_key or {}))
    logger.info("Pushed Prometheus metrics for job '%s' to '%s'.", job_name, gateway)


def start_prometheus_server(port: int = 8001) -> None:
    """
    Start a lightweight Prometheus metrics HTTP server.
    """

    _require_prometheus()
    start_http_server(port)
    hostname = socket.gethostname()
    logger.info("Prometheus metrics server started at http://%s:%d/metrics.", hostname, port)


def create_training_metrics(registry: CollectorRegistry) -> Dict[str, Any]:
    """
    Create a standard set of training metrics.
    """

    metrics = {
        "registry": registry,
        "train_loss": define_gauge("train_loss", "Current training loss value", registry),
        "val_loss": define_gauge("val_loss", "Current validation loss value", registry),
        "epoch_runtime": define_histogram(
            "epoch_runtime_seconds",
            "Training epoch runtime in seconds",
            registry,
        ),
        "epoch_counter": define_counter("epoch_counter", "Number of completed training epochs", registry),
    }
    logger.info("Initialized default training Prometheus metrics.")
    return metrics


def update_and_push_training_metrics(
    metrics: Mapping[str, Any],
    train_loss: float,
    val_loss: Optional[float],
    runtime_sec: float,
    gateway: str = "http://localhost:9091",
    job_name: str = "training_job",
) -> None:
    """
    Update and push the standard training metrics to Prometheus.
    """

    metrics["train_loss"].set(train_loss)
    if val_loss is not None:
        metrics["val_loss"].set(val_loss)
    metrics["epoch_runtime"].observe(runtime_sec)
    metrics["epoch_counter"].inc()

    push_metrics(metrics["registry"], job_name=job_name, gateway=gateway)
    logger.info("Pushed training metrics to Prometheus.")
