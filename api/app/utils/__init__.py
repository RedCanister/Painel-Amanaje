"""
Unified utility package for training, tracking, deployment, and operations.

The modules under ``app.utils`` are designed to work together as a shared
foundation for the rest of the project. Import concrete helpers from each
module directly to avoid pulling heavyweight optional dependencies into places
that do not need them.
"""

UTILITY_MODULES = (
    "airflow_utils",
    "config",
    "deployment_utils",
    "io",
    "logging",
    "metrics",
    "mlflow_utils",
    "monitoring",
    "optuna_utils",
    "plotting",
    "prometheus_utils",
    "retrain_utils",
    "serialization",
    "training",
    "utils",
    "validation",
)

__all__ = ["UTILITY_MODULES"]
