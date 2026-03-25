"""
utils/plotting.py

Reusable plotting utilities for model training, evaluation, and forecasting visualization.

Features:
- Prediction vs. Actual plotting
- Training/validation loss curves
- Metric comparison bar plots
- Automatic MLflow artifact logging
- Works seamessly inside Airflow and FastAPI

Dependencies;
- matplotlib
- numpy
- pandas
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional

from app.utils.io import ensure_dir
from app.utils.logging import get_logger
from app.utils.mlflow_utils import log_artifact

logger = get_logger("plotting")


# Helper: save + log figure

def _save_and_log_plot(fig, filename: str, log_to_mlflow: bool = True) -> None:
    """
    Save figure and optionally log it to MLflow.
    """
    ensure_dir("plots")
    filepath = os.path.join("plots", filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"📊 Plot saved at: {filepath}")
    if log_to_mlflow:
        try:
            log_artifact(filepath, artifact_path="plots")
            logger.info(f"🗂️ Logged plot to MLflow: {filename}")
        except Exception as e:
            logger.warning(f"⚠️ Could not log plot to MLflow: {e}")


# Prediction vs Actual

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actuals",
    log_to_mlflow: bool = True,
) -> None:
    """ 
    Plots predicted vs actual values for regression/time series models.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, color="royalblue", alpha=0.6, label="Predictions")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
    "r--", lw=2, label="Perfect Fit")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    _save_and_log_plot(fig, "predictions_vs_actuals.png", log_to_mlflow)

def plot_time_series_forecast(
    timestamps: List,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual",
    log_to_mlflow: bool = True,
) -> None:
    """
    Plots actual vs predicted series over time.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, y_true, label="Actual", color="black", linewidth=2)
    ax.plot(timestamps, y_pred, label="Forecast", color="deepskyblue", linewidth=2)
    ax.fill_between(timestamps, y_pred, y_true, color="lightblue", alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_and_log_plot(fig, "forecast_plot.png", log_to_mlflow)


# Training/Validation Loss Curves

def plot_loss_curve(
    history: Dict[str, List[float]],
    title: str = "Training and Validation Loss",
    log_to_mlflow: bool = True,
) -> None:

    """
    Plots training and validation loss curves from a history dict.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_and_log_plot(fig, "loss_curve.png", log_to_mlflow)


# Metric Comparison

def plot_metric_comparison(
    metrics: Dict[str, float],
    title: str = "Model Metrics Comparison",
    log_to_mlflow: bool = True,
) -> None:
    """
    Plots metrics (e.g., RMSE, MAE, R2, etc.) as a horizontal bar chart.
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    names, values = list(metrics.keys()), list(metrics.values())
    ax.barh(names, values, color="Cornflowerblue", alpha=0.8)
    ax.set_xlabel("Value")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    _save_and_log_plot(fig, "metric_comparison.png", log_to_mlflow)
