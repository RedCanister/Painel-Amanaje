""" 
utils/logging.py

Centralized and consistent logging configuration for the project

Features:
- Colored console logging (for local development and debugging)
- Optional file logging (rotating handler).
- MLflow integration hook for run-level logging
- Airflow and FastAPI compatibility
- Structured format with timestamps, levels, and module names
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
import mlflow


# ANSI Color Map for Console Output

_COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",    # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[41m",    # Red Background
    "RESET": "\033[0m",    # Reset
}


# Custom Formatter

class ColorFormatter(logging.Formatter):
    """
    Formatter that colorizes log levels for better readability in console.
    """

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        log_fmt = f"%(asctime)s | %(name)s | {color}%(levelname)s{reset} | %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:S")
        return formatter.format(record)


# Logger

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    file_max_bytes: int = 5_000_000,
    file_backup_count: int = 3,
) -> logging.Logger:
    """
    Creates and returns a configured logger.
    - name: module or component name
    - level: logging level(DEBUG, INFO, etc.)
    - log_file: optional file path path to store logs
    """

    logger = logging.getlogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)

    if log_file and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler(
            log_file, maxBytes=file_max_bytes, backupCount=file_backup_count, enconding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M%:%S"
        ))
        logger.addHandler(file_handler)

    return logger


# MLflow 

def log_to_mlflow(logger: logging.Logger, message: str, level: str = "info") -> None:
        """
        Logs a message both locally and to the active MLflow run, if one exists.
        """
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)

        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.log_text(message, artifact_file="logs/run_logs.txt")
        except Exception:
            pass

def init_global_logging(log_dir: Optional[str] = "logs") -> logging.Logger:
    """ 
    Initializes a root-level logger or the entire project.
    Creates a rotating log file in `logs/app.log` by default.
    """

    import os
    os.makedirs(log_dir, exist_ok=True)

    root_logger = get_logger(
        name="app",
        level=logging.INFO,
        log_file=os.path.join(log_dir, "app.log"),
    )

    root_logger.info("✅ Logging initialized.")
    return root_logger