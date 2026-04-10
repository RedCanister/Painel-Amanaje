"""
utils/logging.py

Centralized and consistent logging configuration for the project.

Features:
- Colored console logging for local development.
- Optional rotating file logging.
- Consistent formatting across utility modules.
- Helper to persist selected log messages as MLflow text artifacts.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
    "RESET": "\033[0m",
}


def _coerce_level(level: int | str) -> int:
    """Normalize string and integer log levels into ``logging`` constants."""

    if isinstance(level, int):
        return level

    resolved_level = logging.getLevelName(str(level).upper())
    return resolved_level if isinstance(resolved_level, int) else logging.INFO


class ColorFormatter(logging.Formatter):
    """Formatter that colorizes the level name when the stream supports it."""

    def __init__(self, use_colors: Optional[bool] = None) -> None:
        super().__init__(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        self.use_colors = sys.stdout.isatty() if use_colors is None else use_colors

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_colors:
            return super().format(record)

        original_levelname = record.levelname
        color = _COLORS.get(original_levelname, "")
        reset = _COLORS["RESET"]
        if color:
            record.levelname = f"{color}{original_levelname}{reset}"

        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _has_stream_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler)
        for handler in logger.handlers
    )


def _has_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            try:
                if Path(handler.baseFilename).resolve() == log_file.resolve():
                    return True
            except OSError:
                if Path(handler.baseFilename) == log_file:
                    return True
    return False


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: Optional[str] = None,
    file_max_bytes: int = 5_000_000,
    file_backup_count: int = 3,
    use_colors: Optional[bool] = None,
    reset_handlers: bool = False,
) -> logging.Logger:
    """
    Create or reuse a configured logger.

    Parameters
    ----------
    name:
        Logger name, usually the module or subsystem name.
    level:
        Logging level as either an integer constant or string.
    log_file:
        Optional path for a rotating file handler.
    """

    normalized_level = _coerce_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(normalized_level)
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if not _has_stream_handler(logger):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(normalized_level)
        console_handler.setFormatter(ColorFormatter(use_colors=use_colors))
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not _has_file_handler(logger, log_path):
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=file_max_bytes,
                backupCount=file_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(normalized_level)
            file_handler.setFormatter(
                logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
            )
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(normalized_level)

    return logger


def log_to_mlflow(logger: logging.Logger, message: str, level: str = "info") -> None:
    """
    Log a message locally and, when possible, persist it as an MLflow artifact.
    """

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)

    try:
        import mlflow
    except Exception:
        return

    try:
        if mlflow.active_run():
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
            artifact_file = f"logs/{logger.name}_{timestamp}.log"
            mlflow.log_text(f"{message.rstrip()}\n", artifact_file=artifact_file)
    except Exception:
        logger.debug("Unable to mirror log message to MLflow.", exc_info=True)


def init_global_logging(
    log_dir: Optional[str] = "logs",
    logger_name: str = "app",
    level: int | str = logging.INFO,
) -> logging.Logger:
    """
    Initialize a root application logger backed by a rotating log file.
    """

    directory = Path(log_dir or "logs")
    directory.mkdir(parents=True, exist_ok=True)

    root_logger = get_logger(
        name=logger_name,
        level=level,
        log_file=str(directory / "app.log"),
    )
    root_logger.info("Logging initialized.")
    return root_logger
