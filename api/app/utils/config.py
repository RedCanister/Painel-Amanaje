"""
utils/config.py

Centralized configuration management for the project.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - only used when python-dotenv is unavailable.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

from .io import load_json, load_yaml, save_json
from .logging import get_logger
from .serialization import deep_asdict

logger = get_logger("config")

load_dotenv()


def _parse_env_value(value: str) -> Any:
    """
    Parse environment variable values into structured Python objects when possible.
    """

    stripped = value.strip()
    if not stripped:
        return stripped

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        return stripped


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, preferring values from ``override``.
    """

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _environment_config_candidates(base_path: Path, env: str) -> list[Path]:
    """
    Build candidate filenames for environment-specific configuration overlays.
    """

    suffixes = [base_path.suffix] if base_path.suffix else []
    suffixes.extend(suffix for suffix in (".yaml", ".yml", ".json") if suffix not in suffixes)
    return [base_path.with_name(f"{base_path.stem}.{env}{suffix}") for suffix in suffixes]


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON configuration file into a dictionary.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    extension = config_path.suffix.lower()
    if extension in {".yaml", ".yml"}:
        config = load_yaml(config_path)
    elif extension == ".json":
        config = load_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {extension}")

    if not isinstance(config, dict):
        raise TypeError(f"Configuration at '{path}' must deserialize into a dictionary.")

    logger.info("Loaded config file '%s'.", config_path)
    return config


def merge_env_overrides(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively merge environment variable overrides into a configuration dictionary.

    Nested keys use ``__`` as a separator. For example, ``MODEL__LR=0.001`` maps
    to ``config['model']['lr']``.
    """

    updated_config: Dict[str, Any] = {}

    for key, value in config.items():
        env_key = f"{prefix}{key}".upper().replace(".", "_")
        if isinstance(value, dict):
            updated_config[key] = merge_env_overrides(value, prefix=f"{env_key}__")
            continue

        override = os.getenv(env_key)
        if override is None:
            updated_config[key] = value
            continue

        parsed_override = _parse_env_value(override)
        updated_config[key] = parsed_override
        logger.info("Applied environment override '%s'.", env_key)

    return updated_config


def load_env_config(
    base_config_path: str,
    env_var: str = "APP_ENV",
    default_env: str = "development",
) -> Dict[str, Any]:
    """
    Load base configuration, merge an optional environment-specific overlay,
    and then apply environment variable overrides.
    """

    base_path = Path(base_config_path)
    base_config = load_config(str(base_path))

    environment = os.getenv(env_var, default_env).lower()
    logger.info("Active environment: %s", environment)

    final_config = dict(base_config)
    for candidate in _environment_config_candidates(base_path, environment):
        if candidate.exists():
            overlay = load_config(str(candidate))
            final_config = _deep_merge(final_config, overlay)
            logger.info("Loaded environment-specific config '%s'.", candidate)
            break

    return merge_env_overrides(final_config)


def save_config_snapshot(config: Dict[str, Any], output_path: str = "configs/runtime_config.json") -> None:
    """
    Save the active runtime configuration to disk for reproducibility.
    """

    save_json(deep_asdict(config), output_path)
    logger.info("Saved runtime config snapshot to '%s'.", output_path)
