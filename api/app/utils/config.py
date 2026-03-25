"""
utils/config.py

Centralized configuration management for the project.

Features:
- Load configuration from YAML or JSON
- Merge environment variable overrides
- Support for multiple environments (dev, staging, prod)
- Compatible with airflow, FastAPI, MLflow, an Docker
"""

import os
import json
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from app.utils.io import load_yaml, load_json
from app.utils.logging import get_logger
from app.utils.serialization import deep_asdict

logger = get_logger("config")

# Automatically load environment variables from .env file

load_dotenv()

# Base Loaders

def load_config(path: str) -> Dict[str, Any]:
    """
    Loads configuration file (YAML or JSON) into a dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not fount: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".yaml", ".yml"]:
        config = load_yaml(path)
    elif ext == ".json":
        config = load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {ext}")

    logger.info(f"🧩 Loaded config file: {path}")
    return config

# Env Merge

def merge_env_overrides(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively merges environment variable overrides into a configuration dict.

    Example:
    - If you have an env var MODEL__LR=0.001, it overrides config['model']['lr'].
    """
    updated_config = {}

    for key, value in config.items():
        env_key = f"{prefix}{key}".upper().replace(".", "_")
        if isinstance(value, dict):
            updated_config[key] = merge_env_overrides(value, prefix=f"{env_key}__")
        else:
            override = os.getenv(env_key)
            if override is not None:
                try:
                    override_val = json.loads(override)
                except json.JSONDecodeError:
                    try:
                        override_val = float(override)
                    except ValueError:
                        override_val = override
                
                logger.info(f"🔁 ENV override applied: {env_key}={override_val}")
                updated_config[key] = override_val
            else:
                updated_config[key] = value
    
    return updated_config

# Environment Profiles

def load_env_config(
    base_config_path: str,
    env_var: str = "APP_ENV",
    default_env: str = "development",
) -> Dict[str, Any]:
    """
    Loads configuration for a specific environment.
    Environment is defined by APP_ENV (dev/staging/prod.)
    """
    base_config = load_config(base_config_path)

    # Determine current environment
    env = os.getenv(env_var, default_env).lower()
    logger.info(f"🌍 Active environment: {env}")

    # Try load environment-specific config file
    env_config_path = os.path.splitext(base_config_path)[0] + f".{env}.yaml"
    if os.path.exists(env_config_path):
        env_config = load_yaml(env_config_path)
        logger.info(f"⚙️ Loaded environment-specific config: {env_config_path}")
        base_config.update(env_config)

    # Merge with environment variables
    final_config = merge_env_overrides(base_config)
    return final_config


# Export Helpers

def save_config_snapshot(config: Dict[str, Any], output_path: str = "configs/runtime_config.json") -> None:
    """
    Saves the active runtime configuration snapshot (for MLflow or reprsoducibility).
    """
    from app.utils.io import save_json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(deep_asdict(config), output_path)
    logger.info(f"💾 Runtime config snapshot save: {output_path}")

