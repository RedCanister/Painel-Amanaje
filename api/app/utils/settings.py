from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "debug_mode": False,
    "assistant_visible": True,
}

SAFE_ENVIRONMENT_VARIABLES: dict[str, dict[str, Any]] = {
    "APP_ENV": {
        "label": "Application Environment",
        "description": "Logical app environment used when loading environment-specific configuration overlays.",
        "category": "Application",
        "restart_required": True,
        "default": "development",
    },
    "MLFLOW_TRACKING_URI": {
        "label": "MLflow Tracking URI",
        "description": "Tracking endpoint or local file store used by experiment logging.",
        "category": "MLOps",
        "restart_required": True,
        "default": "",
    },
    "MLFLOW_REGISTRY_URI": {
        "label": "MLflow Registry URI",
        "description": "Model registry endpoint. Leave empty to follow the tracking URI.",
        "category": "MLOps",
        "restart_required": True,
        "default": "",
    },
    "AIRFLOW_URL": {
        "label": "Airflow URL",
        "description": "Airflow web/API URL shown to operators.",
        "category": "Orchestration",
        "restart_required": True,
        "default": "http://localhost:8080",
    },
    "AMANAJE_DASHBOARD_PUBLIC_URL": {
        "label": "Visualization Renderer URL",
        "description": "Internal browser-facing URL used by embedded Visualization plot frames.",
        "category": "Visualization",
        "restart_required": True,
        "default": "http://localhost:8050",
    },
    "AMANAJE_ASSISTANT_ACTIVE_PROVIDER": {
        "label": "Assistant Active Provider",
        "description": "Provider key used when assistant requests select the configured provider automatically.",
        "category": "Assistant",
        "restart_required": True,
        "default": "amanaje_slm",
    },
    "AMANAJE_SLM_ENABLED": {
        "label": "Private SLM Enabled",
        "description": "Enables the private SLM provider when its non-secret runtime configuration is available.",
        "category": "Assistant",
        "restart_required": True,
        "default": "false",
    },
    "AMANAJE_SLM_BASE_URL": {
        "label": "Private SLM Base URL",
        "description": "Base URL for the private SLM provider. API keys are intentionally not editable here.",
        "category": "Assistant",
        "restart_required": True,
        "default": "",
    },
    "AMANAJE_ASSISTANT_MLFLOW_ENABLED": {
        "label": "Assistant MLflow Logging",
        "description": "Controls whether assistant events attempt to log artifacts to MLflow.",
        "category": "Assistant",
        "restart_required": True,
        "default": "true",
    },
}

SECRET_KEY_MARKERS = (
    "API_KEY",
    "AUTH",
    "CONNECTION_STRING",
    "CREDENTIAL",
    "DATABASE_URL",
    "FERNET",
    "PASS",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SQL_ALCHEMY_CONN",
    "TOKEN",
)


class SettingsValidationError(ValueError):
    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def default_settings_state() -> dict[str, Any]:
    return {
        "feature_flags": deepcopy(DEFAULT_FEATURE_FLAGS),
        "env_overrides": {},
        "assistant": {
            "default_model_id": None,
        },
        "updated_at": None,
    }


def _state_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[2] / "runtime_artifacts" / "config" / "settings_state.json"


def _json_safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def load_settings_state(path: str | Path | None = None) -> dict[str, Any]:
    target = _state_path(path)
    state = default_settings_state()
    if not target.exists():
        return state

    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state

    loaded_flags = _json_safe_mapping(loaded.get("feature_flags"))
    for key, default_value in DEFAULT_FEATURE_FLAGS.items():
        state["feature_flags"][key] = bool(loaded_flags.get(key, default_value))

    loaded_overrides = _json_safe_mapping(loaded.get("env_overrides"))
    state["env_overrides"] = {
        str(key).strip().upper(): str(value)
        for key, value in loaded_overrides.items()
        if str(key).strip().upper() in SAFE_ENVIRONMENT_VARIABLES and value not in (None, "")
    }
    loaded_assistant = _json_safe_mapping(loaded.get("assistant"))
    default_model_id = loaded_assistant.get("default_model_id")
    state["assistant"] = {
        "default_model_id": str(default_model_id).strip() if default_model_id not in (None, "") else None,
    }
    state["updated_at"] = loaded.get("updated_at")
    return state


def save_settings_state(state: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = _state_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_flags": {
            key: bool(_json_safe_mapping(state.get("feature_flags")).get(key, default_value))
            for key, default_value in DEFAULT_FEATURE_FLAGS.items()
        },
        "env_overrides": {
            key: str(value)
            for key, value in _json_safe_mapping(state.get("env_overrides")).items()
            if key in SAFE_ENVIRONMENT_VARIABLES and value not in (None, "")
        },
        "assistant": {
            "default_model_id": (
                str(_json_safe_mapping(state.get("assistant")).get("default_model_id")).strip()
                if _json_safe_mapping(state.get("assistant")).get("default_model_id") not in (None, "")
                else None
            ),
        },
        "updated_at": state.get("updated_at") or datetime.now().isoformat(),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def is_secret_like_key(key: str) -> bool:
    normalized = str(key or "").upper()
    return any(marker in normalized for marker in SECRET_KEY_MARKERS)


def _coerce_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise SettingsValidationError(
        "Invalid feature flag value.",
        [{"field": f"feature_flags.{key}", "reason": "must_be_boolean"}],
    )


def _validate_env_key(key: str) -> str:
    normalized = str(key or "").strip().upper()
    if normalized in SAFE_ENVIRONMENT_VARIABLES:
        return normalized

    reason = "secret_key_blocked" if is_secret_like_key(normalized) else "unknown_key"
    raise SettingsValidationError(
        "Invalid environment variable override.",
        [{"field": f"env_overrides.{normalized or '<empty>'}", "reason": reason}],
    )


def update_settings_state(payload: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    state = load_settings_state(path)
    errors: list[dict[str, Any]] = []

    feature_flags = payload.get("feature_flags")
    if feature_flags is not None:
        if not isinstance(feature_flags, Mapping):
            errors.append({"field": "feature_flags", "reason": "must_be_object"})
        else:
            for key, value in feature_flags.items():
                flag_key = str(key or "").strip()
                if flag_key not in DEFAULT_FEATURE_FLAGS:
                    errors.append({"field": f"feature_flags.{flag_key}", "reason": "unknown_flag"})
                    continue
                try:
                    state["feature_flags"][flag_key] = _coerce_bool(value, flag_key)
                except SettingsValidationError as exc:
                    errors.extend(exc.errors)

    env_overrides = payload.get("env_overrides")
    if env_overrides is not None:
        if not isinstance(env_overrides, Mapping):
            errors.append({"field": "env_overrides", "reason": "must_be_object"})
        else:
            for key, value in env_overrides.items():
                try:
                    env_key = _validate_env_key(str(key))
                except SettingsValidationError as exc:
                    errors.extend(exc.errors)
                    continue
                if value is None or str(value).strip() == "":
                    state["env_overrides"].pop(env_key, None)
                    continue
                env_value = str(value).strip()
                if len(env_value) > 2048:
                    errors.append({"field": f"env_overrides.{env_key}", "reason": "value_too_long"})
                    continue
                state["env_overrides"][env_key] = env_value

    assistant = payload.get("assistant")
    if assistant is not None:
        if not isinstance(assistant, Mapping):
            errors.append({"field": "assistant", "reason": "must_be_object"})
        else:
            default_model_id = assistant.get("default_model_id")
            if default_model_id in (None, ""):
                state.setdefault("assistant", {})["default_model_id"] = None
            else:
                value = str(default_model_id).strip()
                if not value:
                    state.setdefault("assistant", {})["default_model_id"] = None
                elif len(value) > 128:
                    errors.append({"field": "assistant.default_model_id", "reason": "value_too_long"})
                else:
                    state.setdefault("assistant", {})["default_model_id"] = value

    if errors:
        raise SettingsValidationError("Invalid settings payload.", errors)

    state["updated_at"] = datetime.now().isoformat()
    save_settings_state(state, path)
    return state


def apply_env_overrides(state: Mapping[str, Any]) -> list[str]:
    applied: list[str] = []
    for key, value in _json_safe_mapping(state.get("env_overrides")).items():
        env_key = _validate_env_key(key)
        os.environ[env_key] = str(value)
        applied.append(env_key)
    return applied


def apply_saved_env_overrides(path: str | Path | None = None) -> list[str]:
    return apply_env_overrides(load_settings_state(path))


def apply_debug_logging(enabled: bool, loggers: Iterable[logging.Logger | None] = ()) -> int:
    level = logging.DEBUG if enabled else logging.INFO
    seen: set[int] = set()
    for logger in loggers:
        if logger is None or id(logger) in seen:
            continue
        seen.add(id(logger))
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    return level


def _redact_runtime_value(key: str, value: Any) -> Any:
    if is_secret_like_key(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(child_key): _redact_runtime_value(str(child_key), child_value) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [_redact_runtime_value(key, item) for item in value]
    return value


def redact_runtime_config(runtime_config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): _redact_runtime_value(str(key), value)
        for key, value in dict(runtime_config or {}).items()
    }


def build_client_settings(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = state or load_settings_state()
    flags = _json_safe_mapping(resolved.get("feature_flags"))
    return {
        "feature_flags": {
            "debug_mode": bool(flags.get("debug_mode", DEFAULT_FEATURE_FLAGS["debug_mode"])),
            "assistant_visible": bool(flags.get("assistant_visible", DEFAULT_FEATURE_FLAGS["assistant_visible"])),
        },
        "services": {
            "plotly_dashboard_url": os.getenv("AMANAJE_DASHBOARD_PUBLIC_URL", "http://localhost:8050"),
        },
        "assistant": _json_safe_mapping(resolved.get("assistant")),
        "updated_at": resolved.get("updated_at"),
    }


def build_settings_response(
    runtime_config: Mapping[str, Any] | None = None,
    *,
    state: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = state or load_settings_state(path)
    flags = build_client_settings(resolved)["feature_flags"]
    saved_overrides = _json_safe_mapping(resolved.get("env_overrides"))
    env_rows: list[dict[str, Any]] = []
    restart_keys: list[str] = []

    for key, metadata in SAFE_ENVIRONMENT_VARIABLES.items():
        saved_value = saved_overrides.get(key)
        process_value = os.getenv(key)
        effective_value = saved_value if saved_value is not None else process_value
        source = "saved_override" if saved_value is not None else ("process" if process_value is not None else "default")
        if saved_value is not None and metadata.get("restart_required"):
            restart_keys.append(key)
        env_rows.append(
            {
                "key": key,
                "label": metadata["label"],
                "description": metadata["description"],
                "category": metadata["category"],
                "value": effective_value if effective_value is not None else metadata.get("default", ""),
                "saved_override": saved_value,
                "process_value": process_value,
                "source": source,
                "editable": True,
                "restart_required": bool(metadata.get("restart_required")),
                "redacted": False,
            }
        )

    warnings: list[str] = []
    if restart_keys:
        warnings.append("Saved environment overrides require an application restart before every subsystem uses them.")

    return {
        "status": "ok",
        "feature_flags": flags,
        "environment_variables": env_rows,
        "runtime_config": redact_runtime_config(runtime_config),
        "assistant": _json_safe_mapping(resolved.get("assistant")),
        "restart_required": {
            "required": bool(restart_keys),
            "keys": restart_keys,
        },
        "warnings": warnings,
        "updated_at": resolved.get("updated_at"),
    }
