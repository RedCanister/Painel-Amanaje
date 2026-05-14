from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from app.utils.assistant_provider import redact_assistant_provider_config


DEFAULT_ASSISTANT_SERVER_BASE_URL = "http://localhost:8091"


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _model_value(model_record: Any, key: str, default: Any = None) -> Any:
    if isinstance(model_record, Mapping):
        return model_record.get(key, default)
    return getattr(model_record, key, default)


def _model_parameters(model_record: Any) -> dict[str, Any]:
    return _as_mapping(_model_value(model_record, "parameters", {}) or {})


def _assistant_config(model_record: Any) -> dict[str, Any]:
    return _as_mapping(_model_parameters(model_record).get("assistant") or {})


def assistant_runtime_base_url() -> str:
    configured = os.getenv("AMANAJE_ASSISTANT_SERVER_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    slm_base_url = os.getenv("AMANAJE_SLM_BASE_URL", "").strip()
    if slm_base_url:
        return slm_base_url.rstrip("/").rsplit("/v1", 1)[0].rstrip("/")
    return DEFAULT_ASSISTANT_SERVER_BASE_URL


def assistant_runtime_admin_token() -> str:
    return os.getenv("AMANAJE_ASSISTANT_SERVER_ADMIN_TOKEN", "").strip()


def assistant_runtime_config() -> dict[str, Any]:
    return {
        "base_url": assistant_runtime_base_url(),
        "admin_token_configured": bool(assistant_runtime_admin_token()),
        "chat_base_url": os.getenv("AMANAJE_SLM_BASE_URL", "").strip() or f"{assistant_runtime_base_url()}/v1",
    }


def _request_json(method: str, path: str, payload: Mapping[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    url = f"{assistant_runtime_base_url()}{path}"
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, default=str).encode("utf-8")
    token = assistant_runtime_admin_token()
    if token:
        headers["X-Amanaje-Assistant-Token"] = token
    request = UrlRequest(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"Assistant runtime returned HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Assistant runtime timed out.") from exc
    except URLError as exc:
        raise RuntimeError(f"Assistant runtime is unavailable: {exc.reason}") from exc
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Assistant runtime returned invalid JSON.") from exc
    return data if isinstance(data, dict) else {"status": "invalid_response", "payload": data}


def assistant_runtime_status(timeout: float = 5.0) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        payload = _request_json("GET", "/admin/models/status", timeout=timeout)
        status = "ok"
        error = None
    except Exception as exc:
        try:
            payload = _request_json("GET", "/health", timeout=timeout)
            status = "health_only"
            error = str(exc)
        except Exception as health_exc:
            payload = {}
            status = "unavailable"
            error = str(health_exc)
    return {
        "status": status,
        "runtime": payload,
        "config": assistant_runtime_config(),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "error": error,
    }


def assistant_model_runtime_payload(
    model_record: Any,
    provider_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assistant = _assistant_config(model_record)
    provider = _as_mapping(provider_config)
    generation_config = {
        **_as_mapping(assistant.get("generation_config") or {}),
        "temperature": provider.get("temperature", assistant.get("temperature", 0.2)),
        "max_new_tokens": provider.get("max_tokens", assistant.get("max_tokens", 1600)),
    }
    extracted_dir = assistant.get("extracted_dir") or assistant.get("bundle_dir")
    model_artifact_path = assistant.get("model_artifact_path") or _model_value(model_record, "path")
    tokenizer_path = assistant.get("tokenizer_path")
    if not extracted_dir and model_artifact_path:
        candidate = Path(str(model_artifact_path))
        extracted_dir = str(candidate.parent) if candidate.suffix else str(candidate)
    return {
        "assistant_model_id": _model_value(model_record, "id"),
        "assistant_model_name": _model_value(model_record, "name"),
        "model_name": str(provider.get("model_name") or assistant.get("model_name") or _model_value(model_record, "name") or "amanaje-assistant"),
        "model_version": str(provider.get("model_version") or assistant.get("model_version") or _model_value(model_record, "version") or "unversioned"),
        "bundle_dir": extracted_dir,
        "extracted_dir": extracted_dir,
        "model_artifact_path": model_artifact_path,
        "tokenizer_path": tokenizer_path,
        "chat_template_path": assistant.get("chat_template_path"),
        "chat_template": assistant.get("chat_template"),
        "generation_config": generation_config,
        "device": assistant.get("device") or "auto",
        "dtype": assistant.get("dtype") or "auto",
        "max_context_tokens": int(assistant.get("max_context_tokens") or provider.get("max_context_tokens") or 4096),
        "trust_remote_code": bool(assistant.get("trust_remote_code") is True),
        "provider_config": redact_assistant_provider_config(provider),
    }


def ensure_assistant_model_runtime_loaded(
    model_record: Any,
    provider_config: Mapping[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    target = assistant_model_runtime_payload(model_record, provider_config)
    started_at = time.perf_counter()
    try:
        status = _request_json("GET", "/admin/models/status", timeout=min(timeout, 10.0))
        active = _as_mapping(status.get("active_model"))
        if active.get("model_name") == target.get("model_name") and active.get("model_version") == target.get("model_version"):
            return {
                "status": "already_loaded",
                "target_model": target.get("model_name"),
                "target_version": target.get("model_version"),
                "runtime": status,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            }
        loaded = _request_json("POST", "/admin/models/load", target, timeout=timeout)
        return {
            "status": "loaded",
            "target_model": target.get("model_name"),
            "target_version": target.get("model_version"),
            "runtime": loaded,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }
    except Exception as exc:
        return {
            "status": "load_failed",
            "target_model": target.get("model_name"),
            "target_version": target.get("model_version"),
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }


def unload_assistant_runtime_model(timeout: float = 15.0) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        payload = _request_json("POST", "/admin/models/unload", timeout=timeout)
        return {
            "status": "unloaded",
            "runtime": payload,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }
    except Exception as exc:
        return {
            "status": "unload_failed",
            "error": str(exc),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }
