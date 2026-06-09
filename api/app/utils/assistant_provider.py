from __future__ import annotations

import json
import os
import time
import textwrap
import uuid
from datetime import datetime
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from app.models.assistant_objects import AssistantDraftRequest, WorkflowDraft
from app.utils.assistant_llmops import (
    ASSISTANT_MLFLOW_EXPERIMENT,
    LOCAL_MODEL_VERSION,
    PROMPT_TEMPLATE_VERSION,
    SAFETY_PROFILE_VERSION,
    EVALUATION_PROFILE,
    assistant_active_provider_name,
    assistant_alignment_contracts,
    assistant_provider_configs,
    assistant_provider_server_contract,
    assistant_slm_runtime_config,
    build_assistant_context_pack,
    build_slm_messages,
    redact_assistant_provider_config,
    stamp_workflow_draft,
)
from app.utils.assistant_safety import review_workflow_draft


class AssistantProvider(Protocol):
    name: str

    def create_draft(self, request: AssistantDraftRequest) -> WorkflowDraft:
        ...


class LocalWorkflowProvider:
    name = "local"

    def create_draft(self, request: AssistantDraftRequest) -> WorkflowDraft:
        started_at = time.perf_counter()
        draft_type = request.target_type or _infer_draft_type(request)
        context_pack = build_assistant_context_pack(request, draft_type)
        draft = _create_local_draft(request, draft_type)
        return stamp_workflow_draft(
            draft,
            context_pack=context_pack,
            provider_name=self.name,
            model_version=LOCAL_MODEL_VERSION,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )


class OpenAICompatibleRuntimeClient:
    def complete(self, messages: list[dict[str, str]], config: dict[str, object]) -> str:
        endpoint = str(config.get("chat_endpoint") or f"{str(config['base_url']).rstrip('/')}/chat/completions")
        body = {
            "model": config["model_name"],
            "messages": messages,
            "temperature": float(config.get("temperature") or 0.2),
            "max_tokens": int(config["max_tokens"]),
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        api_key = str(config.get("api_key") or "").strip()
        api_key_env = str(config.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = UrlRequest(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=float(config["timeout_seconds"])) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"SLM runtime returned HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError("SLM runtime timed out.") from exc
        except URLError as exc:
            raise RuntimeError(f"SLM runtime is unavailable: {exc.reason}") from exc
        data = json.loads(raw)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("SLM runtime response did not include choices[0].message.content.") from exc


class OpenAICompatibleAssistantProvider:
    name = "openai_compatible"

    _cache: dict[str, WorkflowDraft] = {}
    _last_status_by_provider: dict[str, dict[str, object]] = {}

    def __init__(self, name: str, config: dict[str, object] | None = None):
        self.name = name
        self.config = dict(config or {})
        self.runtime_client = OpenAICompatibleRuntimeClient()

    def create_draft(self, request: AssistantDraftRequest) -> WorkflowDraft:
        started_at = time.perf_counter()
        draft_type = request.target_type or _infer_draft_type(request)
        context_pack = build_assistant_context_pack(request, draft_type)
        model_version = str(self.config.get("model_version") or self.name)
        allow_fallback = _coerce_provider_bool(self.config.get("allow_fallback"), True)
        cache_key = f"{self.name}:{model_version}:{context_pack.pack_hash}"
        cached = self._cache.get(cache_key)
        cached_fallback = bool(cached.provider_metadata.get("fallback_provider")) if cached is not None else False
        if cached is not None and (allow_fallback or not cached_fallback):
            self._set_status(
                self.name,
                available=True,
                fallback_active=bool(cached.provider_metadata.get("fallback_provider")),
                last_event="cache_hit",
                context_pack_hash=context_pack.pack_hash,
                latency_ms=0.0,
            )
            return cached.model_copy(
                update={
                    "draft_id": f"draft_{uuid.uuid4().hex[:12]}",
                    "created_at": datetime.now(),
                    "provider_metadata": {**cached.provider_metadata, "cache_hit": True},
                },
                deep=True,
            )

        fallback_reason: str | None = None
        try:
            draft = self._create_structured_draft(request, draft_type, context_pack)
        except Exception as exc:
            fallback_reason = str(exc)
            draft = None

        if draft is None:
            fallback_reason = fallback_reason or "No external SLM runtime is configured; using deterministic local fallback."
            if not allow_fallback:
                self._set_status(
                    self.name,
                    available=False,
                    fallback_active=False,
                    last_event="runtime_unavailable",
                    context_pack_hash=context_pack.pack_hash,
                    error=fallback_reason,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                )
                raise RuntimeError(fallback_reason)
            draft = _create_local_draft(request, draft_type)
            self._set_status(
                self.name,
                available=False,
                fallback_active=True,
                last_event="fallback",
                context_pack_hash=context_pack.pack_hash,
                error=fallback_reason,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )
        else:
            self._set_status(
                self.name,
                available=True,
                fallback_active=False,
                last_event="slm_success",
                context_pack_hash=context_pack.pack_hash,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )

        stamped = stamp_workflow_draft(
            draft,
            context_pack=context_pack,
            provider_name=self.name,
            model_version=model_version,
            fallback_provider="local" if fallback_reason else None,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            extra_metadata={
                "runtime": "openai-compatible" if fallback_reason is None else "bootstrap",
                "fallback_reason": fallback_reason,
                "serving_mode": "separate-service-ready",
                "provider_config": redact_assistant_provider_config(self.config),
                "slm_config": redact_assistant_provider_config(self.config),
            },
        )
        self._cache[cache_key] = stamped
        return stamped

    def _create_structured_draft(
        self,
        request: AssistantDraftRequest,
        draft_type: str,
        context_pack: object,
    ) -> WorkflowDraft | None:
        config = dict(self.config or {})
        if not config["enabled"] and not request.constraints.get("use_configured_slm_runtime"):
            return None
        messages = build_slm_messages(context_pack)  # type: ignore[arg-type]
        content = self._call_openai_compatible_runtime(messages, config)
        payload = _extract_json_object(content)
        if isinstance(payload, dict) and "draft" in payload and isinstance(payload["draft"], dict):
            payload = dict(payload["draft"])
        draft = WorkflowDraft.model_validate(payload)
        draft = draft.model_copy(
            update={
                "session_id": request.session_id,
                "prompt": draft.prompt or request.prompt,
                "provider": self.name,
                "draft_type": draft_type,
                "execution_profile": draft.execution_profile or draft_type,
                "context": {**dict(request.context or {}), **dict(draft.context or {})},
            },
            deep=True,
        )
        review = review_workflow_draft(draft)
        if not review.safety.ok:
            messages = "; ".join(violation.message for violation in review.safety.violations)
            raise RuntimeError(f"SLM draft failed safety review: {messages}")
        return draft

    def _call_openai_compatible_runtime(self, messages: list[dict[str, str]], config: dict[str, object]) -> str:
        return self.runtime_client.complete(messages, config)

    @classmethod
    def _set_status(cls, provider_name: str, **updates: object) -> None:
        configs = assistant_provider_configs()
        config = configs.get(provider_name, {})
        previous = dict(cls._last_status_by_provider.get(provider_name) or {})
        cls._last_status_by_provider[provider_name] = {
            **previous,
            **updates,
            "provider": provider_name,
            "model_version": config.get("model_version"),
            "config": redact_assistant_provider_config(config),
            "updated_at": datetime.now().isoformat(),
        }

    @classmethod
    def status_for(cls, provider_name: str, config: dict[str, object]) -> dict[str, object]:
        last_status = dict(
            cls._last_status_by_provider.get(provider_name)
            or {
                "provider": provider_name,
                "available": False,
                "fallback_active": True,
                "last_event": "not_called",
            }
        )
        return {
            **last_status,
            "provider": provider_name,
            "type": config.get("type"),
            "enabled": bool(config.get("enabled")),
            "model_version": config.get("model_version"),
            "capabilities": config.get("capabilities", {}),
            "supported_draft_types": config.get("supported_draft_types", []),
            "supports_json_mode": bool(config.get("supports_json_mode")),
            "supports_remote_auth": bool(config.get("supports_remote_auth")),
            "health_url": config.get("health_url"),
            "config": redact_assistant_provider_config(config),
            "cache_size": len(cls._cache),
        }


class AmanajeSLMProvider(OpenAICompatibleAssistantProvider):
    name = "amanaje_slm"

    def __init__(self, config: dict[str, object] | None = None):
        super().__init__("amanaje_slm", config or assistant_slm_runtime_config())

    @classmethod
    def status(cls) -> dict[str, object]:
        return cls.status_for("amanaje_slm", assistant_slm_runtime_config())


def get_assistant_provider(name: str | None = None) -> AssistantProvider:
    normalized = (name or "auto").strip().lower()
    if normalized in {"", "auto", "default", "active"}:
        normalized = assistant_active_provider_name()
    configs = assistant_provider_configs()
    if normalized == "local":
        return LocalWorkflowProvider()
    if normalized in {"slm", "amanaje-local"}:
        normalized = "amanaje_slm"
    config = configs.get(normalized)
    if not config:
        available = ", ".join(sorted(configs))
        raise ValueError(f"Assistant provider '{normalized}' is not configured. Available providers: {available}")
    provider_type = str(config.get("type") or "").lower()
    if provider_type == "local":
        return LocalWorkflowProvider()
    if provider_type == "openai_compatible":
        if normalized == "amanaje_slm":
            return AmanajeSLMProvider(config)
        return OpenAICompatibleAssistantProvider(normalized, config)
    raise ValueError(f"Assistant provider '{normalized}' has unsupported type '{provider_type}'.")


def _model_mapping_value(model_record: Any, key: str, default: Any = None) -> Any:
    if isinstance(model_record, Mapping):
        return model_record.get(key, default)
    return getattr(model_record, key, default)


def _assistant_model_slug(model_record: Any) -> str:
    raw_id = _model_mapping_value(model_record, "id")
    raw_name = _model_mapping_value(model_record, "name", "assistant_model")
    label = raw_id if raw_id not in (None, "") else raw_name
    text = "".join(character.lower() if character.isalnum() else "_" for character in str(label))
    return "_".join(part for part in text.split("_") if part) or "assistant_model"


def _coerce_provider_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def assistant_model_to_provider_config(
    model_record: Any,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a registry AssistantModel/LearningModel record into provider config."""

    parameters = _model_mapping_value(model_record, "parameters", {}) or {}
    if not isinstance(parameters, Mapping):
        parameters = {}
    assistant_config = dict(parameters.get("assistant") or {})
    runtime_config = dict(assistant_config.get("runtime_config") or {})
    behavior_profile = dict(assistant_config.get("behavior_profile") or {})
    merged = {
        **dict(assistant_provider_configs().get("amanaje_slm", {})),
        **assistant_config,
        **runtime_config,
        **dict(overrides or {}),
    }
    provider_type = str(
        merged.get("provider_type")
        or merged.get("type")
        or _model_mapping_value(model_record, "provider_type")
        or "openai_compatible"
    ).strip().lower()
    assistant_role = str(merged.get("assistant_role") or "generator").strip().lower()
    if assistant_role not in {"generator", "embedding", "dual"}:
        assistant_role = "generator"
    if assistant_role == "embedding" and provider_type in {"", "openai", "http", "openai_compatible", "embedding", "embeddings"}:
        provider_type = "openai_compatible_embeddings"
    if provider_type in {"slm", "llm", "http", "openai"}:
        provider_type = "openai_compatible"
    model_name = (
        merged.get("model_name")
        or _model_mapping_value(model_record, "model_name")
        or _model_mapping_value(model_record, "name")
        or "amanaje-assistant"
    )
    model_version = (
        merged.get("model_version")
        or _model_mapping_value(model_record, "model_version")
        or f"registry-assistant-model-{_model_mapping_value(model_record, 'id', 'unversioned')}"
    )
    temperature = merged.get("temperature", behavior_profile.get("temperature", 0.2))
    runtime_kind = str(merged.get("runtime_kind") or runtime_config.get("runtime_kind") or "").strip()
    base_url = merged.get("base_url") or os.getenv("AMANAJE_SLM_BASE_URL") or "http://assistant-server:8080/v1"
    if runtime_kind == "pytorch_hf_server" and str(base_url).rstrip("/") == "http://localhost:8080/v1":
        base_url = os.getenv("AMANAJE_SLM_BASE_URL") or "http://assistant-server:8080/v1"
    config = {
        **merged,
        "type": provider_type,
        "provider_type": provider_type,
        "assistant_role": assistant_role,
        "enabled": _coerce_provider_bool(merged.get("enabled"), True),
        "base_url": base_url,
        "chat_endpoint": merged.get("chat_endpoint"),
        "embeddings_endpoint": merged.get("embeddings_endpoint"),
        "health_url": merged.get("health_url"),
        "model_name": str(model_name),
        "model_version": str(model_version),
        "timeout_seconds": float(merged.get("timeout_seconds") or 20),
        "max_tokens": int(merged.get("max_tokens") or 1600),
        "temperature": float(temperature or 0.2),
        "allow_fallback": _coerce_provider_bool(merged.get("allow_fallback"), False),
        "api_key": merged.get("api_key"),
        "api_key_env": merged.get("api_key_env"),
        "supports_json_mode": _coerce_provider_bool(merged.get("supports_json_mode"), True),
        "supports_remote_auth": _coerce_provider_bool(
            merged.get("supports_remote_auth"),
            bool(merged.get("api_key") or merged.get("api_key_env")),
        ),
        "supported_draft_types": list(
            merged.get("supported_draft_types")
            or (["embedding_retrieval"] if assistant_role == "embedding" else None)
            or [
                "dataset_generation",
                "model_generation",
                "registry_object",
                "feature_operations",
                "training_run",
                "study",
            ]
        ),
        "embedding_dimension": int(merged.get("embedding_dimension") or 128),
        "pooling_strategy": str(merged.get("pooling_strategy") or "mean"),
        "max_sequence_tokens": int(merged.get("max_sequence_tokens") or merged.get("max_context_tokens") or 512),
        "query_instruction": merged.get("query_instruction"),
        "document_instruction": merged.get("document_instruction"),
        "normalize_embeddings": _coerce_provider_bool(merged.get("normalize_embeddings"), True),
        "capabilities": {
            "registry_backed": True,
            "assistant_model_id": _model_mapping_value(model_record, "id"),
            "assistant_model_name": _model_mapping_value(model_record, "name"),
            "assistant_role": assistant_role,
            "embedding_body": assistant_role in {"embedding", "dual"},
            "behavior_profile": behavior_profile,
            "reference_policy": dict(assistant_config.get("reference_policy") or {}),
            **dict(merged.get("capabilities") or {}),
        },
        "registry_model": {
            "id": _model_mapping_value(model_record, "id"),
            "name": _model_mapping_value(model_record, "name"),
            "path": _model_mapping_value(model_record, "path"),
            "model_type": _model_mapping_value(model_record, "model_type"),
        },
    }
    return config


def get_assistant_provider_for_model(
    model_record: Any,
    overrides: Mapping[str, Any] | None = None,
) -> AssistantProvider:
    config = assistant_model_to_provider_config(model_record, overrides=overrides)
    provider_type = str(config.get("type") or "").lower()
    if provider_type == "local":
        return LocalWorkflowProvider()
    if provider_type == "openai_compatible":
        return OpenAICompatibleAssistantProvider(f"assistant_model_{_assistant_model_slug(model_record)}", config)
    raise ValueError(f"Assistant model provider type '{provider_type}' is not supported.")


def get_assistant_provider_status() -> dict[str, object]:
    active_provider = assistant_active_provider_name()
    configs = assistant_provider_configs()
    providers: dict[str, object] = {}
    for provider_name, config in configs.items():
        provider_type = str(config.get("type") or "").lower()
        if provider_type == "local":
            providers[provider_name] = {
                "provider": provider_name,
                "type": "local",
                "available": True,
                "enabled": bool(config.get("enabled", True)),
                "fallback_active": False,
                "model_version": config.get("model_version", LOCAL_MODEL_VERSION),
                "fallback_provider": None,
                "capabilities": config.get("capabilities", {}),
                "supported_draft_types": config.get("supported_draft_types", []),
                "config": redact_assistant_provider_config(config),
            }
        elif provider_type == "openai_compatible":
            providers[provider_name] = OpenAICompatibleAssistantProvider.status_for(provider_name, config)
        else:
            providers[provider_name] = {
                "provider": provider_name,
                "type": provider_type,
                "available": False,
                "enabled": bool(config.get("enabled")),
                "error": f"Unsupported provider type '{provider_type}'.",
                "config": redact_assistant_provider_config(config),
            }
    return {
        "active_provider": active_provider,
        "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
        "alignment_contracts": assistant_alignment_contracts(),
        "server_contract": assistant_provider_server_contract(),
        "providers": providers,
    }


def test_assistant_provider_contract(provider_name: str | None = None) -> dict[str, object]:
    started_at = time.perf_counter()
    requested_provider = provider_name or "auto"
    provider = get_assistant_provider(requested_provider)
    request = AssistantDraftRequest(
        prompt="Return a minimal valid WorkflowDraft contract response for a synthetic dataset.",
        target_type="dataset_generation",
        provider=provider.name,
        constraints={"provider_diagnostic": True, "max_references": 0},
    )
    draft = provider.create_draft(request)
    review = review_workflow_draft(draft)
    fallback_used = bool(draft.provider_metadata.get("fallback_provider"))
    return {
        "status": "fallback" if fallback_used else ("passed" if review.safety.ok else "failed"),
        "requested_provider": requested_provider,
        "provider": provider.name,
        "draft_type": draft.draft_type,
        "model_version": draft.model_version,
        "context_pack_hash": draft.context_pack_hash,
        "workflow_draft_valid": bool(draft.draft_id and draft.draft_type and draft.title),
        "safety_ok": bool(review.safety.ok),
        "fallback_used": fallback_used,
        "fallback_provider": draft.provider_metadata.get("fallback_provider"),
        "fallback_reason": draft.provider_metadata.get("fallback_reason"),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "review_status": review.status,
    }


def test_assistant_model_contract(model_record: Any, overrides: Mapping[str, Any] | None = None) -> dict[str, object]:
    started_at = time.perf_counter()
    provider = get_assistant_provider_for_model(model_record, overrides=overrides)
    request = AssistantDraftRequest(
        prompt="Return a minimal valid WorkflowDraft contract response for a registry-selected assistant model.",
        target_type="dataset_generation",
        provider=provider.name,
        assistant_model_id=_model_mapping_value(model_record, "id"),
        constraints={"provider_diagnostic": True, "max_references": 0},
    )
    draft = provider.create_draft(request)
    review = review_workflow_draft(draft)
    fallback_used = bool(draft.provider_metadata.get("fallback_provider"))
    return {
        "status": "fallback" if fallback_used else ("passed" if review.safety.ok else "failed"),
        "assistant_model_id": _model_mapping_value(model_record, "id"),
        "assistant_model_name": _model_mapping_value(model_record, "name"),
        "provider": provider.name,
        "draft_type": draft.draft_type,
        "model_version": draft.model_version,
        "context_pack_hash": draft.context_pack_hash,
        "workflow_draft_valid": bool(draft.draft_id and draft.draft_type and draft.title),
        "safety_ok": bool(review.safety.ok),
        "fallback_used": fallback_used,
        "fallback_provider": draft.provider_metadata.get("fallback_provider"),
        "fallback_reason": draft.provider_metadata.get("fallback_reason"),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        "review_status": review.status,
    }


def _extract_json_object(content: str) -> dict[str, object]:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SLM runtime returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SLM runtime JSON response must be an object.")
    return payload


def _create_local_draft(request: AssistantDraftRequest, draft_type: str) -> WorkflowDraft:
    if draft_type == "model_generation":
        return _model_generation_draft(request)
    if draft_type == "feature_operations":
        return _feature_operations_draft(request)
    if draft_type == "training_run":
        return _training_draft(request)
    if draft_type == "study":
        return _study_draft(request)
    if draft_type == "registry_object":
        return _registry_draft(request)
    return _dataset_generation_draft(request)


def _infer_draft_type(request: AssistantDraftRequest) -> str:
    context = dict(request.context or {})
    prompt = f"{request.workflow_goal or ''} {request.prompt} {context.get('operationId') or ''}".lower()
    if any(token in prompt for token in ("feature", "column", "rename", "transform", "fill missing")):
        return "feature_operations"
    if any(token in prompt for token in ("model", "joblib", "sklearn", "estimator", "random forest", "regressor")):
        return "model_generation"
    if any(token in prompt for token in ("train", "training", "fit model", "launch run")):
        return "training_run"
    if any(token in prompt for token in ("study", "optuna", "hyperparameter", "optimization")):
        return "study"
    if any(token in prompt for token in ("registry", "register", "object", "metadata")):
        return "registry_object"
    return "dataset_generation"


def _dataset_generation_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    prompt_comment = "\n".join(f"# {line.strip()}" for line in request.prompt.splitlines() if line.strip())
    code = textwrap.dedent(
        f"""
        # Assistant dataset draft
        {prompt_comment}

        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        row_count = 100
        df = pd.DataFrame({{
            "feature_a": rng.normal(10, 2, row_count).round(4),
            "feature_b": rng.integers(0, 5, row_count),
            "segment": rng.choice(["baseline", "growth", "risk"], row_count),
        }})
        df["target"] = (df["feature_a"] * 0.7 + df["feature_b"] * 1.5 + rng.normal(0, 0.5, row_count)).round(4)

        csv_text = df.to_csv(index=False)
        name = "assistant_generated_dataset"
        description = "Dataset drafted by the Amanaje assistant."
        object_type = "dataset"
        dataset_type = "dataset"
        version = 1
        connection_string = ""
        """
    ).strip()
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="dataset_generation",
        title="Generated Dataset Draft",
        summary="Creates a tabular dataset and exposes it as csv_text for the existing upload/create flow.",
        prompt=request.prompt,
        provider=request.provider,
        code=code,
        execution_profile="dataset_generation",
        materialized_outputs=["csv_text", "name", "description", "object_type", "dataset_type"],
        assumptions=[
            "The generated dataset is synthetic.",
            "The draft only prepares csv_text; it does not write files or registry rows.",
        ],
        risks=["Synthetic data may not represent the target production distribution."],
        context=request.context,
        next_actions=["Review safety report", "Approve draft", "Execute through the guarded execution path"],
    )


def _model_generation_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    prompt_comment = "\n".join(f"# {line.strip()}" for line in request.prompt.splitlines() if line.strip())
    code = textwrap.dedent(
        f"""
        # Assistant model draft
        {prompt_comment}

        import base64
        import io
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=80, random_state=42)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        joblib_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
        name = "assistant_generated_model"
        description = "Scikit-learn model artifact drafted by the Amanaje assistant."
        object_type = "learning_model"
        path = "generated/assistant_generated_model.joblib"
        version = 1
        model_type = "supervised_model"
        parameters = {{"framework": "sklearn", "estimator_class": "RandomForestRegressor", "n_estimators": 80, "random_state": 42}}
        metrics = {{"task": "regression", "status": "untrained_template"}}
        reference_data = "assistant_generated_dataset.csv"
        input_features = ["feature_a", "feature_b"]
        output_features = ["target"]
        is_trained = False
        is_tested = False
        is_deployed = False
        """
    ).strip()
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="model_generation",
        title="Generated Model Draft",
        summary="Creates a scikit-learn joblib artifact and matching LearningModel metadata for the Create workspace.",
        prompt=request.prompt,
        provider=request.provider,
        code=code,
        execution_profile="model_generation",
        materialized_outputs=["joblib_bytes", "name", "parameters", "metrics", "input_features", "output_features"],
        assumptions=[
            "The first assistant-generated model artifact is a scikit-learn joblib template.",
            "The generated artifact is intended for registry creation, not immediate production deployment.",
        ],
        risks=["The model is not trained until it is paired with a dataset and submitted to the training workflow."],
        context=request.context,
        next_actions=["Review safety report", "Run through guarded execution", "Create the model through the existing upload backend"],
    )


def _feature_operations_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    operations = list(context.get("transforms") or context.get("operations") or [])
    if not operations:
        operations = [{"type": "limit", "value": int(context.get("limitRows") or context.get("limit_rows") or 200)}]
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="feature_operations",
        title="Feature Transformation Draft",
        summary="Prepares visual feature operations for /features/preview before materialization.",
        prompt=request.prompt,
        provider=request.provider,
        execution_profile="feature_operations",
        operations=operations,
        artifacts={"preview_endpoint": "/features/preview", "materialize_endpoint": "/features/materialize"},
        assumptions=["Existing Feature Workspace operation schemas remain the source of truth."],
        risks=["Column names must match the selected dataset before preview can succeed."],
        context=context,
        next_actions=["Preview transforms", "Approve materialization", "Register the derived dataset"],
    )


def _training_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    training_request = {
        "modelId": context.get("modelId") or context.get("model_id"),
        "datasetId": context.get("datasetId") or context.get("dataset_id"),
        "inputType": "Assistant",
        "studyId": context.get("studyId") or context.get("study_id"),
        "parameters": dict(context.get("parameters") or {}),
    }
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="training_run",
        title="Training Run Draft",
        summary="Prepares a reviewed training submission using the existing async run ledger.",
        prompt=request.prompt,
        provider=request.provider,
        execution_profile="training_run",
        training_request=training_request,
        assumptions=["Training will use the selected registry model and dataset after preflight validation."],
        risks=["Training cannot start until modelId and datasetId are present and schema preflight passes."],
        context=context,
        next_actions=["Review selected model/dataset", "Approve training submission", "Track /runs/get/{run_id}"],
    )


def _study_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    study_request = {
        "studyId": context.get("studyId") or context.get("study_id"),
        "nTrials": context.get("nTrials") or context.get("n_trials") or 10,
        "plotResults": context.get("plotResults", True),
        "parameters": dict(context.get("parameters") or {}),
    }
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="study",
        title="Optimization Study Draft",
        summary="Prepares a reviewed Optuna study request for the existing study runner.",
        prompt=request.prompt,
        provider=request.provider,
        execution_profile="study",
        study_request=study_request,
        assumptions=["The existing StudyModel registry record defines the model and dataset pairing."],
        risks=["Search-space quality depends on the current StudyModel parameters."],
        context=context,
        next_actions=["Review study settings", "Approve optimization", "Track the study run ledger"],
    )


def _registry_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    registry_type = str(context.get("registryType") or context.get("modelType") or "DatasetModel")
    payload = dict(context.get("registry_payload") or {})
    if not payload:
        payload = _default_registry_payload(registry_type, context)
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="registry_object",
        title=f"{registry_type} Draft",
        summary="Prepares registry metadata for review before the user submits the manual create form.",
        prompt=request.prompt,
        provider=request.provider,
        registry_type=registry_type,
        registry_payload=payload,
        form_payload=payload,
        execution_profile="registry_object",
        materialized_outputs=["registry_payload"],
        assumptions=["The target registry route will perform final Pydantic and ORM validation."],
        risks=["References to datasets, models, or studies must exist before the object can be saved."],
        context=context,
        next_actions=["Review metadata", "Approve persistence", "Check dependency report before deletion"],
    )


def _slug(value: object, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    text = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in text.split("_") if part) or fallback


def _base_registry_payload(registry_type: str, context: dict) -> dict:
    name = _slug(context.get("name") or context.get("objectName"), f"assistant_{registry_type.lower()}")
    return {
        "name": name,
        "description": context.get("description") or f"{registry_type} drafted by the Amanaje assistant.",
        "size": float(context.get("size") or context.get("objectSize") or 0),
        "date": context.get("date"),
        "version": int(context.get("version") or context.get("objectVersion") or 1),
        "history": [{"operation": "assistant_draft"}],
    }


def _default_registry_payload(registry_type: str, context: dict) -> dict:
    payload = _base_registry_payload(registry_type, context)
    name = payload["name"]

    if registry_type == "AssistantModel":
        return {
            **payload,
            "object_type": "learning_model",
            "path": context.get("path") or context.get("objectPath") or f"runtime_artifacts/assistant_models/{name}",
            "model_type": "assistant_model",
            "parameters": {
                "assistant": {
                    "provider_type": context.get("provider_type") or "openai_compatible",
                    "base_url": context.get("base_url") or "http://localhost:8080/v1",
                    "model_name": context.get("model_name") or name,
                    "model_version": context.get("model_version") or "registry-assistant-draft-0.1",
                    "temperature": float(context.get("temperature") or 0.2),
                    "max_tokens": int(context.get("max_tokens") or 1600),
                    "timeout_seconds": float(context.get("timeout_seconds") or 20),
                    "enabled": bool(context.get("enabled", True)),
                    "supported_draft_types": [
                        "dataset_generation",
                        "model_generation",
                        "registry_object",
                        "feature_operations",
                        "training_run",
                        "study",
                    ],
                    "behavior_profile": dict(context.get("behavior_profile") or {}),
                    "reference_policy": dict(context.get("reference_policy") or {"mode": "automatic_internal_repository"}),
                    "training_profile": dict(context.get("training_profile") or {"curation": "approved_and_corrected_drafts"}),
                }
            },
            "metrics": dict(context.get("metrics") or {"status": "draft", "evaluation_profile": "pending"}),
            "reference_data": context.get("reference_data") or "runtime_artifacts/assistant_datasets/interactions.jsonl",
            "input_features": ["prompt", "context_pack", "target_type"],
            "output_features": ["workflow_draft"],
            "is_trained": bool(context.get("is_trained") or False),
            "is_tested": bool(context.get("is_tested") or False),
            "is_deployed": bool(context.get("is_deployed") or False),
        }

    if registry_type == "AssistantTrainingDatasetModel":
        return {
            **payload,
            "object_type": "dataset",
            "path": context.get("path") or context.get("objectPath") or f"runtime_artifacts/assistant_datasets/snapshots/{name}.csv",
            "dataset_type": "assistant_training_dataset",
            "shape": list(context.get("shape") or [0, 16]),
            "has_features": True,
            "features_list": list(
                context.get("features_list")
                or [
                    "source_type",
                    "label",
                    "workflow_type",
                    "prompt",
                    "context_pack_hash",
                    "draft",
                    "review",
                    "user_edits",
                    "quality_signals",
                    "assistant_success",
                    "assistant_quality_label",
                ]
            ),
            "connection_string": context.get("connection_string") or context.get("manifest_path") or "",
        }

    if registry_type == "LearningModel":
        return {
            **payload,
            "object_type": "learning_model",
            "path": context.get("path") or context.get("objectPath") or f"models/{name}.joblib",
            "model_type": context.get("model_type") or "supervised_model",
            "parameters": dict(context.get("parameters") or {"framework": "sklearn", "estimator_class": "RandomForestRegressor"}),
            "metrics": dict(context.get("metrics") or {"status": "draft"}),
            "reference_data": context.get("reference_data") or context.get("referenceData") or "",
            "input_features": list(context.get("input_features") or ["feature_a", "feature_b"]),
            "output_features": list(context.get("output_features") or ["target"]),
            "is_trained": bool(context.get("is_trained") or False),
            "is_tested": bool(context.get("is_tested") or False),
            "is_deployed": bool(context.get("is_deployed") or False),
        }

    if registry_type == "InferenceModel":
        learning_model_id = context.get("learning_model_id") or context.get("learningModelId") or context.get("model_id") or 1
        dataset_id = context.get("dataset_id") or context.get("datasetId") or 1
        return {
            **payload,
            "object_type": "inference_model",
            "path": context.get("path") or context.get("objectPath") or f"runtime_artifacts/inference/{name}.json",
            "learning_model_id": int(learning_model_id),
            "dataset_id": int(dataset_id),
            "input_features": list(context.get("input_features") or ["feature_a", "feature_b"]),
            "output_features": list(context.get("output_features") or ["target"]),
            "inference_params": dict(context.get("inference_params") or {"status": "draft", "source": "assistant"}),
        }

    if registry_type == "StudyModel":
        learning_model_id = context.get("learning_model_id") or context.get("learningModelId") or context.get("model_id") or 1
        dataset_id = context.get("dataset_id") or context.get("datasetId") or 1
        return {
            **payload,
            "object_type": "study_model",
            "path": context.get("path") or context.get("objectPath") or f"runtime_artifacts/optuna/{name}.json",
            "learning_model_id": int(learning_model_id),
            "dataset_id": int(dataset_id),
            "sampler": context.get("sampler") or "TPESampler",
            "objective": context.get("objective") or "maximize_accuracy",
            "best_trial": dict(context.get("best_trial") or {}),
            "best_params": dict(context.get("best_params") or {}),
            "study_params": dict(context.get("study_params") or {"n_trials": 20, "direction": "maximize", "framework": "sklearn"}),
        }

    if registry_type == "CodeModel":
        return {
            **payload,
            "object_type": "code_model",
            "path": context.get("path") or context.get("objectPath") or f"generated/{name}.py",
            "variables": dict(context.get("variables") or {"source": "assistant"}),
            "code": dict(context.get("code") or {"language": "python", "script": "# Assistant-drafted code document"}),
        }

    return {
        **payload,
        "object_type": "dataset",
        "path": context.get("path") or context.get("objectPath") or f"data/datasets/{name}.csv",
        "dataset_type": context.get("dataset_type") or "dataset",
        "shape": list(context.get("shape") or [100, 4]),
        "has_features": bool(context.get("has_features") if "has_features" in context else True),
        "features_list": list(context.get("features_list") or ["feature_a", "feature_b", "segment", "target"]),
        "connection_string": context.get("connection_string") or "",
    }
