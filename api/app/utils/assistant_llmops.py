from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from app.models.assistant_objects import (
    AssistantContextPack,
    AssistantDraftRequest,
    AssistantReferenceRequest,
    ContentReference,
    GenerationProvenance,
    ReviewDecision,
    WorkflowDraft,
)
from app.utils.assistant_safety import SAFETY_PROFILES
from app.utils.io import ensure_dir, save_json
from app.utils.mlflow_utils import (
    MLFLOW_AVAILABLE,
    active_run_id,
    log_artifact,
    log_json,
    log_metrics,
    log_params,
    managed_run,
    set_tags,
)


CONTEXT_PACK_VERSION = "assistant-context-pack-v1"
PROMPT_TEMPLATE_VERSION = "amanaje-assistant-template-v1"
SAFETY_PROFILE_VERSION = "assistant-safety-v1"
EVALUATION_PROFILE = "assistant-golden-v1"
ASSISTANT_MLFLOW_EXPERIMENT = os.getenv("AMANAJE_ASSISTANT_MLFLOW_EXPERIMENT", "assistant/amanaje_slm")
LOCAL_MODEL_VERSION = "local-workflow-provider-0.2"
SLM_MODEL_VERSION = os.getenv("AMANAJE_SLM_MODEL_VERSION", "amanaje-slm-bootstrap-0.1")
SLM_DEFAULT_BASE_URL = "http://localhost:8080/v1"
SLM_DEFAULT_MODEL_NAME = "amanaje-assistant"
ASSISTANT_ACTIVE_PROVIDER_ENV = "AMANAJE_ASSISTANT_ACTIVE_PROVIDER"
ASSISTANT_PROVIDER_CONFIGS_ENV = "AMANAJE_ASSISTANT_PROVIDER_CONFIGS"
ASSISTANT_PROVIDER_CONFIG_PATH_ENV = "AMANAJE_ASSISTANT_PROVIDER_CONFIG_PATH"
ASSISTANT_REFERENCE_SOURCE_ENV = "AMANAJE_ASSISTANT_REFERENCE_PATHS"
ASSISTANT_DRAFT_TYPE_LIST = [
    "dataset_generation",
    "feature_operations",
    "model_generation",
    "registry_object",
    "study",
    "training_run",
]
ASSISTANT_REFERENCE_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SENSITIVE_KEY_PATTERN = re.compile(r"(api[_-]?key|authorization|bearer|password|secret|token)", re.IGNORECASE)
_SENSITIVE_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|password|secret|token)\s*[:=]\s*['\"]?[^'\"\s,;]+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"),
)

ASSISTANT_ALIGNMENT_CONTRACTS: dict[str, str] = {
    "output_alignment": "Every response must validate as WorkflowDraft and include the required workflow outputs.",
    "safety_alignment": "Generated code must pass assistant_safety.py before approval or execution.",
    "registry_alignment": "Registry drafts must map to the supported manual registry object payloads.",
    "dependency_alignment": "Drafted code must stay inside allowed imports, dependencies, and artifact formats.",
    "mlflow_alignment": "Drafts, evals, context packs, prompt versions, and model versions must be traceable.",
}

ASSISTANT_DRAFT_WRITING_GUIDE: dict[str, Any] = {
    "voice": "Concrete, project-aware, concise, and operational.",
    "titles": "Name the artifact or workflow outcome, not the assistant action.",
    "summaries": "State what the draft creates, how it fits Painel Amanaje, and what remains for user approval.",
    "metadata": "Prefer registry-compatible names, versions, paths, feature names, and dependency declarations.",
    "risks": "Name practical limitations such as synthetic distribution mismatch, missing IDs, or untrained artifacts.",
    "next_actions": "Use short workflow actions that match reviewed draft, guarded run, upload, or registry submit steps.",
    "references": "Use internal reference summaries and cite their reference IDs in metadata when they shape the draft.",
    "originality": "Generate project-specific assets from the request and internal references; do not copy raw reference text.",
}

ASSISTANT_DRAFT_QUALITY_RUBRIC: dict[str, str] = {
    "title_quality": "Title is specific to the requested artifact or workflow.",
    "summary_quality": "Summary is not generic and explains the generated draft's project fit.",
    "metadata_quality": "Registry metadata and materialized outputs match project schemas.",
    "risk_quality": "Risks and assumptions are concrete enough for user review.",
    "reference_quality": "Internal references are summarized, traceable, and not copied raw into prompts.",
}

ASSISTANT_GOLDEN_EVALS: tuple[dict[str, Any], ...] = (
    {"target_type": "dataset_generation", "required_outputs": ["csv_text"]},
    {"target_type": "model_generation", "required_outputs": ["joblib_bytes"]},
    {"target_type": "registry_object", "required_outputs": ["registry_payload"]},
    {"target_type": "training_run", "required_outputs": ["training_request"]},
    {"target_type": "study", "required_outputs": ["study_request"]},
)


SLM_PROMPT_TEMPLATES: dict[str, str] = {
    "dataset_generation": (
        "Draft safe Python code that creates synthetic tabular data and exposes csv_text plus dataset metadata. "
        "Use only allowed dependencies, align names with trusted internal references, and never write files."
    ),
    "model_generation": (
        "Draft safe Python code that creates a scikit-learn joblib artifact as base64 joblib_bytes plus LearningModel metadata. "
        "The artifact is a template, must not train on hidden data, and must match project dependency constraints."
    ),
    "registry_object": (
        "Draft a form-ready registry payload for the selected registry type using internal schema references. "
        "Do not create the object; only return metadata."
    ),
    "feature_operations": (
        "Draft visual feature operations that match the selected dataset context and can be previewed before materialization."
    ),
    "training_run": (
        "Draft a reviewed training request that uses existing selected model and dataset identifiers."
    ),
    "study": (
        "Draft an optimization study request using existing project study settings and safe defaults."
    ),
}


def assistant_alignment_contracts() -> dict[str, str]:
    return dict(ASSISTANT_ALIGNMENT_CONTRACTS)


def assistant_draft_writing_guide() -> dict[str, Any]:
    return dict(ASSISTANT_DRAFT_WRITING_GUIDE)


def assistant_draft_quality_rubric() -> dict[str, str]:
    return dict(ASSISTANT_DRAFT_QUALITY_RUBRIC)


def assistant_active_provider_name() -> str:
    explicit = os.getenv(ASSISTANT_ACTIVE_PROVIDER_ENV, "").strip()
    if explicit:
        return explicit.lower()
    payload = _load_assistant_provider_config_payload()
    configured_active = str(payload.get("active_provider") or payload.get("active") or "").strip()
    if configured_active:
        return configured_active.lower()
    legacy_enabled = os.getenv("AMANAJE_SLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    return "amanaje_slm" if legacy_enabled else "local"


def assistant_provider_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        "local": _local_provider_config(),
        "amanaje_slm": _legacy_slm_runtime_config(),
    }
    payload = _load_assistant_provider_config_payload()
    raw_providers = payload.get("providers", payload)
    if isinstance(raw_providers, Mapping):
        for raw_name, raw_config in raw_providers.items():
            name = str(raw_name or "").strip().lower()
            if not name or name in {"active", "active_provider"}:
                continue
            if isinstance(raw_config, Mapping):
                configs[name] = _normalize_provider_config(name, raw_config, default_enabled=True)
    return configs


def assistant_provider_server_contract() -> dict[str, Any]:
    return {
        "runtime_protocol": "openai_compatible_chat_completions",
        "required_chat_endpoint": "POST /v1/chat/completions",
        "recommended_health_endpoint": "GET /health",
        "response_shape": "choices[0].message.content must be a JSON object that validates as WorkflowDraft.",
        "safety_boundary": "Provider drafts only; Painel Amanaje validates, reviews, executes, and persists.",
        "required_output_contract": "WorkflowDraft",
        "supported_draft_types": list(ASSISTANT_DRAFT_TYPE_LIST),
    }


def redact_assistant_provider_config(config: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in dict(config or {}).items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("api_key", "authorization", "bearer", "password", "secret", "token")):
            if lowered.endswith("_configured"):
                redacted[key] = bool(value)
            else:
                redacted[key] = "[REDACTED]" if value else ""
        elif isinstance(value, Mapping):
            redacted[key] = redact_assistant_provider_config(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_assistant_provider_config(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            redacted[key] = value
    return redacted


def assistant_slm_runtime_config() -> dict[str, Any]:
    return dict(assistant_provider_configs().get("amanaje_slm") or _legacy_slm_runtime_config())


def _local_provider_config() -> dict[str, Any]:
    return {
        "name": "local",
        "type": "local",
        "enabled": True,
        "model_name": "local-workflow-provider",
        "model_version": LOCAL_MODEL_VERSION,
        "supports_json_mode": True,
        "supports_remote_auth": False,
        "supports_health_check": False,
        "supported_draft_types": list(ASSISTANT_DRAFT_TYPE_LIST),
        "fallback_provider": None,
        "capabilities": _provider_capabilities("local"),
    }


def _legacy_slm_runtime_config() -> dict[str, Any]:
    enabled_value = os.getenv("AMANAJE_SLM_ENABLED", "false").strip().lower()
    timeout_value = os.getenv("AMANAJE_SLM_TIMEOUT_SECONDS", "20")
    max_tokens_value = os.getenv("AMANAJE_SLM_MAX_TOKENS", "1600")
    try:
        timeout_seconds = max(1.0, float(timeout_value))
    except ValueError:
        timeout_seconds = 20.0
    try:
        max_tokens = max(256, int(max_tokens_value))
    except ValueError:
        max_tokens = 1600
    base_url = os.getenv("AMANAJE_SLM_BASE_URL", SLM_DEFAULT_BASE_URL).rstrip("/")
    return {
        "name": "amanaje_slm",
        "type": "openai_compatible",
        "enabled": enabled_value in {"1", "true", "yes", "on"},
        "base_url": base_url,
        "chat_endpoint": f"{base_url}/chat/completions",
        "health_url": os.getenv("AMANAJE_SLM_HEALTH_URL", f"{base_url.rsplit('/v1', 1)[0]}/health"),
        "model_name": os.getenv("AMANAJE_SLM_MODEL_NAME", SLM_DEFAULT_MODEL_NAME),
        "timeout_seconds": timeout_seconds,
        "max_tokens": max_tokens,
        "temperature": float(os.getenv("AMANAJE_SLM_TEMPERATURE", "0.2")),
        "api_key_env": "AMANAJE_SLM_API_KEY",
        "api_key_configured": bool(os.getenv("AMANAJE_SLM_API_KEY")),
        "model_version": SLM_MODEL_VERSION,
        "runtime_kind": os.getenv("AMANAJE_SLM_RUNTIME_KIND", "pytorch_hf_server"),
        "base_model_name": os.getenv("AMANAJE_SLM_BASE_MODEL", "local-workflow-template"),
        "adapter_path": os.getenv("AMANAJE_SLM_ADAPTER_PATH", "runtime_artifacts/assistant_models/amanaje_slm"),
        "supports_json_mode": True,
        "supports_remote_auth": bool(os.getenv("AMANAJE_SLM_API_KEY")),
        "supports_health_check": True,
        "supported_draft_types": list(ASSISTANT_DRAFT_TYPE_LIST),
        "fallback_provider": "local",
        "capabilities": _provider_capabilities("openai_compatible"),
    }


def _load_assistant_provider_config_payload() -> dict[str, Any]:
    inline = os.getenv(ASSISTANT_PROVIDER_CONFIGS_ENV, "").strip()
    if inline:
        try:
            payload = json.loads(inline)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    config_path = os.getenv(ASSISTANT_PROVIDER_CONFIG_PATH_ENV, "").strip()
    if config_path:
        path = Path(config_path).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_provider_config(name: str, config: Mapping[str, Any], *, default_enabled: bool) -> dict[str, Any]:
    provider_type = str(config.get("type") or config.get("provider_type") or "openai_compatible").strip().lower()
    if provider_type == "local":
        return {**_local_provider_config(), **dict(config), "name": name, "type": "local", "enabled": True}

    timeout_seconds = _coerce_float(config.get("timeout_seconds"), 20.0, minimum=1.0)
    max_tokens = int(_coerce_float(config.get("max_tokens"), 1600, minimum=256))
    base_url = str(config.get("base_url") or SLM_DEFAULT_BASE_URL).rstrip("/")
    api_key_env = str(config.get("api_key_env") or f"{name.upper()}_API_KEY")
    api_key_configured = bool(config.get("api_key") or os.getenv(api_key_env))
    return {
        "name": name,
        "type": "openai_compatible",
        "enabled": _coerce_bool(config.get("enabled"), default_enabled),
        "base_url": base_url,
        "chat_endpoint": str(config.get("chat_endpoint") or f"{base_url}/chat/completions"),
        "health_url": str(config.get("health_url") or f"{base_url.rsplit('/v1', 1)[0]}/health"),
        "model_name": str(config.get("model_name") or config.get("model") or name),
        "timeout_seconds": timeout_seconds,
        "max_tokens": max_tokens,
        "temperature": _coerce_float(config.get("temperature"), 0.2, minimum=0.0),
        "api_key": config.get("api_key"),
        "api_key_env": api_key_env,
        "api_key_configured": api_key_configured,
        "model_version": str(config.get("model_version") or f"{name}-unversioned"),
        "runtime_kind": str(config.get("runtime_kind") or "external_openai_compatible_server"),
        "base_model_name": config.get("base_model_name"),
        "adapter_path": config.get("adapter_path"),
        "supports_json_mode": _coerce_bool(config.get("supports_json_mode"), True),
        "supports_remote_auth": api_key_configured,
        "supports_health_check": bool(config.get("health_url") or base_url),
        "supported_draft_types": list(config.get("supported_draft_types") or ASSISTANT_DRAFT_TYPE_LIST),
        "fallback_provider": str(config.get("fallback_provider") or "local"),
        "capabilities": _provider_capabilities("openai_compatible"),
    }


def _provider_capabilities(provider_type: str) -> dict[str, Any]:
    if provider_type == "local":
        return {
            "structured_workflow_draft": True,
            "deterministic_fallback": True,
            "openai_compatible": False,
            "remote_server": False,
        }
    return {
        "structured_workflow_draft": True,
        "deterministic_fallback": True,
        "openai_compatible": True,
        "remote_server": True,
    }


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_float(value: Any, default: float, *, minimum: float) -> float:
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


def build_slm_messages(context_pack: AssistantContextPack) -> list[dict[str, str]]:
    target_type = context_pack.target_type or "dataset_generation"
    task_template = SLM_PROMPT_TEMPLATES.get(target_type, SLM_PROMPT_TEMPLATES["dataset_generation"])
    compact_pack = context_pack.model_dump(mode="json")
    return [
        {
            "role": "system",
            "content": (
                "You are the private Painel Amanaje assistant model. "
                "Return exactly one JSON object that validates as WorkflowDraft. "
                "Do not wrap JSON in markdown. Do not execute code. "
                "All generated code must fit the allowed imports and required outputs in the context pack."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task_template,
                    "writing_guide": assistant_draft_writing_guide(),
                    "draft_quality_rubric": assistant_draft_quality_rubric(),
                    "workflow_draft_required_fields": ["draft_type", "title", "summary", "prompt"],
                    "context_pack": compact_pack,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def build_assistant_context_pack(request: AssistantDraftRequest, target_type: str) -> AssistantContextPack:
    context = dict(request.context or {})
    registry_type = context.get("registryType") or context.get("modelType")
    profile = target_type or "dataset_generation"
    profile_config = SAFETY_PROFILES.get(profile, SAFETY_PROFILES["dataset_generation"])
    form_values = _extract_form_values(context)
    references = _build_content_references(context, profile, registry_type)
    reference_pack = build_reference_pack_metadata(references)
    compact_summary = _build_compact_summary(
        target_type=profile,
        registry_type=registry_type,
        form_values=form_values,
        references=references,
    )
    pack = AssistantContextPack(
        version=CONTEXT_PACK_VERSION,
        prompt=request.prompt,
        workflow_goal=request.workflow_goal,
        target_type=profile,
        provider=request.provider,
        registry_type=str(registry_type) if registry_type else None,
        form_values=form_values,
        constraints=dict(request.constraints or {}),
        safety_profile=profile,
        allowed_imports=sorted(profile_config["allowed_imports"]),
        allowed_dependencies=sorted(profile_config["allowed_dependencies"]),
        content_references=references,
        reference_pack_hash=reference_pack["reference_pack_hash"],
        selected_reference_ids=reference_pack["selected_reference_ids"],
        reference_trust_levels=reference_pack["reference_trust_levels"],
        compact_summary=compact_summary,
    )
    return pack.model_copy(update={"pack_hash": context_pack_hash(pack)})


def context_pack_hash(pack: AssistantContextPack) -> str:
    payload = pack.model_dump(
        mode="json",
        exclude={"context_pack_id", "reference_pack_id", "pack_hash", "created_at"},
    )
    for reference in payload.get("content_references", []):
        if isinstance(reference, dict):
            reference.pop("created_at", None)
    normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_reference_pack_metadata(references: Iterable[ContentReference]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    trust_levels: dict[str, str] = {}
    selected_ids: list[str] = []
    for reference in references:
        reference_id = str(reference.reference_id)
        selected_ids.append(reference_id)
        trust_levels[reference_id] = str(reference.trust_level or "project")
        selected.append(
            {
                "reference_id": reference_id,
                "source_type": reference.source_type,
                "name": reference.name,
                "trust_level": reference.trust_level,
                "content_hash": reference.content_hash,
                "tags": sorted(str(tag) for tag in reference.tags),
                "summary": _summarize_text(reference.summary, max_length=220),
            }
        )
    normalized = json.dumps(selected, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "reference_pack_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "selected_reference_ids": selected_ids,
        "reference_trust_levels": trust_levels,
    }


def persist_assistant_context_pack(context_dir: str | Path, draft: WorkflowDraft) -> Path | None:
    payload = dict(draft.artifacts or {}).get("context_pack", {}).get("payload")
    if not payload:
        return None
    directory = ensure_dir(context_dir)
    context_pack_id = draft.context_pack_id or payload.get("context_pack_id") or "context_pack"
    path = directory / f"{context_pack_id}.json"
    save_json(payload, path, indent=2)
    return path


def persist_assistant_eval_result(eval_dir: str | Path, draft: WorkflowDraft, evaluation: Mapping[str, Any]) -> Path:
    directory = ensure_dir(eval_dir)
    draft_id = draft.draft_id or "draft"
    path = directory / f"{draft_id}_eval.json"
    save_json(
        {
            "recorded_at": datetime.now().isoformat(),
            "draft_id": draft.draft_id,
            "draft_type": draft.draft_type,
            "provider": draft.provider,
            "model_version": draft.model_version,
            "context_pack_hash": draft.context_pack_hash,
            "evaluation": dict(evaluation or {}),
        },
        path,
        indent=2,
    )
    return path


def persist_assistant_eval_suite(eval_dir: str | Path, evaluation: Mapping[str, Any]) -> Path:
    directory = ensure_dir(eval_dir)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = directory / f"assistant_golden_eval_{timestamp}.json"
    save_json(
        {
            "recorded_at": datetime.now().isoformat(),
            "evaluation_profile": evaluation.get("evaluation_profile", EVALUATION_PROFILE),
            "provider": evaluation.get("provider"),
            "evaluation": dict(evaluation or {}),
        },
        path,
        indent=2,
    )
    return path


def create_assistant_reference(
    reference_dir: str | Path,
    request: AssistantReferenceRequest,
) -> tuple[ContentReference, Path, Path | None]:
    directory = ensure_dir(reference_dir)
    source_text = str(request.content_text or "")
    redacted_text = _redact_sensitive_text(source_text)
    normalized_text = _normalize_text(redacted_text)
    content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest() if normalized_text else None
    summary = request.summary or _summarize_text(normalized_text)
    reference = ContentReference(
        source_type=request.source_type or "user_text",
        name=request.name or request.file_name or "Assistant Reference",
        summary=summary,
        trust_level=request.trust_level or "user",
        uri=request.uri,
        metadata={
            **dict(request.metadata or {}),
            "file_name": request.file_name,
            "content_length": len(source_text),
            "redacted_length": len(redacted_text),
            "excerpts": _extract_reference_excerpts(normalized_text),
        },
        content_hash=content_hash,
        tags=list(dict.fromkeys(str(tag).strip() for tag in request.tags if str(tag).strip())),
    )
    metadata_path = directory / f"{reference.reference_id}.json"
    text_path = directory / f"{reference.reference_id}.txt" if normalized_text else None
    if text_path is not None:
        text_path.write_text(redacted_text, encoding="utf-8")
        reference.uri = str(text_path)
    save_json(reference.model_dump(mode="json"), metadata_path, indent=2)
    return reference, metadata_path, text_path


def load_assistant_reference(reference_dir: str | Path, reference_id: str) -> ContentReference | None:
    path = Path(reference_dir) / f"{reference_id}.json"
    if not path.exists():
        return None
    try:
        return ContentReference.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def list_assistant_references(
    reference_dir: str | Path,
    *,
    query: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
    source_types: list[str] | tuple[str, ...] | None = None,
    limit: int = 20,
) -> list[ContentReference]:
    directory = Path(reference_dir)
    if not directory.exists():
        return []
    normalized_query = str(query or "").strip().lower()
    wanted_tags = {str(tag).strip().lower() for tag in tags or [] if str(tag).strip()}
    wanted_source_types = {str(item).strip().lower() for item in source_types or [] if str(item).strip()}
    references: list[ContentReference] = []
    for path in sorted(directory.glob("ref_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            reference = ContentReference.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        haystack = " ".join(
            [
                reference.reference_id,
                reference.source_type,
                reference.name or "",
                reference.summary,
                " ".join(reference.tags),
            ]
        ).lower()
        if normalized_query and normalized_query not in haystack:
            continue
        if wanted_tags and not wanted_tags.intersection({tag.lower() for tag in reference.tags}):
            continue
        if wanted_source_types and reference.source_type.lower() not in wanted_source_types:
            continue
        references.append(reference)
        if len(references) >= max(1, min(int(limit or 20), 100)):
            break
    return references


def resolve_assistant_reference_source_paths(
    reference_dir: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> list[Path]:
    configured = os.getenv(ASSISTANT_REFERENCE_SOURCE_ENV, "").strip()
    raw_paths = _split_reference_source_paths(configured)
    if not raw_paths:
        raw_paths = [
            str(Path(reference_dir) / "repository"),
            "assistant_references",
        ]

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    resolved: list[Path] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def assistant_reference_repository_status(
    reference_dir: str | Path,
    *,
    source_paths: Iterable[str | Path] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_paths = [
        Path(path).resolve()
        for path in (source_paths or resolve_assistant_reference_source_paths(reference_dir, base_dir=base_dir))
    ]
    references = list_assistant_references(reference_dir, limit=100)
    return {
        "reference_dir": str(Path(reference_dir)),
        "source_paths": [
            {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
            for path in resolved_paths
        ],
        "stored_reference_count": len(references),
        "stored_reference_ids": [reference.reference_id for reference in references[:20]],
    }


def sync_assistant_reference_repository(
    reference_dir: str | Path,
    *,
    source_paths: Iterable[str | Path] | None = None,
    base_dir: str | Path | None = None,
    max_files: int = 100,
    max_bytes: int = 200_000,
) -> dict[str, Any]:
    directory = ensure_dir(reference_dir)
    resolved_paths = [
        Path(path).resolve()
        for path in (source_paths or resolve_assistant_reference_source_paths(directory, base_dir=base_dir))
    ]
    existing = list_assistant_references(directory, limit=100)
    existing_hashes = {reference.content_hash for reference in existing if reference.content_hash}
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    scanned = 0

    for file_path in _iter_reference_source_files(resolved_paths):
        if scanned >= max(1, max_files):
            break
        scanned += 1
        try:
            size = file_path.stat().st_size
        except OSError:
            skipped.append({"path": str(file_path), "reason": "stat_failed"})
            continue
        if size > max_bytes:
            skipped.append({"path": str(file_path), "reason": "too_large"})
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped.append({"path": str(file_path), "reason": "read_failed"})
            continue
        normalized = _normalize_text(_redact_sensitive_text(text))
        if not normalized:
            skipped.append({"path": str(file_path), "reason": "empty"})
            continue
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if content_hash in existing_hashes:
            skipped.append({"path": str(file_path), "reason": "duplicate_hash"})
            continue

        reference_request = AssistantReferenceRequest(
            source_type="project_file",
            name=file_path.stem,
            content_text=text,
            file_name=file_path.name,
            trust_level="project",
            uri=str(file_path),
            tags=_reference_file_tags(file_path),
            metadata={
                "source_path": str(file_path),
                "source_size": size,
                "source_extension": file_path.suffix.lower(),
            },
        )
        reference, metadata_path, text_path = create_assistant_reference(directory, reference_request)
        existing_hashes.add(reference.content_hash)
        created.append(
            {
                "reference_id": reference.reference_id,
                "source_path": str(file_path),
                "metadata_path": str(metadata_path),
                "text_path": str(text_path) if text_path else None,
                "content_hash": reference.content_hash,
            }
        )

    return {
        "status": "synced",
        "reference_dir": str(directory),
        "source_paths": [str(path) for path in resolved_paths],
        "scanned": scanned,
        "created_count": len(created),
        "skipped_count": len(skipped),
        "created": created,
        "skipped": skipped[:25],
    }


def build_selected_reference_pack(
    reference_dir: str | Path,
    *,
    reference_ids: list[str] | tuple[str, ...] | None = None,
    query: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
    limit: int = 8,
) -> list[ContentReference]:
    seen: set[str] = set()
    selected: list[ContentReference] = []
    for reference_id in reference_ids or []:
        reference = load_assistant_reference(reference_dir, str(reference_id))
        if reference is None or reference.reference_id in seen:
            continue
        selected.append(_compact_reference(reference))
        seen.add(reference.reference_id)

    remaining = max(0, min(int(limit or 8), 12) - len(selected))
    if remaining and (query or tags):
        for reference in _rank_assistant_references(reference_dir, query=query, tags=tags, limit=remaining):
            if reference.reference_id in seen:
                continue
            selected.append(_compact_reference(reference))
            seen.add(reference.reference_id)
            if len(selected) >= min(int(limit or 8), 12):
                break
    return selected


def persist_assistant_promotion_report(
    model_dir: str | Path,
    draft: WorkflowDraft,
    evaluation: Mapping[str, Any],
) -> Path:
    directory = ensure_dir(model_dir)
    report = build_assistant_promotion_report(
        draft=draft,
        evaluation=evaluation,
        provider_status=dict(draft.provider_metadata or {}),
    )
    version_key = _safe_artifact_name(draft.model_version or draft.provider or "assistant")
    draft_key = _safe_artifact_name(draft.draft_id or "draft")
    path = directory / f"{version_key}_{draft_key}_promotion.json"
    save_json(report, path, indent=2)
    return path


def stamp_workflow_draft(
    draft: WorkflowDraft,
    *,
    context_pack: AssistantContextPack,
    provider_name: str,
    model_version: str,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
    evaluation_profile: str = EVALUATION_PROFILE,
    fallback_provider: str | None = None,
    latency_ms: float | None = None,
    cache_hit: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> WorkflowDraft:
    provider_metadata = {
        **dict(draft.provider_metadata or {}),
        "provider": provider_name,
        "base_model_name": os.getenv("AMANAJE_SLM_BASE_MODEL", "local-workflow-template"),
        "adapter_path": os.getenv("AMANAJE_SLM_ADAPTER_PATH", "runtime_artifacts/assistant_models/amanaje_slm"),
        "context_pack_version": context_pack.version,
        "safety_profile_version": SAFETY_PROFILE_VERSION,
        "allowed_dependencies": list(context_pack.allowed_dependencies),
        "reference_count": len(context_pack.content_references),
        "reference_pack_id": context_pack.reference_pack_id,
        "reference_pack_hash": context_pack.reference_pack_hash,
        "selected_reference_ids": list(context_pack.selected_reference_ids),
        "reference_trust_levels": dict(context_pack.reference_trust_levels),
        "reference_hashes": [
            reference.content_hash
            for reference in context_pack.content_references
            if reference.content_hash
        ],
        "fallback_provider": fallback_provider,
        "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
        "cache_hit": cache_hit,
        "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
        "alignment_contracts": list(ASSISTANT_ALIGNMENT_CONTRACTS),
        **dict(extra_metadata or {}),
    }
    provenance = GenerationProvenance(
        provider=provider_name,
        model_version=model_version,
        prompt_template_version=prompt_template_version,
        context_pack_id=context_pack.context_pack_id,
        context_pack_hash=context_pack.pack_hash,
        fallback_provider=fallback_provider,
    )
    artifacts = {
        **dict(draft.artifacts or {}),
        "context_pack": {
            "id": context_pack.context_pack_id,
            "hash": context_pack.pack_hash,
            "version": context_pack.version,
            "payload": context_pack.model_dump(mode="json"),
        },
        "assistant_model_registry_payload": build_assistant_model_registry_payload(
            provider_name=provider_name,
            model_version=model_version,
            provider_metadata=provider_metadata,
        ),
        "assistant_prompt_registry_payload": build_assistant_prompt_registry_payload(
            provider_name=provider_name,
            model_version=model_version,
            provider_metadata=provider_metadata,
        ),
        "alignment_contracts": assistant_alignment_contracts(),
    }
    return draft.model_copy(
        update={
            "provider": provider_name,
            "model_version": model_version,
            "prompt_template_version": prompt_template_version,
            "context_pack_id": context_pack.context_pack_id,
            "context_pack_hash": context_pack.pack_hash,
            "evaluation_profile": evaluation_profile,
            "generation_provenance": provenance.model_dump(mode="json"),
            "provider_metadata": provider_metadata,
            "artifacts": artifacts,
        },
        deep=True,
    )


def build_assistant_model_registry_payload(
    *,
    provider_name: str,
    model_version: str,
    provider_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(provider_name or "amanaje_slm").replace("-", "_")
    adapter_path = str(provider_metadata.get("adapter_path") or f"runtime_artifacts/assistant_models/{name}")
    return {
        "name": f"{name}_assistant",
        "description": "Private assistant model service used by Painel Amanaje.",
        "object_type": "learning_model",
        "size": 0,
        "path": adapter_path,
        "version": 1,
        "model_type": "assistant_model",
        "parameters": {
            "assistant": {
                "provider_type": "openai_compatible",
                "provider": provider_name,
                "model_version": model_version,
                "base_model_name": provider_metadata.get("base_model_name"),
                "adapter_path": adapter_path,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "context_pack_version": provider_metadata.get("context_pack_version"),
                "safety_profile_version": provider_metadata.get("safety_profile_version"),
                "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
                "supported_draft_types": [
                    "dataset_generation",
                    "model_generation",
                    "registry_object",
                    "feature_operations",
                    "training_run",
                    "study",
                ],
            },
        },
        "metrics": {
            "evaluation_profile": EVALUATION_PROFILE,
            "evaluation_score": provider_metadata.get("evaluation_score"),
            "fallback_provider": provider_metadata.get("fallback_provider"),
        },
        "reference_data": "runtime_artifacts/assistant_datasets/interactions.jsonl",
        "input_features": ["prompt", "context_pack", "target_type"],
        "output_features": ["workflow_draft"],
        "is_trained": False,
        "is_tested": False,
        "is_deployed": False,
        "history": [
            {
                "operation": "assistant_model_registered_payload",
                "model_version": model_version,
                "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
            }
        ],
    }


def build_assistant_prompt_registry_payload(
    *,
    provider_name: str,
    model_version: str,
    provider_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(provider_name or "amanaje_slm").replace("-", "_")
    return {
        "name": f"{name}_assistant_prompt_context",
        "description": "Prompt templates, context-pack rules, and alignment contracts for the private assistant.",
        "object_type": "code_model",
        "size": 0,
        "path": "runtime_artifacts/assistant_models/prompt_context.json",
        "version": 1,
        "variables": {
            "provider": provider_name,
            "model_version": model_version,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "context_pack_version": provider_metadata.get("context_pack_version"),
            "safety_profile_version": provider_metadata.get("safety_profile_version"),
            "alignment_contracts": assistant_alignment_contracts(),
            "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
        },
        "code": {
            "language": "json",
            "prompt_templates": dict(SLM_PROMPT_TEMPLATES),
            "context_pack_version": CONTEXT_PACK_VERSION,
        },
        "history": [
            {
                "operation": "assistant_prompt_context_registered_payload",
                "model_version": model_version,
                "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
            }
        ],
    }


def build_assistant_promotion_report(
    *,
    draft: WorkflowDraft,
    evaluation: Mapping[str, Any],
    provider_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    alignment = dict(evaluation.get("alignment_contracts") or {})
    gate_checks = {
        "evaluation_passed": bool(evaluation.get("passed")),
        "output_alignment": bool(alignment.get("output_alignment")),
        "safety_alignment": bool(alignment.get("safety_alignment")),
        "dependency_alignment": bool(alignment.get("dependency_alignment")),
        "mlflow_alignment": bool(alignment.get("mlflow_alignment")),
    }
    if draft.draft_type == "registry_object":
        gate_checks["registry_alignment"] = bool(alignment.get("registry_alignment"))
    eligible = all(gate_checks.values())
    return {
        "recorded_at": datetime.now().isoformat(),
        "eligible_for_promotion": eligible,
        "gate_checks": gate_checks,
        "draft_id": draft.draft_id,
        "draft_type": draft.draft_type,
        "provider": draft.provider,
        "model_version": draft.model_version,
        "context_pack_hash": draft.context_pack_hash,
        "prompt_template_version": draft.prompt_template_version,
        "evaluation_profile": draft.evaluation_profile,
        "mlflow_experiment": ASSISTANT_MLFLOW_EXPERIMENT,
        "provider_status": dict(provider_status or {}),
        "learning_model_payload": dict(draft.artifacts.get("assistant_model_registry_payload") or {}),
        "code_model_payload": dict(draft.artifacts.get("assistant_prompt_registry_payload") or {}),
        "evaluation": dict(evaluation or {}),
    }


def append_assistant_training_example(
    dataset_dir: str | Path,
    *,
    label: str,
    draft: WorkflowDraft,
    review: ReviewDecision | Mapping[str, Any] | None = None,
    approval: Mapping[str, Any] | None = None,
    final_payload: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
    user_edits: Mapping[str, Any] | None = None,
) -> Path:
    directory = ensure_dir(dataset_dir)
    path = directory / "interactions.jsonl"
    review_payload = review.model_dump(mode="json") if isinstance(review, ReviewDecision) else dict(review or {})
    record = {
        "recorded_at": datetime.now().isoformat(),
        "label": label,
        "prompt": draft.prompt,
        "workflow_type": draft.draft_type,
        "provider": draft.provider,
        "model_version": draft.model_version,
        "prompt_template_version": draft.prompt_template_version,
        "context_pack_id": draft.context_pack_id,
        "context_pack_hash": draft.context_pack_hash,
        "draft": draft.model_dump(mode="json"),
        "review": review_payload,
        "approval": dict(approval or {}),
        "final_payload": dict(final_payload or {}),
        "outcome": dict(outcome or {}),
        "user_edits": dict(user_edits or {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def evaluate_draft_contract(draft: WorkflowDraft, review: ReviewDecision | Mapping[str, Any]) -> dict[str, Any]:
    review_payload = review.model_dump(mode="json") if isinstance(review, ReviewDecision) else dict(review or {})
    safety_payload = dict(review_payload.get("safety") or {})
    required_by_type = {
        "dataset_generation": ["csv_text"],
        "model_generation": ["joblib_bytes"],
        "registry_object": ["registry_payload"],
        "training_run": ["training_request"],
        "study": ["study_request"],
    }
    required_outputs = required_by_type.get(draft.draft_type, [])
    checks = {
        "valid_workflow_draft": bool(draft.draft_id and draft.draft_type and draft.title),
        "has_context_pack": bool(draft.context_pack_id and draft.context_pack_hash),
        "has_model_version": bool(draft.model_version),
        "has_prompt_template_version": bool(draft.prompt_template_version),
        "review_approved": bool(review_payload.get("approved")),
    }
    for output_name in required_outputs:
        if output_name == "registry_payload":
            checks[output_name] = bool(draft.registry_payload or draft.form_payload)
        elif output_name == "training_request":
            checks[output_name] = bool(draft.training_request)
        elif output_name == "study_request":
            checks[output_name] = bool(draft.study_request)
        else:
            checks[output_name] = output_name in (draft.code or "")
    required_outputs_ok = all(checks.get(output_name, False) for output_name in required_outputs)
    registry_payload = dict(draft.registry_payload or draft.form_payload or {})
    dependency_evidence = (
        draft.provider_metadata.get("allowed_dependencies")
        or safety_payload.get("allowed_dependencies")
        or SAFETY_PROFILES.get(draft.execution_profile or draft.draft_type, {}).get("allowed_dependencies")
    )
    alignment_contract_checks = {
        "output_alignment": bool(checks["valid_workflow_draft"] and (not required_outputs or required_outputs_ok)),
        "safety_alignment": bool(safety_payload.get("ok")),
        "registry_alignment": draft.draft_type != "registry_object"
        or bool(registry_payload.get("name") and registry_payload.get("object_type") and registry_payload.get("path")),
        "dependency_alignment": bool(safety_payload.get("ok") and dependency_evidence is not None),
        "mlflow_alignment": bool(
            draft.model_version
            and draft.prompt_template_version
            and draft.context_pack_hash
            and draft.evaluation_profile
        ),
    }
    quality_checks = {
        "title_quality": len(str(draft.title or "").strip()) >= 8
        and str(draft.title or "").strip().lower() not in {"review", "draft", "assistant review"},
        "summary_quality": len(str(draft.summary or "").strip()) >= 24,
        "risk_or_assumption_quality": bool(draft.risks or draft.assumptions),
        "next_action_quality": bool(draft.next_actions),
        "reference_traceability": bool(
            dict(draft.artifacts or {}).get("context_pack", {}).get("hash")
            or draft.context_pack_hash
        ),
    }
    return {
        "evaluation_profile": EVALUATION_PROFILE,
        "draft_type": draft.draft_type,
        "passed": all(checks.values()) and all(alignment_contract_checks.values()) and all(quality_checks.values()),
        "checks": checks,
        "draft_quality": quality_checks,
        "draft_quality_rubric": assistant_draft_quality_rubric(),
        "alignment_contracts": alignment_contract_checks,
        "alignment_contract_descriptions": assistant_alignment_contracts(),
    }


def build_assistant_golden_eval_requests(provider: str = "amanaje_slm") -> list[AssistantDraftRequest]:
    return [
        AssistantDraftRequest(
            prompt="Create a compact synthetic dataset with feature_a, feature_b, segment, and target.",
            target_type="dataset_generation",
            provider=provider,
        ),
        AssistantDraftRequest(
            prompt="Create a scikit-learn joblib model artifact draft.",
            target_type="model_generation",
            provider=provider,
        ),
        AssistantDraftRequest(
            prompt="Draft a LearningModel registry object.",
            target_type="registry_object",
            provider=provider,
            context={"registryType": "LearningModel"},
        ),
        AssistantDraftRequest(
            prompt="Prepare a training run.",
            target_type="training_run",
            provider=provider,
            context={"modelId": 1, "datasetId": 1},
        ),
        AssistantDraftRequest(
            prompt="Prepare an optimization study draft.",
            target_type="study",
            provider=provider,
            context={"studyId": 1},
        ),
    ]


def run_assistant_golden_evals(
    provider_factory: Callable[[str], Any],
    *,
    provider: str = "amanaje_slm",
) -> dict[str, Any]:
    assistant_provider = provider_factory(provider)
    results: list[dict[str, Any]] = []
    for request in build_assistant_golden_eval_requests(provider):
        draft = assistant_provider.create_draft(request)
        from app.utils.assistant_safety import review_workflow_draft

        review = review_workflow_draft(draft)
        evaluation = evaluate_draft_contract(draft, review)
        results.append(evaluation)
    return {
        "evaluation_profile": EVALUATION_PROFILE,
        "provider": provider,
        "passed": all(item.get("passed") for item in results),
        "results": results,
    }


def build_assistant_reference_snapshot(
    dataset_dir: str | Path,
    eval_dir: str | Path,
    *,
    max_examples: int = 3,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    eval_path = Path(eval_dir)
    examples: list[dict[str, Any]] = []
    interactions_path = dataset_path / "interactions.jsonl"
    if interactions_path.exists():
        try:
            lines = interactions_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-max(1, max_examples) :]:
                record = json.loads(line)
                examples.append(
                    {
                        "label": record.get("label"),
                        "workflow_type": record.get("workflow_type"),
                        "provider": record.get("provider"),
                        "model_version": record.get("model_version"),
                        "context_pack_hash": record.get("context_pack_hash"),
                        "review_status": dict(record.get("review") or {}).get("status"),
                    }
                )
        except Exception:
            examples = []

    evals: list[dict[str, Any]] = []
    if eval_path.exists():
        try:
            files = sorted(eval_path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
            for file_path in files[: max(1, max_examples)]:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                evaluation = dict(payload.get("evaluation") or {})
                evals.append(
                    {
                        "file": file_path.name,
                        "draft_type": payload.get("draft_type") or evaluation.get("draft_type"),
                        "passed": evaluation.get("passed"),
                        "evaluation_profile": payload.get("evaluation_profile") or evaluation.get("evaluation_profile"),
                    }
                )
        except Exception:
            evals = []

    return {
        "source_type": "assistant_reference_snapshot",
        "interactions_path": str(interactions_path),
        "eval_dir": str(eval_path),
        "recent_examples": examples,
        "recent_evals": evals,
    }


def curate_assistant_training_examples(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    project_examples: Mapping[str, Any] | None = None,
    reference_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = ensure_dir(dataset_dir)
    curated_dir = ensure_dir(output_dir or dataset_path / "curated")
    interactions_path = dataset_path / "interactions.jsonl"
    curated_path = curated_dir / "curated_interactions.jsonl"
    manifest_path = curated_dir / "curation_manifest.json"

    labels: dict[str, int] = {}
    workflow_types: dict[str, int] = {}
    kept_records = 0
    rejected_records = 0
    interaction_records = 0
    project_example_records = 0
    reference_records = 0

    with curated_path.open("w", encoding="utf-8") as handle:
        if interactions_path.exists():
            for line in interactions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    rejected_records += 1
                    continue
                label = str(record.get("label") or "unlabeled")
                workflow_type = str(record.get("workflow_type") or "unknown")
                labels[label] = labels.get(label, 0) + 1
                workflow_types[workflow_type] = workflow_types.get(workflow_type, 0) + 1
                handle.write(json.dumps(_redact_sensitive_values(record), ensure_ascii=False, default=str) + "\n")
                kept_records += 1
                interaction_records += 1
        for record in _project_example_training_records(project_examples):
            label = str(record.get("label") or "positive")
            workflow_type = str(record.get("workflow_type") or "project_example")
            labels[label] = labels.get(label, 0) + 1
            workflow_types[workflow_type] = workflow_types.get(workflow_type, 0) + 1
            handle.write(json.dumps(_redact_sensitive_values(record), ensure_ascii=False, default=str) + "\n")
            kept_records += 1
            project_example_records += 1
        if reference_dir is not None:
            reference_path = Path(reference_dir)
            if reference_path.exists():
                for reference in list_assistant_references(reference_path, limit=2_000):
                    record = _reference_training_record(reference, reference_path)
                    label = str(record.get("label") or "reference_asset")
                    workflow_type = str(record.get("workflow_type") or "reference")
                    labels[label] = labels.get(label, 0) + 1
                    workflow_types[workflow_type] = workflow_types.get(workflow_type, 0) + 1
                    handle.write(json.dumps(_redact_sensitive_values(record), ensure_ascii=False, default=str) + "\n")
                    kept_records += 1
                    reference_records += 1

    manifest = {
        "recorded_at": datetime.now().isoformat(),
        "source_path": str(interactions_path),
        "project_examples_version": project_examples.get("version") if isinstance(project_examples, Mapping) else None,
        "curated_path": str(curated_path),
        "manifest_path": str(manifest_path),
        "record_count": kept_records,
        "interaction_records": interaction_records,
        "project_example_records": project_example_records,
        "reference_records": reference_records,
        "rejected_records": rejected_records,
        "labels": labels,
        "workflow_types": workflow_types,
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "redaction_policy": "Keys containing api_key, authorization, bearer, password, secret, or token are redacted.",
        "ready_for_fine_tuning": kept_records > 0 and labels.get("positive", 0) > 0,
    }
    save_json(manifest, manifest_path, indent=2)
    return manifest


def _project_example_training_records(project_examples: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(project_examples, Mapping):
        return []
    objects = project_examples.get("objects")
    if not isinstance(objects, list):
        return []
    records: list[dict[str, Any]] = []
    catalog_version = project_examples.get("version")
    generated_at = project_examples.get("generated_at")
    for entry in objects:
        if not isinstance(entry, Mapping):
            continue
        object_name = str(entry.get("object_name") or entry.get("object_key") or "ProjectExample")
        family = str(entry.get("family") or "project_examples")
        samples = entry.get("samples")
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            payload = sample.get("payload")
            if payload is None:
                continue
            tier = str(sample.get("tier") or "sample")
            source_hash = _hash_text(
                json.dumps(
                    {
                        "catalog_version": catalog_version,
                        "object_name": object_name,
                        "tier": tier,
                        "payload": _redact_sensitive_values(payload),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
            records.append(
                {
                    "source_type": "project_example_catalog",
                    "label": "positive",
                    "workflow_type": "project_example",
                    "prompt": (
                        f"Use this validated Painel Amanaje {object_name} {tier} example as a schema, "
                        "metadata, and workflow reference for AssistantModel training."
                    ),
                    "context_pack_hash": "",
                    "asset_summary": f"{family}.{object_name} {tier} validated project example.",
                    "content": _compact_training_text(payload),
                    "registry_metadata": {
                        "catalog_version": catalog_version,
                        "generated_at": generated_at,
                        "object_key": entry.get("object_key"),
                        "object_name": object_name,
                        "family": family,
                        "module": entry.get("module"),
                        "tier": tier,
                        "declared_size_mb": sample.get("declared_size_mb"),
                        "rows": sample.get("rows"),
                        "columns": sample.get("columns"),
                        "complexity": sample.get("complexity"),
                    },
                    "source_path": "/examples/catalog",
                    "source_hash": source_hash,
                    "quality_signals": {
                        "validated_catalog_payload": True,
                        "training_use": "assistant_schema_and_workflow_reference",
                    },
                    "assistant_success": True,
                    "assistant_quality_label": 1,
                }
            )
    return records


ASSISTANT_TRAINING_DATASET_FEATURES = [
    "source_type",
    "label",
    "workflow_type",
    "prompt",
    "context_pack_hash",
    "asset_summary",
    "content",
    "registry_metadata",
    "source_path",
    "source_hash",
    "draft",
    "review",
    "user_edits",
    "quality_signals",
    "assistant_success",
    "assistant_quality_label",
]
ASSISTANT_TRAINING_TEXT_LIMIT = 12_000
ASSISTANT_TRAINING_SCRIPT_EXTENSIONS = {".py", ".js", ".ts", ".sql", ".html", ".css", ".sh", ".ps1", ".md", ".txt", ".yaml", ".yml", ".json"}
ASSISTANT_TRAINING_GENERIC_TARGET = "assistant_quality_label"


def _hash_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _jsonl_record_hash(records: list[Mapping[str, Any]]) -> str:
    normalized = "\n".join(
        json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
        for record in records
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_training_hash(value: Any) -> str:
    normalized = json.dumps(
        _redact_sensitive_values(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _assistant_quality_values(record: Mapping[str, Any]) -> tuple[int, float]:
    label = str(record.get("label") or "").strip().lower()
    review = record.get("review") if isinstance(record.get("review"), Mapping) else {}
    quality_signals = record.get("quality_signals") if isinstance(record.get("quality_signals"), Mapping) else {}
    review_status = str(dict(review or {}).get("status") or quality_signals.get("review_status") or "").strip().lower()
    status = str(quality_signals.get("status") or "").strip().lower()

    if bool(dict(review or {}).get("approved")) or bool(quality_signals.get("approved")) or bool(quality_signals.get("passed")):
        return 1, 1.0
    if label in {"positive", "approved", "completed", "success"} or review_status == "approved" or status == "completed":
        return 1, 1.0
    if label in {"needs_revision", "revision", "needs-revision"} or review_status in {"needs_revision", "changes_requested"}:
        return 0, 0.5
    if label in {"rejected", "unsafe", "failed", "error", "negative"} or status in {"failed", "error"}:
        return 0, 0.0
    if label in {"registry_asset", "reference_asset", "context"}:
        return 1, 0.75
    return 0, 0.25


def _csv_cell(value: Any) -> Any:
    redacted = _redact_sensitive_values(value)
    if redacted is None:
        return ""
    if isinstance(redacted, bool):
        return int(redacted)
    if isinstance(redacted, (int, float)):
        return redacted
    if isinstance(redacted, str):
        return _compact_training_text(redacted, max_length=ASSISTANT_TRAINING_TEXT_LIMIT)
    return _compact_training_text(redacted, max_length=ASSISTANT_TRAINING_TEXT_LIMIT)


def _assistant_training_tabular_row(record: Mapping[str, Any]) -> dict[str, Any]:
    assistant_success, assistant_quality_label = _assistant_quality_values(record)
    enriched = {
        **dict(record),
        "assistant_success": assistant_success,
        "assistant_quality_label": assistant_quality_label,
    }
    return {column: _csv_cell(enriched.get(column)) for column in ASSISTANT_TRAINING_DATASET_FEATURES}


def load_assistant_training_jsonl_records(jsonl_path: str | Path) -> list[dict[str, Any]]:
    path = Path(jsonl_path)
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            records.append(dict(payload))
    return records


def materialize_assistant_training_csv_from_records(
    records: Iterable[Mapping[str, Any]],
    csv_path: str | Path,
) -> dict[str, Any]:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_assistant_training_tabular_row(record) for record in records]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSISTANT_TRAINING_DATASET_FEATURES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    text = path.read_text(encoding="utf-8")
    return {
        "tabular_csv_path": str(path),
        "tabular_hash": _hash_text(text),
        "shape": [len(rows), len(ASSISTANT_TRAINING_DATASET_FEATURES)],
        "features_list": list(ASSISTANT_TRAINING_DATASET_FEATURES),
    }


def materialize_assistant_training_csv_from_jsonl(
    jsonl_path: str | Path,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(jsonl_path)
    target_path = Path(csv_path) if csv_path is not None else source_path.with_suffix(".csv")
    return materialize_assistant_training_csv_from_records(load_assistant_training_jsonl_records(source_path), target_path)


def _compact_training_text(value: Any, *, max_length: int = ASSISTANT_TRAINING_TEXT_LIMIT) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = _redact_sensitive_text(value)
    else:
        text = json.dumps(_redact_sensitive_values(value), ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 80].rstrip()}\n\n[TRUNCATED: original_length={len(text)}]"


def _extract_code_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("script", "source", "text", "content", "code"):
            child = value.get(key)
            if isinstance(child, str) and child.strip():
                return child
        return json.dumps(_redact_sensitive_values(value), ensure_ascii=False, indent=2, default=str)
    return ""


def _registry_asset_record(source_name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    payload = _redact_sensitive_values(dict(entry))
    common_metadata = {
        key: payload.get(key)
        for key in ("id", "name", "description", "object_type", "path", "version", "date", "size", "history")
        if key in payload
    }
    source_type = f"registry_{source_name.rstrip('s')}"
    workflow_type = "registry_object"
    content: Any = ""
    asset_summary = _summarize_mapping(payload)
    registry_metadata: dict[str, Any] = dict(common_metadata)

    if source_name == "datasets":
        workflow_type = "dataset_generation"
        registry_metadata.update(
            {
                "dataset_type": payload.get("dataset_type"),
                "shape": payload.get("shape"),
                "has_features": payload.get("has_features"),
                "features_list": payload.get("features_list"),
                "connection_string": payload.get("connection_string"),
            }
        )
        asset_summary = (
            f"Dataset '{payload.get('name')}' type={payload.get('dataset_type')} "
            f"shape={payload.get('shape')} features={payload.get('features_list')}"
        )
        content = registry_metadata
    elif source_name in {"learning_models", "assistant_models"}:
        workflow_type = "model_generation" if source_name == "learning_models" else "assistant_model"
        registry_metadata.update(
            {
                "model_type": payload.get("model_type"),
                "parameters": payload.get("parameters"),
                "metrics": payload.get("metrics"),
                "reference_data": payload.get("reference_data"),
                "input_features": payload.get("input_features"),
                "output_features": payload.get("output_features"),
                "is_trained": payload.get("is_trained"),
                "is_tested": payload.get("is_tested"),
                "is_deployed": payload.get("is_deployed"),
            }
        )
        asset_summary = (
            f"Model '{payload.get('name')}' type={payload.get('model_type')} "
            f"inputs={payload.get('input_features')} outputs={payload.get('output_features')}"
        )
        content = registry_metadata
    elif source_name == "code_models":
        workflow_type = "code_asset"
        code_payload = payload.get("code", {})
        script_text = _extract_code_text(code_payload)
        registry_metadata.update({"variables": payload.get("variables"), "language": dict(code_payload or {}).get("language") if isinstance(code_payload, Mapping) else None})
        asset_summary = f"CodeModel '{payload.get('name')}' with saved script content and variables."
        content = script_text or code_payload
    elif source_name == "inference_models":
        workflow_type = "inference"
        registry_metadata.update(
            {
                "learning_model_id": payload.get("learning_model_id"),
                "dataset_id": payload.get("dataset_id"),
                "input_features": payload.get("input_features"),
                "output_features": payload.get("output_features"),
                "inference_params": payload.get("inference_params"),
            }
        )
        content = registry_metadata
    elif source_name == "study_models":
        workflow_type = "study"
        registry_metadata.update(
            {
                "learning_model_id": payload.get("learning_model_id"),
                "dataset_id": payload.get("dataset_id"),
                "sampler": payload.get("sampler"),
                "objective": payload.get("objective"),
                "best_trial": payload.get("best_trial"),
                "best_params": payload.get("best_params"),
                "study_params": payload.get("study_params"),
            }
        )
        content = registry_metadata

    return {
        "source_type": source_type,
        "label": "registry_asset",
        "workflow_type": workflow_type,
        "prompt": "",
        "context_pack_hash": None,
        "asset_summary": asset_summary,
        "content": _compact_training_text(content),
        "registry_metadata": registry_metadata,
        "source_path": payload.get("path"),
        "source_hash": _stable_training_hash(payload),
        "quality_signals": {
            "registry_source": source_name,
            "has_content": bool(content),
            "has_metadata": bool(registry_metadata),
        },
    }


def _reference_training_record(reference: ContentReference, reference_dir: Path) -> dict[str, Any]:
    metadata = dict(reference.metadata or {})
    text = ""
    uri = str(reference.uri or "")
    if uri:
        try:
            uri_path = Path(uri)
            if uri_path.exists() and uri_path.is_file():
                text = uri_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    if not text:
        text = "\n".join(str(item) for item in metadata.get("excerpts") or [])
    metadata_path = reference_dir / f"{reference.reference_id}.json"
    source_path = metadata.get("source_path") or uri or str(metadata_path)
    return {
        "source_type": "content_reference",
        "label": "reference_asset",
        "workflow_type": "reference",
        "prompt": "",
        "context_pack_hash": None,
        "asset_summary": reference.summary,
        "content": _compact_training_text(text),
        "registry_metadata": {
            "reference_id": reference.reference_id,
            "source_type": reference.source_type,
            "name": reference.name,
            "trust_level": reference.trust_level,
            "uri": reference.uri,
            "tags": reference.tags,
            "metadata": metadata,
        },
        "source_path": source_path,
        "source_hash": reference.content_hash or _stable_training_hash(reference.model_dump(mode="json")),
        "quality_signals": {
            "trust_level": reference.trust_level,
            "tag_count": len(reference.tags),
            "has_content": bool(text),
        },
    }


def _append_training_dataset_record(records: list[dict[str, Any]], payload: Mapping[str, Any]) -> None:
    redacted = _redact_sensitive_values(dict(payload))
    reserved_keys = {
        "source_type",
        "label",
        "workflow_type",
        "draft_type",
        "prompt",
        "context_pack_hash",
        "asset_summary",
        "content",
        "registry_metadata",
        "source_path",
        "source_hash",
        "draft",
        "review",
        "user_edits",
        "quality_signals",
        "assistant_success",
        "assistant_quality_label",
    }
    base_record = {
        "source_type": redacted.get("source_type", "unknown"),
        "label": redacted.get("label", "unlabeled"),
        "workflow_type": redacted.get("workflow_type") or redacted.get("draft_type") or "unknown",
        "prompt": redacted.get("prompt", ""),
        "context_pack_hash": redacted.get("context_pack_hash"),
        "asset_summary": redacted.get("asset_summary", ""),
        "content": redacted.get("content", ""),
        "registry_metadata": redacted.get("registry_metadata", {}),
        "source_path": redacted.get("source_path"),
        "source_hash": redacted.get("source_hash"),
        "draft": redacted.get("draft", {}),
        "review": redacted.get("review", {}),
        "user_edits": redacted.get("user_edits", {}),
        "quality_signals": redacted.get("quality_signals", {}),
        "metadata": {
            key: value
            for key, value in redacted.items()
            if key not in reserved_keys
        },
    }
    assistant_success, assistant_quality_label = _assistant_quality_values(base_record)
    base_record["assistant_success"] = redacted.get("assistant_success", assistant_success)
    base_record["assistant_quality_label"] = redacted.get("assistant_quality_label", assistant_quality_label)
    records.append(
        base_record
    )


def build_assistant_training_dataset_snapshot(
    dataset_dir: str | Path,
    eval_dir: str | Path,
    run_ledger_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_records_per_source: int = 500,
    registry_records: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    reference_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = ensure_dir(dataset_dir)
    eval_path = ensure_dir(eval_dir)
    run_path = ensure_dir(run_ledger_dir)
    snapshot_dir = ensure_dir(output_dir or dataset_path / "snapshots")
    recorded_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = snapshot_dir / f"assistant_training_dataset_{recorded_at}.jsonl"
    csv_path = snapshot_dir / f"assistant_training_dataset_{recorded_at}.csv"
    manifest_path = snapshot_dir / f"assistant_training_dataset_{recorded_at}_manifest.json"
    records: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    interactions_path = dataset_path / "interactions.jsonl"
    if interactions_path.exists():
        source_files.append({"path": str(interactions_path), "sha256": _hash_file(interactions_path), "source_type": "assistant_interactions"})
        for line in interactions_path.read_text(encoding="utf-8").splitlines()[-max_records_per_source:]:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            _append_training_dataset_record(
                records,
                {
                    **record,
                    "source_type": "assistant_interaction",
                    "quality_signals": {
                        "label": record.get("label"),
                        "review_status": dict(record.get("review") or {}).get("status"),
                        "approved": bool(dict(record.get("review") or {}).get("approved")),
                    },
                },
            )

    context_dir = dataset_path / "context_packs"
    if context_dir.exists():
        context_files = sorted(context_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for file_path in context_files[:max_records_per_source]:
            source_files.append({"path": str(file_path), "sha256": _hash_file(file_path), "source_type": "context_pack"})
            try:
                context_pack = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            _append_training_dataset_record(
                records,
                {
                    "source_type": "context_pack",
                    "label": "context",
                    "workflow_type": context_pack.get("target_type", "unknown"),
                    "prompt": context_pack.get("prompt", ""),
                    "context_pack_hash": context_pack.get("pack_hash"),
                    "draft": {},
                    "review": {},
                    "quality_signals": {
                        "reference_count": len(context_pack.get("content_references") or []),
                        "allowed_dependency_count": len(context_pack.get("allowed_dependencies") or []),
                    },
                    "context_pack": context_pack,
                },
            )

    eval_files = sorted(eval_path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True) if eval_path.exists() else []
    for file_path in eval_files[:max_records_per_source]:
        source_files.append({"path": str(file_path), "sha256": _hash_file(file_path), "source_type": "assistant_eval"})
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        evaluation = dict(payload.get("evaluation") or payload)
        _append_training_dataset_record(
            records,
            {
                "source_type": "assistant_eval",
                "label": "positive" if evaluation.get("passed") else "needs_revision",
                "workflow_type": payload.get("draft_type") or evaluation.get("draft_type") or "unknown",
                "prompt": "",
                "context_pack_hash": payload.get("context_pack_hash"),
                "draft": {"draft_id": payload.get("draft_id")},
                "review": {},
                "quality_signals": {
                    "evaluation_profile": evaluation.get("evaluation_profile"),
                    "passed": bool(evaluation.get("passed")),
                    "checks": evaluation.get("checks", {}),
                    "draft_quality": evaluation.get("draft_quality", {}),
                },
            },
        )

    run_files = sorted(run_path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True) if run_path.exists() else []
    for file_path in run_files[:max_records_per_source]:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_type = str(payload.get("run_type") or "")
        if run_type not in {"assistant", "training", "study"}:
            continue
        source_files.append({"path": str(file_path), "sha256": _hash_file(file_path), "source_type": f"{run_type}_run"})
        _append_training_dataset_record(
            records,
            {
                "source_type": f"{run_type}_run",
                "label": "positive" if payload.get("status") == "completed" else str(payload.get("status") or "unknown"),
                "workflow_type": run_type,
                "prompt": dict(payload.get("parameters") or {}).get("prompt", ""),
                "context_pack_hash": dict(payload.get("context") or {}).get("context_pack_hash"),
                "draft": dict(payload.get("result") or {}).get("draft", {}),
                "review": dict(payload.get("result") or {}).get("review", {}),
                "quality_signals": {
                    "status": payload.get("status"),
                    "stage": payload.get("stage"),
                    "metric_count": len(dict(payload.get("metrics") or {})),
                    "artifact_count": len(dict(payload.get("artifacts") or {})),
                },
                "run_summary": {
                    "run_id": payload.get("run_id"),
                    "run_type": run_type,
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "context": payload.get("context", {}),
                    "metrics": payload.get("metrics", {}),
                },
            },
        )

    if reference_dir is not None:
        reference_path = Path(reference_dir)
        if reference_path.exists():
            references = list_assistant_references(reference_path, limit=max_records_per_source)
            for reference in references:
                metadata_path = reference_path / f"{reference.reference_id}.json"
                source_files.append(
                    {
                        "path": str(metadata_path),
                        "sha256": _hash_file(metadata_path),
                        "source_type": "content_reference",
                    }
                )
                _append_training_dataset_record(records, _reference_training_record(reference, reference_path))

    registry_counts: dict[str, int] = {}
    for source_name, entries in dict(registry_records or {}).items():
        normalized_source = _normalize_reference_tag(source_name)
        if normalized_source not in {"datasets", "learning_models", "assistant_models", "code_models", "inference_models", "study_models"}:
            continue
        source_entries = list(entries or [])[:max_records_per_source]
        count = 0
        for entry in source_entries:
            if not isinstance(entry, Mapping):
                continue
            _append_training_dataset_record(records, _registry_asset_record(normalized_source, entry))
            count += 1
        if count:
            registry_counts[normalized_source] = count
            source_files.append(
                {
                    "path": f"registry://{normalized_source}",
                    "sha256": _stable_training_hash(source_entries),
                    "source_type": f"registry_{normalized_source}",
                    "record_count": count,
                }
            )

    dataset_hash = _jsonl_record_hash(records)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    tabular_manifest = materialize_assistant_training_csv_from_records(records, csv_path)

    manifest = {
        "recorded_at": datetime.now().isoformat(),
        "dataset_type": "assistant_training_dataset",
        "materialization_format": "jsonl+csv",
        "dataset_path": str(csv_path),
        "canonical_jsonl_path": str(jsonl_path),
        "tabular_csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "dataset_hash": dataset_hash,
        "tabular_hash": tabular_manifest["tabular_hash"],
        "record_count": len(records),
        "feature_count": len(ASSISTANT_TRAINING_DATASET_FEATURES),
        "features_list": list(ASSISTANT_TRAINING_DATASET_FEATURES),
        "shape": list(tabular_manifest["shape"]),
        "generic_training_target": ASSISTANT_TRAINING_GENERIC_TARGET,
        "source_files": source_files,
        "source_file_count": len(source_files),
        "registry_record_counts": registry_counts,
        "redaction_policy": "Keys containing api_key, authorization, bearer, password, secret, or token are redacted.",
    }
    save_json(manifest, manifest_path, indent=2)
    return manifest


def log_assistant_mlflow_event(
    event_name: str,
    *,
    draft: WorkflowDraft | None = None,
    review: ReviewDecision | Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None = None,
    training_example_path: str | Path | None = None,
    extra_params: Mapping[str, Any] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    enabled = os.getenv("AMANAJE_ASSISTANT_MLFLOW_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return {"status": "disabled", "experiment_name": ASSISTANT_MLFLOW_EXPERIMENT}
    if not MLFLOW_AVAILABLE:
        return {"status": "skipped", "experiment_name": ASSISTANT_MLFLOW_EXPERIMENT, "reason": "mlflow_unavailable"}

    evaluation_payload = dict(evaluation or {})
    review_payload = review.model_dump(mode="json") if isinstance(review, ReviewDecision) else dict(review or {})
    provider_metadata = dict(draft.provider_metadata or {}) if draft is not None else {}
    run_name = f"{event_name}_{draft.draft_type if draft is not None else 'assistant'}"
    tags = {
        "run_type": "assistant",
        "assistant_event": event_name,
        "assistant_provider": str(draft.provider if draft is not None else provider_metadata.get("provider", "unknown")),
        "assistant_model_version": str(draft.model_version if draft is not None else provider_metadata.get("model_version", "")),
        "assistant_evaluation_profile": str(
            draft.evaluation_profile if draft is not None else evaluation_payload.get("evaluation_profile", EVALUATION_PROFILE)
        ),
    }

    params = {
        "event_name": event_name,
        "provider": draft.provider if draft is not None else provider_metadata.get("provider"),
        "draft_type": draft.draft_type if draft is not None else evaluation_payload.get("draft_type"),
        "model_version": draft.model_version if draft is not None else provider_metadata.get("model_version"),
        "base_model_name": provider_metadata.get("base_model_name"),
        "adapter_path": provider_metadata.get("adapter_path"),
        "prompt_template_version": draft.prompt_template_version if draft is not None else PROMPT_TEMPLATE_VERSION,
        "context_pack_version": provider_metadata.get("context_pack_version", CONTEXT_PACK_VERSION),
        "context_pack_hash": draft.context_pack_hash if draft is not None else None,
        "reference_pack_hash": provider_metadata.get("reference_pack_hash"),
        "safety_profile_version": provider_metadata.get("safety_profile_version", SAFETY_PROFILE_VERSION),
        "evaluation_profile": draft.evaluation_profile if draft is not None else evaluation_payload.get("evaluation_profile"),
        "fallback_provider": provider_metadata.get("fallback_provider"),
        **dict(extra_params or {}),
    }
    alignment = dict(evaluation_payload.get("alignment_contracts") or {})
    metrics = {
        "json_valid_rate": float(bool(evaluation_payload.get("checks", {}).get("valid_workflow_draft", True))),
        "safety_pass_rate": float(bool(dict(review_payload.get("safety") or {}).get("ok", True))),
        "eval_pass_rate": float(bool(evaluation_payload.get("passed", True))),
        "fallback_rate": float(bool(provider_metadata.get("fallback_provider"))),
        "approval_rate": float(bool(review_payload.get("approved"))),
        "latency_ms": provider_metadata.get("latency_ms"),
        "output_alignment": float(bool(alignment.get("output_alignment", True))),
        "safety_alignment": float(bool(alignment.get("safety_alignment", True))),
        "registry_alignment": float(bool(alignment.get("registry_alignment", True))),
        "dependency_alignment": float(bool(alignment.get("dependency_alignment", True))),
        "mlflow_alignment": float(bool(alignment.get("mlflow_alignment", True))),
        **dict(extra_metrics or {}),
    }
    artifacts = _normalize_artifact_paths(artifact_paths)
    if training_example_path:
        artifacts["training_example"] = str(training_example_path)

    try:
        with managed_run(ASSISTANT_MLFLOW_EXPERIMENT, run_name=run_name, tags=tags):
            set_tags(tags)
            log_params({key: value for key, value in params.items() if value is not None})
            log_metrics(metrics)
            log_json(
                {
                    "event_name": event_name,
                    "draft": draft.model_dump(mode="json") if draft is not None else None,
                    "review": review_payload,
                    "evaluation": evaluation_payload,
                    "artifact_paths": artifacts,
                    "alignment_contracts": assistant_alignment_contracts(),
                },
                filename=f"{event_name}_assistant_event.json",
                artifact_path="assistant",
            )
            for artifact_path in artifacts.values():
                path = Path(str(artifact_path))
                if path.exists() and path.is_file():
                    log_artifact(str(path), artifact_path="assistant/artifacts")
            return {
                "status": "logged",
                "experiment_name": ASSISTANT_MLFLOW_EXPERIMENT,
                "run_id": active_run_id(),
                "artifact_paths": artifacts,
            }
    except Exception as exc:
        return {
            "status": "failed",
            "experiment_name": ASSISTANT_MLFLOW_EXPERIMENT,
            "error": str(exc),
        }


def _normalize_artifact_paths(artifact_paths: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None) -> dict[str, str]:
    if artifact_paths is None:
        return {}
    if isinstance(artifact_paths, Mapping):
        return {str(key): str(value) for key, value in artifact_paths.items() if value}
    return {f"artifact_{index}": str(value) for index, value in enumerate(artifact_paths) if value}


def _safe_artifact_name(value: Any) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "artifact")).strip("._")
    return normalized or "artifact"


def _split_reference_source_paths(value: str) -> list[str]:
    if not value:
        return []
    chunks: list[str] = []
    for part in value.split(os.pathsep):
        chunks.extend(item.strip() for item in part.split(",") if item.strip())
    return chunks


def _iter_reference_source_files(source_paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        candidates = [source_path] if source_path.is_file() else sorted(source_path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in ASSISTANT_REFERENCE_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _reference_file_tags(file_path: Path) -> list[str]:
    raw_tags = {
        "internal_reference",
        "project_file",
        file_path.suffix.lower().lstrip("."),
        file_path.parent.name.lower(),
    }
    name = file_path.stem.lower()
    for token in ("dataset", "model", "registry", "feature", "training", "study", "mlflow", "schema", "prompt"):
        if token in name or token in str(file_path.parent).lower():
            raw_tags.add(token)
        if token in {"dataset", "model"} and token in name:
            raw_tags.add(f"{token}_generation")
    return sorted(tag for tag in raw_tags if tag)


def _rank_assistant_references(
    reference_dir: str | Path,
    *,
    query: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
    limit: int = 8,
) -> list[ContentReference]:
    candidates = list_assistant_references(reference_dir, limit=100)
    scored = [
        (score, reference)
        for reference in candidates
        if (score := _score_reference(reference, query=query, tags=tags)) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [reference for _score, reference in scored[: max(1, min(int(limit or 8), 12))]]


def _score_reference(
    reference: ContentReference,
    *,
    query: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
) -> int:
    wanted_tags = {_normalize_reference_tag(tag) for tag in tags or [] if _normalize_reference_tag(tag)}
    reference_tags = {_normalize_reference_tag(tag) for tag in reference.tags if _normalize_reference_tag(tag)}
    score = 0
    score += 6 * len(wanted_tags.intersection(reference_tags))
    if reference.trust_level in {"project", "system"}:
        score += 2
    if reference.source_type in {"project_file", "project_doc", "approved_draft", "mlflow_summary"}:
        score += 2
    haystack = " ".join(
        [
            reference.reference_id,
            reference.source_type,
            reference.name or "",
            reference.summary,
            " ".join(reference.tags),
        ]
    ).lower()
    for term in _query_terms(query):
        if term in haystack:
            score += 1
    return score


def _normalize_reference_tag(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def _query_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9_]{3,}", str(query or "").lower())
        if term not in {"the", "and", "for", "with", "from", "this", "that", "draft", "create"}
    }


def _default_reference_id(index: int) -> str:
    return f"ref_inline_{index + 1}"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _summarize_text(value: str, max_length: int = 420) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return "Reference metadata without raw text."
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def _extract_reference_excerpts(value: str, *, max_excerpts: int = 3, max_length: int = 260) -> list[str]:
    normalized = str(value or "").strip()
    if not normalized:
        return []
    chunks = [chunk.strip() for chunk in re.split(r"(?:\r?\n){2,}|(?<=[.!?])\s+", normalized) if chunk.strip()]
    excerpts: list[str] = []
    for chunk in chunks:
        excerpt = _summarize_text(chunk, max_length=max_length)
        if excerpt and excerpt not in excerpts:
            excerpts.append(excerpt)
        if len(excerpts) >= max_excerpts:
            break
    return excerpts


def _redact_sensitive_text(value: str) -> str:
    text = str(value or "")
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1) if match.groups() else 'secret'}=[REDACTED]", text)
    return text


def _compact_reference(reference: ContentReference) -> ContentReference:
    metadata = dict(reference.metadata or {})
    compact_metadata = {
        key: metadata[key]
        for key in (
            "file_name",
            "content_length",
            "redacted_length",
            "excerpts",
            "registry_type",
            "object_id",
            "source_path",
            "source_extension",
        )
        if key in metadata
    }
    return reference.model_copy(
        update={
            "metadata": compact_metadata,
            "summary": _summarize_text(reference.summary, max_length=520),
        },
        deep=True,
    )


def _redact_sensitive_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            text_key = str(key)
            if _SENSITIVE_KEY_PATTERN.search(text_key):
                redacted[text_key] = "[REDACTED]"
            else:
                redacted[text_key] = _redact_sensitive_values(child)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    return value


def _extract_form_values(context: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("registry_payload", "form_payload", "formValues", "form_values"):
        value = context.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    form_keys = {
        "datasetId",
        "dataset_id",
        "modelId",
        "model_id",
        "objectName",
        "objectPath",
        "registryType",
        "modelType",
        "parameters",
        "metrics",
        "input_features",
        "output_features",
    }
    return {key: context[key] for key in sorted(form_keys) if key in context}


def _build_content_references(
    context: Mapping[str, Any],
    profile: str,
    registry_type: Any,
) -> list[ContentReference]:
    references = [
        ContentReference(
            reference_id=f"ref_system_project_schema_{profile}",
            source_type="project_schema",
            name="assistant_workflow_draft",
            summary="WorkflowDraft is the structured output contract for assistant providers.",
            trust_level="system",
            metadata={"target_type": profile},
        ),
        ContentReference(
            reference_id=f"ref_system_safety_{profile}",
            source_type="safety_profile",
            name=profile,
            summary="Allowed imports, dependencies, and required outputs for assistant-reviewed code.",
            trust_level="system",
            metadata=SAFETY_PROFILES.get(profile, {}),
        ),
        ContentReference(
            reference_id="ref_system_alignment_contracts",
            source_type="alignment_contracts",
            name="assistant_alignment_contracts",
            summary="The five alignment contracts that every private assistant draft is evaluated against.",
            trust_level="system",
            metadata=assistant_alignment_contracts(),
        ),
        ContentReference(
            reference_id=f"ref_system_prompt_{profile}",
            source_type="prompt_template",
            name=PROMPT_TEMPLATE_VERSION,
            summary=SLM_PROMPT_TEMPLATES.get(profile, SLM_PROMPT_TEMPLATES["dataset_generation"]),
            trust_level="system",
            metadata={"prompt_template_version": PROMPT_TEMPLATE_VERSION},
        ),
        ContentReference(
            reference_id="ref_system_writing_guide",
            source_type="draft_writing_guide",
            name="amanaje_assistant_writing_guide",
            summary="Project writing guidance for draft titles, summaries, metadata, risks, assumptions, and next actions.",
            trust_level="system",
            metadata=assistant_draft_writing_guide(),
        ),
        ContentReference(
            reference_id="ref_system_quality_rubric",
            source_type="draft_quality_rubric",
            name="amanaje_assistant_quality_rubric",
            summary="Quality checks used to evaluate generated draft text and metadata.",
            trust_level="system",
            metadata=assistant_draft_quality_rubric(),
        ),
        ContentReference(
            reference_id="ref_system_mlflow_tracking",
            source_type="mlflow_tracking",
            name=ASSISTANT_MLFLOW_EXPERIMENT,
            summary="Assistant model versions, evals, training examples, and promotion reports are tracked in MLflow.",
            trust_level="system",
            uri=ASSISTANT_MLFLOW_EXPERIMENT,
            metadata={
                "experiment_name": ASSISTANT_MLFLOW_EXPERIMENT,
                "evaluation_profile": EVALUATION_PROFILE,
                "context_pack_version": CONTEXT_PACK_VERSION,
            },
        ),
    ]
    if registry_type:
        references.append(
            ContentReference(
                reference_id=f"ref_system_registry_{_safe_artifact_name(registry_type)}",
                source_type="registry_schema",
                name=str(registry_type),
                summary=f"Manual registry payload target: {registry_type}.",
                trust_level="system",
                metadata={"registry_type": registry_type},
            )
        )
    for key in (
        "selected_dataset",
        "dataset_summary",
        "selected_model",
        "model_summary",
        "previous_run_summary",
        "mlflow_run_summary",
        "mlflow_experiment_summary",
        "registry_history",
        "assistant_reference_snapshot",
    ):
        value = context.get(key)
        if isinstance(value, Mapping):
            references.append(
                ContentReference(
                    reference_id=f"ref_runtime_{key}",
                    source_type="runtime_context",
                    name=key,
                    summary=_summarize_mapping(value),
                    trust_level="runtime",
                    metadata=dict(value),
                )
            )
    supplied_references = context.get("content_references")
    if isinstance(supplied_references, list):
        for index, value in enumerate(supplied_references[:8]):
            if isinstance(value, Mapping):
                references.append(
                    ContentReference(
                        reference_id=str(value.get("reference_id") or _default_reference_id(index)),
                        source_type=str(value.get("source_type") or "user_supplied_reference"),
                        name=str(value.get("name") or f"reference_{index + 1}"),
                        summary=str(value.get("summary") or _summarize_mapping(value)),
                        trust_level=str(value.get("trust_level") or "user"),
                        uri=value.get("uri"),
                        metadata=dict(value.get("metadata") or {}),
                        content_hash=value.get("content_hash"),
                        tags=list(value.get("tags") or []),
                    )
                )
    return references


def _build_compact_summary(
    *,
    target_type: str,
    registry_type: Any,
    form_values: Mapping[str, Any],
    references: list[ContentReference],
) -> str:
    pieces = [f"target={target_type}"]
    if registry_type:
        pieces.append(f"registry_type={registry_type}")
    if form_values:
        pieces.append(f"form_keys={','.join(sorted(form_values))}")
    pieces.append(f"references={len(references)}")
    return "; ".join(pieces)


def _summarize_mapping(value: Mapping[str, Any]) -> str:
    keys = sorted(str(key) for key in value.keys())
    visible = ", ".join(keys[:8])
    if len(keys) > 8:
        visible += ", ..."
    return f"keys: {visible}" if visible else "empty mapping"
