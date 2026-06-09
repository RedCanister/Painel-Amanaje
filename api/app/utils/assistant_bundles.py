from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from app.utils.assistant_llmops import CONTEXT_PACK_VERSION, PROMPT_TEMPLATE_VERSION, SAFETY_PROFILE_VERSION
from app.utils.io import ensure_dir, save_json


ASSISTANT_MODEL_MANIFEST_NAME = "assistant_model_manifest.json"
ASSISTANT_BUNDLE_REQUIRED_METADATA = (
    "provider_type",
    "model_name",
    "model_version",
    "runtime_kind",
    "base_model_name",
    "supported_draft_types",
    "max_context_tokens",
)
ASSISTANT_BUNDLE_BASE_REQUIRED_METADATA = (
    "provider_type",
    "model_name",
    "model_version",
    "runtime_kind",
    "base_model_name",
    "max_context_tokens",
)
ASSISTANT_MODEL_ASSET_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "model.pt",
}
ASSISTANT_TOKENIZER_ASSET_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
}
ASSISTANT_PROMPT_ASSET_NAMES = {
    "chat_template.txt",
    "chat_template.jinja",
    "prompt_template.txt",
    "prompt_template.json",
}

_SENSITIVE_KEY_PATTERN = re.compile(r"(api[_-]?key|authorization|bearer|password|secret|token)", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"\S+")


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _model_value(model_record: Any, key: str, default: Any = None) -> Any:
    if isinstance(model_record, Mapping):
        return model_record.get(key, default)
    return getattr(model_record, key, default)


def _model_parameters(model_record: Any) -> dict[str, Any]:
    return _as_mapping(_model_value(model_record, "parameters", {}) or {})


def _assistant_parameters(model_record: Any) -> dict[str, Any]:
    return _as_mapping(_model_parameters(model_record).get("assistant") or {})


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("utf-8"))
        try:
            digest.update(_hash_file(item).encode("utf-8"))
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()


def _normalize_zip_members(bundle: zipfile.ZipFile) -> list[str]:
    return [member.filename.replace("\\", "/") for member in bundle.infolist() if not member.is_dir()]


def _member_basename(member: str) -> str:
    return PurePosixPath(member).name


def _find_manifest_member(members: list[str]) -> str | None:
    if ASSISTANT_MODEL_MANIFEST_NAME in members:
        return ASSISTANT_MODEL_MANIFEST_NAME
    matches = [member for member in members if _member_basename(member) == ASSISTANT_MODEL_MANIFEST_NAME]
    return matches[0] if matches else None


def _find_manifest_path(manifest: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\\", "/")
    assets = manifest.get("assets")
    if isinstance(assets, Mapping):
        for key in keys:
            value = assets.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().replace("\\", "/")
    return None


def _member_exists(members: list[str], configured_path: str | None) -> str | None:
    if not configured_path:
        return None
    normalized = configured_path.strip("/").replace("\\", "/")
    if normalized in members:
        return normalized
    matches = [member for member in members if member.endswith(f"/{normalized}") or _member_basename(member) == normalized]
    return matches[0] if matches else None


def _find_named_asset(members: list[str], asset_names: set[str]) -> str | None:
    for member in members:
        if _member_basename(member) in asset_names:
            return member
    return None


def _find_model_asset(members: list[str]) -> str | None:
    exact = _find_named_asset(members, ASSISTANT_MODEL_ASSET_NAMES)
    if exact:
        return exact
    for member in members:
        basename = _member_basename(member)
        if basename.endswith((".safetensors", ".bin", ".pt", ".pth")):
            return member
    return None


def _find_all_named_assets(members: list[str], asset_names: set[str]) -> list[str]:
    return [member for member in members if _member_basename(member) in asset_names]


def _safe_extract_zip(bundle: zipfile.ZipFile, target_dir: Path) -> None:
    target_dir = ensure_dir(target_dir)
    target_root = target_dir.resolve()
    for info in bundle.infolist():
        raw_name = info.filename.replace("\\", "/")
        posix_path = PurePosixPath(raw_name)
        if posix_path.is_absolute() or any(part in {"", ".", ".."} for part in posix_path.parts):
            raise ValueError(f"Unsafe assistant model bundle member path: {info.filename}")
        destination = (target_dir / Path(*posix_path.parts)).resolve()
        if target_root not in destination.parents and destination != target_root:
            raise ValueError(f"Unsafe assistant model bundle member path: {info.filename}")
    bundle.extractall(target_dir)


def _redact_sensitive_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            text_key = str(key)
            redacted[text_key] = "[REDACTED]" if _SENSITIVE_KEY_PATTERN.search(text_key) else _redact_sensitive_values(child)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_values(item) for item in value]
    return value


def _coerce_supported_draft_types(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in value.split(",") if item.strip()]
        value = parsed
    if not isinstance(value, list):
        raise ValueError("assistant_model_manifest.json field 'supported_draft_types' must be a list.")
    draft_types = [str(item).strip() for item in value if str(item).strip()]
    if not draft_types:
        raise ValueError("assistant_model_manifest.json field 'supported_draft_types' cannot be empty.")
    return draft_types


def _assistant_manifest_role(manifest: Mapping[str, Any]) -> str:
    role = str(manifest.get("assistant_role") or manifest.get("role") or "generator").strip().lower()
    return role if role in {"generator", "embedding", "dual"} else "generator"


def _required_manifest_fields(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    role = _assistant_manifest_role(manifest)
    if role == "embedding":
        return (*ASSISTANT_BUNDLE_BASE_REQUIRED_METADATA, "embedding_dimension")
    return ASSISTANT_BUNDLE_REQUIRED_METADATA


def _normalize_provider_type(provider_type: Any, role: str) -> str:
    normalized = str(provider_type or "").strip().lower()
    if role == "embedding":
        if normalized in {"", "openai", "http", "embedding", "embeddings", "openai_compatible"}:
            return "openai_compatible_embeddings"
        if normalized in {"embedding_openai_compatible"}:
            return "openai_compatible_embeddings"
        return normalized
    if normalized in {"openai", "http", "llm", "slm"}:
        return "openai_compatible"
    return normalized or "openai_compatible"


def _coerce_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"assistant_model_manifest.json field '{field_name}' must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"assistant_model_manifest.json field '{field_name}' must be greater than zero.")
    return parsed


def _relative_or_none(root: Path | None, member: str | None) -> str | None:
    if root is None or not member:
        return None
    return str((root / Path(*PurePosixPath(member).parts)).resolve())


def inspect_assistant_model_bundle(
    bundle_path: str | Path,
    *,
    extract_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and optionally extract a PyTorch/HF assistant model bundle.

    This function intentionally does not import torch/transformers or load weights.
    Painel Amanaje stores bundle metadata and delegates generation to a separate
    OpenAI-compatible assistant server.
    """

    path = Path(bundle_path)
    if path.suffix.lower() != ".zip":
        raise ValueError("AssistantModel uploads must be .zip bundles.")
    if not path.exists():
        raise ValueError(f"AssistantModel bundle does not exist: {path}")

    with zipfile.ZipFile(path) as bundle:
        members = _normalize_zip_members(bundle)
        manifest_member = _find_manifest_member(members)
        if manifest_member is None:
            raise ValueError(f"AssistantModel bundle must include {ASSISTANT_MODEL_MANIFEST_NAME}.")
        try:
            manifest = json.loads(bundle.read(manifest_member).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ASSISTANT_MODEL_MANIFEST_NAME} is not valid JSON.") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError(f"{ASSISTANT_MODEL_MANIFEST_NAME} must be a JSON object.")

        assistant_role = _assistant_manifest_role(manifest)
        required_metadata = _required_manifest_fields(manifest)
        missing = [key for key in required_metadata if manifest.get(key) in (None, "", [])]
        if missing:
            raise ValueError(f"AssistantModel manifest is missing required fields: {', '.join(missing)}")

        supported_draft_types = (
            ["embedding_retrieval"]
            if assistant_role == "embedding" and manifest.get("supported_draft_types") in (None, "", [])
            else _coerce_supported_draft_types(manifest.get("supported_draft_types"))
        )
        max_context_tokens = _coerce_positive_int(manifest.get("max_context_tokens"), "max_context_tokens")

        model_asset = _member_exists(
            members,
            _find_manifest_path(manifest, ("model_artifact_path", "model_path", "weights_path", "checkpoint_path")),
        ) or _find_model_asset(members)
        if model_asset is None:
            raise ValueError(
                "AssistantModel bundle must include a model artifact such as model.safetensors, "
                "pytorch_model.bin, adapter_model.safetensors, or model.pt."
            )

        configured_tokenizer = _member_exists(
            members,
            _find_manifest_path(manifest, ("tokenizer_path", "tokenizer_config_path")),
        )
        tokenizer_assets = _find_all_named_assets(members, ASSISTANT_TOKENIZER_ASSET_NAMES)
        if configured_tokenizer and configured_tokenizer not in tokenizer_assets:
            tokenizer_assets.insert(0, configured_tokenizer)
        tokenizer_names = {_member_basename(asset) for asset in tokenizer_assets}
        has_tokenizer = "tokenizer.json" in tokenizer_names or (
            "tokenizer_config.json" in tokenizer_names
            and bool(tokenizer_names.intersection({"vocab.json", "vocab.txt", "merges.txt"}))
        ) or bool(tokenizer_names.intersection({"tokenizer.model", "spiece.model"}))
        if not has_tokenizer:
            raise ValueError(
                "AssistantModel bundle must include tokenizer.json or tokenizer config plus vocab/merges files."
            )

        prompt_asset = _member_exists(
            members,
            _find_manifest_path(manifest, ("chat_template_path", "prompt_template_path")),
        ) or _find_named_asset(members, ASSISTANT_PROMPT_ASSET_NAMES)
        has_prompt = bool(manifest.get("chat_template") or manifest.get("prompt_template_version") or prompt_asset)
        if assistant_role != "embedding" and not has_prompt:
            raise ValueError(
                "AssistantModel bundle must include a chat_template, prompt_template_version, or prompt template file."
            )

        extracted_dir_path = Path(extract_dir) if extract_dir is not None else None
        if extracted_dir_path is not None:
            _safe_extract_zip(bundle, extracted_dir_path)

    manifest_dict = dict(_redact_sensitive_values(dict(manifest)))
    assistant_role = _assistant_manifest_role(manifest)
    provider_type = _normalize_provider_type(manifest.get("provider_type"), assistant_role)
    runtime_kind = str(manifest.get("runtime_kind") or ("embedding_hf_server" if assistant_role == "embedding" else "pytorch_hf_server")).strip()
    base_url = str(manifest.get("base_url") or "http://assistant-server:8080/v1").rstrip("/")
    generation_config = _as_mapping(manifest.get("generation_config") or {})
    prompt_template_version = str(manifest.get("prompt_template_version") or PROMPT_TEMPLATE_VERSION)
    embedding_dimension = (
        _coerce_positive_int(manifest.get("embedding_dimension"), "embedding_dimension")
        if assistant_role == "embedding"
        else manifest.get("embedding_dimension")
    )

    assistant_parameters = {
        "assistant_role": assistant_role,
        "provider_type": provider_type,
        "runtime_kind": runtime_kind,
        "base_url": base_url,
        "chat_endpoint": str(manifest.get("chat_endpoint") or f"{base_url}/chat/completions"),
        "embeddings_endpoint": str(manifest.get("embeddings_endpoint") or f"{base_url}/embeddings"),
        "health_url": manifest.get("health_url") or f"{base_url.rsplit('/v1', 1)[0]}/health",
        "model_name": str(manifest.get("model_name")),
        "model_version": str(manifest.get("model_version")),
        "base_model_name": str(manifest.get("base_model_name")),
        "supported_draft_types": supported_draft_types,
        "max_context_tokens": max_context_tokens,
        "max_sequence_tokens": int(manifest.get("max_sequence_tokens") or max_context_tokens),
        "embedding_dimension": embedding_dimension,
        "pooling_strategy": manifest.get("pooling_strategy") or ("mean" if assistant_role == "embedding" else None),
        "query_instruction": manifest.get("query_instruction"),
        "document_instruction": manifest.get("document_instruction"),
        "normalize_embeddings": bool(manifest.get("normalize_embeddings", True)),
        "max_tokens": int(generation_config.get("max_new_tokens") or manifest.get("max_tokens") or 1600),
        "temperature": float(generation_config.get("temperature") or manifest.get("temperature") or 0.2),
        "timeout_seconds": float(manifest.get("timeout_seconds") or 20),
        "prompt_template_version": prompt_template_version,
        "context_pack_version": str(manifest.get("context_pack_version") or CONTEXT_PACK_VERSION),
        "safety_profile_version": str(manifest.get("safety_profile_version") or SAFETY_PROFILE_VERSION),
        "bundle_path": str(path),
        "bundle_sha256": _hash_file(path),
        "bundle_manifest_member": manifest_member,
        "bundle_manifest_path": _relative_or_none(extracted_dir_path, manifest_member),
        "extracted_dir": str(extracted_dir_path.resolve()) if extracted_dir_path is not None else None,
        "model_artifact_path": _relative_or_none(extracted_dir_path, model_asset) or model_asset,
        "tokenizer_path": _relative_or_none(extracted_dir_path, configured_tokenizer or (tokenizer_assets[0] if tokenizer_assets else None)),
        "tokenizer_assets": [_relative_or_none(extracted_dir_path, asset) or asset for asset in tokenizer_assets],
        "chat_template_path": _relative_or_none(extracted_dir_path, prompt_asset),
        "chat_template": manifest.get("chat_template"),
        "adapter_path": manifest.get("adapter_path") or _relative_or_none(extracted_dir_path, _member_exists(members, "adapter_model.safetensors")),
        "device": manifest.get("device") or manifest.get("device_preference") or "auto",
        "dtype": manifest.get("dtype") or "auto",
        "quantization": manifest.get("quantization") or manifest.get("quantization_config"),
        "generation_config": generation_config,
        "runtime_config": {
            "runtime_kind": runtime_kind,
            "bundle_path": str(path),
            "extracted_dir": str(extracted_dir_path.resolve()) if extracted_dir_path is not None else None,
            "load_in_fastapi": False,
            "server_contract": "openai_compatible_embeddings" if assistant_role == "embedding" else "openai_compatible_chat_completions",
        },
        "capabilities": {
            "registry_backed": True,
            "pytorch_hf_bundle": True,
            "embedding_body": assistant_role in {"embedding", "dual"},
            "separate_server_required": True,
            "loads_in_fastapi": False,
        },
    }

    return {
        "status": "valid",
        "bundle_path": str(path),
        "bundle_sha256": assistant_parameters["bundle_sha256"],
        "manifest_member": manifest_member,
        "manifest": manifest_dict,
        "required_metadata": {key: manifest_dict.get(key) for key in _required_manifest_fields(manifest)},
        "assets": {
            "model_artifact": model_asset,
            "tokenizer_assets": tokenizer_assets,
            "prompt_asset": prompt_asset,
        },
        "assistant_parameters": assistant_parameters,
        "provider_config": {
            "type": provider_type,
            "provider_type": provider_type,
            "runtime_kind": runtime_kind,
            "base_url": assistant_parameters["base_url"],
            "chat_endpoint": assistant_parameters["chat_endpoint"],
            "embeddings_endpoint": assistant_parameters["embeddings_endpoint"],
            "health_url": assistant_parameters["health_url"],
            "model_name": assistant_parameters["model_name"],
            "model_version": assistant_parameters["model_version"],
            "max_tokens": assistant_parameters["max_tokens"],
            "temperature": assistant_parameters["temperature"],
            "timeout_seconds": assistant_parameters["timeout_seconds"],
            "supported_draft_types": supported_draft_types,
            "supports_json_mode": True,
            "supports_remote_auth": bool(manifest.get("api_key_env") or manifest.get("api_key")),
        },
        "warnings": [],
    }


def _directory_members(root: Path) -> list[str]:
    return [
        child.relative_to(root).as_posix()
        for child in sorted(root.rglob("*"))
        if child.is_file()
    ]


def _write_assistant_manifest(directory: Path, manifest: Mapping[str, Any]) -> Path:
    manifest_path = directory / ASSISTANT_MODEL_MANIFEST_NAME
    save_json(dict(manifest), manifest_path, indent=2)
    return manifest_path


def inspect_assistant_model_directory(directory_path: str | Path) -> dict[str, Any]:
    """Validate a directory-style AssistantModel bundle such as a Hugging Face snapshot."""

    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"AssistantModel directory does not exist: {path}")

    members = _directory_members(path)
    manifest_path = path / ASSISTANT_MODEL_MANIFEST_NAME
    if not manifest_path.exists():
        raise ValueError(f"AssistantModel directory must include {ASSISTANT_MODEL_MANIFEST_NAME}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ASSISTANT_MODEL_MANIFEST_NAME} is not valid JSON.") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{ASSISTANT_MODEL_MANIFEST_NAME} must be a JSON object.")

    assistant_role = _assistant_manifest_role(manifest)
    missing = [key for key in _required_manifest_fields(manifest) if manifest.get(key) in (None, "", [])]
    if missing:
        raise ValueError(f"AssistantModel manifest is missing required fields: {', '.join(missing)}")

    supported_draft_types = (
        ["embedding_retrieval"]
        if assistant_role == "embedding" and manifest.get("supported_draft_types") in (None, "", [])
        else _coerce_supported_draft_types(manifest.get("supported_draft_types"))
    )
    max_context_tokens = _coerce_positive_int(manifest.get("max_context_tokens"), "max_context_tokens")
    model_asset = _member_exists(
        members,
        _find_manifest_path(manifest, ("model_artifact_path", "model_path", "weights_path", "checkpoint_path")),
    ) or _find_model_asset(members)
    if model_asset is None:
        raise ValueError(
            "AssistantModel directory must include a model artifact such as a safetensors, bin, pt, or pth file."
        )

    configured_tokenizer = _member_exists(
        members,
        _find_manifest_path(manifest, ("tokenizer_path", "tokenizer_config_path")),
    )
    tokenizer_assets = _find_all_named_assets(members, ASSISTANT_TOKENIZER_ASSET_NAMES)
    if configured_tokenizer and configured_tokenizer not in tokenizer_assets:
        tokenizer_assets.insert(0, configured_tokenizer)
    tokenizer_names = {_member_basename(asset) for asset in tokenizer_assets}
    has_tokenizer = "tokenizer.json" in tokenizer_names or (
        "tokenizer_config.json" in tokenizer_names
        and bool(tokenizer_names.intersection({"vocab.json", "vocab.txt", "merges.txt"}))
    ) or bool(tokenizer_names.intersection({"tokenizer.model", "spiece.model"}))
    if not has_tokenizer:
        raise ValueError(
            "AssistantModel directory must include tokenizer.json, tokenizer.model, or tokenizer config plus vocab/merges files."
        )

    prompt_asset = _member_exists(
        members,
        _find_manifest_path(manifest, ("chat_template_path", "prompt_template_path")),
    ) or _find_named_asset(members, ASSISTANT_PROMPT_ASSET_NAMES)
    has_prompt = bool(manifest.get("chat_template") or manifest.get("prompt_template_version") or prompt_asset)
    if assistant_role != "embedding" and not has_prompt:
        raise ValueError(
            "AssistantModel directory must include a chat_template, prompt_template_version, or prompt template file."
        )

    manifest_dict = dict(_redact_sensitive_values(dict(manifest)))
    provider_type = _normalize_provider_type(manifest.get("provider_type"), assistant_role)
    runtime_kind = str(manifest.get("runtime_kind") or ("embedding_hf_server" if assistant_role == "embedding" else "pytorch_hf_server")).strip()
    base_url = str(manifest.get("base_url") or "http://assistant-server:8080/v1").rstrip("/")
    generation_config = _as_mapping(manifest.get("generation_config") or {})
    prompt_template_version = str(manifest.get("prompt_template_version") or PROMPT_TEMPLATE_VERSION)
    bundle_sha256 = _hash_directory(path)
    embedding_dimension = (
        _coerce_positive_int(manifest.get("embedding_dimension"), "embedding_dimension")
        if assistant_role == "embedding"
        else manifest.get("embedding_dimension")
    )

    assistant_parameters = {
        "assistant_role": assistant_role,
        "provider_type": provider_type,
        "runtime_kind": runtime_kind,
        "base_url": base_url,
        "chat_endpoint": str(manifest.get("chat_endpoint") or f"{base_url}/chat/completions"),
        "embeddings_endpoint": str(manifest.get("embeddings_endpoint") or f"{base_url}/embeddings"),
        "health_url": manifest.get("health_url") or f"{base_url.rsplit('/v1', 1)[0]}/health",
        "model_name": str(manifest.get("model_name")),
        "model_version": str(manifest.get("model_version")),
        "base_model_name": str(manifest.get("base_model_name")),
        "supported_draft_types": supported_draft_types,
        "max_context_tokens": max_context_tokens,
        "max_sequence_tokens": int(manifest.get("max_sequence_tokens") or max_context_tokens),
        "embedding_dimension": embedding_dimension,
        "pooling_strategy": manifest.get("pooling_strategy") or ("mean" if assistant_role == "embedding" else None),
        "query_instruction": manifest.get("query_instruction"),
        "document_instruction": manifest.get("document_instruction"),
        "normalize_embeddings": bool(manifest.get("normalize_embeddings", True)),
        "max_tokens": int(generation_config.get("max_new_tokens") or manifest.get("max_tokens") or 1600),
        "temperature": float(generation_config.get("temperature") or manifest.get("temperature") or 0.2),
        "timeout_seconds": float(manifest.get("timeout_seconds") or 20),
        "prompt_template_version": prompt_template_version,
        "context_pack_version": str(manifest.get("context_pack_version") or CONTEXT_PACK_VERSION),
        "safety_profile_version": str(manifest.get("safety_profile_version") or SAFETY_PROFILE_VERSION),
        "bundle_path": str(path),
        "bundle_sha256": bundle_sha256,
        "bundle_manifest_member": ASSISTANT_MODEL_MANIFEST_NAME,
        "bundle_manifest_path": str(manifest_path.resolve()),
        "extracted_dir": str(path.resolve()),
        "model_artifact_path": _relative_or_none(path, model_asset) or model_asset,
        "tokenizer_path": _relative_or_none(path, configured_tokenizer or (tokenizer_assets[0] if tokenizer_assets else None)),
        "tokenizer_assets": [_relative_or_none(path, asset) or asset for asset in tokenizer_assets],
        "chat_template_path": _relative_or_none(path, prompt_asset),
        "chat_template": manifest.get("chat_template"),
        "adapter_path": manifest.get("adapter_path") or _relative_or_none(path, _member_exists(members, "adapter_model.safetensors")),
        "device": manifest.get("device") or manifest.get("device_preference") or "auto",
        "dtype": manifest.get("dtype") or "auto",
        "quantization": manifest.get("quantization") or manifest.get("quantization_config"),
        "generation_config": generation_config,
        "runtime_config": {
            "runtime_kind": runtime_kind,
            "bundle_path": str(path),
            "extracted_dir": str(path.resolve()),
            "load_in_fastapi": False,
            "server_contract": "openai_compatible_embeddings" if assistant_role == "embedding" else "openai_compatible_chat_completions",
        },
        "capabilities": {
            "registry_backed": True,
            "pytorch_hf_bundle": True,
            "huggingface_directory_bundle": True,
            "embedding_body": assistant_role in {"embedding", "dual"},
            "separate_server_required": True,
            "loads_in_fastapi": False,
        },
    }

    return {
        "status": "valid",
        "bundle_path": str(path),
        "bundle_sha256": bundle_sha256,
        "manifest_member": ASSISTANT_MODEL_MANIFEST_NAME,
        "manifest": manifest_dict,
        "required_metadata": {key: manifest_dict.get(key) for key in _required_manifest_fields(manifest)},
        "assets": {
            "model_artifact": model_asset,
            "tokenizer_assets": tokenizer_assets,
            "prompt_asset": prompt_asset,
        },
        "assistant_parameters": assistant_parameters,
        "provider_config": {
            "type": provider_type,
            "provider_type": provider_type,
            "runtime_kind": runtime_kind,
            "base_url": assistant_parameters["base_url"],
            "chat_endpoint": assistant_parameters["chat_endpoint"],
            "embeddings_endpoint": assistant_parameters["embeddings_endpoint"],
            "health_url": assistant_parameters["health_url"],
            "model_name": assistant_parameters["model_name"],
            "model_version": assistant_parameters["model_version"],
            "max_tokens": assistant_parameters["max_tokens"],
            "temperature": assistant_parameters["temperature"],
            "timeout_seconds": assistant_parameters["timeout_seconds"],
            "supported_draft_types": supported_draft_types,
            "supports_json_mode": True,
            "supports_remote_auth": bool(manifest.get("api_key_env") or manifest.get("api_key")),
        },
        "warnings": [],
    }


def normalize_huggingface_assistant_directory(
    directory_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write/normalize an assistant manifest for a Hugging Face snapshot and validate it."""

    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Hugging Face snapshot directory does not exist: {path}")
    supplied = _as_mapping(metadata)
    manifest_path = path / ASSISTANT_MODEL_MANIFEST_NAME
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    if not isinstance(manifest, Mapping):
        manifest = {}
    manifest = dict(manifest)
    members = _directory_members(path)
    repo_id = str(supplied.get("repo_id") or manifest.get("repo_id") or path.name)
    display_name = str(supplied.get("display_name") or manifest.get("display_name") or repo_id.rsplit("/", 1)[-1])
    assistant_role = str(supplied.get("assistant_role") or manifest.get("assistant_role") or "generator").strip().lower()
    if assistant_role not in {"generator", "embedding", "dual"}:
        assistant_role = "generator"
    model_asset = _member_exists(
        members,
        str(supplied.get("model_artifact_path") or manifest.get("model_artifact_path") or "").strip() or None,
    ) or _find_model_asset(members)
    tokenizer_asset = _member_exists(
        members,
        str(supplied.get("tokenizer_path") or manifest.get("tokenizer_path") or "").strip() or None,
    ) or _find_named_asset(members, ASSISTANT_TOKENIZER_ASSET_NAMES)
    prompt_asset = _member_exists(
        members,
        str(supplied.get("chat_template_path") or manifest.get("chat_template_path") or "").strip() or None,
    ) or _find_named_asset(members, ASSISTANT_PROMPT_ASSET_NAMES)
    manifest.update(
        {
            "assistant_role": assistant_role,
            "provider_type": supplied.get("provider_type")
            or manifest.get("provider_type")
            or ("openai_compatible_embeddings" if assistant_role == "embedding" else "openai_compatible"),
            "runtime_kind": supplied.get("runtime_kind")
            or manifest.get("runtime_kind")
            or ("embedding_hf_server" if assistant_role == "embedding" else "pytorch_hf_server"),
            "model_name": supplied.get("model_name") or manifest.get("model_name") or display_name,
            "model_version": supplied.get("model_version") or manifest.get("model_version") or supplied.get("revision") or "huggingface-import",
            "base_model_name": supplied.get("base_model_name") or manifest.get("base_model_name") or repo_id,
            "supported_draft_types": supplied.get("supported_draft_types")
            or manifest.get("supported_draft_types")
            or (
                ["embedding_retrieval"]
                if assistant_role == "embedding"
                else [
                "dataset_generation",
                "model_generation",
                "registry_object",
                "feature_operations",
                "training_run",
                "study",
                ]
            ),
            "max_context_tokens": supplied.get("max_context_tokens") or manifest.get("max_context_tokens") or 4096,
            "embedding_dimension": supplied.get("embedding_dimension") or manifest.get("embedding_dimension") or (768 if assistant_role == "embedding" else None),
            "pooling_strategy": supplied.get("pooling_strategy") or manifest.get("pooling_strategy") or ("mean" if assistant_role == "embedding" else None),
            "max_sequence_tokens": supplied.get("max_sequence_tokens") or manifest.get("max_sequence_tokens") or supplied.get("max_context_tokens") or manifest.get("max_context_tokens") or 512,
            "query_instruction": supplied.get("query_instruction") or manifest.get("query_instruction"),
            "document_instruction": supplied.get("document_instruction") or manifest.get("document_instruction"),
            "normalize_embeddings": supplied.get("normalize_embeddings") if "normalize_embeddings" in supplied else manifest.get("normalize_embeddings", True),
            "base_url": supplied.get("base_url") or manifest.get("base_url") or "http://assistant-server:8080/v1",
            "prompt_template_version": supplied.get("prompt_template_version")
            or manifest.get("prompt_template_version")
            or PROMPT_TEMPLATE_VERSION,
            "generation_config": {
                **_as_mapping(manifest.get("generation_config") or {}),
                **_as_mapping(supplied.get("generation_config") or {}),
            },
            "repo_id": repo_id,
            "revision": supplied.get("revision") or manifest.get("revision"),
            "display_name": display_name,
        }
    )
    if model_asset:
        manifest["model_artifact_path"] = model_asset
    if tokenizer_asset:
        manifest["tokenizer_path"] = tokenizer_asset
    if prompt_asset:
        manifest["chat_template_path"] = prompt_asset
    if assistant_role != "embedding" and not manifest.get("chat_template") and not manifest.get("chat_template_path"):
        manifest["chat_template"] = (
            "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"
        )
    _write_assistant_manifest(path, manifest)
    return inspect_assistant_model_directory(path)


def assistant_bundle_status_from_model(model_record: Any) -> dict[str, Any]:
    assistant_config = _assistant_parameters(model_record)

    def _path_status(key: str) -> dict[str, Any]:
        value = assistant_config.get(key)
        if not value:
            return {"configured": False, "exists": False, "path": None}
        path = Path(str(value))
        return {
            "configured": True,
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "path": str(path),
        }

    tokenizer_assets = assistant_config.get("tokenizer_assets")
    if isinstance(tokenizer_assets, list):
        tokenizer_asset_status = [
            {"path": str(path), "exists": Path(str(path)).exists()}
            for path in tokenizer_assets
        ]
    else:
        tokenizer_asset_status = []

    bundle_path = _path_status("bundle_path")
    extracted_dir = _path_status("extracted_dir")
    model_artifact = _path_status("model_artifact_path")
    tokenizer = _path_status("tokenizer_path")
    prompt_asset = _path_status("chat_template_path")
    manifest = _path_status("bundle_manifest_path")
    assistant_role = str(assistant_config.get("assistant_role") or "generator").strip().lower()
    ready = bool(
        bundle_path["exists"]
        and (extracted_dir["exists"] or model_artifact["exists"])
        and model_artifact["exists"]
        and (tokenizer["exists"] or any(item["exists"] for item in tokenizer_asset_status))
        and (
            assistant_role == "embedding"
            or prompt_asset["exists"]
            or assistant_config.get("chat_template")
            or assistant_config.get("prompt_template_version")
        )
    )
    return {
        "status": "ready" if ready else ("configured" if bundle_path["configured"] else "missing"),
        "ready": ready,
        "runtime_kind": assistant_config.get("runtime_kind"),
        "provider_type": assistant_config.get("provider_type"),
        "assistant_role": assistant_role,
        "model_name": assistant_config.get("model_name"),
        "model_version": assistant_config.get("model_version"),
        "max_context_tokens": assistant_config.get("max_context_tokens"),
        "embedding_dimension": assistant_config.get("embedding_dimension"),
        "pooling_strategy": assistant_config.get("pooling_strategy"),
        "bundle": bundle_path,
        "extracted_dir": extracted_dir,
        "manifest": manifest,
        "model_artifact": model_artifact,
        "tokenizer": tokenizer,
        "tokenizer_assets": tokenizer_asset_status,
        "prompt_asset": prompt_asset,
        "has_inline_chat_template": bool(assistant_config.get("chat_template")),
        "server": {
            "base_url": assistant_config.get("base_url"),
            "health_url": assistant_config.get("health_url"),
            "separate_server_required": True,
            "loads_in_fastapi": False,
        },
    }


def _load_manifest(manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    path = Path(manifest)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_jsonl_path(manifest: Mapping[str, Any]) -> Path:
    for key in ("canonical_jsonl_path", "dataset_path", "training_dataset_jsonl_path"):
        value = manifest.get(key)
        if value and str(value).lower().endswith(".jsonl"):
            return Path(str(value))
    raise ValueError("Assistant training dataset manifest does not include a canonical JSONL path.")


def _record_prompt(record: Mapping[str, Any]) -> str:
    parts = [
        str(record.get("prompt") or ""),
        str(record.get("workflow_type") or ""),
        str(record.get("asset_summary") or ""),
        str(record.get("content") or ""),
    ]
    return "\n".join(part for part in parts if part.strip()).strip() or "Create a valid Painel Amanaje assistant draft."


def _record_response(record: Mapping[str, Any]) -> str:
    for key in ("draft", "registry_metadata", "review", "quality_signals"):
        value = record.get(key)
        if value:
            return json.dumps(_redact_sensitive_values(value), ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps(_redact_sensitive_values(dict(record)), ensure_ascii=False, sort_keys=True, default=str)


def _estimate_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def prepare_assistant_model_tokenization(
    model_record: Any,
    dataset_manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare instruction-style JSONL examples from canonical assistant JSONL.

    This is a tokenizer preparation contract, not a fine-tuning job. It records
    estimated token counts without requiring transformers in the FastAPI app.
    """

    manifest = _load_manifest(dataset_manifest)
    jsonl_path = _resolve_jsonl_path(manifest)
    if not jsonl_path.exists():
        raise ValueError(f"Assistant training JSONL does not exist: {jsonl_path}")

    assistant_config = _assistant_parameters(model_record)
    max_context_tokens = int(assistant_config.get("max_context_tokens") or manifest.get("max_context_tokens") or 4096)
    output_root = ensure_dir(output_dir)
    recorded_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = _model_value(model_record, "id", "unregistered")
    examples_path = output_root / f"assistant_model_{model_id}_token_examples_{recorded_at}.jsonl"
    manifest_path = output_root / f"assistant_model_{model_id}_token_examples_{recorded_at}_manifest.json"

    total_records = 0
    total_tokens = 0
    max_observed_tokens = 0
    truncation_count = 0
    label_counts: dict[str, int] = {}

    with jsonl_path.open("r", encoding="utf-8") as source, examples_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            prompt = _record_prompt(record)
            response = _record_response(record)
            system = (
                "You are the Painel Amanaje assistant. Return strict WorkflowDraft JSON only. "
                "Use project context, registry schemas, safety rules, and approved dependencies."
            )
            text = f"{system}\n{prompt}\n{response}"
            token_count = _estimate_tokens(text)
            total_records += 1
            total_tokens += token_count
            max_observed_tokens = max(max_observed_tokens, token_count)
            if token_count > max_context_tokens:
                truncation_count += 1
            label = str(record.get("label") or record.get("quality_label") or "unlabeled")
            label_counts[label] = label_counts.get(label, 0) + 1
            target.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                        "metadata": {
                            "source_hash": record.get("source_hash"),
                            "context_pack_hash": record.get("context_pack_hash"),
                            "workflow_type": record.get("workflow_type"),
                            "label": label,
                            "estimated_tokens": token_count,
                            "truncated": token_count > max_context_tokens,
                        },
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    examples_hash = _hash_file(examples_path) if examples_path.exists() else None
    payload = {
        "status": "prepared",
        "assistant_model_id": model_id,
        "assistant_model_name": _model_value(model_record, "name"),
        "created_at": datetime.now().isoformat(),
        "source_jsonl_path": str(jsonl_path),
        "source_manifest_path": str(dataset_manifest) if not isinstance(dataset_manifest, Mapping) else manifest.get("manifest_path"),
        "dataset_hash": manifest.get("dataset_hash"),
        "tabular_csv_path": manifest.get("tabular_csv_path"),
        "examples_path": str(examples_path),
        "examples_hash": examples_hash,
        "manifest_path": str(manifest_path),
        "record_count": total_records,
        "total_estimated_tokens": total_tokens,
        "average_estimated_tokens": (total_tokens / total_records) if total_records else 0,
        "max_observed_tokens": max_observed_tokens,
        "max_context_tokens": max_context_tokens,
        "truncation_count": truncation_count,
        "truncation_rate": (truncation_count / total_records) if total_records else 0,
        "prompt_template_version": assistant_config.get("prompt_template_version") or PROMPT_TEMPLATE_VERSION,
        "context_pack_version": assistant_config.get("context_pack_version") or CONTEXT_PACK_VERSION,
        "dataset_snapshot_hash": manifest.get("dataset_hash"),
        "label_counts": label_counts,
        "notes": "Prepared for assistant fine-tuning outside the FastAPI app; no PyTorch weights were loaded.",
    }
    save_json(payload, manifest_path, indent=2)
    return payload
