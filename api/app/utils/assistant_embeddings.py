from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import numpy as np

from app.models.assistant_objects import ContentReference
from app.utils.assistant_llmops import list_assistant_references
from app.utils.io import ensure_dir, save_json


EMBEDDING_INDEX_VERSION = "assistant-embedding-index-v1"
EMBEDDING_TRAINING_VERSION = "assistant-embedding-training-v1"
DEFAULT_EMBEDDING_DIMENSION = 128
DEFAULT_EMBEDDING_MODEL_VERSION = "keyword-hash-embedding-fallback-v1"
EMBEDDING_PROVIDER_TYPES = {"openai_compatible_embeddings", "embedding_openai_compatible"}
EMBEDDING_ROLES = {"embedding", "dual"}
GENERATOR_ROLES = {"generator", "dual", ""}


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


def assistant_model_role(model_record: Any) -> str:
    assistant = _assistant_config(model_record)
    role = str(assistant.get("assistant_role") or _model_value(model_record, "assistant_role") or "generator").strip().lower()
    return role if role in {"generator", "embedding", "dual"} else "generator"


def assistant_model_matches_role(model_record: Any, role: str | None) -> bool:
    normalized = str(role or "all").strip().lower()
    if normalized in {"", "all", "any"}:
        return True
    model_role = assistant_model_role(model_record)
    if normalized == "embedding":
        return model_role in EMBEDDING_ROLES
    if normalized == "generator":
        return model_role in GENERATOR_ROLES
    return model_role == normalized


def assistant_embedding_model_config(model_record: Any | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    assistant = _assistant_config(model_record) if model_record is not None else {}
    merged = {**assistant, **dict(overrides or {})}
    base_url = str(merged.get("embedding_base_url") or merged.get("base_url") or os.getenv("AMANAJE_EMBEDDING_BASE_URL") or "").rstrip("/")
    if base_url.endswith("/v1"):
        embeddings_endpoint = f"{base_url}/embeddings"
    elif base_url:
        embeddings_endpoint = f"{base_url}/v1/embeddings"
    else:
        embeddings_endpoint = ""
    dimension = int(merged.get("embedding_dimension") or merged.get("dimension") or DEFAULT_EMBEDDING_DIMENSION)
    return {
        "assistant_model_id": _model_value(model_record, "id") if model_record is not None else None,
        "assistant_model_name": _model_value(model_record, "name") if model_record is not None else None,
        "assistant_role": str(merged.get("assistant_role") or "embedding"),
        "provider_type": str(merged.get("provider_type") or "openai_compatible_embeddings"),
        "runtime_kind": str(merged.get("runtime_kind") or "embedding_hf_server"),
        "base_url": base_url,
        "embeddings_endpoint": str(merged.get("embeddings_endpoint") or embeddings_endpoint),
        "model_name": str(merged.get("model_name") or _model_value(model_record, "name", "amanaje-embedding-body")),
        "model_version": str(merged.get("model_version") or _model_value(model_record, "version", DEFAULT_EMBEDDING_MODEL_VERSION)),
        "embedding_dimension": dimension,
        "pooling_strategy": str(merged.get("pooling_strategy") or "mean"),
        "max_sequence_tokens": int(merged.get("max_sequence_tokens") or merged.get("max_context_tokens") or 512),
        "query_instruction": str(merged.get("query_instruction") or "Represent this Painel Amanaje user request for retrieval:"),
        "document_instruction": str(merged.get("document_instruction") or "Represent this Painel Amanaje project reference for retrieval:"),
        "normalize_embeddings": _coerce_bool(merged.get("normalize_embeddings"), True),
        "timeout_seconds": float(merged.get("timeout_seconds") or 15),
        "api_key": merged.get("api_key"),
        "api_key_env": merged.get("api_key_env"),
    }


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _safe_text(value: Any, max_length: int = 1800) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3].rstrip()}..."


def _deterministic_embedding(text: str, dimension: int) -> list[float]:
    vector = np.zeros(max(8, int(dimension or DEFAULT_EMBEDDING_DIMENSION)), dtype=np.float32)
    tokens = [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split() if token]
    if not tokens:
        tokens = ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % vector.shape[0]
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log1p(len(token)))
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return vector.astype(float).tolist()


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class OpenAICompatibleEmbeddingClient:
    def embed(self, texts: list[str], config: Mapping[str, Any]) -> list[list[float]]:
        endpoint = str(config.get("embeddings_endpoint") or "").strip()
        if not endpoint:
            raise RuntimeError("Embedding endpoint is not configured.")
        body = {"model": config.get("model_name"), "input": texts}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = str(config.get("api_key") or "").strip()
        api_key_env = str(config.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = UrlRequest(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=float(config.get("timeout_seconds") or 15)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"Embedding runtime returned HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Embedding runtime timed out.") from exc
        except URLError as exc:
            raise RuntimeError(f"Embedding runtime is unavailable: {exc.reason}") from exc
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list):
            raise RuntimeError("Embedding runtime response must include a data list.")
        vectors = []
        for item in data:
            if not isinstance(item, Mapping) or not isinstance(item.get("embedding"), list):
                raise RuntimeError("Embedding runtime data item missing embedding list.")
            vectors.append([float(value) for value in item["embedding"]])
        return vectors


def embed_texts(texts: list[str], config: Mapping[str, Any] | None = None, *, purpose: str = "document") -> tuple[np.ndarray, dict[str, Any]]:
    embedding_config = assistant_embedding_model_config(overrides=config or {})
    dimension = int(embedding_config["embedding_dimension"])
    instruction_key = "query_instruction" if purpose == "query" else "document_instruction"
    instructed = [f"{embedding_config[instruction_key]}\n{text}" for text in texts]
    runtime_error = None
    vectors: list[list[float]] | None = None
    if embedding_config.get("embeddings_endpoint"):
        try:
            vectors = OpenAICompatibleEmbeddingClient().embed(instructed, embedding_config)
        except Exception as exc:
            runtime_error = str(exc)
    if vectors is None:
        vectors = [_deterministic_embedding(text, dimension) for text in instructed]
    matrix = np.asarray(vectors, dtype=np.float32)
    if bool(embedding_config.get("normalize_embeddings", True)):
        matrix = _normalize_matrix(matrix)
    return matrix, {
        "model_name": embedding_config.get("model_name"),
        "model_version": embedding_config.get("model_version"),
        "embedding_dimension": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape else dimension,
        "provider_type": embedding_config.get("provider_type"),
        "runtime_kind": embedding_config.get("runtime_kind"),
        "runtime_fallback": runtime_error is not None or not embedding_config.get("embeddings_endpoint"),
        "runtime_error": runtime_error,
        "purpose": purpose,
    }


def _reference_record(reference: ContentReference) -> dict[str, Any]:
    excerpts = reference.metadata.get("excerpts") if isinstance(reference.metadata, Mapping) else None
    text = "\n".join(
        part
        for part in [
            reference.name or "",
            reference.summary or "",
            " ".join(str(item) for item in excerpts or []),
            " ".join(reference.tags or []),
        ]
        if part
    )
    return {
        "source_type": reference.source_type,
        "reference_id": reference.reference_id,
        "name": reference.name,
        "summary": reference.summary,
        "text": _safe_text(text),
        "content_hash": reference.content_hash or _hash_text(text),
        "trust_level": reference.trust_level,
        "tags": list(reference.tags or []),
        "uri": reference.uri,
        "metadata": {
            key: value
            for key, value in dict(reference.metadata or {}).items()
            if key in {"source_path", "source_extension", "registry_type", "object_id", "file_name"}
        },
    }


def _project_example_records(project_examples: Mapping[str, Any] | None, *, limit: int = 250) -> list[dict[str, Any]]:
    if not isinstance(project_examples, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for entry in project_examples.get("objects") or []:
        if not isinstance(entry, Mapping):
            continue
        object_name = str(entry.get("object_name") or entry.get("object_key") or "ProjectExample")
        family = str(entry.get("family") or "project_examples")
        for sample in entry.get("samples") or []:
            if not isinstance(sample, Mapping):
                continue
            payload = sample.get("payload")
            if payload is None:
                continue
            tier = str(sample.get("tier") or "sample")
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            source_hash = _hash_text(f"{object_name}:{tier}:{text}")
            rows.append(
                {
                    "source_type": "project_example_catalog",
                    "reference_id": f"example_{object_name}_{tier}_{source_hash[:10]}",
                    "name": f"{object_name} {tier}",
                    "summary": f"{family}.{object_name} validated {tier} example.",
                    "text": _safe_text(text),
                    "content_hash": source_hash,
                    "trust_level": "system",
                    "tags": ["project_example", family, object_name],
                    "uri": None,
                    "metadata": {"object_name": object_name, "tier": tier, "catalog_version": project_examples.get("version")},
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def collect_embedding_source_records(
    reference_dir: str | Path,
    *,
    project_examples: Mapping[str, Any] | None = None,
    limit: int = 2_000,
) -> list[dict[str, Any]]:
    records = [_reference_record(reference) for reference in list_assistant_references(reference_dir, limit=limit)]
    records.extend(_project_example_records(project_examples))
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("content_hash") or record.get("reference_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def rebuild_embedding_index(
    reference_dir: str | Path,
    index_dir: str | Path,
    *,
    embedding_model: Any | None = None,
    embedding_config: Mapping[str, Any] | None = None,
    project_examples: Mapping[str, Any] | None = None,
    limit: int = 2_000,
) -> dict[str, Any]:
    output_dir = ensure_dir(index_dir)
    records = collect_embedding_source_records(reference_dir, project_examples=project_examples, limit=limit)
    config = assistant_embedding_model_config(embedding_model, overrides=embedding_config)
    texts = [str(record.get("text") or record.get("summary") or record.get("name") or "") for record in records]
    started_at = time.perf_counter()
    vectors, embedding_metadata = embed_texts(texts, config, purpose="document") if texts else (
        np.zeros((0, int(config["embedding_dimension"])), dtype=np.float32),
        {"runtime_fallback": True, "embedding_dimension": int(config["embedding_dimension"]), "model_version": config["model_version"]},
    )
    vectors_path = output_dir / "vectors.npz"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    np.savez_compressed(vectors_path, vectors=vectors)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps({**record, "index": index}, ensure_ascii=False, default=str) + "\n")
    index_hash = _hash_file(vectors_path, metadata_path)
    manifest = {
        "status": "ready",
        "version": EMBEDDING_INDEX_VERSION,
        "created_at": datetime.now().isoformat(),
        "index_dir": str(output_dir),
        "vectors_path": str(vectors_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "record_count": len(records),
        "embedding_dimension": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape else int(config["embedding_dimension"]),
        "index_hash": index_hash,
        "embedding_model_id": config.get("assistant_model_id"),
        "embedding_model_name": config.get("assistant_model_name") or config.get("model_name"),
        "embedding_model_version": embedding_metadata.get("model_version") or config.get("model_version"),
        "pooling_strategy": config.get("pooling_strategy"),
        "runtime_fallback": bool(embedding_metadata.get("runtime_fallback")),
        "runtime_error": embedding_metadata.get("runtime_error"),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
    }
    save_json(manifest, manifest_path, indent=2)
    return manifest


def _hash_file(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        if path.exists():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def embedding_index_status(index_dir: str | Path) -> dict[str, Any]:
    directory = Path(index_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {
            "status": "missing",
            "index_dir": str(directory),
            "vectors_path": str(directory / "vectors.npz"),
            "metadata_path": str(directory / "metadata.jsonl"),
            "manifest_path": str(manifest_path),
            "ready": False,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "invalid", "index_dir": str(directory), "ready": False, "error": str(exc)}
    vectors_path = Path(str(manifest.get("vectors_path") or directory / "vectors.npz"))
    metadata_path = Path(str(manifest.get("metadata_path") or directory / "metadata.jsonl"))
    return {
        **manifest,
        "ready": vectors_path.exists() and metadata_path.exists(),
        "vectors_exists": vectors_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }


def _load_index(index_dir: str | Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    status = embedding_index_status(index_dir)
    if not status.get("ready"):
        raise RuntimeError("Assistant embedding index is missing. Rebuild it first.")
    vectors = np.load(str(status["vectors_path"]))["vectors"].astype(np.float32)
    metadata: list[dict[str, Any]] = []
    with Path(str(status["metadata_path"])).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                metadata.append(record)
    return vectors, metadata, status


def search_embedding_index(
    index_dir: str | Path,
    query: str,
    *,
    embedding_model: Any | None = None,
    embedding_config: Mapping[str, Any] | None = None,
    tags: Iterable[str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    vectors, metadata, status = _load_index(index_dir)
    config = assistant_embedding_model_config(embedding_model, overrides=embedding_config)
    query_vector, query_meta = embed_texts([query], config, purpose="query")
    if vectors.size == 0 or not metadata:
        return {**status, "status": "empty", "results": [], "query": query}
    if query_vector.shape[1] != vectors.shape[1]:
        query_vector = np.asarray([_deterministic_embedding(query, vectors.shape[1])], dtype=np.float32)
    query_vector = _normalize_matrix(query_vector)
    scored = vectors @ query_vector[0]
    wanted_tags = {str(tag).strip().lower() for tag in tags or [] if str(tag).strip()}
    rows: list[dict[str, Any]] = []
    for index, score in enumerate(scored.tolist()):
        if index >= len(metadata):
            continue
        record = dict(metadata[index])
        if wanted_tags:
            record_tags = {str(tag).strip().lower() for tag in record.get("tags") or []}
            if not wanted_tags.intersection(record_tags):
                score -= 0.05
        rows.append(
            {
                "score": round(float(score), 6),
                "reference_id": record.get("reference_id"),
                "source_type": record.get("source_type"),
                "name": record.get("name"),
                "summary": record.get("summary"),
                "trust_level": record.get("trust_level"),
                "content_hash": record.get("content_hash"),
                "tags": record.get("tags") or [],
                "metadata": record.get("metadata") or {},
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    selected = rows[: max(1, min(int(limit or 8), 12))]
    retrieval_hash = _hash_text(json.dumps({"query": query, "index_hash": status.get("index_hash"), "selected": selected}, sort_keys=True, default=str))
    return {
        "status": "ok",
        "query": query,
        "results": selected,
        "result_count": len(selected),
        "embedding_index_hash": status.get("index_hash"),
        "embedding_retrieval_hash": retrieval_hash,
        "embedding_model_id": config.get("assistant_model_id"),
        "embedding_model_version": query_meta.get("model_version") or config.get("model_version"),
        "runtime_fallback": bool(query_meta.get("runtime_fallback")),
        "runtime_error": query_meta.get("runtime_error"),
        "index": status,
    }


def prepare_embedding_training_examples(
    dataset_dir: str | Path,
    reference_dir: str | Path,
    output_dir: str | Path,
    *,
    project_examples: Mapping[str, Any] | None = None,
    embedding_model: Any | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    output_root = ensure_dir(output_dir)
    recorded_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    examples_path = output_root / f"embedding_training_examples_{recorded_at}.jsonl"
    manifest_path = output_root / f"embedding_training_examples_{recorded_at}_manifest.json"
    source_docs = collect_embedding_source_records(reference_dir, project_examples=project_examples, limit=2_000)
    interactions_path = dataset_path / "interactions.jsonl"
    queries: list[dict[str, Any]] = []
    if interactions_path.exists():
        for line in interactions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(record.get("label") or "unlabeled").lower()
            queries.append(
                {
                    "query": _safe_text(record.get("prompt") or record.get("workflow_type") or "Painel Amanaje assistant draft"),
                    "label": "negative" if label in {"negative", "unsafe", "rejected"} else "positive",
                    "workflow_type": record.get("workflow_type") or "assistant_interaction",
                    "source_hash": record.get("source_hash") or record.get("context_pack_hash") or _hash_text(json.dumps(record, default=str, sort_keys=True)),
                }
            )
    if not queries:
        for doc in source_docs[:100]:
            queries.append(
                {
                    "query": f"Find project context for {doc.get('name') or doc.get('source_type')}",
                    "label": "positive",
                    "workflow_type": doc.get("source_type") or "reference",
                    "source_hash": doc.get("content_hash"),
                }
            )
    record_count = 0
    label_counts: dict[str, int] = {}
    with examples_path.open("w", encoding="utf-8") as handle:
        for index, query_record in enumerate(queries):
            positive_doc = source_docs[index % len(source_docs)] if source_docs else {}
            negative_doc = source_docs[(index + max(1, len(source_docs) // 2)) % len(source_docs)] if len(source_docs) > 1 else {}
            label = str(query_record.get("label") or "positive")
            payload = {
                "query": query_record.get("query"),
                "positive_text": positive_doc.get("text") or positive_doc.get("summary") or "",
                "negative_text": negative_doc.get("text") or negative_doc.get("summary") or "",
                "label": label,
                "source_type": positive_doc.get("source_type") or "assistant_training",
                "workflow_type": query_record.get("workflow_type"),
                "reference_ids": [item for item in [positive_doc.get("reference_id"), negative_doc.get("reference_id")] if item],
                "source_hash": query_record.get("source_hash") or positive_doc.get("content_hash"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            record_count += 1
            label_counts[label] = label_counts.get(label, 0) + 1
    examples_hash = _hash_file(examples_path)
    config = assistant_embedding_model_config(embedding_model)
    manifest = {
        "status": "prepared",
        "version": EMBEDDING_TRAINING_VERSION,
        "created_at": datetime.now().isoformat(),
        "examples_path": str(examples_path),
        "manifest_path": str(manifest_path),
        "examples_hash": examples_hash,
        "record_count": record_count,
        "label_counts": label_counts,
        "source_document_count": len(source_docs),
        "interaction_query_count": len(queries),
        "embedding_model_id": config.get("assistant_model_id"),
        "embedding_model_name": config.get("assistant_model_name"),
        "embedding_model_version": config.get("model_version"),
        "ready_for_fine_tuning": record_count > 0,
        "notes": "Prepared query-positive-negative examples for embedding fine-tuning outside FastAPI.",
    }
    save_json(manifest, manifest_path, indent=2)
    return manifest
