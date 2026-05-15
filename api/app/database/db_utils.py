import ast
import json
from datetime import datetime, timezone
from typing import Any, Type, List, Optional

from pydantic import BaseModel
from sqlalchemy import DateTime, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.model_orm import AssistantORM, AssistantTrainingDatasetORM, DatasetORM, InferenceORM, LearningORM, ObjectORM, StudyORM

from app.database.db_session import Base
from app.utils.utils import debug_type

# Result.scalars - return _asdict(), _fields(), _mapping()

POLYMORPHIC_IDENTITY_ALIASES = {
    "object_type": {
        "code": "code_model",
        "codes": "code_model",
        "inference": "inference_model",
        "monitor": "inference_model",
        "model": "learning_model",
        "models": "learning_model",
        "study": "study_model",
        "panel": "panel_dashboard",
        "paneldashboard": "panel_dashboard",
        "panel_dashboard": "panel_dashboard",
    },
    "dataset_type": {
        "csv": "dataset",
        "tsv": "dataset",
        "json": "dataset",
        "parquet": "dataset",
        "excel": "dataset",
        "xlsx": "dataset",
        "xls": "dataset",
        "table": "dataset",
        "tabular": "dataset",
        "dataframe": "dataset",
        "assistant_training": "assistant_training_dataset",
        "assistant_training_dataset": "assistant_training_dataset",
        "assistant_dataset": "assistant_training_dataset",
        "assistanttrainingdataset": "assistant_training_dataset",
        "numerical": "numerical_dataset",
        "categorical": "categorical_dataset",
        "timeseries": "timeseries_dataset",
        "time_series": "timeseries_dataset",
        "time-series": "timeseries_dataset",
        "image": "image_dataset",
        "video_audio": "videoaudio_dataset",
        "video-audio": "videoaudio_dataset",
        "text": "text_dataset",
        "mixed": "mixed_dataset",
    },
    "model_type": {
        "assistant": "assistant_model",
        "assistant_llm": "assistant_model",
        "assistant_slm": "assistant_model",
        "assistant_llm_adapter": "assistant_model",
        "assistant_slm_adapter": "assistant_model",
        "llm": "assistant_model",
        "slm": "assistant_model",
        "supervised": "supervised_model",
        "unsupervised": "unsupervised_model",
        "reinforcement": "reinforcement_model",
        "deep": "deep_learning_model",
        "deep_learning": "deep_learning_model",
        "deep-learning": "deep_learning_model",
        "transformer": "transformative_model",
        "transformative": "transformative_model",
        "generative": "generative_model",
        "onnx": "learning_model",
        "sklearn": "learning_model",
        "torch": "learning_model",
        "pytorch": "learning_model",
        "tensorflow": "learning_model",
    },
}

POLYMORPHIC_IDENTITY_TABLES = {
    "object_type": "objects",
    "dataset_type": "datasets",
    "model_type": "learning_models",
}

JSONB_CONTAINER_COLUMNS = {
    "objects": {
        "history": "[",
    },
    "learning_models": {
        "parameters": "{",
        "metrics": "{",
        "input_features": "[",
        "output_features": "[",
    },
    "inference_models": {
        "input_features": "[",
        "output_features": "[",
        "inference_params": "{",
    },
    "code_models": {
        "variables": "{",
        "code": "{",
    },
    "panel_dashboards": {
        "layout": "{",
        "widgets": "[",
        "panel_metadata": "{",
    },
    "study_models": {
        "best_trial": "{",
        "best_params": "{",
        "study_params": "{",
    },
}

ASSISTANT_MODEL_PARAMETER_FIELDS = {
    "provider_type",
    "runtime_kind",
    "base_url",
    "chat_endpoint",
    "health_url",
    "model_name",
    "model_version",
    "base_model_name",
    "api_key_env",
    "bundle_path",
    "bundle_sha256",
    "bundle_manifest_member",
    "bundle_manifest_path",
    "extracted_dir",
    "model_artifact_path",
    "tokenizer_path",
    "tokenizer_assets",
    "chat_template_path",
    "chat_template",
    "adapter_path",
    "device",
    "dtype",
    "quantization",
    "generation_config",
    "max_context_tokens",
    "prompt_template_version",
    "context_pack_version",
    "safety_profile_version",
    "evaluation_profile",
    "supported_draft_types",
    "behavior_profile",
    "reference_policy",
    "training_profile",
    "runtime_config",
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "enabled",
    "is_default",
}


def _parse_container_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    trimmed = value.strip()
    if not trimmed:
        return value

    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(trimmed)
    except (ValueError, SyntaxError):
        return value


def _normalize_datetime_instance(value: Any) -> Any:
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _normalize_datetime_value(value):
    """Convert supported datetime payloads to datetime objects."""
    if not isinstance(value, str):
        return _normalize_datetime_instance(value)

    normalized_value = value.strip()
    if normalized_value.endswith("Z"):
        normalized_value = normalized_value[:-1] + "+00:00"

    try:
        return _normalize_datetime_instance(datetime.fromisoformat(normalized_value))
    except (ValueError, TypeError):
        return value


def _normalize_payload_for_orm(orm_model: Type[ObjectORM], payload: dict) -> dict:
    mapper = getattr(orm_model, "__mapper__", None)
    if mapper is None:
        return payload

    normalized = dict(payload)
    column_by_key = {column.key: column for column in mapper.columns}

    for field, value in list(normalized.items()):
        column = column_by_key.get(field)
        if field == "date" or isinstance(getattr(column, "type", None), DateTime):
            normalized[field] = _normalize_datetime_value(value)
            continue

        column_type = getattr(column, "type", None)
        parsed = _parse_container_literal(value)

        if isinstance(column_type, JSONB):
            if parsed == "":
                normalized[field] = None
            elif isinstance(parsed, (dict, list)):
                normalized[field] = parsed
            continue

        if isinstance(column_type, ARRAY):
            if parsed == "":
                normalized[field] = None
            elif isinstance(parsed, (list, tuple, set)):
                normalized[field] = list(parsed)

    return normalized


def _normalize_polymorphic_identity(discriminator_key: str, value):
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    aliases = POLYMORPHIC_IDENTITY_ALIASES.get(discriminator_key, {})
    return aliases.get(normalized, normalized)


def _resolve_polymorphic_model(orm_model: Type[ObjectORM], payload: dict) -> tuple[Type[ObjectORM], dict]:
    mapper = getattr(orm_model, "__mapper__", None)
    if mapper is None:
        return orm_model, payload

    polymorphic_on = getattr(mapper, "polymorphic_on", None)
    discriminator_key = getattr(polymorphic_on, "key", None)
    if not discriminator_key or discriminator_key not in payload:
        return orm_model, payload

    normalized_identity = _normalize_polymorphic_identity(discriminator_key, payload[discriminator_key])
    if normalized_identity is not None:
        payload[discriminator_key] = normalized_identity

    target_mapper = mapper.polymorphic_map.get(payload[discriminator_key])
    if target_mapper is not None:
        return target_mapper.class_, payload

    base_identity = getattr(mapper, "polymorphic_identity", None)
    if base_identity is not None:
        payload[discriminator_key] = base_identity

    return orm_model, payload


def _sync_legacy_name_payload(orm_model: Type[ObjectORM], payload: dict) -> dict:
    if "name" in payload and hasattr(orm_model, "legacy_name") and "legacy_name" not in payload:
        payload["legacy_name"] = payload["name"]
    return payload


def _pack_assistant_model_payload(orm_model: Type[ObjectORM], payload: dict) -> dict:
    is_assistant_orm = getattr(orm_model, "__name__", "") == "AssistantORM"
    is_assistant_payload = _normalize_polymorphic_identity("model_type", payload.get("model_type")) == "assistant_model"
    if not (is_assistant_orm or is_assistant_payload):
        return payload

    normalized = dict(payload)
    normalized["model_type"] = "assistant_model"
    parameters = _parse_container_literal(normalized.get("parameters") or {})
    if not isinstance(parameters, dict):
        parameters = {}
    assistant_parameters = dict(parameters.get("assistant") or {})
    for field_name in ASSISTANT_MODEL_PARAMETER_FIELDS:
        if field_name in normalized:
            value = normalized.pop(field_name)
            if value is not None:
                assistant_parameters[field_name] = value
    assistant_parameters.setdefault("provider_type", "openai_compatible")
    parameters["assistant"] = assistant_parameters
    normalized["parameters"] = parameters
    return normalized


async def normalize_legacy_polymorphic_identities(db: AsyncSession) -> dict[str, int]:
    """Repair persisted discriminator aliases so ORM polymorphic loading can succeed."""

    normalized_counts: dict[str, int] = {}

    for discriminator_key, aliases in POLYMORPHIC_IDENTITY_ALIASES.items():
        table_name = POLYMORPHIC_IDENTITY_TABLES.get(discriminator_key)
        if not table_name:
            continue

        updated_rows = 0
        for legacy_value, canonical_value in aliases.items():
            result = await db.execute(
                text(
                    f"UPDATE {table_name} "
                    f"SET {discriminator_key} = :canonical_value "
                    f"WHERE lower(trim({discriminator_key})) = :legacy_value"
                ),
                {
                    "canonical_value": canonical_value,
                    "legacy_value": legacy_value,
                },
            )
            updated_rows += result.rowcount or 0

        if updated_rows:
            normalized_counts[discriminator_key] = updated_rows

    if normalized_counts:
        await db.commit()

    return normalized_counts


async def normalize_legacy_jsonb_containers(db: AsyncSession) -> dict[str, int]:
    """Repair JSONB columns that were persisted as JSON strings instead of containers."""

    normalized_counts: dict[str, int] = {}

    for table_name, columns in JSONB_CONTAINER_COLUMNS.items():
        updated_rows = 0

        for column_name, leading_char in columns.items():
            result = await db.execute(
                text(
                    f"UPDATE {table_name} "
                    f"SET {column_name} = ({column_name} #>> '{{}}')::jsonb "
                    f"WHERE {column_name} IS NOT NULL "
                    f"AND jsonb_typeof({column_name}) = 'string' "
                    f"AND left(ltrim({column_name} #>> '{{}}'), 1) = :leading_char"
                ),
                {"leading_char": leading_char},
            )
            updated_rows += result.rowcount or 0

        if updated_rows:
            normalized_counts[table_name] = updated_rows

    if normalized_counts:
        await db.commit()

    return normalized_counts

# Database utilities: create/get/update/delete helpers for ORM models
async def create_entry(db: AsyncSession, orm_model: ObjectORM, data: BaseModel | dict):
    """Create an ORM entry from a Pydantic model or plain dict.

    Accept both a BaseModel instance (with .dict()) or a plain dict so caller
    logic can construct payloads that map to ORM column names.
    """

    # Normalize input to a plain dict. Support Pydantic v2 (.model_dump) and v1 (.dict).
    payload = None

    if isinstance(data, BaseModel) or hasattr(data, "model_dump") or hasattr(data, "dict"):
        try:
            if hasattr(data, "model_dump"):
                payload = data.model_dump()
            elif hasattr(data, "dict"):
                # pydantic v1 dict supports exclude_unset
                payload = data.dict(exclude_unset=True)
            else:
                payload = dict(data)
            print("Has dump or dict.")
        except Exception:
            # fallback to a best-effort conversion
            
            payload = dict(data)
            
            print("Not in format.")
        
    elif isinstance(data, dict):
        payload = data
        print("utils.py:create_entry 45:23 Data is a dict")
    else:
        raise TypeError("data must be a pydantic BaseModel or a dict")

    if not isinstance(payload, dict):
        raise TypeError("normalized payload must be a dict")

    orm_model, payload = _resolve_polymorphic_model(orm_model, payload)
    payload = _pack_assistant_model_payload(orm_model, payload)
    payload = _sync_legacy_name_payload(orm_model, payload)
    payload = _normalize_payload_for_orm(orm_model, payload)

    try:
        obj = orm_model(**payload)
    except TypeError as e:
        print(f"Failed to instantiate {orm_model} with filtered payload: {e}")
        raise

    print("Obj", obj)

    db.add(obj)

    debug_type(obj)
    try:
        print("Creation successful")
        await db.commit()
        await db.refresh(obj)
    except Exception as e:
        print(f"DB commit/refresh failed for {orm_model} with payload {payload}: {e}")
        await db.rollback()
        raise

    return obj

# Resgatando uma entrada de um modelo orm por id ou nome
async def get_entry(db: AsyncSession, orm_model: Type, entry_id: int | str) -> Optional[object]:
    # Try integer id lookup first
    entry = None

    print(f"Getting entry from {orm_model.__name__}...")
    if isinstance(entry_id, int):
        print(f"{orm_model.__name__} item_id == (int)")
        result = await db.execute(select(orm_model).where(orm_model.id == entry_id))
        entry = result.scalars().first()
        debug_type(entry)
        return entry

    # If a numeric string was provided, treat it as an id first. This keeps
    # FastAPI path params like "/get/1" from being misread as a model name.
    if isinstance(entry_id, str):
        normalized_entry_id = entry_id.strip()
        attr = getattr(orm_model, "name", None)
        print(f"{orm_model.__name__} item_id == (str)")

        try:
            iid = int(normalized_entry_id)
            result = await db.execute(select(orm_model).where(orm_model.id == iid))
            entry = result.scalars().first()
            debug_type(entry)
            if entry is not None:
                print("Returned entry by id.")
                return entry
        except (TypeError, ValueError):
            pass

        if attr is not None:
            result = await db.execute(select(orm_model).where(attr == normalized_entry_id))
            entry = result.scalars().first()
            debug_type(entry)
            print("Returned entry by name.")
            return entry

        return None

    return None


# Resgatando todas as entradas de um modelo orm específico
async def get_all_entries(db: AsyncSession, orm_model: Type) -> List:
    "docstring"

    print("List got called", orm_model)
    result = await db.execute(select(orm_model))
    print("Result is printed,", result)
    all_entries = result.scalars().all()

    debug_type(all_entries)

    return all_entries


# Atualizando todas as entradas de um modelo orm específico
async def update_entry(db: AsyncSession, orm_model: ObjectORM, entry_id: int | str, data: BaseModel | dict):
    """Update an ORM entry by id with data from a BaseModel or dict."""

    #debug_type(data)

    obj = await get_entry(db, orm_model, entry_id)

    if not obj:
        return None
    
    if hasattr(data, "dict"):
        # pydantic v2 preferred, fall back to v1
        try:
            updates = data.model_dump(exclude_unset=True)
        except Exception:
            updates = data.dict(exclude_unset=True)
    elif isinstance(data, dict):
        updates = data
    else:
        raise TypeError("data must be a pydantic BaseModel or a dict")

    debug_type(updates)

    updates = _pack_assistant_model_payload(type(obj), updates)
    updates = _normalize_payload_for_orm(type(obj), updates)

    for field, value in updates.items():
        setattr(obj, field, value)
        if field == "name" and hasattr(obj, "legacy_name"):
            setattr(obj, "legacy_name", value)

    await db.commit()
    await db.refresh(obj)

    return obj


async def delete_entry(db: AsyncSession, orm_model:Type, entry_id: int | str) -> bool:
    "docstring"
    
    obj = await get_entry(db, orm_model, entry_id)

    if not obj:
        return False

    await db.delete(obj)
    await db.commit()

    return True


async def get_entry_dependencies(db: AsyncSession, orm_model: Type, entry_id: int | str) -> dict[str, Any]:
    obj = await get_entry(db, orm_model, entry_id)
    if not obj:
        return {
            "exists": False,
            "can_delete": False,
            "dependencies": [],
            "message": f"{getattr(orm_model, '__name__', 'Object')} not found.",
        }

    normalized_id = int(getattr(obj, "id", entry_id))
    dependencies: list[dict[str, Any]] = []

    async def _count(model: Type, column_name: str, label: str) -> None:
        result = await db.execute(select(func.count()).select_from(model).where(getattr(model, column_name) == normalized_id))
        total = int(result.scalar() or 0)
        if total > 0:
            dependencies.append(
                {
                    "label": label,
                    "orm_model": getattr(model, "__name__", str(model)),
                    "count": total,
                    "field": column_name,
                }
            )

    if issubclass(orm_model, LearningORM):
        await _count(StudyORM, "learning_model_id", "Linked studies")
        await _count(InferenceORM, "learning_model_id", "Linked inference pairs")
    elif issubclass(orm_model, DatasetORM):
        await _count(StudyORM, "dataset_id", "Linked studies")
        await _count(InferenceORM, "dataset_id", "Linked inference pairs")

    return {
        "exists": True,
        "can_delete": not dependencies,
        "dependencies": dependencies,
        "message": (
            "This registry item is still referenced by other records."
            if dependencies
            else "No blocking dependencies were found."
        ),
    }
