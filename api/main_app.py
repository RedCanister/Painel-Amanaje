from __future__ import annotations

import ast
import base64
import json
import os
import sys
import textwrap
import traceback
import tracemalloc
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_session import AsyncSessionLocal, get_db, init_models
from app.database.db_utils import (
    create_entry,
    get_all_entries,
    normalize_legacy_jsonb_containers,
    normalize_legacy_polymorphic_identities,
    update_entry,
)
from app.models.model_objects import CodeModel, DatasetModel, LearningModel, StudyModel
from app.models.model_orm import CodeORM, DatasetORM, LearningORM, StudyORM
from app.models.model_registry import ModelRegistry
from app.utils.config import merge_env_overrides, save_config_snapshot
from app.utils.deployment_utils import (
    create_fastapi_serving_file,
    get_model_uri_for_run,
    save_deployment_summary,
    save_model_local,
)
from app.utils.io import ensure_dir, load_csv, save_json
from app.utils.logging import get_logger, init_global_logging, log_to_mlflow
from app.utils.metrics import evaluate_and_log_metrics, pretty_print_metrics
from app.utils.mlflow_utils import (
    configure_tracking,
    get_experiment_summary,
    list_experiments,
    log_json as log_mlflow_json,
    log_metrics as log_mlflow_metrics,
    managed_run,
    resume_run,
    set_tags as set_mlflow_tags,
)
from app.utils.monitoring import (
    detect_data_drift,
    detect_prediction_drift,
    log_monitoring_results_to_mlflow,
    summarize_monitoring_results,
    track_performance,
)
from app.utils.optuna_utils import optimize_with_tracking
from app.utils.plotting import plot_loss_curve, plot_metric_comparison, plot_predictions_vs_actual
from app.utils.retrain_utils import monitor_and_retrain, should_retrain
from app.utils.serialization import to_json
from app.utils.training import (
    TrainingResult,
    run_training_pipeline,
    save_training_summary,
    train_pytorch,
    train_sklearn,
)
from app.utils.utils import (
    build_payload_data,
    build_payload_model,
    get_upload_dir,
    recover_model_params,
    save_file_to_disk,
    debug_type
)
from app.utils.validation import validate_dataframe_columns, validate_input, validate_required_keys

try:
    import mlflow

    MLFLOW_DIRECT_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_DIRECT_AVAILABLE = False


tracemalloc.start()

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


def _ensure_env_path(env_key: str, default_path: Path) -> Path:
    override = os.getenv(env_key)
    return ensure_dir(Path(override) if override else default_path)


RUNTIME_DIR = ensure_dir(PROJECT_ROOT / "runtime_artifacts")

CONFIG_SNAPSHOT_DIR = ensure_dir(RUNTIME_DIR / "config")
TRAINING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "training")
MONITORING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "monitoring")
DEPLOYMENT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "deployment")
OPTUNA_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "optuna")
PLOT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "plots")
SERVING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "serving")

LOG_DIR = ensure_dir(PROJECT_ROOT / "logs")
MLRUNS_DIR = _ensure_env_path("MLFLOW_LOCAL_DIR", PROJECT_ROOT / "mlruns")

DATASET_OPERATION = "datasets"
MODEL_OPERATION = "models"
ALLOWED_UPLOAD_OPERATIONS = {DATASET_OPERATION, MODEL_OPERATION}
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
CONTROL_PARAMETER_KEYS = {
    "framework",
    "input_features",
    "output_features",
    "test_size",
    "random_state",
    "estimator_class",
    "experiment_name",
    "run_name",
    "task_type",
    "epochs",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "hidden_layers",
    "dropout",
    "n_trials",
    "direction",
    "metric",
    "search_space",
    "objective_metric",
    "plot_results",
}


APP_LOGGER = init_global_logging(log_dir=str(LOG_DIR), logger_name="painel_amanaje")
LOGGER = get_logger("main_app", log_file=str(LOG_DIR / "main_app.log"))


def _build_runtime_config() -> dict[str, Any]:
    fallback_tracking_uri = MLRUNS_DIR.resolve().as_uri()
    base_config = {
        "app": {"name": "Painel Amanaje API", "version": "0.3.0"},
        "mlflow": {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", fallback_tracking_uri),
            "registry_uri": os.getenv(
                "MLFLOW_REGISTRY_URI",
                os.getenv("MLFLOW_TRACKING_URI", fallback_tracking_uri),
            ),
        },
        "artifacts": {
            "runtime_dir": str(RUNTIME_DIR),
            "training_dir": str(TRAINING_ARTIFACT_DIR),
            "monitoring_dir": str(MONITORING_ARTIFACT_DIR),
            "deployment_dir": str(DEPLOYMENT_ARTIFACT_DIR),
            "optuna_dir": str(OPTUNA_ARTIFACT_DIR),
            "plots_dir": str(PLOT_ARTIFACT_DIR),
            "serving_dir": str(SERVING_ARTIFACT_DIR),
        },
    }
    runtime_config = merge_env_overrides(base_config)
    save_config_snapshot(runtime_config, str(CONFIG_SNAPSHOT_DIR / "runtime_config.json"))
    return runtime_config


RUNTIME_CONFIG = _build_runtime_config()
configure_tracking(
    tracking_uri=RUNTIME_CONFIG["mlflow"]["tracking_uri"],
    registry_uri=RUNTIME_CONFIG["mlflow"]["registry_uri"],
)


class ExecuteRequest(BaseModel):
    code: str = ""
    save: bool = False
    objectName: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str = ""
    operationId: str = ""


class TrainingRequest(BaseModel):
    datasetId: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputType: str = "Manual"
    studyId: Optional[int | str] = None
    model_config = ConfigDict(extra="allow")


class StudyOptimizationRequest(BaseModel):
    nTrials: Optional[int] = None
    outputDir: Optional[str] = None
    plotResults: bool = True
    model_config = ConfigDict(extra="allow")


class OnnxPrepareRequest(BaseModel):
    model_id: Optional[int] = None
    dataset_id: Optional[int] = None
    study_id: Optional[int | str] = None
    framework: Optional[str] = None
    export_target: str = "onnx"
    parameters: dict[str, Any] = Field(default_factory=dict)


class ProductionStartRequest(BaseModel):
    modelId: Optional[int] = None
    datasetId: Optional[int] = None
    model_config = ConfigDict(extra="allow")


class ProductionMonitorRequest(BaseModel):
    modelId: Optional[int] = None
    datasetId: Optional[int] = None
    autoRetrain: bool = False
    tolerance: float = 0.05
    model_config = ConfigDict(extra="allow")


@dataclass
class PreparedDataset:
    dataframe: pd.DataFrame
    feature_frame: pd.DataFrame
    target_series: pd.Series
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    input_features: list[str]
    output_feature: str
    task_type: str
    label_mapping: dict[str, int] = field(default_factory=dict)


app = FastAPI(title="Painel Amanaje API", version="0.3.0")

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


PRODUCTION_STATE: dict[str, Any] = {
    "running": False,
    "status": "inactive",
    "model_id": None,
    "dataset_id": None,
    "started_at": None,
    "stopped_at": None,
    "deployment": {},
    "metrics": {},
    "monitoring": {},
    "active_model": None,
    "logs": [],
}


def _register_models() -> None:
    registry_pairs = (
        (DatasetModel, DatasetORM),
        (LearningModel, LearningORM),
        (CodeModel, CodeORM),
        (StudyModel, StudyORM),
    )
    for pydantic_model, orm_model in registry_pairs:
        if ModelRegistry._registry.get(pydantic_model) is not orm_model:
            ModelRegistry.register_model(pydantic_model, orm_model)


def _include_registry_routers() -> None:
    existing_prefixes = {route.path for route in app.router.routes}
    for router in ModelRegistry.generate_all_routers():
        first_route = next(iter(router.routes), None)
        if first_route and first_route.path not in existing_prefixes:
            app.include_router(router)
            existing_prefixes.update(route.path for route in router.routes)


def _render_page(template_name: str, request_: Request, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(
            name = template_name, 
            request = request_,
            context = context 
        )


def _json_payload(payload: Any) -> Any:
    return json.loads(to_json(payload))


def _json_response(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(_json_payload(payload), status_code=status_code)


def _json_error(detail: str, status_code: int = 400, **extra: Any) -> JSONResponse:
    return _json_response({"status": "error", "detail": detail, **extra}, status_code=status_code)


async def _database_healthcheck() -> tuple[bool, Optional[str]]:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        return False, str(exc)


def _append_history(history: Any, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = list(history) if isinstance(history, list) else []
    items.append(dict(entry))
    return items


def _append_production_log(message: str) -> None:
    timestamped_message = f"{datetime.now().isoformat()} | {message}"
    PRODUCTION_STATE["logs"] = list(PRODUCTION_STATE.get("logs", []))[-99:] + [timestamped_message]
    LOGGER.info(message)


def _serialize_dataset_summary(dataset: Any) -> dict[str, Any]:
    return {
        "id": getattr(dataset, "id", None),
        "name": getattr(dataset, "name", None),
        "description": getattr(dataset, "description", None),
        "dataset_type": getattr(dataset, "dataset_type", None),
        "shape": getattr(dataset, "shape", None),
        "size": getattr(dataset, "size", None),
        "features_list": getattr(dataset, "features_list", None),
        "has_features": getattr(dataset, "has_features", None),
        "path": getattr(dataset, "path", None),
    }


def _serialize_model_summary(model: Any) -> dict[str, Any]:
    return {
        "id": getattr(model, "id", None),
        "name": getattr(model, "name", None),
        "description": getattr(model, "description", None),
        "model_type": getattr(model, "model_type", None),
        "is_trained": getattr(model, "is_trained", False),
        "is_tested": getattr(model, "is_tested", False),
        "is_deployed": getattr(model, "is_deployed", False),
        "metrics": getattr(model, "metrics", {}) or {},
        "parameters": getattr(model, "parameters", {}) or {},
        "input_features": getattr(model, "input_features", None),
        "output_features": getattr(model, "output_features", None),
        "size": getattr(model, "size", None),
        "path": getattr(model, "path", None),
    }


def _serialize_study_summary(study: Any) -> dict[str, Any]:
    return {
        "id": getattr(study, "id", None),
        "name": getattr(study, "name", None),
        "description": getattr(study, "description", None),
        "learning_model_id": getattr(study, "learning_model_id", None),
        "dataset_id": getattr(study, "dataset_id", None),
        "sampler": getattr(study, "sampler", None),
        "objective": getattr(study, "objective", None),
        "best_trial": getattr(study, "best_trial", None),
        "best_params": getattr(study, "best_params", None),
        "study_params": getattr(study, "study_params", None),
        "path": getattr(study, "path", None),
    }


def _build_training_tracking_context(
    prepared: PreparedDataset,
    model_record: Any,
    parameters: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any]:
    train_rows, train_columns = prepared.x_train.shape
    test_rows, test_columns = prepared.x_test.shape
    return {
        "framework": framework,
        "task_type": prepared.task_type,
        "model": {
            "id": getattr(model_record, "id", None),
            "name": getattr(model_record, "name", None),
            "model_type": getattr(model_record, "model_type", None),
        },
        "dataset_profile": {
            "input_features": list(prepared.input_features),
            "output_feature": prepared.output_feature,
            "feature_count": len(prepared.input_features),
            "train_shape": [int(train_rows), int(train_columns)],
            "test_shape": [int(test_rows), int(test_columns)],
            "label_mapping": dict(prepared.label_mapping),
        },
        "parameters": dict(parameters),
    }


def _log_training_tracking_context(
    result: TrainingResult,
    prepared: PreparedDataset,
    model_record: Any,
    parameters: Mapping[str, Any],
    *,
    framework: str,
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    with resume_run(result.run_id) as tracking_active:
        if not tracking_active:
            return

        set_mlflow_tags(
            {
                "run_type": "training",
                "framework": framework,
                "task_type": prepared.task_type,
                "model_id": str(getattr(model_record, "id", "")),
                "model_name": str(getattr(model_record, "name", "")),
                "feature_count": str(len(prepared.input_features)),
                "output_feature": prepared.output_feature,
            }
        )
        payload = _build_training_tracking_context(
            prepared,
            model_record,
            parameters,
            framework=framework,
        )
        if extra_payload:
            payload["extra"] = dict(extra_payload)
        log_mlflow_json(payload, filename="training_context.json", artifact_path="context")


def _json_safe_exec_value(value: Any) -> tuple[Any, dict[str, Any]]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, {}
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii"), {"encoding": "base64"}
    if isinstance(value, tuple):
        return list(value), {"source_type": "tuple"}
    if isinstance(value, list):
        return value, {}
    if isinstance(value, dict):
        return value, {}
    if hasattr(value, "tolist"):
        try:
            return value.tolist(), {"source_type": type(value).__name__}
        except Exception:
            return None, {}
    return None, {}


def _truncate_preview(value: str, max_length: int = 160) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if len(value) > max_length:
        metadata["original_length"] = len(value)
        return f"{value[: max_length - 3]}...", metadata
    return value, metadata


def _classify_variable_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, pd.DataFrame):
        return "dataframe"
    if isinstance(value, pd.Series):
        return "series"
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, (list, tuple, set)):
        return "sequence"
    if isinstance(value, (datetime, date, Path)):
        return "scalar"
    module_name = type(value).__module__
    type_name = type(value).__name__.lower()
    if module_name.startswith("numpy"):
        if "ndarray" in type_name:
            return "ndarray"
        return "number"
    return "object"


def _serialize_exec_value(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 3,
    max_items: int = 25,
) -> tuple[Any, dict[str, Any]]:
    metadata: dict[str, Any] = {}

    if value is None or isinstance(value, (str, int, float, bool)):
        return value, metadata
    if isinstance(value, (datetime, date)):
        metadata["serialization"] = "isoformat"
        return value.isoformat(), metadata
    if isinstance(value, Path):
        metadata["serialization"] = "path"
        return str(value), metadata
    if isinstance(value, bytes):
        metadata["encoding"] = "base64"
        metadata["byte_length"] = len(value)
        return base64.b64encode(value).decode("ascii"), metadata
    if depth >= max_depth:
        metadata["serialization"] = "repr"
        preview, preview_meta = _truncate_preview(repr(value))
        metadata.update(preview_meta)
        return preview, metadata

    if isinstance(value, pd.DataFrame):
        rows, columns = value.shape
        metadata.update(
            {
                "rows": int(rows),
                "columns": list(map(str, value.columns.tolist())),
                "shape": [int(rows), int(columns)],
                "dtypes": {str(col): str(dtype) for col, dtype in value.dtypes.items()},
            }
        )
        limited_df = value.head(max_items)
        if len(limited_df) < len(value):
            metadata["truncated"] = True
            metadata["returned_rows"] = int(len(limited_df))
        return limited_df.to_dict(orient="records"), metadata

    if isinstance(value, pd.Series):
        metadata.update(
            {
                "length": int(len(value)),
                "dtype": str(value.dtype),
                "name": None if value.name is None else str(value.name),
            }
        )
        limited_series = value.head(max_items)
        if len(limited_series) < len(value):
            metadata["truncated"] = True
            metadata["returned_items"] = int(len(limited_series))
        return limited_series.tolist(), metadata

    if isinstance(value, dict):
        items = list(value.items())
        if len(items) > max_items:
            metadata["truncated"] = True
            metadata["returned_items"] = max_items
            metadata["total_items"] = len(items)
            items = items[:max_items]
        return {
            str(key): _serialize_exec_value(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)[0]
            for key, item in items
        }, metadata

    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        if isinstance(value, tuple):
            metadata["source_type"] = "tuple"
        elif isinstance(value, set):
            metadata["source_type"] = "set"
        if len(sequence) > max_items:
            metadata["truncated"] = True
            metadata["returned_items"] = max_items
            metadata["total_items"] = len(sequence)
            sequence = sequence[:max_items]
        return [
            _serialize_exec_value(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)[0]
            for item in sequence
        ], metadata

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            scalar = value.item()
            metadata["source_type"] = type(value).__name__
            return _serialize_exec_value(scalar, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass

    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            array_like = value.tolist()
            metadata["source_type"] = type(value).__name__
            shape = getattr(value, "shape", None)
            if shape is not None:
                metadata["shape"] = [int(item) for item in shape]
            dtype = getattr(value, "dtype", None)
            if dtype is not None:
                metadata["dtype"] = str(dtype)
            return _serialize_exec_value(array_like, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass

    preview, preview_meta = _truncate_preview(repr(value))
    metadata.update(preview_meta)
    metadata["serialization"] = "repr"
    return preview, metadata


def _preview_exec_value(serialized_value: Any, kind: str) -> str:
    if isinstance(serialized_value, str):
        return serialized_value
    if kind == "dataframe" and isinstance(serialized_value, list):
        return f"DataFrame rows={len(serialized_value)}"
    if kind == "series" and isinstance(serialized_value, list):
        return f"Series length={len(serialized_value)}"
    try:
        preview = json.dumps(serialized_value, ensure_ascii=False)
    except TypeError:
        preview = str(serialized_value)
    preview, _ = _truncate_preview(preview)
    return preview


def _extract_variables(namespace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extracted_variables: dict[str, dict[str, Any]] = {}
    for name, variable in namespace.items():
        if name.startswith("_") or hasattr(variable, "__spec__"):
            continue
        try:
            var_type = type(variable).__name__
            python_type = f"{type(variable).__module__}.{var_type}"
            kind = _classify_variable_kind(variable)
            raw_value, metadata = _serialize_exec_value(variable)
            preview = _preview_exec_value(raw_value, kind)
            legacy_raw_value, legacy_metadata = _json_safe_exec_value(variable)
            metadata.update({key: value for key, value in legacy_metadata.items() if key not in metadata})
            payload = {
                "name": name,
                "type": var_type,
                "python_type": python_type,
                "kind": kind,
                "value": preview,
                "preview": preview,
                "metadata": metadata or None,
            }
            if raw_value is not None:
                payload["raw_value"] = raw_value
            elif legacy_raw_value is not None:
                payload["raw_value"] = legacy_raw_value
            extracted_variables[name] = payload
        except Exception as exc:
            extracted_variables[name] = {
                "name": name,
                "type": type(variable).__name__,
                "kind": "error",
                "value": f"<Error serializing: {exc}>",
            }
    return extracted_variables


EDITOR_METADATA_FIELDS = {
    "name",
    "description",
    "object_type",
    "path",
    "date",
    "version",
    "history",
    "dataset_type",
    "connection_string",
    "model_type",
    "parameters",
    "metrics",
    "reference_data",
    "input_features",
    "output_features",
    "is_trained",
    "is_tested",
    "is_deployed",
    "csv_text",
    "pytorch_bytes",
    "torch_bytes",
    "model_bytes",
}


def _extract_editor_metadata(variables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field_name in EDITOR_METADATA_FIELDS:
        payload = variables.get(field_name)
        if not payload:
            continue
        if "raw_value" in payload:
            metadata[field_name] = payload["raw_value"]
        elif "value" in payload:
            metadata[field_name] = payload["value"]
    return metadata


def _extract_defined_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def _normalize_code_document(
    code: str,
    variables: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now_iso = datetime.now().isoformat()
    metadata = metadata or {}
    version = metadata.get("version", 1)
    if not isinstance(version, int):
        try:
            version = int(version)
        except (TypeError, ValueError):
            version = 1

    history = metadata.get("history")
    if not isinstance(history, list):
        history = [{"operation": "execute", "when": now_iso}]

    return {
        "name": metadata.get("name") or f"code_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "description": metadata.get("description") or "Code created in the editor",
        "object_type": metadata.get("object_type") or "code_model",
        "size": 0.0,
        "path": metadata.get("path") or "editor",
        "date": metadata.get("date") or now_iso,
        "version": version,
        "history": history,
        "variables": variables,
        "metadata": metadata,
        "code": {"script": code, "language": "python", "line_count": len(code.splitlines())},
    }


def _resolve_fs_path(path_value: str | Path | None, default_parent: Optional[Path] = None) -> Optional[Path]:
    if path_value is None or str(path_value).strip() == "":
        return None
    raw_path = Path(path_value)
    candidates = [raw_path]
    if default_parent is not None:
        candidates.append(default_parent / raw_path)
    candidates.extend([PROJECT_ROOT / raw_path, REPO_ROOT / raw_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if default_parent is not None:
        return (default_parent / raw_path).resolve()
    return raw_path.resolve()


def _coerce_feature_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(value)]


def _normalize_numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _default_output_feature(df: pd.DataFrame) -> str:
    if not len(df.columns):
        raise ValueError("Dataset does not contain any columns.")
    return str(df.columns[-1])


def _infer_task_type(target: pd.Series, model_record: Any, parameters: Mapping[str, Any]) -> str:
    explicit = str(parameters.get("task_type") or "").strip().lower()
    if explicit in {"classification", "regression"}:
        return explicit

    metrics = getattr(model_record, "metrics", {}) or {}
    metric_keys = {str(key).lower() for key in metrics.keys()}
    if {"accuracy", "precision", "recall", "f1"} & metric_keys:
        return "classification"
    if {"rmse", "mse", "mae", "r2"} & metric_keys:
        return "regression"
    if pd.api.types.is_bool_dtype(target) or target.dtype == object:
        return "classification"

    non_null = target.dropna()
    if non_null.empty:
        return "regression"
    unique_count = int(non_null.nunique())
    if pd.api.types.is_integer_dtype(non_null) and unique_count <= max(10, int(len(non_null) * 0.05) or 2):
        return "classification"
    return "regression"


def _normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized[column] = series.astype("int64") / 1_000_000_000
            continue
        if series.dtype == object:
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            if parsed.notna().sum() and parsed.notna().mean() >= 0.8:
                normalized[column] = parsed.astype("int64") / 1_000_000_000

    normalized = pd.get_dummies(normalized, dummy_na=True)
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    numeric_means = normalized.mean(numeric_only=True)
    normalized = normalized.fillna(numeric_means).fillna(0.0)
    return normalized.astype(float)


def _encode_target(target: pd.Series, task_type: str) -> tuple[pd.Series, dict[str, int]]:
    if task_type == "classification":
        categorical = target.astype("category")
        mapping = {str(category): int(index) for index, category in enumerate(categorical.cat.categories)}
        encoded = pd.Series(categorical.cat.codes, index=target.index, name=target.name).astype(int)
        return encoded, mapping

    numeric_target = pd.to_numeric(target, errors="coerce")
    fill_value = float(numeric_target.median()) if numeric_target.notna().any() else 0.0
    return numeric_target.fillna(fill_value).astype(float), {}


def _load_dataset_frame_from_record(dataset_record: Any) -> pd.DataFrame:
    path_value = getattr(dataset_record, "path", None)
    resolved_path = _resolve_fs_path(path_value, default_parent=REPO_ROOT)
    if resolved_path is None:
        raise FileNotFoundError("Dataset path is not defined.")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")
    return load_csv(resolved_path)


def _prepare_dataset_for_training(
    dataset_record: Any,
    model_record: Any,
    parameters: Mapping[str, Any],
) -> PreparedDataset:
    dataframe = _load_dataset_frame_from_record(dataset_record)
    requested_input_features = _coerce_feature_list(parameters.get("input_features"))
    requested_output_features = _coerce_feature_list(parameters.get("output_features"))
    stored_input_features = _coerce_feature_list(getattr(model_record, "input_features", None))
    stored_output_features = _coerce_feature_list(getattr(model_record, "output_features", None))

    output_features = requested_output_features or stored_output_features
    output_feature = output_features[0] if output_features else _default_output_feature(dataframe)
    input_features = requested_input_features or stored_input_features
    if not input_features:
        input_features = [column for column in dataframe.columns if str(column) != output_feature]

    validate_dataframe_columns(dataframe, [*input_features, output_feature])

    feature_frame = _normalize_feature_frame(dataframe[input_features])
    target_series = dataframe[output_feature]
    task_type = _infer_task_type(target_series, model_record, parameters)
    encoded_target, label_mapping = _encode_target(target_series, task_type)

    from sklearn.model_selection import train_test_split

    test_size = float(parameters.get("test_size", DEFAULT_TEST_SIZE))
    random_state = int(parameters.get("random_state", DEFAULT_RANDOM_STATE))
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    stratify = None
    if task_type == "classification" and encoded_target.nunique() > 1:
        value_counts = encoded_target.value_counts()
        if not value_counts.empty and int(value_counts.min()) > 1:
            stratify = encoded_target

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        encoded_target,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    return PreparedDataset(
        dataframe=dataframe,
        feature_frame=feature_frame,
        target_series=target_series,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        input_features=[str(item) for item in input_features],
        output_feature=str(output_feature),
        task_type=task_type,
        label_mapping=label_mapping,
    )


def _extract_model_hyperparameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if key not in CONTROL_PARAMETER_KEYS and value is not None
    }


def _select_sklearn_estimator(estimator_class: Optional[str], task_type: str) -> Any:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

    estimator_lookup = {
        "gradientboostingclassifier": GradientBoostingClassifier,
        "gradientboostingregressor": GradientBoostingRegressor,
        "lasso": Lasso,
        "linearregression": LinearRegression,
        "logisticregression": LogisticRegression,
        "randomforestclassifier": RandomForestClassifier,
        "randomforestregressor": RandomForestRegressor,
        "ridge": Ridge,
    }

    if estimator_class:
        estimator = estimator_lookup.get(estimator_class.replace(".", "").lower())
        if estimator is not None:
            return estimator

    return LogisticRegression if task_type == "classification" else RandomForestRegressor


def _log_model_artifact_to_mlflow(model: Any, framework: str) -> Optional[str]:
    if not MLFLOW_DIRECT_AVAILABLE or mlflow is None or not mlflow.active_run():
        return None

    package_root = ensure_dir(RUNTIME_DIR / "mlflow_models")
    package_dir = ensure_dir(package_root / f"{framework}_{uuid.uuid4().hex}")

    try:
        if framework == "sklearn":
            mlflow.sklearn.save_model(model, path=str(package_dir))
        elif framework == "pytorch":
            mlflow.pytorch.save_model(model, path=str(package_dir))
        else:
            return None

        mlflow.log_artifacts(str(package_dir), artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        return get_model_uri_for_run(run_id, artifact_path="model")
    except Exception:
        LOGGER.exception("Unable to log %s model artifact to MLflow.", framework)
        return None


def _predict_with_result(result: TrainingResult, features: pd.DataFrame, task_type: str) -> np.ndarray:
    if result.framework == "sklearn":
        return np.asarray(result.model.predict(features))

    if result.framework != "pytorch":
        raise ValueError(f"Unsupported training framework: {result.framework}")

    import torch

    model = result.model
    model.eval()
    tensor_x = torch.as_tensor(features.to_numpy(dtype=np.float32))
    with torch.no_grad():
        outputs = model(tensor_x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if task_type == "classification":
            if outputs.ndim == 1 or outputs.shape[-1] == 1:
                predictions = (torch.sigmoid(outputs).reshape(-1) >= 0.5).int().cpu().numpy()
            else:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            predictions = outputs.reshape(-1).cpu().numpy()
    model.train()
    return np.asarray(predictions)


def _train_with_sklearn(
    prepared: PreparedDataset,
    model_record: Any,
    parameters: Mapping[str, Any],
) -> tuple[TrainingResult, np.ndarray]:
    estimator_class = parameters.get("estimator_class")
    estimator_factory = _select_sklearn_estimator(
        estimator_class=str(estimator_class) if estimator_class else None,
        task_type=prepared.task_type,
    )
    estimator_instance = estimator_factory()
    hyperparameters = _extract_model_hyperparameters(parameters)
    if estimator_factory.__name__ == "LogisticRegression":
        hyperparameters.setdefault("max_iter", 1000)

    experiment_name = str(parameters.get("experiment_name") or f"{model_record.name}_training")
    run_name = str(parameters.get("run_name") or f"{model_record.name}_{uuid.uuid4().hex[:8]}")
    tags = {
        "framework": "sklearn",
        "model_name": str(getattr(model_record, "name", "model")),
        "model_id": str(getattr(model_record, "id", "")),
        "task_type": prepared.task_type,
    }

    result = run_training_pipeline(
        lambda: train_sklearn(
            estimator_instance,
            prepared.x_train,
            prepared.y_train,
            params=hyperparameters,
            random_state=int(parameters.get("random_state", DEFAULT_RANDOM_STATE)),
            eval_data=(prepared.x_test, prepared.y_test),
            task_type=prepared.task_type,
            experiment_name=None,
            run_name=None,
            tags=None,
            log_to_mlflow=True,
        ),
        experiment_name=experiment_name,
        run_name=run_name,
        params=parameters,
        tags=tags,
        log_to_mlflow=True,
    )

    predictions = np.asarray(result.model.predict(prepared.x_test))
    _log_training_tracking_context(
        result,
        prepared,
        model_record,
        parameters,
        framework="sklearn",
        extra_payload={"predictions_preview": predictions[:10].tolist()},
    )

    return result, predictions


def _train_with_pytorch(
    prepared: PreparedDataset,
    model_record: Any,
    parameters: Mapping[str, Any],
) -> tuple[TrainingResult, np.ndarray]:
    import torch
    import torch.nn as nn

    class FeedForwardNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            current_dim = input_dim
            for _ in range(max(hidden_layers, 1)):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs)

    x_train = prepared.x_train.to_numpy(dtype=np.float32)
    x_test = prepared.x_test.to_numpy(dtype=np.float32)
    y_train_array = prepared.y_train.to_numpy()
    y_test_array = prepared.y_test.to_numpy()

    hidden_dim = int(parameters.get("hidden_dim", 64))
    hidden_layers = int(parameters.get("hidden_layers", 2))
    dropout = float(parameters.get("dropout", 0.1))
    learning_rate = float(parameters.get("learning_rate", 0.001))
    batch_size = int(parameters.get("batch_size", 32))
    epochs = int(parameters.get("epochs", 20))

    if prepared.task_type == "classification":
        n_classes = int(pd.Series(prepared.y_train).nunique())
        if n_classes <= 2:
            output_dim = 1
            y_train_tensor = torch.as_tensor(y_train_array.astype(np.float32)).reshape(-1, 1)
            y_test_tensor = torch.as_tensor(y_test_array.astype(np.float32)).reshape(-1, 1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            output_dim = n_classes
            y_train_tensor = torch.as_tensor(y_train_array.astype(np.int64))
            y_test_tensor = torch.as_tensor(y_test_array.astype(np.int64))
            criterion = nn.CrossEntropyLoss()
    else:
        output_dim = 1
        y_train_tensor = torch.as_tensor(y_train_array.astype(np.float32)).reshape(-1, 1)
        y_test_tensor = torch.as_tensor(y_test_array.astype(np.float32)).reshape(-1, 1)
        criterion = nn.MSELoss()

    model = FeedForwardNet(
        input_dim=int(x_train.shape[1]),
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    experiment_name = str(parameters.get("experiment_name") or f"{model_record.name}_training")
    run_name = str(parameters.get("run_name") or f"{model_record.name}_{uuid.uuid4().hex[:8]}")
    tags = {
        "framework": "pytorch",
        "model_name": str(getattr(model_record, "name", "model")),
        "model_id": str(getattr(model_record, "id", "")),
        "task_type": prepared.task_type,
    }

    result = run_training_pipeline(
        lambda: train_pytorch(
            model,
            torch.as_tensor(x_train),
            y_train_tensor,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            val_data=(torch.as_tensor(x_test), y_test_tensor),
            params={
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            },
            experiment_name=None,
            run_name=None,
            tags=None,
            log_to_mlflow=True,
        ),
        experiment_name=experiment_name,
        run_name=run_name,
        params=parameters,
        tags=tags,
        log_to_mlflow=True,
    )

    predictions = _predict_with_result(result, prepared.x_test, prepared.task_type)
    eval_metrics = evaluate_and_log_metrics(
        y_true=prepared.y_test.to_numpy(),
        y_pred=predictions,
        task_type=prepared.task_type,
        prefix="eval_",
        log_to_mlflow=True,
    )
    result.metrics.update(eval_metrics)
    _log_training_tracking_context(
        result,
        prepared,
        model_record,
        parameters,
        framework="pytorch",
        extra_payload={
            "predictions_preview": predictions[:10].tolist(),
            "evaluation_metrics": dict(eval_metrics),
        },
    )
    pretty_print_metrics(result.metrics)
    return result, predictions


def _generate_training_artifacts(
    job_id: str,
    result: TrainingResult,
    prepared: PreparedDataset,
    predictions: np.ndarray,
    *,
    log_to_mlflow: bool,
) -> dict[str, Any]:
    artifact_dir = ensure_dir(TRAINING_ARTIFACT_DIR / job_id)
    plot_dir = ensure_dir(PLOT_ARTIFACT_DIR / job_id)
    artifact_paths: list[str] = []
    model_artifact_path = None
    serving_file = None
    model_uri = None

    summary_path = artifact_dir / "training_summary.json"
    save_training_summary(result, str(summary_path))
    artifact_paths.append(str(summary_path))

    try:
        model_artifact_path = save_model_local(result.model, str(artifact_dir / f"model_{result.framework}.pkl"))
        artifact_paths.append(model_artifact_path)
    except Exception:
        LOGGER.exception("Unable to persist the trained model locally for job %s.", job_id)

    try:
        artifact_paths.append(
            plot_metric_comparison(
                result.metrics,
                title="Training Metrics",
                output_dir=str(plot_dir),
                log_to_mlflow=log_to_mlflow,
            )
        )
    except Exception:
        LOGGER.exception("Unable to create metric comparison plot for job %s.", job_id)

    if result.framework == "pytorch" and result.history:
        try:
            artifact_paths.append(
                plot_loss_curve(
                    result.history,
                    title="PyTorch Loss Curve",
                    output_dir=str(plot_dir),
                    log_to_mlflow=log_to_mlflow,
                )
            )
        except Exception:
            LOGGER.exception("Unable to create loss plot for job %s.", job_id)

    if prepared.task_type == "regression":
        try:
            artifact_paths.append(
                plot_predictions_vs_actual(
                    prepared.y_test.to_numpy(),
                    predictions,
                    title="Predictions vs Actual",
                    output_dir=str(plot_dir),
                    log_to_mlflow=log_to_mlflow,
                )
            )
        except Exception:
            LOGGER.exception("Unable to create prediction plot for job %s.", job_id)

    model_uri = _log_model_artifact_to_mlflow(result.model, result.framework) if log_to_mlflow else None
    if model_uri:
        try:
            serving_file = create_fastapi_serving_file(
                str(SERVING_ARTIFACT_DIR / f"{job_id}_service.py"),
                model_uri=model_uri,
            )
            artifact_paths.append(serving_file)
        except Exception:
            LOGGER.exception("Unable to create serving file for job %s.", job_id)

    result.artifact_paths.extend(artifact_paths)
    return {
        "summary_path": str(summary_path),
        "artifact_paths": artifact_paths,
        "model_artifact_path": model_artifact_path,
        "serving_file": serving_file,
        "model_uri": model_uri,
    }


def _compute_monitoring_snapshot(
    job_id: str,
    model_record: Any,
    prepared: PreparedDataset,
    result: TrainingResult,
    predictions: np.ndarray,
    tolerance: float = 0.05,
    *,
    log_to_mlflow: bool,
) -> dict[str, Any]:
    drift_results: dict[str, bool] = {}
    degradation_results: dict[str, bool] = {}
    prediction_drift: dict[str, Any] = {}

    try:
        drift_results = detect_data_drift(prepared.x_train, prepared.x_test)
    except Exception:
        LOGGER.exception("Data drift detection failed for job %s.", job_id)

    try:
        train_predictions = _predict_with_result(result, prepared.x_train, prepared.task_type)
        drift_detected, p_value = detect_prediction_drift(
            np.asarray(train_predictions).reshape(-1),
            np.asarray(predictions).reshape(-1),
        )
        prediction_drift = {"detected": drift_detected, "p_value": p_value}
    except Exception:
        LOGGER.exception("Prediction drift detection failed for job %s.", job_id)

    previous_metrics = _normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {})
    current_metrics = _normalize_numeric_metrics(result.metrics)
    if previous_metrics:
        try:
            degradation_results = track_performance(
                current_metrics=current_metrics,
                previous_metrics=previous_metrics,
                tolerance=tolerance,
            )
        except Exception:
            LOGGER.exception("Performance monitoring failed for job %s.", job_id)

    if log_to_mlflow:
        try:
            log_monitoring_results_to_mlflow(drift_results, degradation_results)
        except Exception:
            LOGGER.exception("Unable to mirror monitoring results to MLflow for job %s.", job_id)

    summary = summarize_monitoring_results(drift_results, degradation_results)
    retrain_recommended = should_retrain(drift_results, degradation_results)
    snapshot = {
        "job_id": job_id,
        "drift_results": drift_results,
        "prediction_drift": prediction_drift,
        "degradation_results": degradation_results,
        "summary": summary,
        "retrain_recommended": retrain_recommended,
        "evaluated_at": datetime.now().isoformat(),
    }
    save_json(snapshot, MONITORING_ARTIFACT_DIR / job_id / "monitoring_summary.json")
    return snapshot


def _merge_training_parameters(
    model_record: Any,
    payload: TrainingRequest,
    study_record: Any | None,
) -> dict[str, Any]:
    parameters = dict(getattr(model_record, "parameters", {}) or {})
    if study_record is not None:
        parameters.update(dict(getattr(study_record, "study_params", {}) or {}))
        parameters.update(dict(getattr(study_record, "best_params", {}) or {}))
    parameters.update(dict(payload.parameters or {}))
    if "framework" not in parameters:
        parameters["framework"] = "pytorch" if "deep" in str(getattr(model_record, "model_type", "")).lower() else "sklearn"
    return parameters


def _run_training_workflow(
    model_record: Any,
    dataset_record: Any,
    parameters: Mapping[str, Any],
    *,
    job_id: str,
) -> dict[str, Any]:
    prepared = _prepare_dataset_for_training(dataset_record, model_record, parameters)
    framework = str(parameters.get("framework", "sklearn")).strip().lower()
    if framework == "onnx":
        framework = str(parameters.get("training_backend") or parameters.get("base_framework") or "sklearn").strip().lower()

    if framework in {"sklearn", "scikit-learn", "scikitlearn"}:
        result, predictions = _train_with_sklearn(prepared, model_record, parameters)
        framework = "sklearn"
    elif framework in {"pytorch", "torch"}:
        result, predictions = _train_with_pytorch(prepared, model_record, parameters)
        framework = "pytorch"
    else:
        raise ValueError(f"Unsupported training framework: {framework}")

    result.framework = framework
    pretty_print_metrics(result.metrics)
    with resume_run(result.run_id) as tracking_active:
        artifacts = _generate_training_artifacts(
            job_id,
            result,
            prepared,
            predictions,
            log_to_mlflow=tracking_active,
        )
        monitoring = _compute_monitoring_snapshot(
            job_id,
            model_record,
            prepared,
            result,
            predictions,
            log_to_mlflow=tracking_active,
        )
    return {
        "framework": framework,
        "result": result,
        "predictions": predictions,
        "prepared": prepared,
        "artifacts": artifacts,
        "monitoring": monitoring,
    }


def _suggest_default_search_space(framework: str, task_type: str) -> dict[str, Any]:
    if framework == "pytorch":
        return {
            "hidden_dim": {"type": "int", "low": 16, "high": 128, "step": 16},
            "hidden_layers": {"type": "int", "low": 1, "high": 4, "step": 1},
            "dropout": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.1},
            "learning_rate": {"type": "float", "low": 0.0005, "high": 0.01, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        }
    if task_type == "classification":
        return {
            "estimator_class": {"type": "categorical", "choices": ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]},
            "max_depth": {"type": "int", "low": 2, "high": 12, "step": 1},
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 25},
        }
    return {
        "estimator_class": {"type": "categorical", "choices": ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor", "Ridge"]},
        "max_depth": {"type": "int", "low": 2, "high": 12, "step": 1},
        "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 25},
    }


def _suggest_trial_parameters(trial: Any, search_space: Mapping[str, Any]) -> dict[str, Any]:
    suggested: dict[str, Any] = {}
    for parameter_name, config in search_space.items():
        if not isinstance(config, Mapping):
            continue
        parameter_type = str(config.get("type", "categorical")).lower()
        if parameter_type == "int":
            suggested[parameter_name] = trial.suggest_int(
                parameter_name,
                int(config.get("low", 1)),
                int(config.get("high", 10)),
                step=int(config.get("step", 1)),
            )
        elif parameter_type == "float":
            low = float(config.get("low", 0.001))
            high = float(config.get("high", 1.0))
            log_scale = bool(config.get("log", False))
            if "step" in config and not log_scale:
                suggested[parameter_name] = trial.suggest_float(parameter_name, low, high, step=float(config["step"]))
            else:
                suggested[parameter_name] = trial.suggest_float(parameter_name, low, high, log=log_scale)
        elif parameter_type == "categorical":
            suggested[parameter_name] = trial.suggest_categorical(parameter_name, list(config.get("choices", [])))
    return suggested


def _dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [str(column) for column in df.select_dtypes(include=[np.number]).columns]
    categorical_columns = [
        str(column)
        for column in df.columns
        if str(column) not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    datetime_columns = [str(column) for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
    target = _default_output_feature(df)
    recommended_inputs = [str(column) for column in df.columns if str(column) != target]
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "recommended_target": target,
        "recommended_inputs": recommended_inputs,
    }


def _dataset_analysis_summary(df: pd.DataFrame, df_path: str) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for column_name in df.columns:
        column = df[column_name]
        if pd.api.types.is_numeric_dtype(column):
            stats[str(column_name)] = {
                "type": "numeric",
                "mean": None if column.empty else float(column.mean()),
                "std": None if column.empty else float(column.std()),
                "min": None if column.empty else float(column.min()),
                "max": None if column.empty else float(column.max()),
                "null_count": int(column.isnull().sum()),
            }
            continue
        mode = column.mode(dropna=True)
        stats[str(column_name)] = {
            "type": "categorical",
            "unique_values": int(column.nunique(dropna=True)),
            "null_count": int(column.isnull().sum()),
            "mode": None if mode.empty else str(mode.iloc[0]),
        }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": int(df.isnull().sum().sum()),
        "stats": stats,
        "path": df_path,
        "profile": _dataset_profile(df),
    }


def _build_feature_extraction_summary(df: pd.DataFrame) -> dict[str, Any]:
    profile = _dataset_profile(df)
    return {
        "feature_candidates": profile["recommended_inputs"],
        "target_candidate": profile["recommended_target"],
        "numeric_feature_count": len(profile["numeric_columns"]),
        "categorical_feature_count": len(profile["categorical_columns"]),
        "datetime_feature_count": len(profile["datetime_columns"]),
    }


def _augment_model_analysis(model_record: Any) -> dict[str, Any]:
    summary = _serialize_model_summary(model_record)
    path_value = getattr(model_record, "path", None)
    path = _resolve_fs_path(path_value, default_parent=REPO_ROOT)
    if path and path.exists():
        try:
            summary["artifact_metadata"] = recover_model_params(path)
        except Exception:
            LOGGER.exception("Unable to recover model metadata for %s.", path)
    return summary


async def _optimize_study(
    study_record: Any,
    db: AsyncSession,
    request_payload: StudyOptimizationRequest,
) -> dict[str, Any]:
    learning_model = await LearningModel.read(db, int(study_record.learning_model_id))
    dataset_model = await DatasetModel.read(db, int(study_record.dataset_id))
    if learning_model is None or dataset_model is None:
        raise ValueError("The selected study is not linked to a valid model and dataset.")

    base_parameters = dict(getattr(learning_model, "parameters", {}) or {})
    study_parameters = dict(getattr(study_record, "study_params", {}) or {})
    merged_parameters = {**base_parameters, **study_parameters}
    prepared = _prepare_dataset_for_training(dataset_model, learning_model, merged_parameters)
    framework = str(merged_parameters.get("framework", "sklearn")).strip().lower()
    metric_name = str(
        study_parameters.get("metric")
        or study_parameters.get("objective_metric")
        or ("accuracy" if prepared.task_type == "classification" else "rmse")
    )
    if not metric_name.startswith("eval_"):
        metric_name = f"eval_{metric_name}"

    direction = str(study_parameters.get("direction", "maximize")).lower()
    n_trials = int(request_payload.nTrials or study_parameters.get("n_trials", 15))
    search_space = study_parameters.get("search_space") or _suggest_default_search_space(framework, prepared.task_type)
    output_dir = request_payload.outputDir or str(OPTUNA_ARTIFACT_DIR / f"study_{study_record.id}")
    experiment_name = str(study_parameters.get("experiment_name") or f"{learning_model.name}_study")
    study_context = {
        "study_id": getattr(study_record, "id", None),
        "study_name": getattr(study_record, "name", None),
        "framework": framework,
        "task_type": prepared.task_type,
        "objective_metric": metric_name,
        "direction": direction,
        "search_space": search_space,
        "model": _serialize_model_summary(learning_model),
        "dataset": _serialize_dataset_summary(dataset_model),
    }

    def objective(trial: Any) -> float:
        trial_parameters = {**merged_parameters, **_suggest_trial_parameters(trial, search_space)}
        if framework == "pytorch":
            training_result, _ = _train_with_pytorch(prepared, learning_model, trial_parameters)
        else:
            training_result, _ = _train_with_sklearn(prepared, learning_model, trial_parameters)
        value = training_result.metrics.get(metric_name)
        if value is None:
            fallback_metrics = _normalize_numeric_metrics(training_result.metrics)
            if not fallback_metrics:
                raise ValueError(f"No numeric metric was produced for study objective '{metric_name}'.")
            value = max(fallback_metrics.values()) if direction == "maximize" else min(fallback_metrics.values())
        return float(value)

    optimization = optimize_with_tracking(
        objective_fn=objective,
        study_name=str(getattr(study_record, "name", f"study_{study_record.id}")),
        experiment_name=experiment_name,
        direction=direction,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
        output_dir=output_dir,
        plot_results=bool(request_payload.plotResults),
        objective_metric=metric_name,
        search_space=search_space,
        study_context=study_context,
        tags={
            "framework": framework,
            "learning_model_id": str(learning_model.id),
            "dataset_id": str(dataset_model.id),
            "study_id": str(study_record.id),
        },
    )

    summary = optimization["summary"]
    await update_entry(
        db,
        StudyORM,
        study_record.id,
        {
            "best_trial": {"value": summary.get("best_value")},
            "best_params": summary.get("best_params", {}),
            "history": _append_history(
                getattr(study_record, "history", None),
                {
                    "operation": "optimization",
                    "when": datetime.now().isoformat(),
                    "best_value": summary.get("best_value"),
                },
            ),
            "path": str(Path(output_dir) / "study_summary.json"),
        },
    )

    return {
        "study": _serialize_study_summary(await StudyModel.read(db, study_record.id)),
        "summary": summary,
        "plot_paths": optimization.get("plot_paths", []),
        "output_dir": output_dir,
        "objective_metric": metric_name,
    }


async def _build_live_monitoring_snapshot(
    db: AsyncSession,
    model_id: int,
    dataset_id: int,
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    model_record = await LearningModel.read(db, model_id)
    dataset_record = await DatasetModel.read(db, dataset_id)
    if model_record is None or dataset_record is None:
        raise ValueError("The selected monitoring pair is invalid.")

    parameters = dict(getattr(model_record, "parameters", {}) or {})
    prepared = _prepare_dataset_for_training(dataset_record, model_record, parameters)
    path_value = getattr(model_record, "path", None)
    path = _resolve_fs_path(path_value, default_parent=REPO_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError("The selected model does not have a persisted artifact path.")

    import pickle

    with Path(path).open("rb") as file_handle:
        trained_model = pickle.load(file_handle)

    framework = "pytorch" if "torch" in str(path).lower() else str(parameters.get("framework", "sklearn")).lower()
    training_result = TrainingResult(
        model=trained_model,
        framework=framework,
        metrics=_normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
        parameters=parameters,
    )

    predictions = _predict_with_result(training_result, prepared.x_test, prepared.task_type)
    current_metrics = evaluate_and_log_metrics(
        y_true=prepared.y_test.to_numpy(),
        y_pred=predictions,
        task_type=prepared.task_type,
        prefix="live_",
        log_to_mlflow=False,
    )
    drift_results = detect_data_drift(prepared.x_train, prepared.x_test)
    degradation_results = track_performance(
        current_metrics=current_metrics,
        previous_metrics=_normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
        tolerance=tolerance,
    )
    drift_detected, p_value = detect_prediction_drift(
        _predict_with_result(training_result, prepared.x_train, prepared.task_type).reshape(-1),
        predictions.reshape(-1),
    )
    summary = summarize_monitoring_results(drift_results, degradation_results)
    retrain_recommended = should_retrain(drift_results, degradation_results)
    snapshot = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "metrics": current_metrics,
        "drift_results": drift_results,
        "degradation_results": degradation_results,
        "prediction_drift": {"detected": drift_detected, "p_value": p_value},
        "summary": summary,
        "retrain_recommended": retrain_recommended,
        "evaluated_at": datetime.now().isoformat(),
    }
    try:
        with managed_run(
            experiment_name=f"{model_record.name}_production_monitoring",
            run_name=f"production_monitor_{model_id}_{uuid.uuid4().hex[:8]}",
            tags={
                "stage": "production_monitoring",
                "model_id": str(model_id),
                "dataset_id": str(dataset_id),
            },
        ):
            log_mlflow_metrics(current_metrics)
            log_monitoring_results_to_mlflow(drift_results, degradation_results, prefix="production_")
            log_mlflow_json(
                snapshot,
                filename="production_monitoring_summary.json",
                artifact_path="monitoring",
            )
    except Exception:
        LOGGER.exception("Unable to log production monitoring results to MLflow.")
    return snapshot


async def _trigger_retraining_from_monitoring(
    db: AsyncSession,
    model_id: int,
    dataset_id: int,
    tolerance: float,
) -> dict[str, Any]:
    live_snapshot = await _build_live_monitoring_snapshot(db, model_id, dataset_id, tolerance=tolerance)
    model_record = await LearningModel.read(db, model_id)
    dataset_record = await DatasetModel.read(db, dataset_id)
    if model_record is None or dataset_record is None:
        raise ValueError("The selected retraining pair is invalid.")

    parameters = dict(getattr(model_record, "parameters", {}) or {})

    def retrain_callable() -> TrainingResult:
        workflow = _run_training_workflow(
            model_record,
            dataset_record,
            parameters,
            job_id=f"retrain_{model_id}_{uuid.uuid4().hex[:8]}",
        )
        return workflow["result"]

    return monitor_and_retrain(
        drift_results=live_snapshot["drift_results"],
        degradation_results=live_snapshot["degradation_results"],
        train_func=retrain_callable,
        train_params={},
        experiment_name=f"{model_record.name}_retraining",
        run_name=f"retrain_{model_id}",
        push_metrics_enabled=False,
    )


_register_models()
_include_registry_routers()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await init_models()
        APP_LOGGER.info("Database models initialized.")
    except Exception:
        LOGGER.exception("Database initialization failed during startup.")

    try:
        async with AsyncSessionLocal() as db:
            normalized_identities = await normalize_legacy_polymorphic_identities(db)
        if normalized_identities:
            LOGGER.info("Normalized legacy polymorphic identities: %s", normalized_identities)
    except Exception:
        LOGGER.exception("Legacy polymorphic identity normalization failed during startup.")

    try:
        async with AsyncSessionLocal() as db:
            normalized_jsonb = await normalize_legacy_jsonb_containers(db)
        if normalized_jsonb:
            LOGGER.info("Normalized legacy JSONB containers: %s", normalized_jsonb)
    except Exception:
        LOGGER.exception("Legacy JSONB normalization failed during startup.")

    save_config_snapshot(RUNTIME_CONFIG, str(CONFIG_SNAPSHOT_DIR / "runtime_config.json"))
    LOGGER.info("Runtime configuration snapshot saved.")


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request) -> HTMLResponse:
    return _render_page("base_template.html", request, )


@app.get("/health", response_class=JSONResponse)
async def health_live() -> JSONResponse:
    return _json_response(
        {
            "status": "ok",
            "app": RUNTIME_CONFIG["app"],
            "checks": {
                "static_dir": STATIC_DIR.exists(),
                "templates_dir": TEMPLATES_DIR.exists(),
            },
        }
    )


@app.get("/health/ready", response_class=JSONResponse)
async def health_ready() -> JSONResponse:
    database_ready, database_error = await _database_healthcheck()
    status_code = 200 if database_ready else 503
    payload = {
        "status": "ready" if database_ready else "degraded",
        "checks": {
            "database": "ok" if database_ready else "error",
        },
    }
    if database_error:
        LOGGER.warning("Database readiness check failed: %s", database_error)
    return _json_response(payload, status_code=status_code)


@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request) -> HTMLResponse:
    return _render_page("base_red.html", request, )


@app.get("/create", response_class=HTMLResponse)
async def page_create(request: Request) -> HTMLResponse:
    return _render_page("base_red copy.html", request, )


@app.get("/training", response_class=HTMLResponse)
async def page_training(request: Request) -> HTMLResponse:
    return _render_page("base_green.html", request, )


@app.get("/optimization", response_class=HTMLResponse)
async def page_optimization(request: Request) -> HTMLResponse:
    return _render_page("base_green.html", request, )


@app.get("/editor", response_class=HTMLResponse)
async def page_editor(request: Request) -> HTMLResponse:
    return _render_page("base_editor.html", request, )


@app.get("/production", response_class=HTMLResponse)
async def page_production(request: Request) -> HTMLResponse:
    return _render_page("base_blue.html", request, )


@app.get("/registry", response_class=HTMLResponse)
async def page_registry(request: Request) -> HTMLResponse:
    return _render_page("base_purple.html", request, )


@app.post("/upload/{operation_id}", response_class=JSONResponse)
async def post_upload(
    operation_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    form = await request.form()
    upload_file = form.get("file")
    normalized_operation = str(form.get("operationId") or operation_id).lower()

    if normalized_operation not in ALLOWED_UPLOAD_OPERATIONS:
        return _json_error(
            f"operation_id must be one of: {', '.join(sorted(ALLOWED_UPLOAD_OPERATIONS))}",
            status_code=400,
        )
    if upload_file is None:
        return _json_error("no file provided", status_code=400)

    object_name = form.get("objectName") or getattr(upload_file, "filename", "unnamed")
    description = form.get("description") or ""

    try:
        upload_dir = Path(get_upload_dir(normalized_operation))
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = getattr(upload_file, "filename", "uploaded.bin")
        destination_path = upload_dir / filename
        size_mb = await save_file_to_disk(upload_file, str(destination_path))
        now = datetime.now()

        raw_history = form.get("history")
        try:
            history = json.loads(raw_history) if raw_history else [{"operation": "upload", "when": now.isoformat()}]
        except (TypeError, json.JSONDecodeError):
            history = [{"operation": "upload", "when": now.isoformat()}]

        raw_version = form.get("version")
        try:
            version = int(raw_version) if raw_version not in (None, "") else 1
        except (TypeError, ValueError):
            version = 1

        common_payload = {
            "name": object_name,
            "description": description or f"Uploaded file {filename}",
            "object_type": "learning_model" if normalized_operation == MODEL_OPERATION else "dataset",
            "size": size_mb,
            "path": str(destination_path),
            "date": now,
            "version": version,
            "history": history,
        }

        if normalized_operation == MODEL_OPERATION:
            payload = build_payload_model(common_payload, form)
            orm_model = LearningORM
        else:
            payload, dataframe = await build_payload_data(common_payload, form, upload_file)
            payload["history"] = _append_history(history, {"operation": "profiled", "when": now.isoformat()})
            save_json(
                _dataset_analysis_summary(dataframe, str(destination_path)),
                TRAINING_ARTIFACT_DIR / f"dataset_{object_name}_profile.json",
            )
            orm_model = DatasetORM

        created_object = await create_entry(db, orm_model, payload)
        return _json_response({"status": "ok", "id": getattr(created_object, "id", None), "path": str(destination_path)})
    except ValueError as exc:
        return _json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Upload failed for operation '%s'.", normalized_operation)
        return _json_error(str(exc), status_code=500)


@app.post("/execute", response_class=JSONResponse)
async def post_execute(request: Request) -> JSONResponse:
    try:
        body = ExecuteRequest.model_validate(await request.json())
        code = body.code.strip()
    except Exception:
        raw = await request.body()
        code = raw.decode("utf-8").strip() if raw else ""

    if not code:
        return _json_response(
            {"status": "error", "variables": {}, "stdout": "", "stderr": "", "error": "No code provided"},
            status_code=400,
        )

    def _safe_import(module_name: str) -> Any:
        try:
            return __import__(module_name)
        except Exception:
            return None

    namespace = {
        "__builtins__": __builtins__,
        "np": _safe_import("numpy"),
        "pd": _safe_import("pandas"),
        "sk": _safe_import("sklearn"),
        "torch": _safe_import("torch"),
        "ox": _safe_import("onnx"),
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    error_message = None
    extracted_variables: dict[str, dict[str, Any]] = {}
    try:
        exec(code, namespace)
        extracted_variables = _extract_variables(namespace)
    except Exception:
        error_message = traceback.format_exc()
    finally:
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    editor_metadata = _extract_editor_metadata(extracted_variables)
    normalized_document = _normalize_code_document(code=code, variables=extracted_variables, metadata=editor_metadata)
    defined_names = _extract_defined_names(code)

    return _json_response(
        {
            "status": "success" if error_message is None else "error",
            "variables": extracted_variables,
            "metadata": editor_metadata,
            "document": normalized_document,
            "summary": {
                "variable_count": len(extracted_variables),
                "defined_name_count": len(defined_names),
                "defined_names": defined_names,
                "has_error": error_message is not None,
            },
            "stdout": stdout_output,
            "stderr": stderr_output,
            "error": error_message,
        }
    )


@app.post("/generate", response_class=JSONResponse)
async def post_generate(request: Request) -> JSONResponse:
    try:
        payload = validate_input(await request.json(), GenerateRequest)
    except Exception as exc:
        return _json_error(f"Invalid JSON payload: {exc}", status_code=400)

    prompt = payload.prompt.strip()
    operation_id = payload.operationId.lower()
    if not prompt:
        return _json_error("prompt is required", status_code=400)

    prompt_comment = "\n".join(f"# {line}" for line in prompt.splitlines() if line.strip())
    is_model_request = operation_id == MODEL_OPERATION or any(
        token in prompt.lower() for token in ("model", "onnx", "train", "sklearn", "torch")
    )

    if is_model_request:
        script = textwrap.dedent(
            f"""
            # Generated model template
            {prompt_comment}

            name = "generated_model"
            description = "Generated from the editor prompt"
            object_type = "learning_model"
            path = "generated/generated_model.pt"
            version = 1
            model_type = "supervised_model"
            parameters = {{"framework": "pytorch", "input_size": 1, "output_size": 1}}
            metrics = {{"task": "regression"}}
            reference_data = "sample_dataset.csv"
            input_features = ["input"]
            output_features = ["target"]
            is_trained = False
            is_tested = False
            is_deployed = False
            pytorch_bytes = ""
            """
        ).strip()
    else:
        script = textwrap.dedent(
            f"""
            # Generated dataset template
            {prompt_comment}

            import pandas as pd

            df = pd.DataFrame({{
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [10.0, 20.0, 30.0],
                "target": [0.1, 0.4, 0.9]
            }})

            csv_text = df.to_csv(index=False)

            name = "generated_dataset"
            description = "Generated from the editor prompt"
            object_type = "dataset"
            path = "generated/generated_dataset.csv"
            version = 1
            dataset_type = "dataset"
            connection_string = ""
            """
        ).strip()

    return _json_response({"script": script})


@app.post("/training/{model_id}", response_class=JSONResponse)
async def post_training(
    model_id: int,
    payload: TrainingRequest,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        validated_payload = validate_input(payload.model_dump(), TrainingRequest)
        if validated_payload.datasetId <= 0:
            return _json_error("datasetId must be a positive integer", status_code=400)

        learning_model = await LearningModel.read(db, model_id)
        dataset_model = await DatasetModel.read(db, validated_payload.datasetId)
        if learning_model is None:
            return _json_error("Learning model not found", status_code=404)
        if dataset_model is None:
            return _json_error("Dataset not found", status_code=404)

        study_record = None
        if validated_payload.studyId not in (None, "", 0, "0"):
            study_record = await StudyModel.read(db, validated_payload.studyId)
            if study_record is None:
                return _json_error("StudyModel not found", status_code=404)

        merged_parameters = _merge_training_parameters(learning_model, validated_payload, study_record)
        validate_required_keys(merged_parameters, ["framework"])

        job_id = f"train_{model_id}_{validated_payload.datasetId}_{uuid.uuid4().hex[:8]}"
        workflow = _run_training_workflow(learning_model, dataset_model, merged_parameters, job_id=job_id)
        result: TrainingResult = workflow["result"]
        prepared: PreparedDataset = workflow["prepared"]
        monitoring = workflow["monitoring"]
        artifacts = workflow["artifacts"]

        updated_model = await update_entry(
            db,
            LearningORM,
            model_id,
            {
                "parameters": {**dict(getattr(learning_model, "parameters", {}) or {}), **result.parameters, **merged_parameters},
                "metrics": result.metrics,
                "reference_data": getattr(dataset_model, "name", None),
                "input_features": prepared.input_features,
                "output_features": [prepared.output_feature],
                "is_trained": True,
                "is_tested": True,
                "path": artifacts["model_artifact_path"] or getattr(learning_model, "path", None),
                "history": _append_history(
                    getattr(learning_model, "history", None),
                    {
                        "operation": "training",
                        "when": datetime.now().isoformat(),
                        "dataset_id": validated_payload.datasetId,
                        "framework": result.framework,
                        "job_id": job_id,
                    },
                ),
            },
        )

        if study_record is not None:
            await update_entry(
                db,
                StudyORM,
                study_record.id,
                {
                    "history": _append_history(
                        getattr(study_record, "history", None),
                        {
                            "operation": "training_run",
                            "when": datetime.now().isoformat(),
                            "job_id": job_id,
                            "model_id": model_id,
                            "dataset_id": validated_payload.datasetId,
                        },
                    )
                },
            )

        log_to_mlflow(LOGGER, f"Completed training job {job_id} for model {model_id}.")

        return _json_response(
            {
                "status": "completed",
                "job_id": job_id,
                "model_id": model_id,
                "dataset_id": validated_payload.datasetId,
                "input_type": validated_payload.inputType,
                "study_id": validated_payload.studyId,
                "framework": result.framework,
                "metrics": result.metrics,
                "parameters": merged_parameters,
                "monitoring": monitoring,
                "artifacts": artifacts,
                "run_id": result.run_id,
                "model": _serialize_model_summary(updated_model),
            }
        )
    except Exception as exc:
        LOGGER.exception("Training failed for model_id=%s.", model_id)
        return _json_error(str(exc), status_code=500)


@app.post("/studies/{study_id}/optimize", response_class=JSONResponse)
async def post_optimize_study(
    study_id: int,
    payload: StudyOptimizationRequest = Body(default=StudyOptimizationRequest()),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        study_record = await StudyModel.read(db, study_id)
        if study_record is None:
            return _json_error("StudyModel not found", status_code=404)
        return _json_response({"status": "completed", **(await _optimize_study(study_record, db, payload))})
    except Exception as exc:
        LOGGER.exception("Optuna optimization failed for study_id=%s.", study_id)
        return _json_error(str(exc), status_code=500)


@app.get("/features", response_class=JSONResponse)
async def list_features(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return _json_response([_serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/features/extract", response_class=JSONResponse)
async def extract_feature_candidates(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None:
            return _json_error("Dataset not found", status_code=404)
        dataframe = _load_dataset_frame_from_record(dataset)
        summary = _build_feature_extraction_summary(dataframe)
        save_json(summary, TRAINING_ARTIFACT_DIR / f"dataset_{dataset_id}_features.json")
        return _json_response({"dataset_id": dataset_id, "dataset_name": getattr(dataset, "name", None), "summary": summary})
    except Exception as exc:
        LOGGER.exception("Feature extraction failed for dataset_id=%s.", dataset_id)
        return _json_error(str(exc), status_code=500)


@app.get("/list/dataset", response_class=JSONResponse)
async def list_dataset(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return _json_response([_serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/list/model", response_class=JSONResponse)
async def list_model(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    models = await get_all_entries(db, LearningORM)
    return _json_response([_serialize_model_summary(model) for model in models])


@app.get("/analysis/data", response_class=JSONResponse)
async def analysis_dataset(
    dataset_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if dataset_id is None:
            datasets = await get_all_entries(db, DatasetORM)
            return _json_response(
                {
                    "available_datasets": [
                        {"id": dataset.id, "name": dataset.name, "shape": dataset.shape}
                        for dataset in datasets
                    ]
                }
            )

        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None or not getattr(dataset, "path", None):
            return _json_error("Dataset not found", status_code=404)

        df_path = str(_resolve_fs_path(dataset.path, default_parent=REPO_ROOT))
        df = _load_dataset_frame_from_record(dataset)
        summary = _dataset_analysis_summary(df, df_path)
        save_json(summary, TRAINING_ARTIFACT_DIR / f"dataset_{dataset_id}_analysis.json")
        return _json_response({"dataset_id": dataset_id, "summary": summary})
    except Exception as exc:
        LOGGER.exception("Dataset analysis failed for dataset_id=%s.", dataset_id)
        return _json_error(str(exc), status_code=500)


@app.get("/analysis/model", response_class=JSONResponse)
async def analysis_model(
    model_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if model_id is None:
            models = await get_all_entries(db, LearningORM)
            return _json_response(
                {
                    "available_models": [
                        {"id": model.id, "name": model.name, "type": model.model_type, "is_trained": model.is_trained}
                        for model in models
                    ]
                }
            )

        model = await LearningModel.read(db, model_id)
        if model is None:
            return _json_error("Model not found", status_code=404)
        return _json_response({"model_id": model_id, "summary": _augment_model_analysis(model)})
    except Exception as exc:
        LOGGER.exception("Model analysis failed for model_id=%s.", model_id)
        return _json_error(str(exc), status_code=500)


@app.get("/mlflow/experiments", response_class=JSONResponse)
async def mlflow_list_experiments() -> JSONResponse:
    try:
        return _json_response(list_experiments())
    except Exception as exc:
        LOGGER.exception("Unable to list MLflow experiments.")
        return _json_error(str(exc), status_code=500)


@app.get("/mlflow/experiments/{exp_id}", response_class=JSONResponse)
async def mlflow_get_experiment(exp_id: str) -> JSONResponse:
    try:
        return _json_response(get_experiment_summary(exp_id))
    except Exception as exc:
        LOGGER.exception("Unable to load MLflow experiment '%s'.", exp_id)
        return _json_error(str(exc), status_code=500)


@app.post("/onnx/prepare", response_class=JSONResponse)
async def onnx_prepare(payload: OnnxPrepareRequest) -> JSONResponse:
    prepared_payload = {
        "model_id": payload.model_id,
        "dataset_id": payload.dataset_id,
        "study_id": payload.study_id,
        "framework": payload.framework,
        "export_target": payload.export_target,
        "parameters": payload.parameters,
        "prepared_at": datetime.now().isoformat(),
    }
    save_json(prepared_payload, DEPLOYMENT_ARTIFACT_DIR / f"onnx_prepare_{uuid.uuid4().hex[:8]}.json")
    return _json_response({"status": "prepared", "payload": prepared_payload})


@app.get("/onnx/validate/{model_id}", response_class=JSONResponse)
async def onnx_validate(model_id: int, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await LearningModel.read(db, model_id)
        if model_record is None:
            return _json_error("Model not found", status_code=404)

        path = _resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
        if path is None or not path.exists():
            fallback = _resolve_fs_path(PROJECT_ROOT / "model.onnx")
            path = fallback if fallback and fallback.exists() else path
        if path is None or not path.exists():
            return _json_error("No ONNX artifact was found for this model.", status_code=404)

        return _json_response({"status": "validated", "model_id": model_id, "path": str(path), "metadata": recover_model_params(path)})
    except Exception as exc:
        LOGGER.exception("ONNX validation failed for model_id=%s.", model_id)
        return _json_error(str(exc), status_code=500)


@app.get("/production/status", response_class=JSONResponse)
async def production_status() -> JSONResponse:
    return _json_response(PRODUCTION_STATE)


@app.post("/production/start", response_class=JSONResponse)
async def production_start(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}
        payload = validate_input(payload_data, ProductionStartRequest)
        model_id = payload.modelId
        dataset_id = payload.datasetId

        if model_id is None:
            models = await get_all_entries(db, LearningORM)
            trained_models = [model for model in models if getattr(model, "is_trained", False)]
            if not trained_models:
                return _json_error("No trained model is available for production.", status_code=404)
            trained_models.sort(key=lambda model: getattr(model, "id", 0), reverse=True)
            model_id = trained_models[0].id

        model_record = await LearningModel.read(db, model_id)
        if model_record is None:
            return _json_error("Model not found", status_code=404)

        if dataset_id is None:
            datasets = await get_all_entries(db, DatasetORM)
            if datasets:
                datasets.sort(key=lambda dataset: getattr(dataset, "id", 0), reverse=True)
                dataset_id = datasets[0].id

        deployment_summary = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "model_name": getattr(model_record, "name", None),
            "artifact_path": getattr(model_record, "path", None),
            "started_at": datetime.now().isoformat(),
            "status": "active",
        }
        if getattr(model_record, "path", None):
            try:
                resolved_artifact_path = _resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
                deployment_summary["artifact_metadata"] = recover_model_params(resolved_artifact_path or getattr(model_record, "path"))
            except Exception:
                LOGGER.exception("Unable to recover deployment artifact metadata.")

        save_deployment_summary(deployment_summary, str(DEPLOYMENT_ARTIFACT_DIR / f"deployment_{model_id}.json"))

        PRODUCTION_STATE.update(
            {
                "running": True,
                "status": "active",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "started_at": deployment_summary["started_at"],
                "stopped_at": None,
                "deployment": deployment_summary,
                "metrics": _normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
                "active_model": _serialize_model_summary(model_record),
            }
        )
        _append_production_log(f"Production started with model_id={model_id} dataset_id={dataset_id}.")

        await update_entry(
            db,
            LearningORM,
            model_id,
            {
                "is_deployed": True,
                "history": _append_history(
                    getattr(model_record, "history", None),
                    {"operation": "production_start", "when": deployment_summary["started_at"], "dataset_id": dataset_id},
                ),
            },
        )
        return _json_response({"status": "started", **PRODUCTION_STATE})
    except Exception as exc:
        LOGGER.exception("Production start failed.")
        return _json_error(str(exc), status_code=500)


@app.post("/production/stop", response_class=JSONResponse)
async def production_stop(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_id = PRODUCTION_STATE.get("model_id")
        stopped_at = datetime.now().isoformat()
        if model_id is not None:
            model_record = await LearningModel.read(db, int(model_id))
            if model_record is not None:
                await update_entry(
                    db,
                    LearningORM,
                    int(model_id),
                    {
                        "is_deployed": False,
                        "history": _append_history(
                            getattr(model_record, "history", None),
                            {"operation": "production_stop", "when": stopped_at},
                        ),
                    },
                )
        PRODUCTION_STATE.update({"running": False, "status": "inactive", "stopped_at": stopped_at})
        _append_production_log("Production stopped.")
        return _json_response({"status": "stopped", **PRODUCTION_STATE})
    except Exception as exc:
        LOGGER.exception("Production stop failed.")
        return _json_error(str(exc), status_code=500)


@app.post("/production/monitor", response_class=JSONResponse)
async def production_monitor(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}
        payload = validate_input(payload_data, ProductionMonitorRequest)
        model_id = payload.modelId or PRODUCTION_STATE.get("model_id")
        dataset_id = payload.datasetId or PRODUCTION_STATE.get("dataset_id")
        if model_id is None or dataset_id is None:
            return _json_error("Select a model and dataset before monitoring production.", status_code=400)

        snapshot = await _build_live_monitoring_snapshot(db, int(model_id), int(dataset_id), tolerance=float(payload.tolerance))
        PRODUCTION_STATE["monitoring"] = snapshot
        PRODUCTION_STATE["metrics"] = snapshot["metrics"]
        _append_production_log(f"Production monitoring refreshed for model_id={model_id} dataset_id={dataset_id}.")
        return _json_response({"status": "ok", "monitoring": snapshot, "production": PRODUCTION_STATE})
    except Exception as exc:
        LOGGER.exception("Production monitoring failed.")
        return _json_error(str(exc), status_code=500)


@app.post("/production/retrain", response_class=JSONResponse)
async def production_retrain(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}
        payload = validate_input(payload_data, ProductionMonitorRequest)
        model_id = payload.modelId or PRODUCTION_STATE.get("model_id")
        dataset_id = payload.datasetId or PRODUCTION_STATE.get("dataset_id")
        if model_id is None or dataset_id is None:
            return _json_error("Select a model and dataset before retraining.", status_code=400)

        retraining_result = await _trigger_retraining_from_monitoring(
            db,
            int(model_id),
            int(dataset_id),
            tolerance=float(payload.tolerance),
        )
        _append_production_log(f"Retraining flow executed for model_id={model_id} dataset_id={dataset_id}.")
        return _json_response({"status": "ok", "retraining": retraining_result})
    except Exception as exc:
        LOGGER.exception("Production retraining failed.")
        return _json_error(str(exc), status_code=500)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _json_response({"status": "error", "detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(_: Request, exc: ResponseValidationError) -> JSONResponse:
    LOGGER.exception("Response validation failed.", exc_info=exc)
    return _json_response({"status": "error", "detail": str(exc)}, status_code=500)


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception", exc_info=exc)
    error_detail = str(exc) if os.getenv("DEBUG") else "An error occurred"
    return _json_response({"detail": error_detail}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
