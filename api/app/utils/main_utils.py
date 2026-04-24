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
import warnings
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
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_session import AsyncSessionLocal, get_db, init_models, wait_for_database
from app.database.db_utils import (
    create_entry,
    get_all_entries,
    normalize_legacy_jsonb_containers,
    normalize_legacy_polymorphic_identities,
    update_entry,
)
from app.models.model_objects import CodeModel, DatasetModel, InferenceModel, LearningModel, StudyModel
from app.models.model_orm import CodeORM, DatasetORM, InferenceORM, LearningORM, StudyORM
from app.models.model_registry import ModelRegistry
from app.utils.config import merge_env_overrides, save_config_snapshot
from app.utils.deployment_utils import (
    create_fastapi_serving_file,
    get_model_uri_for_run,
    save_deployment_summary,
    save_model_local,
)
from app.utils.artifact_utils import (
    get_model_capability_matrix,
    inspect_model_artifact,
    load_runtime_artifact,
)
from app.utils.io import ensure_dir, save_json
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
from app.utils.run_ledger import create_run_entry, list_run_entries, read_run_entry, update_run_entry
from app.utils.serialization import to_json
from app.utils.tabular_utils import (
    apply_feature_operations,
    build_dataset_analysis_summary,
    build_feature_extraction_summary,
    build_column_explorer,
    dataset_profile,
    get_dataset_capability_matrix,
    load_tabular_from_path,
)
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

try:
    import torch
    import torch.nn as nn

    TORCH_RUNTIME_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_RUNTIME_AVAILABLE = False


if TORCH_RUNTIME_AVAILABLE and nn is not None and torch is not None:

    class FeedForwardNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            hidden_layers: int,
            output_dim: int,
            dropout: float,
        ) -> None:
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

else:

    class FeedForwardNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not available in the current environment.")


tracemalloc.start()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
RUNTIME_DIR = ensure_dir(PROJECT_ROOT / "runtime_artifacts")

CONFIG_SNAPSHOT_DIR = ensure_dir(RUNTIME_DIR / "config")
TRAINING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "training")
MONITORING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "monitoring")
DEPLOYMENT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "deployment")
OPTUNA_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "optuna")
PLOT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "plots")
SERVING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "serving")
EXPORT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "exports")
FEATURE_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "features")
RUN_LEDGER_DIR = ensure_dir(RUNTIME_DIR / "runs")

LOG_DIR = ensure_dir(PROJECT_ROOT / "logs")
MLRUNS_DIR = ensure_dir(PROJECT_ROOT / "mlruns")
ACTIVITY_LOG_PATH = MONITORING_ARTIFACT_DIR / "activity_log.jsonl"

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

REGISTRY_MODEL_MAP: dict[str, dict[str, Any]] = {
    "datasetmodel": {"model": DatasetModel, "orm": DatasetORM, "label": "DatasetModel"},
    "dataset": {"model": DatasetModel, "orm": DatasetORM, "label": "DatasetModel"},
    "learningmodel": {"model": LearningModel, "orm": LearningORM, "label": "LearningModel"},
    "learning": {"model": LearningModel, "orm": LearningORM, "label": "LearningModel"},
    "inferencemodel": {"model": InferenceModel, "orm": InferenceORM, "label": "InferenceModel"},
    "inference": {"model": InferenceModel, "orm": InferenceORM, "label": "InferenceModel"},
    "studymodel": {"model": StudyModel, "orm": StudyORM, "label": "StudyModel"},
    "study": {"model": StudyModel, "orm": StudyORM, "label": "StudyModel"},
    "codemodel": {"model": CodeModel, "orm": CodeORM, "label": "CodeModel"},
    "code": {"model": CodeModel, "orm": CodeORM, "label": "CodeModel"},
}


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


def _register_models() -> None:
    registry_pairs = (
        (DatasetModel, DatasetORM),
        (LearningModel, LearningORM),
        (InferenceModel, InferenceORM),
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


def _create_application() -> FastAPI:
    fastapi_app = FastAPI(title="Painel Amanaje API", version="0.3.0")
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    fastapi_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return fastapi_app


app = _create_application()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
_register_models()
_include_registry_routers()

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


PRODUCTION_STATE: dict[str, Any] = {
    "running": False,
    "status": "inactive",
    "active_watch_context_id": None,
    "inference_id": None,
    "model_id": None,
    "dataset_id": None,
    "started_at": None,
    "stopped_at": None,
    "deployment": {},
    "metrics": {},
    "monitoring": {},
    "active_model": None,
    "active_inference": None,
    "simulation": {},
    "logs": [],
    "watchlist": {},
}

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


def _append_history(history: Any, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = list(history) if isinstance(history, list) else []
    items.append(dict(entry))
    return items


def _append_production_log(message: str) -> None:
    timestamped_message = f"{datetime.now().isoformat()} | {message}"
    PRODUCTION_STATE["logs"] = list(PRODUCTION_STATE.get("logs", []))[-99:] + [timestamped_message]
    LOGGER.info(message)


def _watch_context_id(
    *,
    model_id: Any = None,
    dataset_id: Any = None,
    inference_id: Any = None,
) -> str:
    return f"inference:{inference_id or 'na'}|model:{model_id or 'na'}|dataset:{dataset_id or 'na'}"


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
    parameters = dict(getattr(model, "parameters", {}) or {})
    artifact_manifest = parameters.get("artifact_manifest") if isinstance(parameters.get("artifact_manifest"), Mapping) else None
    return {
        "id": getattr(model, "id", None),
        "name": getattr(model, "name", None),
        "description": getattr(model, "description", None),
        "model_type": getattr(model, "model_type", None),
        "is_trained": getattr(model, "is_trained", False),
        "is_tested": getattr(model, "is_tested", False),
        "is_deployed": getattr(model, "is_deployed", False),
        "metrics": getattr(model, "metrics", {}) or {},
        "parameters": parameters,
        "input_features": getattr(model, "input_features", None),
        "output_features": getattr(model, "output_features", None),
        "size": getattr(model, "size", None),
        "path": getattr(model, "path", None),
        "artifact_manifest": artifact_manifest,
        "runtime_capabilities": dict((artifact_manifest or {}).get("runtime_capabilities", {}) or {}),
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


def _normalize_registry_type(registry_type: str) -> str:
    return "".join(character for character in str(registry_type or "").lower() if character.isalnum())


def _get_registry_binding(registry_type: str) -> dict[str, Any]:
    normalized = _normalize_registry_type(registry_type)
    binding = REGISTRY_MODEL_MAP.get(normalized)
    if binding is None:
        raise ValueError(f"Unsupported registry type: {registry_type}")
    return binding


def _slugify(value: str) -> str:
    text = "".join(character.lower() if character.isalnum() else "_" for character in str(value or "").strip())
    return "_".join(part for part in text.split("_") if part) or "artifact"


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(to_json(value))
    except Exception:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return value


def _read_activity_log(limit: Optional[int] = None) -> list[dict[str, Any]]:
    if not ACTIVITY_LOG_PATH.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with ACTIVITY_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
    except OSError:
        LOGGER.exception("Unable to read the lifetime activity log.")
        return []

    return entries[-limit:] if limit else entries


def _append_activity_log(
    message: str,
    *,
    event_type: str,
    details: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message,
        "details": _json_safe(dict(details or {})),
    }
    ACTIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ACTIVITY_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        LOGGER.exception("Unable to append to the lifetime activity log.")

    PRODUCTION_STATE["logs"] = list(PRODUCTION_STATE.get("logs", []))[-99:] + [f"{timestamp} | {message}"]
    LOGGER.info(message)
    return entry


def _build_mlflow_health_snapshot() -> dict[str, Any]:
    tracking_uri = str(RUNTIME_CONFIG.get("mlflow", {}).get("tracking_uri", ""))
    health = {
        "tracking_uri": tracking_uri,
        "status": "ok",
        "warnings": [],
        "repair_hints": [],
    }

    if not MLFLOW_DIRECT_AVAILABLE or mlflow is None:
        health["status"] = "degraded"
        health["warnings"].append("MLflow is not importable in the current environment.")
        health["repair_hints"].append("Install the MLflow dependencies from api/requirements.txt and restart the app.")
        return health

    try:
        experiments = list_experiments()
        stale_locations = [
            experiment["artifact_location"]
            for experiment in experiments
            if str(experiment.get("artifact_location") or "").startswith("file:///app/mlruns/")
        ]
        if stale_locations:
            health["status"] = "warning"
            health["warnings"].append(
                f"{len(stale_locations)} experiment(s) still reference container-local paths such as file:///app/mlruns/."
            )
            health["repair_hints"].append(
                "Repoint the tracking URI or repair the stored artifact_location entries so they match the current workspace."
            )
    except Exception as exc:
        health["status"] = "degraded"
        health["warnings"].append(str(exc))
        health["repair_hints"].append(
            "Check whether the tracking URI is reachable and whether the mlruns directory still exists after the last reset."
        )

    if tracking_uri.startswith("file:///app/mlruns/"):
        health["status"] = "warning"
        health["warnings"].append("The active tracking URI still points to a container-local /app/mlruns path.")
        health["repair_hints"].append("Update MLFLOW_TRACKING_URI to the current workspace mlruns directory.")

    return health


async def _find_inference_record(
    db: AsyncSession,
    *,
    inference_id: Optional[int] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
) -> Any | None:
    if inference_id not in (None, 0, "0", ""):
        return await InferenceModel.read(db, int(inference_id))

    if model_id is None or dataset_id is None:
        return None

    inference_records = await get_all_entries(db, InferenceORM)
    for record in inference_records:
        if int(getattr(record, "learning_model_id", 0) or 0) == int(model_id) and int(getattr(record, "dataset_id", 0) or 0) == int(dataset_id):
            return record
    return None


def _serialize_inference_summary(record: Any) -> dict[str, Any]:
    params = dict(getattr(record, "inference_params", {}) or {})
    latest_metrics = _normalize_numeric_metrics(params.get("latest_metrics", {}) or {})
    latest_monitoring = params.get("monitoring", {}) or {}
    simulation_defaults = params.get("simulation_defaults", {}) or {}
    artifact_manifest = params.get("artifact_manifest") if isinstance(params.get("artifact_manifest"), Mapping) else {}
    return {
        "id": getattr(record, "id", None),
        "name": getattr(record, "name", None),
        "description": getattr(record, "description", None),
        "learning_model_id": getattr(record, "learning_model_id", None),
        "dataset_id": getattr(record, "dataset_id", None),
        "input_features": getattr(record, "input_features", None),
        "output_features": getattr(record, "output_features", None),
        "path": getattr(record, "path", None),
        "status": params.get("status", "idle"),
        "latest_metrics": latest_metrics,
        "latest_monitoring": latest_monitoring,
        "simulation_defaults": simulation_defaults,
        "artifact_manifest": artifact_manifest,
        "runtime_capabilities": dict((artifact_manifest or {}).get("runtime_capabilities", {}) or {}),
        "history": list(getattr(record, "history", None) or []),
        "inference_params": params,
    }


async def _upsert_inference_record(
    db: AsyncSession,
    *,
    model_record: Any,
    dataset_record: Any,
    input_features: Optional[list[str]] = None,
    output_features: Optional[list[str]] = None,
    inference_updates: Optional[Mapping[str, Any]] = None,
    history_entry: Optional[Mapping[str, Any]] = None,
    artifact_path: Optional[str] = None,
) -> Any:
    existing = await _find_inference_record(
        db,
        model_id=int(getattr(model_record, "id")),
        dataset_id=int(getattr(dataset_record, "id")),
    )
    inference_params = {
        **dict(getattr(existing, "inference_params", {}) or {}),
        **dict(inference_updates or {}),
    }
    record_name = f"{getattr(model_record, 'name', 'model')} :: {getattr(dataset_record, 'name', 'dataset')}"
    payload = {
        "name": getattr(existing, "name", None) or record_name,
        "description": getattr(existing, "description", None) or f"Inference pair for {record_name}",
        "object_type": "inference_model",
        "size": float(getattr(model_record, "size", 0.0) or 0.0),
        "path": artifact_path or getattr(existing, "path", None) or getattr(model_record, "path", None) or f"runtime_artifacts/inference/{_slugify(record_name)}.json",
        "date": datetime.now(),
        "version": getattr(existing, "version", None) or getattr(model_record, "version", None) or 1,
        "history": _append_history(getattr(existing, "history", None), history_entry or {"operation": "sync", "when": datetime.now().isoformat()}),
        "learning_model_id": int(getattr(model_record, "id")),
        "dataset_id": int(getattr(dataset_record, "id")),
        "input_features": input_features if input_features is not None else getattr(existing, "input_features", None),
        "output_features": output_features if output_features is not None else getattr(existing, "output_features", None),
        "inference_params": inference_params or {},
    }

    if existing is not None:
        return await update_entry(db, InferenceORM, existing.id, payload)
    return await create_entry(db, InferenceORM, payload)


def _build_bar_plot(title: str, series: list[tuple[str, Any]], *, description: Optional[str] = None) -> dict[str, Any]:
    points = []
    for label, value in series:
        if value in (None, ""):
            continue
        try:
            points.append({"label": str(label), "value": float(value)})
        except (TypeError, ValueError):
            continue
    return {"kind": "bar", "title": title, "description": description, "series": points}


def _build_line_plot(title: str, series: list[tuple[Any, Any]], *, description: Optional[str] = None) -> dict[str, Any]:
    points = []
    for label, value in series:
        try:
            points.append({"label": str(label), "value": float(value)})
        except (TypeError, ValueError):
            continue
    return {"kind": "line", "title": title, "description": description, "series": points}


def _build_dataset_plots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    profile = dict(summary.get("profile", {}) or {})
    plots.append(
        _build_bar_plot(
            "Column Types",
            [
                ("Numeric", len(profile.get("numeric_columns", []) or [])),
                ("Categorical", len(profile.get("categorical_columns", []) or [])),
                ("Datetime", len(profile.get("datetime_columns", []) or [])),
            ],
            description="Automatic type balance from the extracted dataset profile.",
        )
    )

    stats = dict(summary.get("stats", {}) or {})
    missing_series = sorted(
        (
            (column_name, details.get("null_count", 0))
            for column_name, details in stats.items()
            if isinstance(details, Mapping)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    plots.append(
        _build_bar_plot(
            "Missing Values by Column",
            missing_series,
            description="Top columns ranked by missing-value count.",
        )
    )

    mean_series = [
        (column_name, details.get("mean"))
        for column_name, details in stats.items()
        if isinstance(details, Mapping) and details.get("type") == "numeric"
    ][:8]
    plots.append(
        _build_bar_plot(
            "Numeric Mean Snapshot",
            mean_series,
            description="Quick mean comparison for the first numeric columns in the analysis.",
        )
    )
    return [plot for plot in plots if plot["series"]]


def _build_model_plots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    plots = [
        _build_bar_plot(
            "Model Metrics",
            list(_normalize_numeric_metrics(summary.get("metrics", {}) or {}).items()),
            description="All numeric metrics currently persisted for this learning model.",
        ),
        _build_bar_plot(
            "Lifecycle Flags",
            [
                ("Trained", 1 if summary.get("is_trained") else 0),
                ("Tested", 1 if summary.get("is_tested") else 0),
                ("Deployed", 1 if summary.get("is_deployed") else 0),
            ],
            description="Binary view of the current model lifecycle state.",
        ),
    ]
    return [plot for plot in plots if plot["series"]]


def _build_study_plots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    study_params = dict(summary.get("study_params", {}) or {})
    plots.append(
        _build_bar_plot(
            "Study Controls",
            [
                ("Trials", study_params.get("n_trials")),
                ("Best Value", (summary.get("best_trial") or {}).get("value") if isinstance(summary.get("best_trial"), Mapping) else None),
            ],
            description="Optimization controls and the latest best objective value.",
        )
    )
    plots.append(
        _build_bar_plot(
            "Best Numeric Parameters",
            [
                (key, value)
                for key, value in dict(summary.get("best_params", {}) or {}).items()
                if isinstance(value, (int, float))
            ],
            description="Only numeric best parameters are charted automatically.",
        )
    )
    return [plot for plot in plots if plot["series"]]


def _build_inference_plots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    monitoring_summary = dict((summary.get("latest_monitoring") or {}).get("summary", {}) or {})
    plots = [
        _build_bar_plot(
            "Latest Inference Metrics",
            list((summary.get("latest_metrics") or {}).items()),
            description="Most recent numeric metrics captured for this inference pair.",
        ),
        _build_bar_plot(
            "Monitoring Snapshot",
            [
                ("Drifted Features", monitoring_summary.get("drifted_features", 0)),
                ("Degraded Metrics", monitoring_summary.get("degraded_metrics", 0)),
                ("Runs Logged", len(summary.get("history", []) or [])),
            ],
            description="Current monitoring pressure and inference activity volume.",
        ),
    ]
    return [plot for plot in plots if plot["series"]]


def _build_code_plots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    variables = dict(summary.get("variables", {}) or {})
    document = dict(summary.get("code", {}) or {})
    return [
        _build_bar_plot(
            "Code Asset Footprint",
            [
                ("Variables", len(variables)),
                ("Document Keys", len(document)),
            ],
            description="Stored object counts extracted from the saved code payload.",
        )
    ]


def _build_extraction_manifest(dataset: Any, dataframe: pd.DataFrame) -> dict[str, Any]:
    preview_rows = _json_safe(dataframe.head(20).replace({np.nan: None}).to_dict(orient="records"))
    column_details = _json_safe(build_column_explorer(dataframe))

    return {
        "dataset_id": getattr(dataset, "id", None),
        "dataset_name": getattr(dataset, "name", None),
        "table_name": f"dataset_{getattr(dataset, 'id', 'new')}_{_slugify(getattr(dataset, 'name', 'dataset'))}",
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "columns": column_details,
        "preview_rows": preview_rows,
        "available_edits": [
            "rename_columns",
            "cast_types",
            "fill_missing_values",
            "drop_columns",
            "filter_rows",
            "sort_by",
            "limit_rows",
        ],
        "analysis": _dataset_analysis_summary(
            dataframe,
            str(_resolve_fs_path(getattr(dataset, "path", None), default_parent=REPO_ROOT) or getattr(dataset, "path", "")),
        ),
    }


async def _build_registry_analysis(db: AsyncSession, registry_type: str, item_id: int) -> dict[str, Any]:
    binding = _get_registry_binding(registry_type)
    record = await binding["model"].read(db, item_id)
    if record is None:
        raise ValueError(f"{binding['label']} not found")

    normalized_type = _normalize_registry_type(registry_type)
    if normalized_type in {"datasetmodel", "dataset"}:
        dataframe = _load_dataset_frame_from_record(record)
        summary = _dataset_analysis_summary(
            dataframe,
            str(_resolve_fs_path(getattr(record, "path", None), default_parent=REPO_ROOT) or getattr(record, "path", "")),
        )
        plots = _build_dataset_plots(summary)
    elif normalized_type in {"learningmodel", "learning"}:
        summary = _augment_model_analysis(record)
        plots = _build_model_plots(summary)
    elif normalized_type in {"studymodel", "study"}:
        summary = _serialize_study_summary(record)
        plots = _build_study_plots(summary)
    elif normalized_type in {"inferencemodel", "inference"}:
        summary = _serialize_inference_summary(record)
        plots = _build_inference_plots(summary)
    else:
        summary = _json_safe(ModelRegistry._orm_to_dict(record))
        plots = _build_code_plots(summary)

    return {
        "registry_type": binding["label"],
        "item_id": item_id,
        "summary": summary,
        "plots": plots,
    }


async def _resolve_runtime_context(
    db: AsyncSession,
    *,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    inference_id: Optional[int] = None,
) -> tuple[Any | None, Any | None, Any | None]:
    inference_record = await _find_inference_record(db, inference_id=inference_id, model_id=model_id, dataset_id=dataset_id)
    if inference_record is not None:
        model_id = model_id or int(getattr(inference_record, "learning_model_id", 0) or 0)
        dataset_id = dataset_id or int(getattr(inference_record, "dataset_id", 0) or 0)

    model_record = await LearningModel.read(db, model_id) if model_id not in (None, 0, "0", "") else None
    dataset_record = await DatasetModel.read(db, dataset_id) if dataset_id not in (None, 0, "0", "") else None
    return inference_record, model_record, dataset_record


def _prepare_runtime_execution(model_record: Any, dataset_record: Any) -> tuple[PreparedDataset, TrainingResult]:
    parameters = dict(getattr(model_record, "parameters", {}) or {})
    prepared = _prepare_dataset_for_training(dataset_record, model_record, parameters)
    path = _resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
    if path is None or not path.exists():
        raise FileNotFoundError("The selected model does not have a persisted artifact path.")

    runtime_payload = load_runtime_artifact(path, parameters=parameters)
    trained_model = runtime_payload["model"]
    framework = str(runtime_payload.get("framework") or parameters.get("framework", "sklearn")).strip().lower()
    if framework == "onnx":
        framework = str(
            parameters.get("training_backend") or parameters.get("base_framework") or "sklearn"
        ).strip().lower()

    runtime_result = TrainingResult(
        model=trained_model,
        framework=framework,
        metrics=_normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
        parameters=parameters,
    )
    runtime_result.parameters = {**parameters, "artifact_manifest": runtime_payload.get("manifest", {})}
    return prepared, runtime_result


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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
                except TypeError:
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
    dataframe, _parser_report = load_tabular_from_path(resolved_path)
    return dataframe


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
    artifact_manifest: dict[str, Any] = {}

    summary_path = artifact_dir / "training_summary.json"
    save_training_summary(result, str(summary_path))
    artifact_paths.append(str(summary_path))

    try:
        if result.framework == "sklearn":
            try:
                import joblib
            except Exception as exc:
                raise RuntimeError(f"joblib is required to persist sklearn models: {exc}") from exc
            target_path = artifact_dir / "model_sklearn.joblib"
            joblib.dump(result.model, target_path)
            model_artifact_path = str(target_path)
        elif result.framework == "pytorch":
            if not TORCH_RUNTIME_AVAILABLE or torch is None:
                raise RuntimeError("PyTorch is not available in the current environment.")
            target_path = artifact_dir / "model_pytorch.pt"
            result.model.eval()
            example_input = torch.as_tensor(prepared.x_train.head(2).to_numpy(dtype=np.float32))
            try:
                exported_model = torch.jit.script(result.model)
            except Exception:
                exported_model = torch.jit.trace(result.model, example_input)
            exported_model.save(str(target_path))
            companion_spec = {
                "framework": "pytorch",
                "artifact_format": ".pt",
                "input_features": list(prepared.input_features),
                "output_feature": prepared.output_feature,
                "task_type": prepared.task_type,
                "label_mapping": dict(prepared.label_mapping),
            }
            companion_path = artifact_dir / "model_spec.json"
            save_json(companion_spec, companion_path)
            artifact_paths.append(str(companion_path))
            model_artifact_path = str(target_path)
        else:
            model_artifact_path = save_model_local(result.model, str(artifact_dir / f"model_{result.framework}.pkl"))
        artifact_paths.append(model_artifact_path)
        artifact_manifest = inspect_model_artifact(
            model_artifact_path,
            load_runtime=False,
            parameters=result.parameters,
        )
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
        "artifact_manifest": artifact_manifest,
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
    return dataset_profile(df)


def _dataset_analysis_summary(df: pd.DataFrame, df_path: str) -> dict[str, Any]:
    parser_report = {}
    try:
        _, parser_report = load_tabular_from_path(df_path)
    except Exception:
        parser_report = {}
    return build_dataset_analysis_summary(df, df_path, parser_report=parser_report)


def _build_feature_extraction_summary(df: pd.DataFrame) -> dict[str, Any]:
    return build_feature_extraction_summary(df)


def _augment_model_analysis(model_record: Any) -> dict[str, Any]:
    summary = _serialize_model_summary(model_record)
    path_value = getattr(model_record, "path", None)
    path = _resolve_fs_path(path_value, default_parent=REPO_ROOT)
    if path and path.exists():
        try:
            manifest = inspect_model_artifact(path, load_runtime=False, parameters=getattr(model_record, "parameters", {}) or {})
            summary["artifact_metadata"] = manifest
            summary["artifact_manifest"] = manifest
            summary["runtime_capabilities"] = manifest.get("runtime_capabilities", {})
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

    runtime_payload = load_runtime_artifact(path, parameters=parameters)
    trained_model = runtime_payload["model"]
    framework = str(runtime_payload.get("framework") or parameters.get("framework", "sklearn")).lower()
    training_result = TrainingResult(
        model=trained_model,
        framework=framework,
        metrics=_normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
        parameters={**parameters, "artifact_manifest": runtime_payload.get("manifest", {})},
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
        "artifact_manifest": runtime_payload.get("manifest", {}),
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


async def run_startup_tasks() -> None:
    try:
        await wait_for_database()
        APP_LOGGER.info("Database connection is ready.")
    except Exception:
        LOGGER.exception("Database did not become ready during startup.")
        raise

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


@app.on_event("startup")
async def startup_event() -> None:
    await run_startup_tasks()


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
    error_detail = str(exc) or exc.__class__.__name__
    return _json_response({"detail": error_detail}, status_code=500)
