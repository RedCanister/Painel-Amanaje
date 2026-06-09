from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import inspect
import json
import math
import os
import re
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
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect as sqlalchemy_inspect

from app.database.db_session import AsyncSessionLocal, get_db, init_models, wait_for_database
from app.database.db_utils import (
    create_entry,
    get_all_entries,
    normalize_legacy_jsonb_containers,
    normalize_legacy_polymorphic_identities,
    update_entry,
)
from app.models.model_objects import AssistantModel, AssistantTrainingDatasetModel, CodeModel, DatasetModel, InferenceModel, LearningModel, PanelDashboardModel, StudyModel
from app.models.model_orm import AssistantORM, AssistantTrainingDatasetORM, CodeORM, DatasetORM, InferenceORM, LearningORM, PanelDashboardORM, StudyORM
from app.models.model_registry import ModelRegistry
from app.utils.config import merge_env_overrides, save_config_snapshot
from app.utils.deployment_utils import (
    create_fastapi_serving_file,
    get_model_uri_for_run,
    save_deployment_summary,
    save_model_local,
)
from app.utils.artifact_utils import (
    RuntimeArtifactDependencyError,
    get_model_capability_matrix,
    inspect_model_artifact,
    load_runtime_artifact,
)
from app.utils.accelerators import resolve_torch_device
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
from app.utils.plot_registry import (
    build_bar_plot_spec,
    build_dataset_plot_specs,
    build_line_plot_spec,
    build_table_plot_spec,
    legacy_plot_specs_for_job,
    list_legacy_plot_artifacts,
    resolve_legacy_plot_path,
)
from app.utils.plotting import plot_loss_curve, plot_metric_comparison, plot_predictions_vs_actual
from app.utils.retrain_utils import monitor_and_retrain, should_retrain
from app.utils.run_ledger import (
    append_run_terminal_line,
    build_run_terminal_lines,
    create_run_entry,
    list_run_entries,
    read_run_entry,
    summarize_run_entry,
    update_run_entry,
)
from app.utils.serialization import to_json
from app.utils.settings import (
    apply_debug_logging,
    apply_env_overrides,
    apply_saved_env_overrides,
    build_client_settings,
    load_settings_state,
    register_provider_token_env_var,
)
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
    import sqlparse

    SQLPARSE_AVAILABLE = True
except Exception:
    sqlparse = None  # type: ignore[assignment]
    SQLPARSE_AVAILABLE = False

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
SETTINGS_STATE_PATH = CONFIG_SNAPSHOT_DIR / "settings_state.json"
TRAINING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "training")
MONITORING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "monitoring")
DEPLOYMENT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "deployment")
OPTUNA_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "optuna")
PLOT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "plots")
SERVING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "serving")
EXPORT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "exports")
FEATURE_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "features")
STORE_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "store")
RUN_LEDGER_DIR = ensure_dir(RUNTIME_DIR / "runs")
ASSISTANT_MODEL_DIR = ensure_dir(RUNTIME_DIR / "assistant_models")
ASSISTANT_DATASET_DIR = ensure_dir(RUNTIME_DIR / "assistant_datasets")
ASSISTANT_EVAL_DIR = ensure_dir(RUNTIME_DIR / "assistant_evals")

LOG_DIR = ensure_dir(PROJECT_ROOT / "logs")
MLRUNS_DIR = ensure_dir(PROJECT_ROOT / "mlruns")
ACTIVITY_LOG_PATH = MONITORING_ARTIFACT_DIR / "activity_log.jsonl"


def _register_store_provider_token_env_vars() -> None:
    providers_path = STORE_ARTIFACT_DIR / "providers.json"
    if not providers_path.exists():
        return
    try:
        payload = json.loads(providers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    providers = payload.get("providers") if isinstance(payload, Mapping) else {}
    if not isinstance(providers, Mapping):
        return
    for provider in providers.values():
        if not isinstance(provider, Mapping):
            continue
        token_env_var = str(provider.get("token_env_var") or "").strip()
        if not token_env_var:
            continue
        try:
            register_provider_token_env_var(token_env_var, provider_name=str(provider.get("name") or ""))
        except Exception:
            APP_LOGGER.warning("Ignoring invalid Store provider token_env_var while loading settings.", exc_info=True)

DATASET_OPERATION = "datasets"
MODEL_OPERATION = "models"
ASSISTANT_MODEL_OPERATION = "assistant-models"
ALLOWED_UPLOAD_OPERATIONS = {DATASET_OPERATION, MODEL_OPERATION, ASSISTANT_MODEL_OPERATION}
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
SIMULATION_DEFAULT_STEPS = 12
SIMULATION_MAX_STEPS = 120
SQL_FEATURE_SOURCE_RELATION = "source_dataset"
SQL_FEATURE_SECONDARY_RELATION = "secondary_dataset"
SQL_FEATURE_RESULT_ALIAS = "amanaje_sql_feature_result"
SQL_FEATURE_BLOCKED_KEYWORDS = {
    "ALTER",
    "CALL",
    "COPY",
    "CREATE",
    "DELETE",
    "DO",
    "DROP",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "INSERT",
    "MERGE",
    "REINDEX",
    "REVOKE",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
}
SIMULATION_MAX_SCENARIOS = 6
SIMULATION_MAX_ROWS = 600
SIMULATION_MAX_SENSITIVITY_FEATURES = 8
SIMULATION_FAMILY_LABELS = {
    "regression": "Regression",
    "classification": "Classification",
    "unsupervised": "Unsupervised",
    "recommendation": "Recommendation",
    "geospatial": "Geospatial",
    "llm_slm": "LLM/SLM",
    "unsupported": "Unsupported",
}
SIMULATION_FAMILY_MODES = {
    "regression": [
        {
            "id": "tabular_what_if",
            "label": "What-if Analysis",
            "description": "Baseline, feature overrides, prediction path, and sensitivity checks.",
        }
    ],
    "classification": [
        {
            "id": "classification_threshold",
            "label": "Class Probability",
            "description": "Scenario prediction with class probabilities, threshold, and top-k class review.",
        }
    ],
    "unsupervised": [
        {
            "id": "unsupervised_probe",
            "label": "Cluster / Anomaly Probe",
            "description": "Inspect assignment, anomaly score, and embeddings when the runtime exposes them.",
        }
    ],
    "recommendation": [
        {
            "id": "recommendation_top_n",
            "label": "Top-N Ranking",
            "description": "Score or recommend ranked items for a user, item, or context payload.",
        }
    ],
    "geospatial": [
        {
            "id": "geospatial_sweep",
            "label": "Point / Grid Sweep",
            "description": "Vary coordinates and return map-ready point predictions when latitude and longitude are available.",
        }
    ],
    "llm_slm": [
        {
            "id": "llm_prompt",
            "label": "Prompt Trial",
            "description": "Prompt, generation controls, response metadata, and readiness for assistant runtimes.",
        }
    ],
    "unsupported": [
        {
            "id": "runtime_readiness",
            "label": "Runtime Readiness",
            "description": "Explain which runtime capabilities are needed before simulation can execute.",
        }
    ],
}
SIMULATION_OUTPUT_KINDS = {
    "regression": "continuous",
    "classification": "categorical",
    "unsupervised": "assignment_or_score",
    "recommendation": "ranking",
    "geospatial": "map_ready_prediction",
    "llm_slm": "generated_text",
    "unsupported": "unknown",
}
SIMULATION_GEO_LATITUDE_NAMES = {"lat", "latitude", "y", "geo_lat", "gps_lat"}
SIMULATION_GEO_LONGITUDE_NAMES = {"lon", "lng", "longitude", "x", "geo_lon", "gps_lon"}
CONTROL_PARAMETER_KEYS = {
    "framework",
    "input_features",
    "output_features",
    "test_size",
    "random_state",
    "estimator_class",
    "estimator_params",
    "fit_params",
    "fit_kwargs",
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
    "metrics_to_track",
    "plot_results",
    "training_mode",
    "training_backend",
    "base_framework",
    "device",
    "algorithm",
    "base_model_id",
    "base_artifact_path",
    "base_run_id",
    "transfer_strategy",
    "warm_start_increment",
    "training_lineage",
    "artifact_manifest",
}
SKLEARN_SUPERVISED_FAMILIES = {"classifier", "regressor"}
SKLEARN_UNSUPERVISED_FAMILIES = {"clustering", "clusterer", "transformer", "outlier", "unsupervised", "estimator"}
SKLEARN_UNSUPERVISED_TASK_TYPES = {
    "clustering",
    "clusterer",
    "cluster",
    "transformer",
    "outlier",
    "outlier_detection",
    "anomaly",
    "anomaly_detection",
    "unsupervised",
    "estimator",
}
SKLEARN_ESTIMATOR_LOOKUP_CACHE: dict[str, Any] | None = None

APP_LOGGER = init_global_logging(log_dir=str(LOG_DIR), logger_name="painel_amanaje")
LOGGER = get_logger("main_app", log_file=str(LOG_DIR / "main_app.log"))
_register_store_provider_token_env_vars()
apply_saved_env_overrides(SETTINGS_STATE_PATH)
apply_debug_logging(
    bool(load_settings_state(SETTINGS_STATE_PATH).get("feature_flags", {}).get("debug_mode")),
    (APP_LOGGER, LOGGER),
)

REGISTRY_MODEL_MAP: dict[str, dict[str, Any]] = {
    "datasetmodel": {"model": DatasetModel, "orm": DatasetORM, "label": "DatasetModel"},
    "dataset": {"model": DatasetModel, "orm": DatasetORM, "label": "DatasetModel"},
    "assistanttrainingdatasetmodel": {
        "model": AssistantTrainingDatasetModel,
        "orm": AssistantTrainingDatasetORM,
        "label": "AssistantTrainingDatasetModel",
    },
    "assistanttrainingdataset": {
        "model": AssistantTrainingDatasetModel,
        "orm": AssistantTrainingDatasetORM,
        "label": "AssistantTrainingDatasetModel",
    },
    "learningmodel": {"model": LearningModel, "orm": LearningORM, "label": "LearningModel"},
    "learning": {"model": LearningModel, "orm": LearningORM, "label": "LearningModel"},
    "assistantmodel": {"model": AssistantModel, "orm": AssistantORM, "label": "AssistantModel"},
    "assistant": {"model": AssistantModel, "orm": AssistantORM, "label": "AssistantModel"},
    "assistant_model": {"model": AssistantModel, "orm": AssistantORM, "label": "AssistantModel"},
    "inferencemodel": {"model": InferenceModel, "orm": InferenceORM, "label": "InferenceModel"},
    "inference": {"model": InferenceModel, "orm": InferenceORM, "label": "InferenceModel"},
    "studymodel": {"model": StudyModel, "orm": StudyORM, "label": "StudyModel"},
    "study": {"model": StudyModel, "orm": StudyORM, "label": "StudyModel"},
    "codemodel": {"model": CodeModel, "orm": CodeORM, "label": "CodeModel"},
    "code": {"model": CodeModel, "orm": CodeORM, "label": "CodeModel"},
    "paneldashboardmodel": {"model": PanelDashboardModel, "orm": PanelDashboardORM, "label": "PanelDashboardModel"},
    "paneldashboard": {"model": PanelDashboardModel, "orm": PanelDashboardORM, "label": "PanelDashboardModel"},
    "panel": {"model": PanelDashboardModel, "orm": PanelDashboardORM, "label": "PanelDashboardModel"},
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
            "exports_dir": str(EXPORT_ARTIFACT_DIR),
            "features_dir": str(FEATURE_ARTIFACT_DIR),
            "store_dir": str(STORE_ARTIFACT_DIR),
            "assistant_models_dir": str(ASSISTANT_MODEL_DIR),
            "assistant_datasets_dir": str(ASSISTANT_DATASET_DIR),
            "assistant_evals_dir": str(ASSISTANT_EVAL_DIR),
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
        (AssistantTrainingDatasetModel, AssistantTrainingDatasetORM),
        (LearningModel, LearningORM),
        (AssistantModel, AssistantORM),
        (InferenceModel, InferenceORM),
        (CodeModel, CodeORM),
        (StudyModel, StudyORM),
        (PanelDashboardModel, PanelDashboardORM),
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
    registryContext: dict[str, Any] = Field(default_factory=dict)


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
    target_series: pd.Series | pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series | pd.DataFrame
    y_test: pd.Series | pd.DataFrame
    input_features: list[str]
    output_feature: str
    task_type: str
    output_features: list[str] = field(default_factory=list)
    label_mapping: dict[str, Any] = field(default_factory=dict)


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

def apply_runtime_settings_state(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved_state = dict(state or load_settings_state(SETTINGS_STATE_PATH))
    apply_env_overrides(resolved_state)
    feature_flags = resolved_state.get("feature_flags", {}) if isinstance(resolved_state, Mapping) else {}
    apply_debug_logging(bool(feature_flags.get("debug_mode")), (APP_LOGGER, LOGGER))
    return resolved_state


def is_debug_mode_enabled() -> bool:
    try:
        state = load_settings_state(SETTINGS_STATE_PATH)
    except Exception:
        return False
    flags = state.get("feature_flags", {}) if isinstance(state, Mapping) else {}
    return bool(flags.get("debug_mode"))


def _render_page(template_name: str, request_: Request, **context: Any) -> HTMLResponse:
    context.setdefault("amanaje_settings", build_client_settings(load_settings_state(SETTINGS_STATE_PATH)))
    return templates.TemplateResponse(
            name = template_name, 
            request = request_,
            context = context 
        )


def _strict_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return _strict_json_safe(value.model_dump())
    if getattr(value, "__mapper__", None) is not None:
        try:
            inspected = sqlalchemy_inspect(value)
            return {
                attr.key: _strict_json_safe(getattr(value, attr.key))
                for attr in inspected.mapper.column_attrs
            }
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _strict_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strict_json_safe(item) for item in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _strict_json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _strict_json_safe(value.item())
        except Exception:
            pass
    return value


def _json_payload(payload: Any) -> Any:
    sanitized = _strict_json_safe(payload)
    try:
        return json.loads(json.dumps(sanitized, ensure_ascii=False, allow_nan=False, default=str))
    except (TypeError, ValueError):
        return json.loads(to_json(sanitized))


def _json_response(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(_json_payload(payload), status_code=status_code)


def _request_debug_context(request_: Request | None) -> dict[str, Any] | None:
    if request_ is None:
        return None
    try:
        return {
            "method": request_.method,
            "path": request_.url.path,
            "query": request_.url.query,
        }
    except Exception:
        return None


def _debug_type_payload(value: Any) -> dict[str, Any]:
    try:
        return debug_type(value)
    except Exception as exc:
        return {
            "type": str(type(value)),
            "debug_type_error": str(exc),
        }


def _build_debug_error_payload(
    detail: str,
    status_code: int,
    *,
    exc: BaseException | None = None,
    request_: Request | None = None,
    debug_subject: Any = None,
) -> dict[str, Any] | None:
    if not is_debug_mode_enabled():
        return None

    active_exception = exc
    if active_exception is None:
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type is not None and isinstance(exc_value, BaseException):
            active_exception = exc_value

    if active_exception is not None:
        error_type = active_exception.__class__.__name__
        traceback_text = "".join(
            traceback.format_exception(
                type(active_exception),
                active_exception,
                active_exception.__traceback__,
            )
        )
        subject = active_exception if debug_subject is None else debug_subject
    else:
        error_type = "ErrorResponse"
        traceback_text = "".join(traceback.format_stack(limit=12))
        subject = debug_subject if debug_subject is not None else {"detail": detail, "status_code": status_code}

    return {
        "error_type": error_type,
        "traceback": traceback_text,
        "request": _request_debug_context(request_),
        "debug_type": _debug_type_payload(subject),
    }


def _json_error(
    detail: str,
    status_code: int = 400,
    *,
    exc: BaseException | None = None,
    request: Request | None = None,
    debug_subject: Any = None,
    **extra: Any,
) -> JSONResponse:
    payload = {"status": "error", "detail": detail, **extra}
    debug_payload = _build_debug_error_payload(
        detail,
        status_code,
        exc=exc,
        request_=request,
        debug_subject=debug_subject,
    )
    if debug_payload is not None:
        payload["debug"] = debug_payload
    return _json_response(payload, status_code=status_code)


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
        return _json_payload(value)
    except Exception:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return _strict_json_safe(value)


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
    return build_bar_plot_spec(title, series, description=description)


def _build_line_plot(title: str, series: list[tuple[Any, Any]], *, description: Optional[str] = None) -> dict[str, Any]:
    return build_line_plot_spec(title, series, description=description)


def _build_table_plot(
    title: str,
    rows: list[Mapping[str, Any]],
    *,
    description: Optional[str] = None,
    source: Optional[Mapping[str, Any]] = None,
    max_rows: int = 20,
) -> dict[str, Any]:
    return build_table_plot_spec(title, rows, description=description, source=source, max_rows=max_rows)


def _build_dataset_plots(summary: Mapping[str, Any], dataframe: Optional[pd.DataFrame] = None) -> list[dict[str, Any]]:
    return build_dataset_plot_specs(
        summary,
        dataframe=dataframe,
        source={"domain": "dataset_analysis"},
    )


def _build_model_plots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    parameters = dict(summary.get("parameters", {}) or {})
    training_lineage = dict(parameters.get("training_lineage", {}) or {})
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
    plots.extend(legacy_plot_specs_for_job(PLOT_ARTIFACT_DIR, training_lineage.get("latest_run_id")))
    return [plot for plot in plots if plot.get("kind") == "legacy_image" or plot.get("series")]


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
    inference_params = dict(summary.get("inference_params", {}) or {})
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
    plots.extend(legacy_plot_specs_for_job(PLOT_ARTIFACT_DIR, inference_params.get("last_job_id")))
    try:
        artifact_model_id = int(summary.get("learning_model_id")) if summary.get("learning_model_id") not in (None, "") else None
    except (TypeError, ValueError):
        artifact_model_id = None
    model_artifacts = list_legacy_plot_artifacts(PLOT_ARTIFACT_DIR, model_id=artifact_model_id)
    seen_plot_ids = {plot.get("id") for plot in plots}
    plots.extend([plot for plot in model_artifacts if plot.get("id") not in seen_plot_ids][:8])
    return [plot for plot in plots if plot.get("kind") == "legacy_image" or plot.get("series")]


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


def _list_plot_artifacts(
    *,
    job_id: Optional[str] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    inference_id: Optional[int] = None,
    kind: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    return list_legacy_plot_artifacts(
        PLOT_ARTIFACT_DIR,
        job_id=job_id,
        model_id=model_id,
        dataset_id=dataset_id,
        inference_id=inference_id,
        kind=kind,
        limit=limit,
    )


def _resolve_plot_artifact_path(plot_id: str) -> Path:
    return resolve_legacy_plot_path(PLOT_ARTIFACT_DIR, plot_id)


def _build_extraction_manifest(dataset: Any, dataframe: pd.DataFrame, *, preview_row_limit: int = 20) -> dict[str, Any]:
    preview_rows = _json_safe(dataframe.head(preview_row_limit).replace({np.nan: None}).to_dict(orient="records"))
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


def _metric_reference_for_key(key: str) -> dict[str, Any]:
    normalized = str(key or "").strip().lower()
    if "missing" in normalized or "null" in normalized:
        return {
            "unit": "%",
            "reference": "0% is ideal",
            "expected_range": "0-100%",
            "direction": "lower_is_better",
            "explanation": "Raw missingness ratio. Use it to compare how much of the table still needs imputation or cleanup.",
        }
    if "row" in normalized or "column" in normalized or "feature" in normalized:
        return {
            "unit": "count",
            "reference": "dataset shape",
            "expected_range": ">= 0",
            "direction": "context_only",
            "explanation": "Raw count shown for reference. Compare against previous versions of the same dataset rather than an absolute target.",
        }
    if "cardinality" in normalized or "unique" in normalized:
        return {
            "unit": "count",
            "reference": "column uniqueness",
            "expected_range": ">= 0",
            "direction": "context_only",
            "explanation": "Higher values usually mean the column behaves more like an identifier and may need encoding or exclusion.",
        }
    return {
        "unit": "raw",
        "reference": "raw value",
        "expected_range": "context dependent",
        "direction": "context_only",
        "explanation": "This value is kept raw so the user can interpret it against the selected dataset, model, or production watch context.",
    }


def _build_metric_cards(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for key, value in dict(metrics or {}).items():
        reference = _metric_reference_for_key(str(key))
        cards.append(
            {
                "key": str(key),
                "label": str(key).replace("_", " ").title(),
                "value": _json_safe(value),
                **reference,
            }
        )
    return cards


def _coerce_preview_row_limit(value: Any, default: int = 20, maximum: int = 5000) -> int:
    try:
        row_limit = int(value)
    except (TypeError, ValueError):
        row_limit = default
    return max(1, min(row_limit, maximum))


def _feature_sql_fingerprint(sql_query: str) -> str:
    return hashlib.sha256(sql_query.encode("utf-8")).hexdigest()[:12]


def _is_sql_feature_payload(payload: Mapping[str, Any]) -> bool:
    mode = str(payload.get("mode") or payload.get("transformMode") or "").strip().lower()
    return mode == "sql" or bool(str(payload.get("sqlQuery") or payload.get("sql_query") or "").strip())


def _feature_sql_query_from_payload(payload: Mapping[str, Any]) -> str:
    return str(payload.get("sqlQuery") or payload.get("sql_query") or "").strip()


def _sql_statements(sql_query: str) -> list[str]:
    if SQLPARSE_AVAILABLE and sqlparse is not None:
        return [statement.strip().rstrip(";").strip() for statement in sqlparse.split(sql_query) if statement.strip()]
    return [statement.strip() for statement in sql_query.split(";") if statement.strip()]


def _sql_visible_token_values(statement: str) -> list[str]:
    if SQLPARSE_AVAILABLE and sqlparse is not None:
        parsed = sqlparse.parse(statement)
        if not parsed:
            return []
        values: list[str] = []
        for token in parsed[0].flatten():
            if token.is_whitespace:
                continue
            token_type = token.ttype
            if token_type in sqlparse.tokens.Comment:
                continue
            if token_type in sqlparse.tokens.Literal.String:
                continue
            values.append(str(token.value))
        return values
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", statement)


def _first_sql_keyword(statement: str) -> str:
    if SQLPARSE_AVAILABLE and sqlparse is not None:
        parsed = sqlparse.parse(statement)
        if parsed:
            for token in parsed[0].tokens:
                if token.is_whitespace or token.ttype in sqlparse.tokens.Comment:
                    continue
                return token.normalized.upper()
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", statement)
    return match.group(0).upper() if match else ""


def validate_feature_sql_query(sql_query: str) -> str:
    if not sql_query.strip():
        raise ValueError("SQL query is required for SQL feature mode.")

    statements = _sql_statements(sql_query)
    if len(statements) != 1:
        raise ValueError("SQL feature mode accepts exactly one statement.")

    statement = statements[0].strip().rstrip(";").strip()
    first_keyword = _first_sql_keyword(statement)
    if first_keyword not in {"SELECT", "WITH"}:
        raise ValueError("SQL feature mode only accepts SELECT or WITH queries.")

    visible_tokens = _sql_visible_token_values(statement)
    normalized_words = {token.upper() for token in visible_tokens if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)}
    blocked = sorted(SQL_FEATURE_BLOCKED_KEYWORDS.intersection(normalized_words))
    if blocked:
        raise ValueError(f"SQL feature mode does not allow: {', '.join(blocked)}.")

    visible_sql = " ".join(visible_tokens).lower()
    if not re.search(rf"\b{re.escape(SQL_FEATURE_SOURCE_RELATION)}\b", visible_sql):
        raise ValueError(f"SQL query must reference {SQL_FEATURE_SOURCE_RELATION}.")

    return statement


def build_feature_sql_operation_metadata(sql_query: str, *, has_secondary: bool = False) -> dict[str, Any]:
    statement = validate_feature_sql_query(sql_query)
    metadata = {
        "mode": "sql",
        "source_relation": SQL_FEATURE_SOURCE_RELATION,
        "sql_query": statement,
        "sql_hash": _feature_sql_fingerprint(statement),
    }
    if has_secondary:
        metadata["secondary_relation"] = SQL_FEATURE_SECONDARY_RELATION
    return metadata


async def execute_feature_sql_query(
    db: AsyncSession,
    dataframe: pd.DataFrame,
    sql_query: str,
    *,
    preview_rows: Any = None,
    secondary_dataframe: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    statement = validate_feature_sql_query(sql_query)
    bind = getattr(db, "bind", None)
    if bind is None:
        raise RuntimeError("PostgreSQL connection is required for SQL feature mode.")

    outer_query = f"SELECT * FROM ({statement}) AS {SQL_FEATURE_RESULT_ALIAS}"
    row_limit = _coerce_preview_row_limit(preview_rows) if preview_rows not in (None, "", 0, "0") else None
    if row_limit is not None:
        outer_query = f"{outer_query}\nLIMIT {row_limit}"

    def _run_sql(sync_conn: Any) -> pd.DataFrame:
        try:
            sync_conn.exec_driver_sql("SET LOCAL search_path TO pg_temp")
            dataframe.to_sql(
                SQL_FEATURE_SOURCE_RELATION,
                con=sync_conn,
                schema="pg_temp",
                if_exists="replace",
                index=False,
            )
            if secondary_dataframe is not None:
                secondary_dataframe.to_sql(
                    SQL_FEATURE_SECONDARY_RELATION,
                    con=sync_conn,
                    schema="pg_temp",
                    if_exists="replace",
                    index=False,
                )
            return pd.read_sql_query(text(outer_query), sync_conn)
        finally:
            try:
                sync_conn.exec_driver_sql(f"DROP TABLE IF EXISTS pg_temp.{SQL_FEATURE_SOURCE_RELATION}")
                sync_conn.exec_driver_sql(f"DROP TABLE IF EXISTS pg_temp.{SQL_FEATURE_SECONDARY_RELATION}")
            except Exception:
                LOGGER.debug("Unable to drop temporary SQL feature source table.", exc_info=True)

    async with bind.begin() as conn:
        result_frame = await conn.run_sync(_run_sql)

    summary = {
        "shape_before": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "shape_after": [int(result_frame.shape[0]), int(result_frame.shape[1])],
        "steps": [
            {
                "operation": "sql_query",
                "source_relation": SQL_FEATURE_SOURCE_RELATION,
                "secondary_relation": SQL_FEATURE_SECONDARY_RELATION if secondary_dataframe is not None else None,
                "sql_hash": _feature_sql_fingerprint(statement),
                "preview_rows": row_limit,
            }
        ],
        "preview_rows": _json_safe(result_frame.head(row_limit or 20).replace({np.nan: None}).to_dict(orient="records")),
        "column_explorer": build_column_explorer(result_frame),
    }
    return result_frame, summary


def _coerce_feature_operations_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    operations = dict(payload.get("operations") or {})
    if operations:
        return operations

    transformed: dict[str, Any] = {
        "rename_columns": {},
        "cast_types": {},
        "fill_missing_values": {},
        "drop_columns": [],
        "filters": [],
        "sort_by": {},
    }
    transforms = payload.get("transforms")
    if not isinstance(transforms, list):
        limit_rows = payload.get("limitRows")
        if limit_rows not in (None, ""):
            transformed["limit_rows"] = int(limit_rows)
        return transformed

    for transform in transforms:
        if not isinstance(transform, Mapping):
            continue
        transform_type = str(transform.get("type") or "").strip().lower()
        if transform_type == "rename":
            source = str(transform.get("column") or "").strip()
            target = str(transform.get("value") or "").strip()
            if source and target:
                transformed.setdefault("rename_columns", {})[source] = target
        elif transform_type == "cast":
            source = str(transform.get("column") or "").strip()
            target = str(transform.get("value") or "").strip()
            if source and target:
                transformed.setdefault("cast_types", {})[source] = target
        elif transform_type == "fill":
            source = str(transform.get("column") or "").strip()
            if source:
                transformed.setdefault("fill_missing_values", {})[source] = transform.get("value")
        elif transform_type == "drop":
            source = str(transform.get("column") or "").strip()
            if source:
                transformed.setdefault("drop_columns", []).append(source)
        elif transform_type == "filter":
            source = str(transform.get("column") or "").strip()
            operator = str(transform.get("operator") or "==").strip()
            if source:
                transformed.setdefault("filters", []).append(
                    {"column": source, "operator": operator, "value": transform.get("value")}
                )
        elif transform_type == "sort":
            source = str(transform.get("column") or "").strip()
            if source:
                transformed["sort_by"] = {
                    "column": source,
                    "ascending": bool(transform.get("ascending", True)),
                }
        elif transform_type == "limit":
            transformed["limit_rows"] = int(transform.get("value") or 0)
        elif transform_type == "math":
            source = str(transform.get("column") or "").strip()
            operator = str(transform.get("operator") or "").strip().lower()
            if source and operator:
                transformed.setdefault("math_operations", []).append(
                    {
                        "column": source,
                        "operator": operator,
                        "value": transform.get("value"),
                        "value_mode": str(transform.get("value_mode") or transform.get("operand_mode") or "").strip().lower(),
                        "right_column": str(transform.get("right_column") or transform.get("secondary_column") or "").strip(),
                        "target": str(transform.get("target") or source).strip() or source,
                    }
                )
        elif transform_type == "normalize":
            columns = transform.get("columns")
            if columns is None and transform.get("column"):
                columns = [transform.get("column")]
            transformed.setdefault("normalize", []).append(
                {
                    "mode": str(transform.get("mode") or "").strip().lower(),
                    "columns": columns,
                    "min": transform.get("min", 0),
                    "max": transform.get("max", 1),
                    "target_suffix": transform.get("target_suffix") or transform.get("suffix") or "",
                }
            )
        elif transform_type in {"date_range", "datetime_range", "index_range"}:
            source = str(transform.get("column") or "").strip()
            if source:
                transformed.setdefault("date_range_filters", []).append(
                    {
                        "column": source,
                        "start": transform.get("start"),
                        "end": transform.get("end"),
                    }
                )

    return transformed


def _build_transform_suggestions(column_explorer: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for column in column_explorer[:24]:
        column_name = str(column.get("name") or "")
        missing_ratio = float(column.get("null_ratio") or 0.0)
        unique_ratio = float(column.get("unique_ratio") or 0.0)
        family = str(column.get("family") or "")
        if missing_ratio >= 0.2:
            suggestions.append(
                {
                    "type": "fill",
                    "column": column_name,
                    "reason": f"{column_name} has {round(missing_ratio * 100, 2)}% missing values.",
                }
            )
        if column.get("is_datetime_candidate"):
            suggestions.append(
                {
                    "type": "cast",
                    "column": column_name,
                    "value": "datetime64[ns]",
                    "reason": f"{column_name} looks like a datetime candidate.",
                }
            )
        if family == "categorical" and unique_ratio >= 0.9:
            suggestions.append(
                {
                    "type": "drop",
                    "column": column_name,
                    "reason": f"{column_name} is highly unique and may behave like an identifier.",
                }
            )
    return suggestions[:16]


def _build_feature_workspace_payload(
    dataset: Any,
    dataframe: pd.DataFrame,
    *,
    preview: Optional[Mapping[str, Any]] = None,
    operations: Optional[Mapping[str, Any]] = None,
    preview_row_limit: int = 20,
) -> dict[str, Any]:
    manifest = _build_extraction_manifest(dataset, dataframe, preview_row_limit=preview_row_limit)
    analysis = dict(manifest.get("analysis", {}) or {})
    column_explorer = list(manifest.get("columns", []) or [])
    profile = dict(analysis.get("profile", {}) or {})
    metric_cards = _build_metric_cards(
        {
            "row_count": manifest.get("shape", [0, 0])[0],
            "column_count": manifest.get("shape", [0, 0])[1],
            "missing_ratio": profile.get("missing_ratio", 0.0),
            "numeric_columns": len(profile.get("numeric_columns", []) or []),
            "categorical_columns": len(profile.get("categorical_columns", []) or []),
            "datetime_candidates": len(profile.get("datetime_candidates", []) or []),
        }
    )
    return {
        "dataset_id": getattr(dataset, "id", None),
        "dataset_name": getattr(dataset, "name", None),
        "manifest": manifest,
        "preview": _json_safe(preview or {}),
        "preview_rows": list(manifest.get("preview_rows", []) or []),
        "column_explorer": column_explorer,
        "metric_cards": metric_cards,
        "plots": _build_dataset_plots(analysis, dataframe=dataframe),
        "summary_cards": metric_cards,
        "transform_suggestions": _build_transform_suggestions(column_explorer),
        "operations": _json_safe(operations or {}),
    }


def _resolve_training_feature_selection(
    dataframe: pd.DataFrame,
    model_record: Any,
    parameters: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    requested_input_features = _coerce_feature_list(parameters.get("input_features"))
    requested_output_features = _coerce_feature_list(parameters.get("output_features"))
    stored_input_features = _coerce_feature_list(getattr(model_record, "input_features", None))
    stored_output_features = _coerce_feature_list(getattr(model_record, "output_features", None))
    unsupervised = _is_unsupervised_sklearn_request(parameters)

    output_features = requested_output_features or stored_output_features
    if not output_features and not unsupervised:
        output_features = [_default_output_feature(dataframe)]
    output_feature_set = {str(feature) for feature in output_features}
    input_features = requested_input_features or stored_input_features
    if not input_features:
        input_features = [column for column in dataframe.columns if str(column) not in output_feature_set]
    else:
        input_features = [column for column in input_features if str(column) not in output_feature_set]
    return input_features, output_features


def _build_training_preflight(
    dataset_record: Any,
    model_record: Any,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    dataframe = _load_dataset_frame_from_record(dataset_record)
    available_columns = [str(column) for column in dataframe.columns]
    input_features, output_features = _resolve_training_feature_selection(dataframe, model_record, parameters)
    output_feature = output_features[0] if output_features else None
    required_columns = [*input_features, *output_features]
    missing_columns = [column for column in required_columns if column not in available_columns]

    recommended_output = None if _is_unsupervised_sklearn_request(parameters) else (_default_output_feature(dataframe) if available_columns else None)
    recommended_inputs = [column for column in available_columns if column != recommended_output]
    return {
        "ok": not missing_columns,
        "dataset_id": getattr(dataset_record, "id", None),
        "dataset_name": getattr(dataset_record, "name", None),
        "model_id": getattr(model_record, "id", None),
        "model_name": getattr(model_record, "name", None),
        "input_features": input_features,
        "output_feature": output_feature,
        "output_features": output_features,
        "required_columns": required_columns,
        "available_columns": available_columns,
        "missing_columns": missing_columns,
        "recommended_pairing": {
            "dataset_id": getattr(dataset_record, "id", None),
            "dataset_name": getattr(dataset_record, "name", None),
            "model_id": getattr(model_record, "id", None),
            "model_name": getattr(model_record, "name", None),
            "input_features": recommended_inputs,
            "output_feature": recommended_output,
        },
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
        plots = _build_dataset_plots(summary, dataframe=dataframe)
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


def _simulation_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _simulation_value(value: Any) -> Any:
    try:
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return round(numeric_value, 6)
    except (TypeError, ValueError):
        pass
    return _json_safe(value)


def _simulation_row_payload(row: pd.Series | Mapping[str, Any]) -> dict[str, Any]:
    items = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {str(key): _simulation_value(value) for key, value in items.items()}


def _simulation_feature_columns(prepared: PreparedDataset) -> list[str]:
    return [str(column) for column in prepared.feature_frame.columns]


def _validate_simulation_features(
    values: Mapping[str, Any] | list[str] | tuple[str, ...],
    feature_columns: list[str],
    *,
    context: str,
) -> None:
    candidate_names = values.keys() if isinstance(values, Mapping) else values
    available = {str(column) for column in feature_columns}
    unknown = sorted({str(name) for name in candidate_names if str(name) not in available})
    if unknown:
        preview = ", ".join(feature_columns[:12])
        suffix = f" Available features include: {preview}." if preview else ""
        raise ValueError(f"Unknown simulation feature(s) in {context}: {', '.join(unknown)}.{suffix}")


def _feature_is_numeric(series: pd.Series, value: Any) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _simulation_feature_stats(feature_frame: pd.DataFrame, baseline: pd.Series) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for column in feature_frame.columns:
        series = feature_frame[column]
        baseline_value = baseline.get(column)
        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "family": "numeric" if _feature_is_numeric(series, baseline_value) else "categorical",
            "baseline": _simulation_value(baseline_value),
            "sample_values": [_simulation_value(value) for value in series.head(5).tolist()],
        }
        if item["family"] == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            item.update(
                {
                    "min": _simulation_value(numeric_series.min()) if numeric_series.notna().any() else None,
                    "max": _simulation_value(numeric_series.max()) if numeric_series.notna().any() else None,
                    "mean": _simulation_value(numeric_series.mean()) if numeric_series.notna().any() else None,
                    "std": _simulation_value(numeric_series.std()) if numeric_series.notna().any() else None,
                }
            )
        else:
            mode = series.mode(dropna=True)
            item["mode"] = None if mode.empty else _simulation_value(mode.iloc[0])
            try:
                item["unique_values"] = int(series.nunique(dropna=True))
            except Exception:
                item["unique_values"] = None
        stats.append(item)
    return stats


def _baseline_from_source(feature_frame: pd.DataFrame, baseline_source: str) -> pd.Series:
    if feature_frame.empty:
        raise ValueError("The selected dataset does not contain rows for simulation.")

    source = str(baseline_source or "last").strip().lower()
    if source in {"first", "head"}:
        return feature_frame.head(1).iloc[0].copy()
    if source in {"mean", "average", "median"}:
        values: dict[str, Any] = {}
        for column in feature_frame.columns:
            series = feature_frame[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().any():
                    values[column] = float(numeric.median() if source == "median" else numeric.mean())
                    continue
            mode = series.mode(dropna=True)
            values[column] = feature_frame.tail(1).iloc[0][column] if mode.empty else mode.iloc[0]
        return pd.Series(values, index=feature_frame.columns)
    return feature_frame.tail(1).iloc[0].copy()


def _build_simulation_baseline(
    prepared: PreparedDataset,
    *,
    baseline_source: str = "last",
    baseline_row: Optional[Mapping[str, Any]] = None,
) -> pd.Series:
    feature_columns = _simulation_feature_columns(prepared)
    baseline = _baseline_from_source(prepared.feature_frame, baseline_source)
    baseline_overrides = _simulation_mapping(baseline_row)
    if baseline_overrides:
        _validate_simulation_features(baseline_overrides, feature_columns, context="baselineRow")
        for feature_name, value in baseline_overrides.items():
            baseline[str(feature_name)] = value
    return baseline


def _coerce_simulation_int(value: Any, default: int, *, minimum: int = 1, maximum: int = SIMULATION_MAX_STEPS) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _coerce_simulation_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        return default
    return parsed


def _scenario_feature_overrides(raw_scenario: Mapping[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    overrides = _simulation_mapping(raw_scenario.get("overrides"))
    if not overrides:
        overrides = _simulation_mapping(raw_scenario.get("scenario"))
    if not overrides:
        controls = {
            "name",
            "label",
            "steps",
            "amplitude",
            "trend",
            "stepPlan",
            "baselineRow",
            "baselineSource",
            "sensitivity",
        }
        overrides = {
            str(key): value
            for key, value in raw_scenario.items()
            if str(key) in set(feature_columns) and str(key) not in controls
        }
    return overrides


def _normalize_simulation_scenarios(payload: Mapping[str, Any], feature_columns: list[str]) -> list[dict[str, Any]]:
    raw_scenarios = payload.get("scenarios")
    scenario_items: list[Mapping[str, Any]] = []
    if isinstance(raw_scenarios, list):
        for item in raw_scenarios[:SIMULATION_MAX_SCENARIOS]:
            if isinstance(item, Mapping):
                scenario_items.append(item)
    elif isinstance(raw_scenarios, Mapping):
        for name, overrides in list(raw_scenarios.items())[:SIMULATION_MAX_SCENARIOS]:
            scenario_items.append({"name": name, "overrides": overrides})

    if not scenario_items:
        scenario_items = [
            {
                "name": payload.get("scenarioName") or payload.get("name") or "Scenario",
                "overrides": payload.get("scenario", {}),
                "steps": payload.get("steps"),
                "amplitude": payload.get("amplitude"),
                "trend": payload.get("trend"),
                "stepPlan": payload.get("stepPlan"),
            }
        ]

    max_steps_per_scenario = max(1, min(SIMULATION_MAX_STEPS, SIMULATION_MAX_ROWS // max(len(scenario_items), 1)))
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(scenario_items, start=1):
        overrides = _scenario_feature_overrides(raw, feature_columns)
        _validate_simulation_features(overrides, feature_columns, context=f"scenario {index}")
        step_plan = _simulation_mapping(payload.get("stepPlan"))
        step_plan.update(_simulation_mapping(raw.get("stepPlan")))
        delta_config = _simulation_mapping(step_plan.get("deltas"))
        if delta_config:
            _validate_simulation_features(delta_config, feature_columns, context=f"scenario {index} stepPlan.deltas")
        planned_features = _coerce_feature_list(step_plan.get("features"))
        if planned_features:
            _validate_simulation_features(planned_features, feature_columns, context=f"scenario {index} stepPlan.features")

        normalized.append(
            {
                "name": str(raw.get("name") or raw.get("label") or f"Scenario {index}"),
                "overrides": overrides,
                "steps": _coerce_simulation_int(
                    raw.get("steps", payload.get("steps", SIMULATION_DEFAULT_STEPS)),
                    SIMULATION_DEFAULT_STEPS,
                    maximum=max_steps_per_scenario,
                ),
                "amplitude": _coerce_simulation_float(raw.get("amplitude", payload.get("amplitude", 0.05)), 0.05),
                "trend": str(raw.get("trend", payload.get("trend", "up")) or "up").strip().lower(),
                "step_plan": step_plan,
            }
        )
    return normalized


def _apply_direct_overrides(row: pd.Series, overrides: Mapping[str, Any]) -> pd.Series:
    updated = row.copy()
    for feature_name, value in overrides.items():
        updated[str(feature_name)] = value
    return updated


def _direction_value(trend: str) -> float:
    return -1.0 if str(trend or "").lower() in {"down", "decrease", "negative", "lower"} else 1.0


def _delta_config_for_feature(
    feature_name: str,
    *,
    default_amplitude: float,
    default_trend: str,
    step_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    delta_map = _simulation_mapping(step_plan.get("deltas"))
    if feature_name in delta_map:
        raw_delta = delta_map[feature_name]
        if isinstance(raw_delta, Mapping):
            return {
                "value": _coerce_simulation_float(raw_delta.get("value", default_amplitude), default_amplitude),
                "mode": str(raw_delta.get("mode", "relative") or "relative").lower(),
                "trend": str(raw_delta.get("trend", default_trend) or default_trend).lower(),
            }
        return {
            "value": _coerce_simulation_float(raw_delta, default_amplitude),
            "mode": "relative",
            "trend": default_trend,
        }

    planned_features = _coerce_feature_list(step_plan.get("features"))
    if planned_features and feature_name not in planned_features:
        return None
    return {"value": default_amplitude, "mode": "relative", "trend": default_trend}


def _apply_step_plan(
    scenario_base: pd.Series,
    *,
    step_index: int,
    steps: int,
    amplitude: float,
    trend: str,
    step_plan: Mapping[str, Any],
) -> pd.Series:
    fraction = step_index / max(steps, 1)
    row = scenario_base.copy()
    for feature_name in row.index:
        config = _delta_config_for_feature(
            str(feature_name),
            default_amplitude=amplitude,
            default_trend=trend,
            step_plan=step_plan,
        )
        if config is None:
            continue
        try:
            numeric_value = float(row[feature_name])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_value):
            continue

        value = _coerce_simulation_float(config.get("value"), amplitude)
        direction = _direction_value(str(config.get("trend") or trend))
        mode = str(config.get("mode") or "relative").lower()
        if mode in {"absolute", "add", "linear"}:
            row[feature_name] = numeric_value + direction * value * fraction
        elif mode in {"target", "set"}:
            row[feature_name] = numeric_value + (value - numeric_value) * fraction
        else:
            row[feature_name] = numeric_value * (1.0 + direction * value * fraction)
    return row


def _render_simulation_prediction(prediction: Any, prepared: PreparedDataset) -> Any:
    if prepared.task_type == "classification":
        reverse_mapping = {value: key for key, value in prepared.label_mapping.items()}
        try:
            return reverse_mapping.get(int(prediction), int(prediction))
        except (TypeError, ValueError):
            return _simulation_value(prediction)
    return _simulation_value(prediction)


def _prediction_delta(first_prediction: Any, final_prediction: Any) -> Any:
    try:
        return round(float(final_prediction) - float(first_prediction), 6)
    except (TypeError, ValueError):
        return None


def _predict_probabilities_with_result(
    result: TrainingResult,
    features: pd.DataFrame,
    prepared: PreparedDataset,
) -> list[dict[str, Any]] | None:
    if getattr(result, "framework", None) != "sklearn" or not hasattr(getattr(result, "model", None), "predict_proba"):
        return None
    try:
        probabilities = np.asarray(result.model.predict_proba(features))
    except Exception:
        return None
    classes = list(getattr(result.model, "classes_", range(probabilities.shape[-1] if probabilities.ndim > 1 else 0)))
    reverse_mapping = {value: key for key, value in prepared.label_mapping.items()}
    rendered_rows: list[dict[str, Any]] = []
    for row in probabilities:
        rendered: dict[str, Any] = {}
        for index, probability in enumerate(np.asarray(row).reshape(-1)):
            raw_class = classes[index] if index < len(classes) else index
            try:
                label = reverse_mapping.get(int(raw_class), raw_class)
            except (TypeError, ValueError):
                label = raw_class
            rendered[str(label)] = _simulation_value(probability)
        rendered_rows.append(rendered)
    return rendered_rows


def _changed_features(baseline: pd.Series, scenario_base: pd.Series) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for feature_name in scenario_base.index:
        baseline_value = baseline.get(feature_name)
        scenario_value = scenario_base.get(feature_name)
        if _simulation_value(baseline_value) == _simulation_value(scenario_value):
            continue
        changes.append(
            {
                "feature": str(feature_name),
                "baseline": _simulation_value(baseline_value),
                "scenario": _simulation_value(scenario_value),
                "delta": _prediction_delta(baseline_value, scenario_value),
            }
        )
    return changes


def _plot_has_renderable_content(plot: Mapping[str, Any]) -> bool:
    if plot.get("kind") == "legacy_image":
        return True
    if plot.get("figure"):
        return True
    if plot.get("series"):
        return True
    if plot.get("rows"):
        return True
    return False


def _changed_feature_delta_series(changes: list[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    series: list[tuple[str, Any]] = []
    for change in changes:
        delta = change.get("delta")
        try:
            numeric_delta = abs(float(delta))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_delta):
            continue
        series.append((str(change.get("feature") or "feature"), numeric_delta))
    return series


def _simulation_step_table_plot(
    title: str,
    rows: list[Mapping[str, Any]],
    *,
    description: Optional[str] = None,
    max_rows: int = 40,
) -> dict[str, Any]:
    return _build_table_plot(
        title,
        rows,
        description=description or "Tabular simulation results for plot decks and saved panels.",
        source={"domain": "production_simulation"},
        max_rows=max_rows,
    )


def _build_sensitivity_summary(
    payload: Mapping[str, Any],
    *,
    baseline: pd.Series,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    output_feature: str,
) -> dict[str, Any]:
    settings = _simulation_mapping(payload.get("sensitivity"))
    if not settings or settings.get("enabled") is False:
        return {"enabled": False, "features": [], "plots": []}

    feature_columns = _simulation_feature_columns(prepared)
    requested_features = _coerce_feature_list(settings.get("features"))
    if requested_features:
        _validate_simulation_features(requested_features, feature_columns, context="sensitivity.features")
    else:
        requested_features = [
            column
            for column in feature_columns
            if _feature_is_numeric(prepared.feature_frame[column], baseline.get(column))
        ]
    requested_features = requested_features[:SIMULATION_MAX_SENSITIVITY_FEATURES]
    amplitude = abs(_coerce_simulation_float(settings.get("amplitude", payload.get("amplitude", 0.05)), 0.05))
    rows: list[pd.Series] = []
    row_labels: list[tuple[str, str]] = []
    for feature_name in requested_features:
        try:
            base_value = float(baseline[feature_name])
        except (TypeError, ValueError):
            continue
        lower = baseline.copy()
        upper = baseline.copy()
        lower[feature_name] = base_value * (1.0 - amplitude)
        upper[feature_name] = base_value * (1.0 + amplitude)
        rows.extend([lower, upper])
        row_labels.extend([(feature_name, "lower"), (feature_name, "upper")])

    if not rows:
        return {"enabled": True, "features": [], "plots": []}

    frame = pd.DataFrame(rows, columns=feature_columns)
    predictions = np.asarray(_predict_with_result(runtime_result, frame, prepared.task_type)).reshape(-1)
    rendered_predictions = [_render_simulation_prediction(prediction, prepared) for prediction in predictions]
    grouped: dict[str, dict[str, Any]] = {}
    for (feature_name, side), prediction in zip(row_labels, rendered_predictions):
        grouped.setdefault(feature_name, {"feature": feature_name, "output_feature": output_feature})[side] = prediction

    feature_rows: list[dict[str, Any]] = []
    for feature_name in requested_features:
        item = grouped.get(feature_name)
        if not item:
            continue
        item["impact"] = _prediction_delta(item.get("lower"), item.get("upper"))
        try:
            item["absolute_impact"] = abs(float(item["impact"]))
        except (TypeError, ValueError, KeyError):
            item["absolute_impact"] = None
        feature_rows.append(item)
    feature_rows.sort(key=lambda item: float(item.get("absolute_impact") or 0.0), reverse=True)
    impact_series = [
        (item["feature"], item["absolute_impact"])
        for item in feature_rows
        if isinstance(item.get("absolute_impact"), (int, float))
    ]
    return {
        "enabled": True,
        "amplitude": amplitude,
        "features": feature_rows,
        "plots": [
            _build_bar_plot(
                "Sensitivity Impact",
                impact_series,
                description="Absolute prediction movement when one feature is nudged around the baseline.",
            )
        ]
        if impact_series
        else [],
    }


def _normalize_simulation_family(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "supervised_regression": "regression",
        "tabular_regression": "regression",
        "regressor": "regression",
        "supervised_classification": "classification",
        "tabular_classification": "classification",
        "classifier": "classification",
        "clustering": "unsupervised",
        "cluster": "unsupervised",
        "anomaly": "unsupervised",
        "anomaly_detection": "unsupervised",
        "recommender": "recommendation",
        "recommendations": "recommendation",
        "ranking": "recommendation",
        "geo": "geospatial",
        "spatial": "geospatial",
        "llm": "llm_slm",
        "slm": "llm_slm",
        "assistant": "llm_slm",
        "assistant_model": "llm_slm",
        "text_generation": "llm_slm",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in SIMULATION_FAMILY_LABELS else None


def _simulation_profile_from_record(inference_record: Any | None, model_record: Any | None) -> dict[str, Any]:
    candidates: list[Any] = []
    if inference_record is not None:
        params = dict(getattr(inference_record, "inference_params", {}) or {})
        candidates.extend([params.get("simulation_profile"), params.get("simulationProfile")])
    if model_record is not None:
        params = dict(getattr(model_record, "parameters", {}) or {})
        candidates.extend([params.get("simulation_profile"), params.get("simulationProfile")])
    for candidate in candidates:
        profile = _simulation_mapping(candidate)
        if profile:
            return profile
    return {}


def _simulation_feature_names_from_prepared(prepared: PreparedDataset | None, model_record: Any | None = None) -> list[str]:
    if prepared is not None:
        return _simulation_feature_columns(prepared)
    return _coerce_feature_list(getattr(model_record, "input_features", None))


def _simulation_geo_features(feature_columns: list[str]) -> dict[str, str | None]:
    normalized = {str(column).strip().lower(): str(column) for column in feature_columns}
    latitude = None
    longitude = None
    for name in SIMULATION_GEO_LATITUDE_NAMES:
        if name in normalized:
            latitude = normalized[name]
            break
    for name in SIMULATION_GEO_LONGITUDE_NAMES:
        if name in normalized:
            longitude = normalized[name]
            break
    return {"latitude": latitude, "longitude": longitude}


def _simulation_capability_flags(
    runtime_result: TrainingResult | Any | None,
    *,
    feature_columns: Optional[list[str]] = None,
    runtime_error: BaseException | None = None,
) -> dict[str, Any]:
    model = getattr(runtime_result, "model", None) if runtime_result is not None else None
    parameters = dict(getattr(runtime_result, "parameters", {}) or {}) if runtime_result is not None else {}
    manifest = parameters.get("artifact_manifest", {}) if isinstance(parameters.get("artifact_manifest"), Mapping) else {}
    manifest_capabilities = dict((manifest or {}).get("runtime_capabilities", {}) or {})
    framework = str(getattr(runtime_result, "framework", "") or parameters.get("framework", "") or "").strip().lower()
    method_names = [
        "predict",
        "predict_proba",
        "transform",
        "decision_function",
        "score_samples",
        "recommend",
        "predict_for_user",
        "score_items",
        "generate",
        "generate_text",
        "chat",
    ]
    methods = {name: callable(getattr(model, name, None)) for name in method_names}
    geo_features = _simulation_geo_features(feature_columns or [])
    flags = {
        "runtime_ready": runtime_error is None,
        "runtime_error": str(runtime_error) if runtime_error else None,
        "framework": framework or None,
        "predict": bool(manifest_capabilities.get("predict") or methods["predict"] or framework in {"sklearn", "pytorch"}),
        "predict_proba": bool(manifest_capabilities.get("predict_proba") or methods["predict_proba"]),
        "transform": bool(manifest_capabilities.get("transform") or methods["transform"]),
        "decision_function": bool(manifest_capabilities.get("decision_function") or methods["decision_function"]),
        "score_samples": bool(manifest_capabilities.get("score_samples") or methods["score_samples"]),
        "recommend": bool(manifest_capabilities.get("recommend") or methods["recommend"]),
        "score_items": bool(manifest_capabilities.get("score_items") or methods["score_items"] or methods["predict_for_user"]),
        "llm_generation": bool(
            manifest_capabilities.get("chat")
            or manifest_capabilities.get("generate")
            or manifest_capabilities.get("requires_assistant_server")
            or methods["generate"]
            or methods["generate_text"]
            or methods["chat"]
        ),
        "geospatial_features": bool(geo_features.get("latitude") and geo_features.get("longitude")),
        "latitude_feature": geo_features.get("latitude"),
        "longitude_feature": geo_features.get("longitude"),
        "manifest": manifest_capabilities,
    }
    return flags


def _simulation_search_blob(
    *,
    model_record: Any | None,
    prepared: PreparedDataset | None,
    output_features: list[str],
    feature_columns: list[str],
) -> str:
    parameters = dict(getattr(model_record, "parameters", {}) or {}) if model_record is not None else {}
    manifest = parameters.get("artifact_manifest", {}) if isinstance(parameters.get("artifact_manifest"), Mapping) else {}
    values = [
        getattr(model_record, "model_type", None),
        getattr(model_record, "object_type", None),
        getattr(model_record, "name", None),
        parameters.get("task_type"),
        parameters.get("model_family"),
        parameters.get("simulation_family"),
        parameters.get("algorithm"),
        parameters.get("estimator_class"),
        parameters.get("framework"),
        manifest.get("artifact_format") if isinstance(manifest, Mapping) else None,
        manifest.get("loader") if isinstance(manifest, Mapping) else None,
        getattr(prepared, "task_type", None),
        *output_features,
        *feature_columns,
    ]
    return " ".join(str(value).lower() for value in values if value not in (None, ""))


def _detect_simulation_family(
    *,
    model_record: Any | None,
    inference_record: Any | None,
    prepared: PreparedDataset | None,
    runtime_result: TrainingResult | Any | None,
    payload: Optional[Mapping[str, Any]] = None,
) -> str:
    saved_profile = _simulation_profile_from_record(inference_record, model_record)
    payload_profile = _simulation_mapping((payload or {}).get("simulationProfile") or (payload or {}).get("simulation_profile"))
    explicit = (
        _normalize_simulation_family((payload or {}).get("family"))
        or _normalize_simulation_family(payload_profile.get("family"))
        or _normalize_simulation_family(saved_profile.get("family"))
    )
    if explicit:
        return explicit

    output_features = (
        _coerce_feature_list(getattr(inference_record, "output_features", None))
        or _coerce_feature_list(getattr(model_record, "output_features", None))
        or ([prepared.output_feature] if prepared is not None and getattr(prepared, "output_feature", None) else [])
    )
    feature_columns = _simulation_feature_names_from_prepared(prepared, model_record)
    blob = _simulation_search_blob(
        model_record=model_record,
        prepared=prepared,
        output_features=output_features,
        feature_columns=feature_columns,
    )
    if any(token in blob for token in ("assistant", "llm", "slm", "language_model", "text_generation", "chat")):
        return "llm_slm"
    if any(token in blob for token in ("recommend", "recommender", "ranking", "collaborative", "lightfm", "implicit")):
        return "recommendation"
    if any(token in blob for token in ("unsupervised", "cluster", "kmeans", "dbscan", "anomaly", "isolationforest", "pca", "embedding")):
        return "unsupervised"
    if any(token in blob for token in ("geospatial", "geo_", "spatial", "latitude", "longitude")) and _simulation_capability_flags(
        runtime_result,
        feature_columns=feature_columns,
    ).get("geospatial_features"):
        return "geospatial"
    if prepared is not None and getattr(prepared, "task_type", None) == "classification":
        return "classification"
    if prepared is not None and getattr(prepared, "task_type", None) == "regression":
        return "regression"
    return "unsupported"


def _simulation_disabled_reasons(family: str, capabilities: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not capabilities.get("runtime_ready", True):
        reasons.append(str(capabilities.get("runtime_error") or "Runtime is not ready."))
    if family in {"regression", "classification"} and not capabilities.get("predict"):
        reasons.append("This mode requires a runtime predict path.")
    elif family == "unsupervised" and not any(
        capabilities.get(name) for name in ("predict", "transform", "decision_function", "score_samples")
    ):
        reasons.append("This mode requires predict, transform, decision_function, or score_samples.")
    elif family == "recommendation" and not any(capabilities.get(name) for name in ("recommend", "score_items")):
        reasons.append("This mode requires a recommend or item-scoring runtime method.")
    elif family == "geospatial":
        if not capabilities.get("geospatial_features"):
            reasons.append("Latitude and longitude feature columns are required.")
        if not capabilities.get("predict"):
            reasons.append("Geospatial sweeps require a runtime predict path.")
    elif family == "llm_slm" and not capabilities.get("llm_generation"):
        reasons.append("Prompt simulation requires an assistant/LLM generation runtime.")
    elif family == "unsupported":
        reasons.append("No simulation adapter matched this model type or output.")
    return reasons


def _simulation_controls_schema(
    family: str,
    *,
    feature_stats: Optional[list[dict[str, Any]]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    feature_stats = feature_stats or []
    capabilities = capabilities or {}
    numeric_features = [item["name"] for item in feature_stats if item.get("family") == "numeric"]
    categorical_features = [item["name"] for item in feature_stats if item.get("family") != "numeric"]
    family_fields: dict[str, list[dict[str, Any]]] = {
        "regression": [
            {"name": "focusFeature", "label": "Focus Feature", "type": "select", "options": numeric_features[:12]},
            {"name": "explainDeltas", "label": "Explain Deltas", "type": "checkbox", "default": True},
        ],
        "classification": [
            {"name": "threshold", "label": "Decision Threshold", "type": "number", "default": 0.5, "min": 0, "max": 1, "step": 0.01},
            {"name": "topK", "label": "Top Classes", "type": "number", "default": 3, "min": 1, "max": 10, "step": 1},
        ],
        "unsupervised": [
            {"name": "scoreMethod", "label": "Score Method", "type": "select", "options": ["auto", "predict", "decision_function", "score_samples", "transform"]},
            {"name": "includeTransform", "label": "Include Embedding", "type": "checkbox", "default": True},
        ],
        "recommendation": [
            {"name": "userId", "label": "User ID", "type": "text", "default": ""},
            {"name": "itemId", "label": "Item ID", "type": "text", "default": ""},
            {"name": "topN", "label": "Top N", "type": "number", "default": 10, "min": 1, "max": 50, "step": 1},
            {"name": "excludeSeen", "label": "Exclude Seen", "type": "checkbox", "default": True},
        ],
        "geospatial": [
            {"name": "latitudeFeature", "label": "Latitude Feature", "type": "select", "options": [capabilities.get("latitude_feature")] + numeric_features},
            {"name": "longitudeFeature", "label": "Longitude Feature", "type": "select", "options": [capabilities.get("longitude_feature")] + numeric_features},
            {"name": "radius", "label": "Radius", "type": "number", "default": 0.01, "min": 0, "step": 0.001},
            {"name": "gridSize", "label": "Grid Size", "type": "number", "default": 5, "min": 1, "max": 25, "step": 1},
        ],
        "llm_slm": [
            {"name": "system", "label": "System Context", "type": "textarea", "default": ""},
            {"name": "prompt", "label": "Prompt", "type": "textarea", "default": ""},
            {"name": "temperature", "label": "Temperature", "type": "number", "default": 0.2, "min": 0, "max": 2, "step": 0.1},
            {"name": "maxTokens", "label": "Max Tokens", "type": "number", "default": 512, "min": 1, "max": 8192, "step": 1},
        ],
        "unsupported": [
            {"name": "notes", "label": "Readiness Notes", "type": "textarea", "default": ""},
        ],
    }
    fields = family_fields.get(family, family_fields["unsupported"])
    for field in fields:
        if isinstance(field.get("options"), list):
            deduped = []
            for option in field["options"]:
                if option in (None, "") or option in deduped:
                    continue
                deduped.append(option)
            field["options"] = deduped
    return {
        "version": 1,
        "family": family,
        "sections": [
            {"id": "baseline", "label": "Baseline", "controls": ["baselineSource", "featureOverrides"]},
            {"id": "scenario", "label": "Scenario", "controls": ["scenarioName", "steps", "amplitude", "trend"]},
            {"id": "family", "label": SIMULATION_FAMILY_LABELS.get(family, "Model"), "fields": fields},
        ],
        "feature_groups": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
    }


def _build_simulation_profile(
    *,
    model_record: Any | None,
    inference_record: Any | None,
    prepared: PreparedDataset | None,
    runtime_result: TrainingResult | Any | None,
    payload: Optional[Mapping[str, Any]] = None,
    feature_stats: Optional[list[dict[str, Any]]] = None,
    runtime_error: BaseException | None = None,
) -> dict[str, Any]:
    feature_columns = _simulation_feature_names_from_prepared(prepared, model_record)
    capabilities = _simulation_capability_flags(runtime_result, feature_columns=feature_columns, runtime_error=runtime_error)
    family = _detect_simulation_family(
        model_record=model_record,
        inference_record=inference_record,
        prepared=prepared,
        runtime_result=runtime_result,
        payload=payload,
    )
    saved_profile = _simulation_profile_from_record(inference_record, model_record)
    modes = [dict(item) for item in SIMULATION_FAMILY_MODES.get(family, SIMULATION_FAMILY_MODES["unsupported"])]
    disabled_reasons = _simulation_disabled_reasons(family, capabilities)
    for mode in modes:
        mode["enabled"] = not disabled_reasons
    requested_mode = str((payload or {}).get("mode") or "").strip()
    saved_mode = str(saved_profile.get("default_mode") or saved_profile.get("mode") or "").strip()
    default_mode = requested_mode or saved_mode or (modes[0]["id"] if modes else "runtime_readiness")
    if default_mode not in {mode["id"] for mode in modes}:
        default_mode = modes[0]["id"] if modes else "runtime_readiness"
    return {
        "family": family,
        "label": SIMULATION_FAMILY_LABELS.get(family, "Unsupported"),
        "output_kind": SIMULATION_OUTPUT_KINDS.get(family, "unknown"),
        "supported_modes": modes,
        "default_mode": default_mode,
        "capabilities": capabilities,
        "disabled_reasons": disabled_reasons,
        "controls_schema": _simulation_controls_schema(family, feature_stats=feature_stats, capabilities=capabilities),
        "result_schema": {
            "primary_result": True,
            "series": family in {"regression", "classification", "geospatial"},
            "probabilities": family == "classification",
            "ranking": family == "recommendation",
            "geojson": family == "geospatial",
            "generated_text": family == "llm_slm",
        },
    }


def _build_unavailable_simulation_context(
    *,
    model_record: Any | None,
    dataset_record: Any | None,
    inference_record: Any | None,
    runtime_error: BaseException,
) -> dict[str, Any]:
    inference_summary = _serialize_inference_summary(inference_record) if inference_record is not None else {}
    model_summary = _serialize_model_summary(model_record) if model_record is not None else {}
    dataset_summary = _serialize_dataset_summary(dataset_record) if dataset_record is not None else {}
    profile = _build_simulation_profile(
        model_record=model_record,
        inference_record=inference_record,
        prepared=None,
        runtime_result=None,
        runtime_error=runtime_error,
    )
    return {
        "model": model_summary,
        "dataset": dataset_summary,
        "inference": inference_summary,
        "model_id": getattr(model_record, "id", None),
        "dataset_id": getattr(dataset_record, "id", None),
        "inference_id": getattr(inference_record, "id", None),
        "task_type": None,
        "input_features": _coerce_feature_list(getattr(model_record, "input_features", None)),
        "runtime_features": _coerce_feature_list(getattr(model_record, "input_features", None)),
        "output_features": _coerce_feature_list(getattr(inference_record, "output_features", None))
        or _coerce_feature_list(getattr(model_record, "output_features", None)),
        "baseline_source": "last",
        "baseline": {},
        "feature_stats": [],
        "simulation_defaults": dict(inference_summary.get("simulation_defaults") or {}),
        "latest_simulation": dict((inference_summary.get("inference_params") or {}).get("latest_simulation") or {}),
        "runtime_ready": False,
        "runtime_framework": None,
        "runtime_capabilities": {},
        "simulation_profile": profile,
    }


def _build_simulation_context(
    *,
    model_record: Any,
    dataset_record: Any,
    inference_record: Any | None,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    baseline_source: str = "last",
) -> dict[str, Any]:
    baseline = _build_simulation_baseline(prepared, baseline_source=baseline_source)
    feature_stats = _simulation_feature_stats(prepared.feature_frame, baseline)
    inference_summary = _serialize_inference_summary(inference_record) if inference_record is not None else {}
    model_summary = _serialize_model_summary(model_record)
    dataset_summary = _serialize_dataset_summary(dataset_record)
    runtime_parameters = dict(getattr(runtime_result, "parameters", {}) or {})
    runtime_capabilities = dict((runtime_parameters.get("artifact_manifest", {}) or {}).get("runtime_capabilities", {}) or {})
    simulation_profile = _build_simulation_profile(
        model_record=model_record,
        inference_record=inference_record,
        prepared=prepared,
        runtime_result=runtime_result,
        feature_stats=feature_stats,
    )
    return {
        "model": model_summary,
        "dataset": dataset_summary,
        "inference": inference_summary,
        "model_id": getattr(model_record, "id", None),
        "dataset_id": getattr(dataset_record, "id", None),
        "inference_id": getattr(inference_record, "id", None),
        "task_type": prepared.task_type,
        "input_features": list(prepared.input_features),
        "runtime_features": _simulation_feature_columns(prepared),
        "output_features": _coerce_feature_list(getattr(inference_record, "output_features", None))
        or _coerce_feature_list(getattr(model_record, "output_features", None))
        or [prepared.output_feature],
        "baseline_source": baseline_source or "last",
        "baseline": _simulation_row_payload(baseline),
        "feature_stats": feature_stats,
        "simulation_defaults": dict(inference_summary.get("simulation_defaults") or {}),
        "latest_simulation": dict((inference_summary.get("inference_params") or {}).get("latest_simulation") or {}),
        "runtime_ready": True,
        "runtime_framework": getattr(runtime_result, "framework", None),
        "runtime_capabilities": runtime_capabilities,
        "simulation_profile": simulation_profile,
    }


def _run_tabular_simulation_adapter(
    payload: Mapping[str, Any],
    *,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    model_record: Any,
    dataset_record: Any,
    inference_record: Any | None,
    simulation_profile: Optional[Mapping[str, Any]] = None,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    feature_columns = _simulation_feature_columns(prepared)
    if prepared.feature_frame.empty:
        raise ValueError("The selected dataset does not contain rows for simulation.")

    baseline_source = str(payload.get("baselineSource") or payload.get("baseline_source") or "last").strip().lower()
    baseline = _build_simulation_baseline(
        prepared,
        baseline_source=baseline_source,
        baseline_row=_simulation_mapping(payload.get("baselineRow") or payload.get("baseline_row")),
    )
    scenarios = _normalize_simulation_scenarios(payload, feature_columns)
    output_features = _coerce_feature_list(getattr(inference_record, "output_features", None)) or _coerce_feature_list(getattr(model_record, "output_features", None))
    output_feature = output_features[0] if output_features else prepared.output_feature or "prediction"

    scenario_summaries: list[dict[str, Any]] = []
    all_plot_points: list[tuple[str, Any]] = []
    all_table_rows: list[dict[str, Any]] = []
    drift_series: list[tuple[str, Any]] = []
    scenario_plot_deck: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_base = _apply_direct_overrides(baseline, scenario["overrides"])
        rows = [
            _apply_step_plan(
                scenario_base,
                step_index=step_index,
                steps=int(scenario["steps"]),
                amplitude=float(scenario["amplitude"]),
                trend=str(scenario["trend"]),
                step_plan=scenario["step_plan"],
            )
            for step_index in range(1, int(scenario["steps"]) + 1)
        ]
        simulation_frame = pd.DataFrame(rows, columns=feature_columns)
        predictions = np.asarray(_predict_with_result(runtime_result, simulation_frame, prepared.task_type)).reshape(-1)
        probabilities = _predict_probabilities_with_result(runtime_result, simulation_frame, prepared)
        series: list[dict[str, Any]] = []
        scenario_plot_points: list[tuple[str, Any]] = []
        scenario_table_rows: list[dict[str, Any]] = []
        changed_features = _changed_features(baseline, scenario_base)
        for step_index, prediction in enumerate(predictions[: len(simulation_frame)], start=1):
            rendered_prediction = _render_simulation_prediction(prediction, prepared)
            row_payload = {
                "scenario": scenario["name"],
                "step": step_index,
                "prediction": rendered_prediction,
                "output_feature": output_feature,
                "inputs": _simulation_row_payload(simulation_frame.iloc[step_index - 1]),
            }
            if probabilities and step_index - 1 < len(probabilities):
                row_payload["probabilities"] = probabilities[step_index - 1]
            series.append(row_payload)
            table_row = {
                "scenario": scenario["name"],
                "step": step_index,
                output_feature: rendered_prediction,
                "changed_features": len(changed_features),
            }
            if probabilities and step_index - 1 < len(probabilities):
                table_row["probabilities"] = probabilities[step_index - 1]
            scenario_table_rows.append(table_row)
            all_table_rows.append(table_row)
            if isinstance(rendered_prediction, (int, float)):
                point = (f"{scenario['name']} #{step_index}", rendered_prediction)
                all_plot_points.append(point)
                scenario_plot_points.append((f"Step {step_index}", rendered_prediction))

        prediction_summary = {
            "label": output_feature,
            "first_prediction": series[0]["prediction"] if series else None,
            "final_prediction": series[-1]["prediction"] if series else None,
            "prediction_delta": _prediction_delta(series[0]["prediction"], series[-1]["prediction"]) if series else None,
            "step_count": len(series),
        }
        final_row = simulation_frame.tail(1).iloc[0] if not simulation_frame.empty else scenario_base
        scenario_drift_series: list[tuple[str, Any]] = []
        for feature_name in feature_columns[:8]:
            try:
                drift_value = abs(float(final_row[feature_name]) - float(baseline[feature_name]))
            except (TypeError, ValueError):
                continue
            scenario_drift_series.append((feature_name, drift_value))
            drift_series.append((f"{scenario['name']} · {feature_name}", drift_value))
        scenario_plots = [
            _build_line_plot(
                f"{scenario['name']} Prediction Path",
                scenario_plot_points,
                description=f"Per-step predictions for the {scenario['name']} scenario.",
            ),
            _build_bar_plot(
                f"{scenario['name']} Changed Features",
                _changed_feature_delta_series(changed_features) or scenario_drift_series,
                description="Feature movement applied in this scenario.",
            ),
            _simulation_step_table_plot(
                f"{scenario['name']} Step Results",
                scenario_table_rows,
                description="Step-level simulation rows for this scenario.",
            ),
        ]
        scenario_plots = [plot for plot in scenario_plots if _plot_has_renderable_content(plot)]
        scenario_plot_deck.extend(scenario_plots)
        scenario_summaries.append(
            {
                "name": scenario["name"],
                "steps": scenario["steps"],
                "trend": scenario["trend"],
                "amplitude": scenario["amplitude"],
                "overrides": _json_safe(scenario["overrides"]),
                "changed_features": changed_features,
                "prediction_summary": prediction_summary,
                "series": series,
                "table_rows": scenario_table_rows,
                "plots": scenario_plots,
            }
        )

    primary = scenario_summaries[0] if scenario_summaries else {"series": [], "prediction_summary": {}}
    sensitivity = _build_sensitivity_summary(
        payload,
        baseline=baseline,
        prepared=prepared,
        runtime_result=runtime_result,
        output_feature=output_feature,
    )
    plots = [
        _build_line_plot(
            "Simulated Prediction Path",
            all_plot_points,
            description="Future-step predictions generated from the current production pair.",
        ),
        _build_bar_plot(
            "Feature Drift Applied",
            drift_series,
            description="Absolute change applied between the baseline row and the final simulated step.",
        ),
    ]
    if all_table_rows:
        plots.append(
            _simulation_step_table_plot(
                "Simulation Step Results",
                all_table_rows,
                description="All scenario step rows produced by the simulation.",
            )
        )
    plots.extend(sensitivity.get("plots", []))
    plots.extend(scenario_plot_deck)
    plots = [plot for plot in plots if _plot_has_renderable_content(plot)]

    result = {
        "model_id": getattr(model_record, "id", None),
        "dataset_id": getattr(dataset_record, "id", None),
        "inference_id": getattr(inference_record, "id", None),
        "task_type": prepared.task_type,
        "family": (simulation_profile or {}).get("family") or prepared.task_type,
        "mode": mode or (simulation_profile or {}).get("default_mode") or "tabular_what_if",
        "simulation_profile": dict(simulation_profile or {}),
        "steps": primary.get("steps") or 0,
        "trend": primary.get("trend"),
        "amplitude": primary.get("amplitude"),
        "baseline_source": baseline_source,
        "output_features": output_features,
        "output_feature": output_feature,
        "prediction_summary": primary.get("prediction_summary") or {},
        "baseline": _simulation_row_payload(baseline),
        "feature_stats": _simulation_feature_stats(prepared.feature_frame, baseline),
        "scenarios": scenario_summaries,
        "series": primary.get("series") or [],
        "table_rows": all_table_rows,
        "sensitivity": sensitivity,
        "limits": {
            "max_scenarios": SIMULATION_MAX_SCENARIOS,
            "max_rows": SIMULATION_MAX_ROWS,
            "max_steps": SIMULATION_MAX_STEPS,
        },
        "plots": plots,
    }
    family_payload = _simulation_mapping(payload.get("familyPayload") or payload.get("family_payload"))
    if (simulation_profile or {}).get("family") == "classification" or prepared.task_type == "classification":
        top_k = _coerce_simulation_int(family_payload.get("topK") or family_payload.get("top_k"), 3, minimum=1, maximum=10)
        threshold = _coerce_simulation_float(family_payload.get("threshold"), 0.5)
        final_probabilities = None
        if primary.get("series"):
            final_probabilities = (primary.get("series") or [{}])[-1].get("probabilities")
        if isinstance(final_probabilities, Mapping):
            ranked = sorted(final_probabilities.items(), key=lambda item: float(item[1] or 0.0), reverse=True)
        else:
            ranked = []
        result["classification"] = {
            "threshold": threshold,
            "top_k": top_k,
            "final_probabilities": dict(final_probabilities or {}),
            "top_classes": [
                {"class": str(label), "probability": _simulation_value(probability)}
                for label, probability in ranked[:top_k]
            ],
        }
        result["prediction_summary"]["final_probabilities"] = dict(final_probabilities or {})
        probability_plots: list[dict[str, Any]] = []
        for scenario in result.get("scenarios", []):
            scenario_series = scenario.get("series") if isinstance(scenario, Mapping) else []
            scenario_probabilities = (scenario_series or [{}])[-1].get("probabilities") if scenario_series else None
            if not isinstance(scenario_probabilities, Mapping):
                continue
            probability_plot = _build_bar_plot(
                f"{scenario.get('name', 'Scenario')} Class Probabilities",
                list(scenario_probabilities.items()),
                description="Final-step class probability distribution for this scenario.",
            )
            if not _plot_has_renderable_content(probability_plot):
                continue
            scenario.setdefault("plots", []).append(probability_plot)
            probability_plots.append(probability_plot)
        result["plots"] = [plot for plot in [*result.get("plots", []), *probability_plots] if _plot_has_renderable_content(plot)]
    else:
        result["regression"] = {
            "delta_explained": bool(family_payload.get("explainDeltas", True)),
            "focus_feature": family_payload.get("focusFeature"),
        }
    result["primary_result"] = {
        "label": output_feature,
        "value": result["prediction_summary"].get("final_prediction"),
        "delta": result["prediction_summary"].get("prediction_delta"),
    }
    return result


def _simulation_readiness_result(
    payload: Mapping[str, Any],
    *,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    model_record: Any,
    dataset_record: Any,
    inference_record: Any | None,
    simulation_profile: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    baseline_source = str(payload.get("baselineSource") or payload.get("baseline_source") or "last").strip().lower()
    baseline = _build_simulation_baseline(
        prepared,
        baseline_source=baseline_source,
        baseline_row=_simulation_mapping(payload.get("baselineRow") or payload.get("baseline_row")),
    )
    output_features = _coerce_feature_list(getattr(inference_record, "output_features", None)) or _coerce_feature_list(getattr(model_record, "output_features", None))
    output_feature = output_features[0] if output_features else getattr(prepared, "output_feature", None) or "prediction"
    family = str(simulation_profile.get("family") or "unsupported")
    reasons = list(simulation_profile.get("disabled_reasons") or [])
    readiness_rows = [
        {
            "family": family,
            "mode": mode,
            "reason": reason,
            "ready": False,
        }
        for reason in (reasons or ["Runtime capability metadata is incomplete for this simulation mode."])
    ]
    readiness_plot = _simulation_step_table_plot(
        "Simulation Readiness",
        readiness_rows,
        description="Why this simulation mode is not ready to run.",
    )
    return {
        "model_id": getattr(model_record, "id", None),
        "dataset_id": getattr(dataset_record, "id", None),
        "inference_id": getattr(inference_record, "id", None),
        "task_type": getattr(prepared, "task_type", None),
        "family": family,
        "mode": mode,
        "status": "not_ready",
        "simulation_profile": dict(simulation_profile),
        "readiness": {
            "ready": False,
            "family": family,
            "mode": mode,
            "disabled_reasons": reasons,
            "capabilities": dict(simulation_profile.get("capabilities") or {}),
        },
        "steps": 0,
        "trend": None,
        "amplitude": None,
        "baseline_source": baseline_source,
        "output_features": output_features,
        "output_feature": output_feature,
        "prediction_summary": {
            "label": output_feature,
            "first_prediction": None,
            "final_prediction": None,
            "prediction_delta": None,
            "step_count": 0,
        },
        "primary_result": {
            "label": SIMULATION_FAMILY_LABELS.get(family, "Simulation"),
            "value": "Runtime not ready",
            "delta": None,
        },
        "baseline": _simulation_row_payload(baseline),
        "feature_stats": _simulation_feature_stats(prepared.feature_frame, baseline),
        "scenarios": [],
        "series": [],
        "table_rows": [],
        "sensitivity": {"enabled": False, "features": [], "plots": []},
        "limits": {
            "max_scenarios": SIMULATION_MAX_SCENARIOS,
            "max_rows": SIMULATION_MAX_ROWS,
            "max_steps": SIMULATION_MAX_STEPS,
        },
        "plots": [readiness_plot] if _plot_has_renderable_content(readiness_plot) else [],
    }


def _run_unsupervised_simulation_adapter(
    payload: Mapping[str, Any],
    *,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    model_record: Any,
    dataset_record: Any,
    inference_record: Any | None,
    simulation_profile: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    capabilities = simulation_profile.get("capabilities") or {}
    if simulation_profile.get("disabled_reasons"):
        return _simulation_readiness_result(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=simulation_profile,
            mode=mode,
        )

    feature_columns = _simulation_feature_columns(prepared)
    baseline_source = str(payload.get("baselineSource") or payload.get("baseline_source") or "last").strip().lower()
    baseline = _build_simulation_baseline(
        prepared,
        baseline_source=baseline_source,
        baseline_row=_simulation_mapping(payload.get("baselineRow") or payload.get("baseline_row")),
    )
    family_payload = _simulation_mapping(payload.get("familyPayload") or payload.get("family_payload"))
    scenario = _simulation_mapping(payload.get("scenario"))
    if scenario:
        _validate_simulation_features(scenario, feature_columns, context="scenario")
    probe_row = _apply_direct_overrides(baseline, scenario)
    frame = pd.DataFrame([probe_row], columns=feature_columns)
    model = getattr(runtime_result, "model", None)
    result_payload: dict[str, Any] = {}
    if capabilities.get("predict") and callable(getattr(model, "predict", None)):
        try:
            prediction = np.asarray(model.predict(frame)).reshape(-1)
            result_payload["assignment"] = _simulation_value(prediction[0] if len(prediction) else None)
        except Exception as exc:
            result_payload["assignment_error"] = str(exc)
    if capabilities.get("decision_function") and callable(getattr(model, "decision_function", None)):
        try:
            score = np.asarray(model.decision_function(frame)).reshape(-1)
            result_payload["decision_score"] = _simulation_value(score[0] if len(score) else None)
        except Exception as exc:
            result_payload["decision_score_error"] = str(exc)
    if capabilities.get("score_samples") and callable(getattr(model, "score_samples", None)):
        try:
            score = np.asarray(model.score_samples(frame)).reshape(-1)
            result_payload["sample_score"] = _simulation_value(score[0] if len(score) else None)
        except Exception as exc:
            result_payload["sample_score_error"] = str(exc)
    if family_payload.get("includeTransform", True) and capabilities.get("transform") and callable(getattr(model, "transform", None)):
        try:
            transformed = np.asarray(model.transform(frame))
            result_payload["embedding"] = _json_safe(transformed[:1].tolist())
        except Exception as exc:
            result_payload["embedding_error"] = str(exc)

    if not result_payload:
        updated_profile = dict(simulation_profile)
        updated_profile["disabled_reasons"] = ["The runtime advertised an unsupervised capability, but no callable method returned a result."]
        return _simulation_readiness_result(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=updated_profile,
            mode=mode,
        )

    output_feature = "assignment" if "assignment" in result_payload else next(iter(result_payload.keys()), "unsupervised_result")
    series = [
        {
            "scenario": payload.get("scenarioName") or "Probe",
            "step": 1,
            "prediction": result_payload.get(output_feature),
            "output_feature": output_feature,
            "inputs": _simulation_row_payload(probe_row),
            "unsupervised": result_payload,
        }
    ]
    changed_features = _changed_features(baseline, probe_row)
    table_rows = [
        {
            "scenario": series[0]["scenario"],
            "step": 1,
            output_feature: series[0]["prediction"],
            "changed_features": len(changed_features),
        }
    ]
    scenario_plots = [
        _simulation_step_table_plot(
            f"{series[0]['scenario']} Unsupervised Result",
            [{**table_rows[0], **{key: _json_safe(value) for key, value in result_payload.items()}}],
            description="Callable unsupervised model outputs for the probe row.",
        ),
        _build_bar_plot(
            f"{series[0]['scenario']} Changed Features",
            _changed_feature_delta_series(changed_features),
            description="Feature changes applied to the unsupervised probe row.",
        ),
    ]
    scenario_plots = [plot for plot in scenario_plots if _plot_has_renderable_content(plot)]
    return {
        "model_id": getattr(model_record, "id", None),
        "dataset_id": getattr(dataset_record, "id", None),
        "inference_id": getattr(inference_record, "id", None),
        "task_type": getattr(prepared, "task_type", None),
        "family": "unsupervised",
        "mode": mode,
        "status": "simulated",
        "simulation_profile": dict(simulation_profile),
        "steps": 1,
        "baseline_source": baseline_source,
        "output_features": [output_feature],
        "output_feature": output_feature,
        "prediction_summary": {
            "label": output_feature,
            "first_prediction": series[0]["prediction"],
            "final_prediction": series[0]["prediction"],
            "prediction_delta": None,
            "step_count": 1,
        },
        "primary_result": {"label": output_feature, "value": series[0]["prediction"], "delta": None},
        "baseline": _simulation_row_payload(baseline),
        "feature_stats": _simulation_feature_stats(prepared.feature_frame, baseline),
        "scenarios": [
            {
                "name": series[0]["scenario"],
                "steps": 1,
                "changed_features": changed_features,
                "prediction_summary": {"final_prediction": series[0]["prediction"], "step_count": 1},
                "series": series,
                "table_rows": table_rows,
                "plots": scenario_plots,
            }
        ],
        "series": series,
        "table_rows": table_rows,
        "unsupervised": result_payload,
        "sensitivity": {"enabled": False, "features": [], "plots": []},
        "limits": {"max_scenarios": SIMULATION_MAX_SCENARIOS, "max_rows": SIMULATION_MAX_ROWS, "max_steps": SIMULATION_MAX_STEPS},
        "plots": scenario_plots,
    }


def _run_recommendation_simulation_adapter(
    payload: Mapping[str, Any],
    *,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    model_record: Any,
    dataset_record: Any,
    inference_record: Any | None,
    simulation_profile: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    if simulation_profile.get("disabled_reasons"):
        return _simulation_readiness_result(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=simulation_profile,
            mode=mode,
        )
    family_payload = _simulation_mapping(payload.get("familyPayload") or payload.get("family_payload"))
    model = getattr(runtime_result, "model", None)
    top_n = _coerce_simulation_int(family_payload.get("topN") or family_payload.get("top_n"), 10, minimum=1, maximum=50)
    user_id = family_payload.get("userId") or family_payload.get("user_id")
    recommendations: Any = None
    recommend = getattr(model, "recommend", None)
    if callable(recommend):
        for args in ((user_id, top_n), (user_id,), ()):
            try:
                recommendations = recommend(*args)
                break
            except TypeError:
                continue
    if recommendations is None:
        updated_profile = dict(simulation_profile)
        updated_profile["disabled_reasons"] = ["The recommender adapter could not call a compatible recommend(user_id, top_n) method."]
        return _simulation_readiness_result(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=updated_profile,
            mode=mode,
        )
    rows = _json_safe(recommendations)
    if not isinstance(rows, list):
        rows = [rows]
    rows = rows[:top_n]
    recommendation_rows = [
        row if isinstance(row, Mapping) else {"item": row}
        for row in rows
    ]
    recommendation_plot = _simulation_step_table_plot(
        "Recommendation Results",
        recommendation_rows,
        description="Top recommendation rows returned by the runtime adapter.",
    )
    recommendation_plots = [recommendation_plot] if _plot_has_renderable_content(recommendation_plot) else []
    return {
        "model_id": getattr(model_record, "id", None),
        "dataset_id": getattr(dataset_record, "id", None),
        "inference_id": getattr(inference_record, "id", None),
        "task_type": getattr(prepared, "task_type", None),
        "family": "recommendation",
        "mode": mode,
        "status": "simulated",
        "simulation_profile": dict(simulation_profile),
        "steps": len(rows),
        "baseline_source": str(payload.get("baselineSource") or payload.get("baseline_source") or "last").strip().lower(),
        "output_features": ["recommendations"],
        "output_feature": "recommendations",
        "prediction_summary": {"label": "recommendations", "first_prediction": rows[0] if rows else None, "final_prediction": rows[0] if rows else None, "prediction_delta": None, "step_count": len(rows)},
        "primary_result": {"label": "Recommendations", "value": len(rows), "delta": None},
        "baseline": {},
        "feature_stats": _simulation_feature_stats(prepared.feature_frame, _baseline_from_source(prepared.feature_frame, "last")),
        "scenarios": [
            {
                "name": "Recommendations",
                "steps": len(rows),
                "changed_features": [],
                "prediction_summary": {"final_prediction": rows[0] if rows else None, "step_count": len(rows)},
                "series": [],
                "table_rows": recommendation_rows,
                "plots": recommendation_plots,
            }
        ],
        "series": [],
        "table_rows": rows,
        "recommendation": {"user_id": user_id, "top_n": top_n, "items": rows},
        "sensitivity": {"enabled": False, "features": [], "plots": []},
        "limits": {"max_scenarios": SIMULATION_MAX_SCENARIOS, "max_rows": SIMULATION_MAX_ROWS, "max_steps": SIMULATION_MAX_STEPS},
        "plots": recommendation_plots,
    }


def _simulation_geojson_from_series(series: list[dict[str, Any]], capabilities: Mapping[str, Any]) -> dict[str, Any] | None:
    latitude_feature = capabilities.get("latitude_feature")
    longitude_feature = capabilities.get("longitude_feature")
    if not latitude_feature or not longitude_feature:
        return None
    features = []
    for row in series[:SIMULATION_MAX_ROWS]:
        inputs = row.get("inputs") or {}
        try:
            latitude = float(inputs.get(latitude_feature))
            longitude = float(inputs.get(longitude_feature))
        except (TypeError, ValueError):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": {
                    "scenario": row.get("scenario"),
                    "step": row.get("step"),
                    "prediction": row.get("prediction"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features} if features else None


def _run_simulation_workbench(
    payload: Mapping[str, Any],
    *,
    prepared: PreparedDataset,
    runtime_result: TrainingResult,
    model_record: Any,
    dataset_record: Any,
    inference_record: Any | None,
) -> dict[str, Any]:
    feature_columns = _simulation_feature_columns(prepared)
    baseline_source = str(payload.get("baselineSource") or payload.get("baseline_source") or "last").strip().lower()
    baseline = _build_simulation_baseline(prepared, baseline_source=baseline_source)
    feature_stats = _simulation_feature_stats(prepared.feature_frame, baseline)
    simulation_profile = _build_simulation_profile(
        model_record=model_record,
        inference_record=inference_record,
        prepared=prepared,
        runtime_result=runtime_result,
        payload=payload,
        feature_stats=feature_stats,
    )
    mode = str(payload.get("mode") or simulation_profile.get("default_mode") or "tabular_what_if").strip()
    family = str(simulation_profile.get("family") or "unsupported")

    if family in {"regression", "classification"}:
        return _run_tabular_simulation_adapter(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=simulation_profile,
            mode=mode,
        )

    if family == "geospatial" and not simulation_profile.get("disabled_reasons"):
        result = _run_tabular_simulation_adapter(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=simulation_profile,
            mode=mode,
        )
        result["geojson"] = _simulation_geojson_from_series(result.get("series") or [], simulation_profile.get("capabilities") or {})
        return result

    if family == "unsupervised":
        return _run_unsupervised_simulation_adapter(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=simulation_profile,
            mode=mode,
        )

    if family == "recommendation":
        return _run_recommendation_simulation_adapter(
            payload,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
            simulation_profile=simulation_profile,
            mode=mode,
        )

    return _simulation_readiness_result(
        payload,
        prepared=prepared,
        runtime_result=runtime_result,
        model_record=model_record,
        dataset_record=dataset_record,
        inference_record=inference_record,
        simulation_profile=simulation_profile,
        mode=mode,
    )


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
            "output_features": list(prepared.output_features or ([prepared.output_feature] if prepared.output_feature else [])),
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


def _safe_sklearn_estimator_family(estimator_or_class: Any) -> str:
    estimator_type = str(getattr(estimator_or_class, "_estimator_type", "") or "").strip().lower()
    if estimator_type == "classifier":
        return "classifier"
    if estimator_type == "regressor":
        return "regressor"
    if estimator_type in {"clusterer", "cluster"}:
        return "clustering"
    if estimator_type in {"outlier_detector", "outlier"}:
        return "outlier"

    try:
        from sklearn.base import is_classifier, is_regressor

        if is_classifier(estimator_or_class):
            return "classifier"
        if is_regressor(estimator_or_class):
            return "regressor"
    except Exception:
        pass

    if hasattr(estimator_or_class, "transform") or hasattr(estimator_or_class, "fit_transform"):
        return "transformer"
    if hasattr(estimator_or_class, "fit_predict"):
        return "clustering"
    if hasattr(estimator_or_class, "score_samples") or hasattr(estimator_or_class, "decision_function"):
        return "outlier"
    return "estimator"


def _normalise_sklearn_task_family(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "classification": "classifier",
        "regression": "regressor",
        "classifier": "classifier",
        "regressor": "regressor",
        "cluster": "clustering",
        "clusterer": "clustering",
        "clustering": "clustering",
        "outlier_detection": "outlier",
        "anomaly": "outlier",
        "anomaly_detection": "outlier",
        "unsupervised": "unsupervised",
        "transform": "transformer",
        "transformer": "transformer",
    }
    return aliases.get(normalized, normalized)


def _is_unsupervised_sklearn_request(parameters: Mapping[str, Any]) -> bool:
    explicit_task = _normalise_sklearn_task_family(str(parameters.get("task_type") or ""))
    if explicit_task in SKLEARN_UNSUPERVISED_FAMILIES:
        return True
    framework = str(parameters.get("framework") or "sklearn").strip().lower()
    if framework not in {"sklearn", "scikit-learn", "scikitlearn"}:
        return False
    estimator_class = parameters.get("estimator_class") or parameters.get("algorithm")
    if not estimator_class:
        return False
    try:
        estimator = _select_sklearn_estimator(str(estimator_class), task_type="regression")
        return _safe_sklearn_estimator_family(estimator) in SKLEARN_UNSUPERVISED_FAMILIES
    except Exception:
        return False


def _infer_task_type(target: pd.Series, model_record: Any, parameters: Mapping[str, Any]) -> str:
    explicit = str(parameters.get("task_type") or "").strip().lower()
    explicit_family = _normalise_sklearn_task_family(explicit)
    if explicit_family == "classifier":
        return "classification"
    if explicit_family == "regressor":
        return "regression"
    if explicit_family in SKLEARN_UNSUPERVISED_FAMILIES:
        return explicit_family

    if str(parameters.get("framework") or "sklearn").strip().lower() in {"sklearn", "scikit-learn", "scikitlearn"}:
        estimator_class = parameters.get("estimator_class") or parameters.get("algorithm")
        if estimator_class:
            try:
                estimator = _select_sklearn_estimator(str(estimator_class), task_type="regression")
                family = _safe_sklearn_estimator_family(estimator)
                if family == "classifier":
                    return "classification"
                if family == "regressor":
                    return "regression"
                if family in SKLEARN_UNSUPERVISED_FAMILIES:
                    return family
            except Exception:
                pass

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


def _encode_training_target(
    target: pd.Series | pd.DataFrame,
    task_type: str,
) -> tuple[pd.Series | pd.DataFrame, dict[str, Any]]:
    if isinstance(target, pd.DataFrame):
        if task_type == "classification":
            encoded_columns: dict[str, pd.Series] = {}
            mappings: dict[str, dict[str, int]] = {}
            for column in target.columns:
                encoded, mapping = _encode_target(target[column], task_type)
                encoded_columns[str(column)] = encoded
                mappings[str(column)] = mapping
            return pd.DataFrame(encoded_columns, index=target.index), mappings

        numeric = target.apply(pd.to_numeric, errors="coerce")
        fill_values = numeric.median(numeric_only=True).fillna(0.0)
        return numeric.fillna(fill_values).fillna(0.0).astype(float), {}

    return _encode_target(target, task_type)


def _read_assistant_training_manifest(dataset_record: Any) -> dict[str, Any]:
    manifest_path = getattr(dataset_record, "connection_string", None)
    resolved_manifest = _resolve_fs_path(manifest_path, default_parent=REPO_ROOT)
    if resolved_manifest is None or not resolved_manifest.exists():
        return {}
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _assistant_training_path_candidates(dataset_record: Any, manifest: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    path_value = getattr(dataset_record, "path", None)
    tabular_candidates = [
        path_value,
        manifest.get("tabular_csv_path"),
        manifest.get("dataset_path") if str(manifest.get("dataset_path") or "").lower().endswith(".csv") else None,
    ]
    jsonl_candidates = [
        manifest.get("canonical_jsonl_path"),
        manifest.get("jsonl_path"),
        path_value if str(path_value or "").lower().endswith(".jsonl") else None,
        manifest.get("dataset_path") if str(manifest.get("dataset_path") or "").lower().endswith(".jsonl") else None,
    ]
    return [item for item in tabular_candidates if item], [item for item in jsonl_candidates if item]


def _load_assistant_training_dataset_frame_from_record(dataset_record: Any) -> pd.DataFrame:
    manifest = _read_assistant_training_manifest(dataset_record)
    tabular_candidates, jsonl_candidates = _assistant_training_path_candidates(dataset_record, manifest)

    for path_value in tabular_candidates:
        resolved_path = _resolve_fs_path(path_value, default_parent=REPO_ROOT)
        if resolved_path is not None and resolved_path.exists():
            dataframe, _parser_report = load_tabular_from_path(resolved_path)
            return dataframe

    for jsonl_value in jsonl_candidates:
        resolved_jsonl = _resolve_fs_path(jsonl_value, default_parent=REPO_ROOT)
        if resolved_jsonl is None or not resolved_jsonl.exists():
            continue
        preferred_csv = manifest.get("tabular_csv_path") or resolved_jsonl.with_suffix(".csv")
        resolved_csv = _resolve_fs_path(preferred_csv, default_parent=REPO_ROOT)
        if resolved_csv is None:
            resolved_csv = resolved_jsonl.with_suffix(".csv")
        from app.utils.assistant_llmops import materialize_assistant_training_csv_from_jsonl

        materialize_assistant_training_csv_from_jsonl(resolved_jsonl, resolved_csv)
        dataframe, _parser_report = load_tabular_from_path(resolved_csv)
        return dataframe

    missing_path = getattr(dataset_record, "path", None) or manifest.get("tabular_csv_path") or manifest.get("canonical_jsonl_path")
    raise FileNotFoundError(f"Assistant training dataset file not found: {missing_path}")


def _load_dataset_frame_from_record(dataset_record: Any) -> pd.DataFrame:
    if str(getattr(dataset_record, "dataset_type", "") or "").lower() == "assistant_training_dataset":
        return _load_assistant_training_dataset_frame_from_record(dataset_record)
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
    input_features, output_features = _resolve_training_feature_selection(dataframe, model_record, parameters)
    output_feature = output_features[0] if output_features else ""

    validate_dataframe_columns(dataframe, [*input_features, *output_features])

    feature_frame = _normalize_feature_frame(dataframe[input_features])
    first_target = dataframe[output_feature] if output_feature else pd.Series(index=dataframe.index, dtype=float, name="")
    task_type = _infer_task_type(first_target, model_record, parameters)
    is_unsupervised = task_type in SKLEARN_UNSUPERVISED_FAMILIES
    if is_unsupervised:
        target_series: pd.Series | pd.DataFrame = pd.Series(index=dataframe.index, dtype=float, name=output_feature)
        encoded_target: pd.Series | pd.DataFrame = target_series
        label_mapping: dict[str, Any] = {}
    else:
        target_series = dataframe[output_features] if len(output_features) > 1 else dataframe[output_feature]
        encoded_target, label_mapping = _encode_training_target(target_series, task_type)

    from sklearn.model_selection import train_test_split

    test_size = float(parameters.get("test_size", DEFAULT_TEST_SIZE))
    random_state = int(parameters.get("random_state", DEFAULT_RANDOM_STATE))
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    stratify = None
    if task_type == "classification" and isinstance(encoded_target, pd.Series) and encoded_target.nunique() > 1:
        value_counts = encoded_target.value_counts()
        if not value_counts.empty and int(value_counts.min()) > 1:
            stratify = encoded_target

    if is_unsupervised:
        x_train, x_test = train_test_split(
            feature_frame,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        y_train = pd.Series(index=x_train.index, dtype=float, name=output_feature)
        y_test = pd.Series(index=x_test.index, dtype=float, name=output_feature)
    else:
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
        output_features=[str(item) for item in output_features],
        task_type=task_type,
        label_mapping=label_mapping,
    )


def _extract_model_hyperparameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if key not in CONTROL_PARAMETER_KEYS and value is not None
    }


def _sklearn_all_estimator_lookup() -> dict[str, Any]:
    global SKLEARN_ESTIMATOR_LOOKUP_CACHE
    if SKLEARN_ESTIMATOR_LOOKUP_CACHE is not None:
        return dict(SKLEARN_ESTIMATOR_LOOKUP_CACHE)
    try:
        from sklearn.utils import all_estimators
    except Exception as exc:
        raise RuntimeError(f"scikit-learn is required for sklearn estimator discovery: {exc}") from exc

    lookup: dict[str, Any] = {}
    for estimator_name, estimator_class in all_estimators():
        keys = {
            estimator_name.lower(),
            estimator_name.replace("_", "").lower(),
            f"{estimator_class.__module__}.{estimator_class.__name__}".lower(),
        }
        for key in keys:
            lookup[key] = estimator_class
    SKLEARN_ESTIMATOR_LOOKUP_CACHE = dict(lookup)
    return lookup


def _safe_import_sklearn_estimator(reference: str) -> Any | None:
    if not reference or "." not in reference:
        return None
    module_name, _, class_name = reference.rpartition(".")
    if not module_name.startswith("sklearn."):
        return None
    module = importlib.import_module(module_name)
    estimator = getattr(module, class_name, None)
    if inspect.isclass(estimator) and callable(getattr(estimator, "fit", None)):
        return estimator
    return None


def _select_sklearn_estimator(estimator_class: Optional[str], task_type: str) -> Any:
    if estimator_class:
        reference = str(estimator_class).strip()
        estimator = _safe_import_sklearn_estimator(reference)
        if estimator is not None:
            return estimator
        lookup = _sklearn_all_estimator_lookup()
        normalized = reference.lower()
        estimator = lookup.get(normalized) or lookup.get(reference.replace(".", "").replace("_", "").lower())
        if estimator is not None:
            return estimator
        raise ValueError(f"Unknown sklearn estimator_class '{estimator_class}'.")

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    return RandomForestClassifier if task_type == "classification" else RandomForestRegressor


def _sklearn_constructor_param_names(estimator_or_class: Any) -> set[str] | None:
    if hasattr(estimator_or_class, "get_params"):
        try:
            return set(estimator_or_class.get_params(deep=True).keys())
        except Exception:
            pass
    try:
        signature = inspect.signature(estimator_or_class)
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
            names.add(name)
    return names


def _mapping_parameter(parameters: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in keys:
        value = parameters.get(key)
        if isinstance(value, Mapping):
            merged.update(dict(value))
    return merged


def _partition_sklearn_parameters(
    parameters: Mapping[str, Any],
    estimator_or_class: Any,
) -> dict[str, Any]:
    valid_names = _sklearn_constructor_param_names(estimator_or_class)
    explicit_estimator_params = _mapping_parameter(parameters, "estimator_params")
    legacy_candidates = {
        key: value
        for key, value in parameters.items()
        if key not in CONTROL_PARAMETER_KEYS and value is not None
    }

    accepted: dict[str, Any] = {}
    ignored: dict[str, Any] = {}
    for source in (legacy_candidates, explicit_estimator_params):
        for key, value in source.items():
            if valid_names is None or key in valid_names:
                accepted[key] = value
            else:
                ignored[key] = value

    fit_kwargs = _mapping_parameter(parameters, "fit_params", "fit_kwargs")
    return {
        "estimator_params": accepted,
        "fit_kwargs": fit_kwargs,
        "ignored_params": ignored,
        "valid_param_names": sorted(valid_names or []),
        "explicit_estimator_params": explicit_estimator_params,
        "legacy_estimator_candidates": legacy_candidates,
    }


def _sklearn_estimator_capabilities(estimator_or_class: Any) -> dict[str, bool]:
    return {
        "fit": hasattr(estimator_or_class, "fit"),
        "predict": hasattr(estimator_or_class, "predict"),
        "predict_proba": hasattr(estimator_or_class, "predict_proba"),
        "fit_predict": hasattr(estimator_or_class, "fit_predict"),
        "transform": hasattr(estimator_or_class, "transform"),
        "fit_transform": hasattr(estimator_or_class, "fit_transform"),
        "score": hasattr(estimator_or_class, "score"),
        "score_samples": hasattr(estimator_or_class, "score_samples"),
        "decision_function": hasattr(estimator_or_class, "decision_function"),
    }


def _sklearn_prediction_outputs(model: Any, features: pd.DataFrame, task_type: str) -> tuple[np.ndarray, dict[str, Any]]:
    family = _normalise_sklearn_task_family(task_type)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(features)), {"output_method": "predict"}
    if family == "transformer" and hasattr(model, "transform"):
        return np.asarray(model.transform(features)), {"output_method": "transform"}
    if hasattr(model, "score_samples"):
        return np.asarray(model.score_samples(features)), {"output_method": "score_samples"}
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features)), {"output_method": "decision_function"}
    if hasattr(model, "fit_predict"):
        return np.asarray(model.fit_predict(features)), {
            "output_method": "fit_predict",
            "warning": "fit_predict was used because the estimator does not expose predict.",
        }
    if family == "transformer" and hasattr(model, "fit_transform"):
        return np.asarray(model.fit_transform(features)), {
            "output_method": "fit_transform",
            "warning": "fit_transform was used because the estimator does not expose transform.",
        }
    return np.asarray([]), {"output_method": None, "warning": "Estimator does not expose prediction-like outputs."}


def _normalise_training_mode(value: Any) -> str:
    normalized = str(value or "transfer").strip().lower()
    return normalized if normalized in {"transfer", "fresh"} else "transfer"


def _dataset_snapshot_hash(dataset_record: Any) -> Optional[str]:
    manifest_path = getattr(dataset_record, "connection_string", None)
    if manifest_path:
        try:
            resolved_manifest = _resolve_fs_path(manifest_path, default_parent=REPO_ROOT)
            manifest = json.loads(Path(resolved_manifest or manifest_path).read_text(encoding="utf-8"))
            dataset_hash = manifest.get("dataset_hash")
            if dataset_hash:
                return str(dataset_hash)
        except Exception:
            pass
    path = _resolve_fs_path(getattr(dataset_record, "path", None), default_parent=REPO_ROOT)
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _public_transfer_context(context: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in dict(context or {}).items()
        if key not in {"base_model"}
    }
    manifest = public.get("base_artifact_manifest")
    if isinstance(manifest, Mapping):
        public["base_artifact_manifest"] = {
            "artifact_format": manifest.get("artifact_format"),
            "framework": manifest.get("framework"),
            "loader": manifest.get("loader"),
            "warnings": list(manifest.get("warnings") or []),
        }
    return public


def _build_transfer_learning_context(
    model_record: Any,
    dataset_record: Any,
    parameters: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any]:
    mode = _normalise_training_mode(parameters.get("training_mode"))
    strategy = str(parameters.get("transfer_strategy") or "auto")
    base_artifact_path = parameters.get("base_artifact_path") or getattr(model_record, "path", None)
    base_run_id = parameters.get("base_run_id") or dict(parameters.get("training_lineage") or {}).get("latest_run_id")
    context: dict[str, Any] = {
        "training_mode": mode,
        "transfer_strategy": strategy,
        "base_model_id": parameters.get("base_model_id") or getattr(model_record, "id", None),
        "base_run_id": base_run_id,
        "base_artifact_path": str(base_artifact_path) if base_artifact_path else None,
        "dataset_snapshot_hash": _dataset_snapshot_hash(dataset_record),
        "transfer_status": "pending",
        "base_model_loaded": False,
    }
    if str(getattr(dataset_record, "dataset_type", "") or "").lower() == "assistant_training_dataset":
        manifest = _read_assistant_training_manifest(dataset_record)
        context.update(
            {
                "assistant_training_dataset": {
                    "manifest_path": getattr(dataset_record, "connection_string", None),
                    "canonical_jsonl_path": manifest.get("canonical_jsonl_path"),
                    "tabular_csv_path": manifest.get("tabular_csv_path") or getattr(dataset_record, "path", None),
                    "dataset_hash": manifest.get("dataset_hash"),
                    "tabular_hash": manifest.get("tabular_hash"),
                    "materialization_format": manifest.get("materialization_format"),
                }
            }
        )
    if mode == "fresh":
        context["transfer_status"] = "fresh_requested"
        return context
    if not base_artifact_path:
        context["transfer_status"] = "no_base_artifact"
        return context
    path = _resolve_fs_path(base_artifact_path, default_parent=REPO_ROOT)
    if path is None or not path.exists():
        context["transfer_status"] = "missing_base_artifact"
        return context
    if framework == "onnx":
        context["transfer_status"] = "onnx_transfer_uses_backend_or_fresh"
        return context
    try:
        runtime_payload = load_runtime_artifact(path, parameters=parameters)
    except Exception as exc:
        context["transfer_status"] = "incompatible_base_artifact"
        context["transfer_error"] = str(exc)
        return context
    loaded_framework = str(runtime_payload.get("framework") or "").strip().lower()
    expected_framework = "pytorch" if framework in {"pytorch", "torch"} else "sklearn"
    if loaded_framework != expected_framework:
        context["transfer_status"] = "incompatible_framework"
        context["loaded_framework"] = loaded_framework
        context["expected_framework"] = expected_framework
        return context
    context["base_model"] = runtime_payload.get("model")
    context["base_artifact_manifest"] = runtime_payload.get("manifest", {})
    context["base_model_loaded"] = True
    context["loaded_framework"] = loaded_framework
    context["transfer_status"] = "base_loaded"
    return context


def _prepare_sklearn_transfer_estimator(
    base_model: Any,
    parameters: Mapping[str, Any],
) -> tuple[Any, str]:
    if base_model is None or not hasattr(base_model, "fit"):
        return None, "no_base_model"
    if hasattr(base_model, "get_params") and hasattr(base_model, "set_params"):
        try:
            params = base_model.get_params(deep=True)
            updates: dict[str, Any] = {}
            if "warm_start" in params:
                updates["warm_start"] = True
                if "n_estimators" in params:
                    increment = int(parameters.get("warm_start_increment") or 10)
                    updates["n_estimators"] = int(params.get("n_estimators") or 0) + max(1, increment)
            if updates:
                base_model.set_params(**updates)
                return base_model, "warm_start_enabled"
        except Exception:
            return base_model, "base_loaded_refit"
    return base_model, "base_loaded_refit"


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
        predictions, metadata = _sklearn_prediction_outputs(result.model, features, task_type)
        if predictions.size == 0:
            raise ValueError(metadata.get("warning") or "The selected sklearn model does not expose prediction outputs.")
        return predictions

    if result.framework != "pytorch":
        raise ValueError(f"Unsupported training framework: {result.framework}")

    import torch

    model = result.model
    model.eval()
    tensor_x = torch.as_tensor(features.to_numpy(dtype=np.float32))
    try:
        model_device = next(model.parameters()).device
        tensor_x = tensor_x.to(model_device)
    except Exception:
        pass
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
    *,
    transfer_context: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> tuple[TrainingResult, np.ndarray]:
    estimator_class = parameters.get("estimator_class") or parameters.get("algorithm")
    estimator_factory = _select_sklearn_estimator(
        estimator_class=str(estimator_class) if estimator_class else None,
        task_type=prepared.task_type,
    )
    transfer_context = dict(transfer_context or {})
    estimator_target: Any = estimator_factory
    clone_estimator = True
    if transfer_context.get("base_model_loaded"):
        transferred_estimator, sklearn_transfer_status = _prepare_sklearn_transfer_estimator(
            transfer_context.get("base_model"),
            parameters,
        )
        if transferred_estimator is not None:
            estimator_target = transferred_estimator
            clone_estimator = False
            transfer_context["transfer_status"] = sklearn_transfer_status
    partitioned_params = _partition_sklearn_parameters(parameters, estimator_target)
    hyperparameters = dict(partitioned_params["estimator_params"])
    if estimator_factory.__name__ == "LogisticRegression":
        hyperparameters.setdefault("max_iter", 1000)

    estimator_family = _safe_sklearn_estimator_family(estimator_factory)
    estimator_metadata = {
        "requested_estimator_class": str(estimator_class) if estimator_class else None,
        "resolved_estimator_class": estimator_factory.__name__,
        "resolved_estimator_module": estimator_factory.__module__,
        "estimator_family": estimator_family,
        "estimator_capabilities": _sklearn_estimator_capabilities(estimator_factory),
        "accepted_estimator_params": dict(hyperparameters),
        "ignored_unknown_params": dict(partitioned_params["ignored_params"]),
        "valid_estimator_params": list(partitioned_params["valid_param_names"]),
        "fit_kwargs": dict(partitioned_params["fit_kwargs"]),
    }

    experiment_name = str(parameters.get("experiment_name") or f"{model_record.name}_training")
    run_name = str(parameters.get("run_name") or f"{model_record.name}_{uuid.uuid4().hex[:8]}")
    tags = {
        "framework": "sklearn",
        "model_name": str(getattr(model_record, "name", "model")),
        "model_id": str(getattr(model_record, "id", "")),
        "task_type": prepared.task_type,
    }

    if progress_callback is not None:
        progress_callback({"stage": "training", "message": "Fitting the scikit-learn estimator."})
    result = run_training_pipeline(
        lambda: train_sklearn(
            estimator_target,
            prepared.x_train,
            prepared.y_train,
            params=hyperparameters,
            random_state=int(parameters.get("random_state", DEFAULT_RANDOM_STATE)),
            fit_kwargs=partitioned_params["fit_kwargs"],
            eval_data=(prepared.x_test, prepared.y_test),
            task_type=prepared.task_type,
            experiment_name=None,
            run_name=None,
            tags=None,
            log_to_mlflow=True,
            clone_estimator=clone_estimator,
            metrics_to_track=parameters.get("metrics_to_track"),
            objective_metric=parameters.get("objective_metric"),
        ),
        experiment_name=experiment_name,
        run_name=run_name,
        params=parameters,
        tags=tags,
        log_to_mlflow=True,
    )
    result.metadata["sklearn"] = {
        **estimator_metadata,
        "training_metadata": dict(result.metadata),
    }
    result.metadata["ignored_unknown_params"] = dict(partitioned_params["ignored_params"])
    if partitioned_params["ignored_params"]:
        result.metadata.setdefault("warnings", [])
        result.metadata["warnings"] = list(result.metadata["warnings"]) + [
            f"Ignored unknown sklearn parameters: {', '.join(sorted(partitioned_params['ignored_params']))}"
        ]

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "evaluating",
                "message": "Evaluating model predictions on the validation split.",
                "counters": {"fit_duration_sec": result.metrics.get("fit_duration_sec")},
            }
        )
    predictions, prediction_metadata = _sklearn_prediction_outputs(result.model, prepared.x_test, prepared.task_type)
    result.metadata["sklearn"]["prediction_output"] = prediction_metadata
    _log_training_tracking_context(
        result,
        prepared,
        model_record,
        parameters,
        framework="sklearn",
        extra_payload={
            "predictions_preview": predictions[:10].tolist(),
            "transfer_learning": _public_transfer_context(transfer_context),
        },
    )
    result.metadata["transfer_learning"] = _public_transfer_context(transfer_context)

    return result, predictions


def _train_with_pytorch(
    prepared: PreparedDataset,
    model_record: Any,
    parameters: Mapping[str, Any],
    *,
    transfer_context: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
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
    device_resolution = resolve_torch_device(parameters.get("device", "auto"))

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
    transfer_context = dict(transfer_context or {})
    if transfer_context.get("base_model_loaded"):
        try:
            state_dict = transfer_context["base_model"].state_dict()
            model.load_state_dict(state_dict, strict=False)
            transfer_context["transfer_status"] = "base_weights_loaded"
        except Exception as exc:
            transfer_context["transfer_status"] = "base_weights_incompatible"
            transfer_context["transfer_error"] = str(exc)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    experiment_name = str(parameters.get("experiment_name") or f"{model_record.name}_training")
    run_name = str(parameters.get("run_name") or f"{model_record.name}_{uuid.uuid4().hex[:8]}")
    tags = {
        "framework": "pytorch",
        "model_name": str(getattr(model_record, "name", "model")),
        "model_id": str(getattr(model_record, "id", "")),
        "task_type": prepared.task_type,
        "requested_device": str(device_resolution["requested_device"]),
        "resolved_device": str(device_resolution["resolved_device"]),
        "cuda_available": str(device_resolution["cuda_available"]),
    }

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "resolving_accelerator",
                "message": (
                    f"Requested device {device_resolution['requested_device']} resolved to "
                    f"{device_resolution['resolved_device']}."
                ),
                "counters": {
                    "cuda_available": device_resolution.get("cuda_available"),
                    "device_count": device_resolution.get("device_count"),
                    "fallback_applied": device_resolution.get("fallback_applied"),
                },
            }
        )
        progress_callback(
            {
                "stage": "training",
                "message": "Running PyTorch training epochs.",
                "counters": {"epochs_total": epochs, "epochs_completed": 0},
            }
        )
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
            device=str(device_resolution["resolved_device"]),
            params={
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "requested_device": device_resolution["requested_device"],
                "resolved_device": device_resolution["resolved_device"],
                "cuda_available": device_resolution["cuda_available"],
                "device_fallback_applied": device_resolution["fallback_applied"],
            },
            experiment_name=None,
            run_name=None,
            tags=None,
            log_to_mlflow=True,
            progress_callback=(
                None
                if progress_callback is None
                else lambda event: progress_callback(
                    {
                        "stage": "training",
                        "message": f"Completed epoch {event.get('epoch')} of {event.get('epochs_total')}.",
                        "counters": {
                            "epochs_total": event.get("epochs_total"),
                            "epochs_completed": event.get("epoch"),
                            "train_loss": event.get("train_loss"),
                            "val_loss": event.get("val_loss"),
                        },
                    }
                )
            ),
        ),
        experiment_name=experiment_name,
        run_name=run_name,
        params={
            **dict(parameters),
            "requested_device": device_resolution["requested_device"],
            "resolved_device": device_resolution["resolved_device"],
            "cuda_available": device_resolution["cuda_available"],
        },
        tags=tags,
        log_to_mlflow=True,
    )

    if progress_callback is not None:
        progress_callback({"stage": "evaluating", "message": "Scoring PyTorch predictions on the validation split."})
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
            "transfer_learning": _public_transfer_context(transfer_context),
        },
    )
    result.parameters = {
        **dict(result.parameters or {}),
        "device": parameters.get("device", "auto"),
        "requested_device": device_resolution["requested_device"],
        "resolved_device": device_resolution["resolved_device"],
        "cuda_available": device_resolution["cuda_available"],
        "device_fallback_applied": device_resolution["fallback_applied"],
        "device_fallback_reason": device_resolution["fallback_reason"],
    }
    result.metadata["accelerator"] = device_resolution
    result.metadata["transfer_learning"] = _public_transfer_context(transfer_context)
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
            try:
                result.model = result.model.to("cpu")
            except Exception:
                pass
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
                "output_features": list(prepared.output_features or ([prepared.output_feature] if prepared.output_feature else [])),
                "task_type": prepared.task_type,
                "label_mapping": dict(prepared.label_mapping),
                "accelerator": dict(result.metadata.get("accelerator") or {}),
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
            actual_values = np.asarray(prepared.y_test)
            predicted_values = np.asarray(predictions)
            if actual_values.ndim > 1:
                actual_values = actual_values.reshape(-1)
                predicted_values = predicted_values.reshape(-1)
            artifact_paths.append(
                plot_predictions_vs_actual(
                    actual_values,
                    predicted_values,
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
    parameters["training_mode"] = _normalise_training_mode(parameters.get("training_mode"))
    parameters.setdefault("transfer_strategy", "auto")
    return parameters


def _run_training_workflow(
    model_record: Any,
    dataset_record: Any,
    parameters: Mapping[str, Any],
    *,
    job_id: str,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback({"stage": "preparing_dataset", "message": "Preparing dataset splits and feature encodings."})
    prepared = _prepare_dataset_for_training(dataset_record, model_record, parameters)
    requested_framework = str(parameters.get("framework", "sklearn")).strip().lower()
    framework = requested_framework
    if framework == "onnx":
        framework = str(parameters.get("training_backend") or parameters.get("base_framework") or "sklearn").strip().lower()

    transfer_context = _build_transfer_learning_context(
        model_record,
        dataset_record,
        parameters,
        framework=framework,
    )
    if requested_framework == "onnx":
        transfer_context["requested_framework"] = "onnx"
        transfer_context["backend_framework"] = framework

    if framework in {"sklearn", "scikit-learn", "scikitlearn"}:
        result, predictions = _train_with_sklearn(
            prepared,
            model_record,
            parameters,
            transfer_context=transfer_context,
            progress_callback=progress_callback,
        )
        framework = "sklearn"
    elif framework in {"pytorch", "torch"}:
        result, predictions = _train_with_pytorch(
            prepared,
            model_record,
            parameters,
            transfer_context=transfer_context,
            progress_callback=progress_callback,
        )
        framework = "pytorch"
    else:
        raise ValueError(f"Unsupported training framework: {framework}")

    result.framework = framework
    result.metadata["transfer_learning"] = _public_transfer_context(transfer_context)
    pretty_print_metrics(result.metrics)
    with resume_run(result.run_id) as tracking_active:
        if progress_callback is not None:
            progress_callback({"stage": "persisting_artifacts", "message": "Persisting artifacts, summaries, and runtime bundles."})
        artifacts = _generate_training_artifacts(
            job_id,
            result,
            prepared,
            predictions,
            log_to_mlflow=tracking_active,
        )
        if progress_callback is not None:
            progress_callback({"stage": "logging_mlflow", "message": "Logging monitoring summaries and artifact context."})
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
        "transfer_learning": _public_transfer_context(transfer_context),
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
    *,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    learning_model = await LearningModel.read(db, int(study_record.learning_model_id))
    dataset_model = await DatasetModel.read(db, int(study_record.dataset_id))
    if learning_model is None or dataset_model is None:
        raise ValueError("The selected study is not linked to a valid model and dataset.")

    base_parameters = dict(getattr(learning_model, "parameters", {}) or {})
    study_parameters = dict(getattr(study_record, "study_params", {}) or {})
    request_parameters = dict(request_payload.model_dump(exclude_none=True) or {})
    merged_parameters = {**base_parameters, **study_parameters, **request_parameters}
    if progress_callback is not None:
        progress_callback({"stage": "validating_schema", "message": "Validating the linked model and dataset schema for the study."})
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
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "optimizing",
                "message": "Running Optuna trials for the selected study.",
                "counters": {"total_trials": n_trials},
            }
        )

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
        progress_callback=progress_callback,
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
async def http_exception_handler(request_: Request, exc: HTTPException) -> JSONResponse:
    return _json_error(
        str(exc.detail),
        status_code=exc.status_code,
        exc=exc,
        request=request_,
        debug_subject=exc.detail,
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request_: Request, exc: ResponseValidationError) -> JSONResponse:
    LOGGER.exception("Response validation failed.", exc_info=exc)
    return _json_error(str(exc), status_code=500, exc=exc, request=request_)


@app.exception_handler(Exception)
async def generic_exception_handler(request_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception", exc_info=exc)
    error_detail = str(exc) or exc.__class__.__name__
    return _json_error(error_detail, status_code=500, exc=exc, request=request_)
