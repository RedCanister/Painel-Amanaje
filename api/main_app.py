from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
import traceback
import time
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from fastapi import Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_session import get_db
from app.database.db_utils import create_entry, get_all_entries, get_entry, get_entry_dependencies, update_entry
from app.models.assistant_objects import (
    AssistantApprovalRequest,
    AssistantDraftRequest,
    AssistantFeedbackRequest,
    AssistantReferenceRequest,
    AssistantReferenceSearchRequest,
    AssistantReviewRequest,
    AssistantSessionRequest,
    AssistantSubmitRequest,
    ExecutionRunRequest,
    WorkflowDraft,
)
from app.models.model_objects import AssistantModel, AssistantTrainingDatasetModel, DatasetModel, LearningModel, StudyModel
from app.models.model_orm import AssistantORM, AssistantTrainingDatasetORM, CodeORM, DatasetORM, InferenceORM, LearningORM, PanelDashboardORM, StudyORM
from app.utils.deployment_utils import save_deployment_summary
from app.utils.assistant_llmops import (
    assistant_alignment_contracts,
    assistant_reference_repository_status,
    append_assistant_training_example,
    build_assistant_reference_snapshot,
    build_assistant_training_dataset_snapshot,
    build_selected_reference_pack,
    create_assistant_reference,
    curate_assistant_training_examples,
    evaluate_draft_contract,
    list_assistant_references,
    log_assistant_mlflow_event,
    persist_assistant_context_pack,
    persist_assistant_eval_suite,
    persist_assistant_eval_result,
    persist_assistant_promotion_report,
    resolve_assistant_reference_source_paths,
    run_assistant_golden_evals,
    sync_assistant_reference_repository,
)
from app.utils.assistant_bundles import (
    assistant_bundle_status_from_model,
    inspect_assistant_model_bundle,
    inspect_assistant_model_directory,
    normalize_huggingface_assistant_directory,
    prepare_assistant_model_tokenization,
)
from app.utils.assistant_provider import (
    assistant_model_to_provider_config,
    get_assistant_provider,
    get_assistant_provider_for_model,
    get_assistant_provider_status,
    redact_assistant_provider_config,
    test_assistant_model_contract,
    test_assistant_provider_contract,
)
from app.utils.assistant_safety import SAFETY_PROFILES, review_code_safety, review_workflow_draft
from app.utils.assistant_runtime import (
    assistant_runtime_config,
    assistant_runtime_status,
    ensure_assistant_model_runtime_loaded,
    unload_assistant_runtime_model,
)
from app.utils.accelerators import get_torch_accelerator_status
from app.utils.editor_execution import execute_editor_code
from app.utils.example_catalog import build_example_catalog, find_example_object
from app.utils.io import save_json
from app.utils.log_stream import LogSourceValidationError, collect_log_snapshot
from app.utils.logging import log_to_mlflow
from app.utils.mlflow_utils import get_experiment_summary, list_experiments
from app.utils.netron_viewer import (
    NetronUnavailableError,
    build_netron_status,
    start_netron_viewer,
    stop_netron_viewer,
)
from app.utils.operation_queue import (
    QueueUnavailableError,
    cancel_rq_job,
    get_queue_runtime_snapshot,
    enqueue_assistant_operation,
    enqueue_editor_execution,
    enqueue_study_run,
    enqueue_training_run,
    get_worker_runtime_status,
)
from app.utils.settings import (
    SettingsValidationError,
    build_settings_response,
    load_settings_state,
    register_provider_token_env_var,
    update_settings_state,
)
from app.utils import store_utils
from app.utils.utils import build_payload_data, build_payload_model, get_upload_dir, recover_model_params, save_file_to_disk
from app.utils.validation import validate_input, validate_required_keys
import app.utils.main_utils as _utils

app = _utils.app
LOGGER = _utils.LOGGER
PROJECT_ROOT = _utils.PROJECT_ROOT
REPO_ROOT = _utils.REPO_ROOT
TRAINING_ARTIFACT_DIR = _utils.TRAINING_ARTIFACT_DIR
DEPLOYMENT_ARTIFACT_DIR = _utils.DEPLOYMENT_ARTIFACT_DIR
EXPORT_ARTIFACT_DIR = _utils.EXPORT_ARTIFACT_DIR
FEATURE_ARTIFACT_DIR = _utils.FEATURE_ARTIFACT_DIR
STORE_ARTIFACT_DIR = _utils.STORE_ARTIFACT_DIR
RUN_LEDGER_DIR = _utils.RUN_LEDGER_DIR
ASSISTANT_MODEL_DIR = _utils.ASSISTANT_MODEL_DIR
ASSISTANT_DATASET_DIR = _utils.ASSISTANT_DATASET_DIR
ASSISTANT_EVAL_DIR = _utils.ASSISTANT_EVAL_DIR
ASSISTANT_REFERENCE_DIR = ASSISTANT_DATASET_DIR / "references"
PRODUCTION_STATE = _utils.PRODUCTION_STATE
ACCELERATOR_STATUS_CACHE: dict[str, Any] = {"expires_at": 0.0, "payload": None}
SECRET_TRACEBACK_PATTERN = re.compile(
    r"(?i)(token|password|passwd|secret|api[_-]?key|authorization)(\s*[:=]\s*)([^\s,;]+)"
)
DATASET_OPERATION = _utils.DATASET_OPERATION
MODEL_OPERATION = _utils.MODEL_OPERATION
ASSISTANT_MODEL_OPERATION = _utils.ASSISTANT_MODEL_OPERATION
ALLOWED_UPLOAD_OPERATIONS = _utils.ALLOWED_UPLOAD_OPERATIONS
TrainingRequest = _utils.TrainingRequest
StudyOptimizationRequest = _utils.StudyOptimizationRequest
OnnxPrepareRequest = _utils.OnnxPrepareRequest
_render_page = _utils._render_page
_append_activity_log = _utils._append_activity_log
_read_activity_log = _utils._read_activity_log
_serialize_inference_summary = _utils._serialize_inference_summary
_upsert_inference_record = _utils._upsert_inference_record
_build_dataset_plots = _utils._build_dataset_plots
_build_model_plots = _utils._build_model_plots
_build_inference_plots = _utils._build_inference_plots
_build_bar_plot = _utils._build_bar_plot
_build_line_plot = _utils._build_line_plot
_list_plot_artifacts = _utils._list_plot_artifacts
_resolve_plot_artifact_path = _utils._resolve_plot_artifact_path
_find_inference_record = _utils._find_inference_record
_build_extraction_manifest = _utils._build_extraction_manifest
_build_registry_analysis = _utils._build_registry_analysis
_get_registry_binding = _utils._get_registry_binding
_json_safe = _utils._json_safe
_normalize_registry_type = _utils._normalize_registry_type
_resolve_runtime_context = _utils._resolve_runtime_context
_prepare_runtime_execution = _utils._prepare_runtime_execution
STORE_DATASET_DIR = STORE_ARTIFACT_DIR / "datasets"
STORE_MANIFEST_DIR = STORE_ARTIFACT_DIR / "manifests"
STORE_PROVIDERS_PATH = STORE_ARTIFACT_DIR / "providers.json"


def _queue_unavailable_response(
    exc: QueueUnavailableError,
    *,
    run_id: str,
    ledger: Mapping[str, Any],
) -> JSONResponse:
    return _utils._json_error(
        str(exc),
        status_code=503,
        run_id=run_id,
        ledger=ledger,
        **exc.to_payload(),
    )


ASSISTANT_BACKGROUND_OPERATIONS = {
    "draft",
    "references_sync",
    "datasets_rebuild",
    "evals_run",
    "training_curate",
    "model_runtime_load",
    "model_activate",
    "model_training_curate",
    "model_training_tokenize",
    "model_training_attach",
}


def _normalize_assistant_operation(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace("/", "_")


def _json_response_payload(response: JSONResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except Exception:
        payload = {"status": "error", "detail": "Unable to decode JSON response."}
    if response.status_code >= 400:
        detail = payload.get("detail") or payload.get("error") or payload.get("message") or f"HTTP {response.status_code}"
        raise ValueError(str(detail))
    return payload


async def _call_assistant_operation(operation: str, payload_data: Mapping[str, Any]) -> dict[str, Any]:
    from app.database.db_session import _require_sessionmaker

    payload = dict(payload_data or {})
    if operation == "references_sync":
        max_files = int(payload.get("max_files") or payload.get("maxFiles") or 100)
        return _json_response_payload(await assistant_sync_references(max_files=max(1, min(max_files, 1000))))
    if operation == "evals_run":
        provider = str(payload.get("provider") or "amanaje_slm")
        return _json_response_payload(await assistant_run_evals(provider=provider))
    if operation == "training_curate":
        return _json_response_payload(await assistant_curate_training())

    async with _require_sessionmaker()() as db:
        if operation == "draft":
            request_payload = AssistantDraftRequest.model_validate(payload.get("payload") or payload)
            return _json_response_payload(await assistant_draft(request_payload, db=db))
        if operation == "datasets_rebuild":
            return _json_response_payload(await assistant_datasets_rebuild(db=db))

        assistant_model_id = payload.get("assistant_model_id") or payload.get("assistantModelId") or payload.get("model_id")
        if assistant_model_id in (None, ""):
            raise ValueError("assistant_model_id is required for this assistant operation.")
        if operation == "model_runtime_load":
            return _json_response_payload(await assistant_model_runtime_load(assistant_model_id, db=db))
        if operation == "model_activate":
            return _json_response_payload(await assistant_model_activate(assistant_model_id, db=db))
        if operation == "model_training_curate":
            return _json_response_payload(await assistant_model_curate_training(assistant_model_id, db=db))
        if operation == "model_training_tokenize":
            return _json_response_payload(await assistant_model_tokenize_training_dataset(assistant_model_id, db=db))
        if operation == "model_training_attach":
            rebuild = bool(payload.get("rebuild", False))
            return _json_response_payload(await assistant_model_attach_training_dataset(assistant_model_id, rebuild=rebuild, db=db))

    raise ValueError(f"Unsupported assistant operation: {operation}")


async def _execute_assistant_operation_job(run_id: str, *, operation: str, payload_data: Mapping[str, Any]) -> None:
    operation = _normalize_assistant_operation(operation)
    try:
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="running",
            stage=operation,
            event_message=f"Assistant operation '{operation}' started.",
        )
        result = await _call_assistant_operation(operation, payload_data)
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="completed",
            stage="completed",
            merge={
                "result": result,
                "metrics": {
                    "assistant_operation_completed": 1.0,
                    "assistant_operation_result_size": float(len(json.dumps(result, default=str))),
                },
            },
            event_message=f"Assistant operation '{operation}' completed.",
        )
    except Exception as exc:
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage="failed",
            merge={"error": str(exc), "result": {"operation": operation, "status": "failed"}},
            event_message=f"Assistant operation '{operation}' failed: {exc}",
        )
        raise


BACKGROUND_RUN_TASKS: dict[str, asyncio.Task[Any]] = {}

APPROVED_EXECUTION_OUTPUT_FIELDS = {
    "assistant_bundle_bytes",
    "assistant_manifest",
    "bundle_bytes",
    "connection_string",
    "csv_text",
    "dataset_type",
    "description",
    "history",
    "input_features",
    "is_deployed",
    "is_tested",
    "is_trained",
    "joblib_bytes",
    "metrics",
    "model_bytes",
    "model_type",
    "name",
    "object_type",
    "onnx_bytes",
    "output_features",
    "parameters",
    "path",
    "pickle_bytes",
    "pytorch_bytes",
    "reference_data",
    "torch_bytes",
    "torchscript_bytes",
    "version",
    "zip_bytes",
}

SAFE_EXECUTION_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _prefers_gpu_queue(parameters: Mapping[str, Any]) -> bool:
    framework = str(parameters.get("framework") or "").strip().lower()
    device = str(parameters.get("device") or "auto").strip().lower()
    return framework in {"pytorch", "torch"} and device == "cuda"


def _normalize_upload_operation(operation_id: str) -> str:
    normalized = str(operation_id or "").strip().lower().replace("_", "-")
    aliases = {
        "dataset": DATASET_OPERATION,
        "data": DATASET_OPERATION,
        "model": MODEL_OPERATION,
        "learning-model": MODEL_OPERATION,
        "learningmodel": MODEL_OPERATION,
        "assistant": ASSISTANT_MODEL_OPERATION,
        "assistant-model": ASSISTANT_MODEL_OPERATION,
        "assistantmodel": ASSISTANT_MODEL_OPERATION,
        "assistant-models": ASSISTANT_MODEL_OPERATION,
        "llm": ASSISTANT_MODEL_OPERATION,
        "slm": ASSISTANT_MODEL_OPERATION,
    }
    return aliases.get(normalized, normalized)


PANEL_DEFAULT_OBJECTIVE = "Understand the current question through connected data, models, plots, simulations, metrics, and notes."
PANEL_WIDGET_KINDS = {
    "dataset",
    "dash_workspace",
    "learning_model",
    "plot",
    "study",
    "inference",
    "simulation",
    "prediction",
    "filter_control",
    "metric",
    "metadata",
    "note",
    "custom_json",
}
PANEL_WIDGET_SIZES = {"wide", "full"}


class PanelDashboardPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    description: Optional[str] = None
    objective: Optional[str] = None
    tint: str = "amanaje"
    layout: dict[str, Any] = Field(default_factory=dict)
    widgets: list[dict[str, Any]] = Field(default_factory=list)
    panel_metadata: dict[str, Any] = Field(default_factory=dict)


def _panel_widget_id(prefix: str = "widget") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _coerce_panel_dict(value: Any) -> dict[str, Any]:
    return _json_safe(value) if isinstance(value, Mapping) else {}


def _coerce_panel_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 24) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _coerce_panel_widget_size(value: Any) -> str:
    size = str(value or "").strip().lower()
    return "full" if size == "full" else "wide"


def _coerce_panel_widgets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    widgets: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        widget = dict(item)
        kind = str(widget.get("kind") or widget.get("type") or "metadata").strip().lower()
        widget["id"] = str(widget.get("id") or _panel_widget_id(f"widget_{index + 1}"))
        widget["kind"] = kind if kind in PANEL_WIDGET_KINDS else "metadata"
        widget["title"] = str(widget.get("title") or widget["kind"].replace("_", " ").title())
        widget["size"] = _coerce_panel_widget_size(widget.get("size"))
        widget["source"] = _coerce_panel_dict(widget.get("source"))
        widget["settings"] = _coerce_panel_dict(widget.get("settings"))
        widget["cache"] = _coerce_panel_dict(widget.get("cache"))
        widgets.append(_json_safe(widget))
    return widgets


def _default_panel_widgets() -> list[dict[str, Any]]:
    return [
        {
            "id": "objective_focus",
            "kind": "metadata",
            "title": "Objective",
            "size": "full",
            "source": {},
            "settings": {"mode": "objective"},
        },
        {
            "id": "dash_visualization_studio",
            "kind": "dash_workspace",
            "title": "Dash Visualization Studio",
            "size": "full",
            "source": {},
            "settings": {
                "path": "/",
                "description": "Interactive Dash workspace for plots, extensions, and panel visualizations.",
            },
        },
        {
            "id": "dataset_signal",
            "kind": "dataset",
            "title": "Dataset Signal",
            "size": "wide",
            "source": {},
            "settings": {"variant": "summary"},
        },
        {
            "id": "model_signal",
            "kind": "learning_model",
            "title": "Learning Model Signal",
            "size": "wide",
            "source": {},
            "settings": {"variant": "summary"},
        },
        {
            "id": "plot_signal",
            "kind": "plot",
            "title": "Plot Artifact",
            "size": "wide",
            "source": {},
            "settings": {"variant": "dash"},
        },
        {
            "id": "metric_stack",
            "kind": "metric",
            "title": "Metric Stack",
            "size": "wide",
            "source": {},
            "settings": {"metrics": []},
        },
        {
            "id": "interpretation_notes",
            "kind": "note",
            "title": "Interpretation Notes",
            "size": "wide",
            "source": {},
            "settings": {"text": ""},
        },
    ]


def _default_panel_layout(widgets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    resolved_widgets = widgets or _default_panel_widgets()
    return {
        "version": 1,
        "columns": 12,
        "density": "comfortable",
        "order": [str(widget.get("id")) for widget in resolved_widgets if widget.get("id")],
    }


def _coerce_panel_layout(value: Any, widgets: list[dict[str, Any]]) -> dict[str, Any]:
    layout = _coerce_panel_dict(value)
    if not layout:
        return _default_panel_layout(widgets)
    layout["version"] = _coerce_panel_int(layout.get("version"), 1, maximum=99)
    layout["columns"] = _coerce_panel_int(layout.get("columns"), 12, maximum=24)
    order = layout.get("order")
    if not isinstance(order, list):
        layout["order"] = [str(widget.get("id")) for widget in widgets if widget.get("id")]
    else:
        layout["order"] = [str(item) for item in order if item is not None]
    return _json_safe(layout)


def _next_panel_version(value: Any) -> int:
    return _coerce_panel_int(value, 1, maximum=999_999) + 1


def _panel_dashboard_to_dict(dashboard: Any) -> dict[str, Any]:
    widgets = _coerce_panel_widgets(getattr(dashboard, "widgets", None))
    layout = _coerce_panel_layout(getattr(dashboard, "layout", None), widgets)
    date_value = getattr(dashboard, "date", None)
    return {
        "id": getattr(dashboard, "id", None),
        "name": getattr(dashboard, "name", None),
        "description": getattr(dashboard, "description", None),
        "object_type": getattr(dashboard, "object_type", "panel_dashboard"),
        "size": getattr(dashboard, "size", 0.0),
        "path": getattr(dashboard, "path", None),
        "date": date_value.isoformat() if hasattr(date_value, "isoformat") else date_value,
        "version": getattr(dashboard, "version", None),
        "history": getattr(dashboard, "history", None) or [],
        "objective": getattr(dashboard, "objective", None),
        "tint": getattr(dashboard, "tint", None) or "amanaje",
        "layout": layout,
        "widgets": widgets,
        "panel_metadata": getattr(dashboard, "panel_metadata", None) or {},
    }


def _panel_summary(dashboard: Any) -> dict[str, Any]:
    payload = _panel_dashboard_to_dict(dashboard)
    return {
        "id": payload["id"],
        "name": payload["name"],
        "objective": payload["objective"],
        "tint": payload["tint"],
        "updated_at": payload["date"],
        "widget_count": len(payload["widgets"]),
    }


def _panel_compact_object(obj: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    date_value = getattr(obj, "date", None)
    payload = {
        "id": getattr(obj, "id", None),
        "name": getattr(obj, "name", None),
        "description": getattr(obj, "description", None),
        "object_type": getattr(obj, "object_type", None),
        "path": getattr(obj, "path", None),
        "date": date_value.isoformat() if hasattr(date_value, "isoformat") else date_value,
    }
    payload.update(dict(extra or {}))
    return _json_safe(payload)


def _panel_payload_for_create(payload: PanelDashboardPayload) -> dict[str, Any]:
    widgets = _coerce_panel_widgets(payload.widgets) or _default_panel_widgets()
    name = (payload.name or "Amanaje Panel").strip() or "Amanaje Panel"
    return {
        "name": name,
        "description": payload.description or "Saved Painel Amanaje objective dashboard.",
        "object_type": "panel_dashboard",
        "size": 0.0,
        "path": f"panel://{_utils._slugify(name)}-{uuid.uuid4().hex[:8]}",
        "version": 1,
        "history": [{"operation": "panel_dashboard_created", "created_at": datetime.now().isoformat()}],
        "objective": payload.objective or PANEL_DEFAULT_OBJECTIVE,
        "tint": payload.tint or "amanaje",
        "layout": _coerce_panel_layout(payload.layout, widgets),
        "widgets": widgets,
        "panel_metadata": _coerce_panel_dict(payload.panel_metadata),
    }


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request) -> HTMLResponse:
    return _render_page("base_panel.html", request, )


@app.get("/panel", response_class=HTMLResponse)
async def page_panel(request: Request) -> HTMLResponse:
    return _render_page("base_panel.html", request, )


@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request) -> HTMLResponse:
    return _render_page("base_red.html", request, )


@app.get("/create", response_class=HTMLResponse)
async def page_create(request: Request) -> HTMLResponse:
    return _render_page("base_create.html", request, )


@app.get("/feature", response_class=HTMLResponse)
async def page_feature(request: Request) -> HTMLResponse:
    return _render_page("base_feature.html", request, )


@app.get("/store", response_class=HTMLResponse)
async def page_store(request: Request) -> HTMLResponse:
    return _render_page("base_store.html", request, )


@app.get("/training", response_class=HTMLResponse)
async def page_training(request: Request) -> HTMLResponse:
    return _render_page("base_green.html", request, )


@app.get("/onnx", response_class=HTMLResponse)
async def page_onnx(request: Request) -> HTMLResponse:
    return _render_page("base_onnx.html", request, )

#
@app.get("/optimization", response_class=HTMLResponse)
async def page_optimization(request: Request) -> HTMLResponse:
    return _render_page("base_green.html", request, )


@app.get("/editor", response_class=HTMLResponse)
async def page_editor(request: Request) -> HTMLResponse:
    return _render_page("base_editor.html", request, )


@app.get("/production", response_class=HTMLResponse)
async def page_production(request: Request) -> HTMLResponse:
    return _render_page("base_blue.html", request, )


@app.get("/operations", response_class=HTMLResponse)
async def page_operations(request: Request) -> HTMLResponse:
    return _render_page("base_operations.html", request, )


@app.get("/visualization", response_class=HTMLResponse)
@app.get("/plot", response_class=HTMLResponse)
async def page_visualization(request: Request) -> HTMLResponse:
    return _render_page("base_plot.html", request, )


@app.get("/registry", response_class=HTMLResponse)
async def page_registry(request: Request) -> HTMLResponse:
    return _render_page("base_purple.html", request, )


@app.get("/assistant", response_class=HTMLResponse)
async def page_assistant(request: Request) -> HTMLResponse:
    return _render_page("base_assistant.html", request, )


@app.get("/settings", response_class=HTMLResponse)
async def page_settings(request: Request) -> HTMLResponse:
    return _render_page("base_settings.html", request, )


@app.get("/panel/context", response_class=JSONResponse)
async def panel_context(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    learning_models = await get_all_entries(db, LearningORM)
    studies = await get_all_entries(db, StudyORM)
    inferences = await get_all_entries(db, InferenceORM)
    code_models = await get_all_entries(db, CodeORM)
    plots = _list_plot_artifacts()[:80]
    runs = _utils.list_run_entries(RUN_LEDGER_DIR, limit=60)

    return _utils._json_response(
        {
            "status": "ok",
            "datasets": [
                _panel_compact_object(
                    item,
                    {
                        "dataset_type": getattr(item, "dataset_type", None),
                        "shape": getattr(item, "shape", None),
                        "features_list": getattr(item, "features_list", None),
                        "has_features": getattr(item, "has_features", None),
                    },
                )
                for item in datasets
            ],
            "learning_models": [
                _panel_compact_object(
                    item,
                    {
                        "model_type": getattr(item, "model_type", None),
                        "metrics": getattr(item, "metrics", None) or {},
                        "parameters": getattr(item, "parameters", None) or {},
                        "input_features": getattr(item, "input_features", None),
                        "output_features": getattr(item, "output_features", None),
                        "is_trained": getattr(item, "is_trained", None),
                        "is_tested": getattr(item, "is_tested", None),
                        "is_deployed": getattr(item, "is_deployed", None),
                    },
                )
                for item in learning_models
            ],
            "studies": [
                _panel_compact_object(
                    item,
                    {
                        "learning_model_id": getattr(item, "learning_model_id", None),
                        "dataset_id": getattr(item, "dataset_id", None),
                        "sampler": getattr(item, "sampler", None),
                        "objective": getattr(item, "objective", None),
                        "best_trial": getattr(item, "best_trial", None),
                        "best_params": getattr(item, "best_params", None),
                    },
                )
                for item in studies
            ],
            "inferences": [
                _panel_compact_object(
                    item,
                    {
                        "learning_model_id": getattr(item, "learning_model_id", None),
                        "dataset_id": getattr(item, "dataset_id", None),
                        "input_features": getattr(item, "input_features", None),
                        "output_features": getattr(item, "output_features", None),
                        "inference_params": getattr(item, "inference_params", None) or {},
                    },
                )
                for item in inferences
            ],
            "code_models": [
                _panel_compact_object(
                    item,
                    {
                        "variables": getattr(item, "variables", None) or {},
                        "code": getattr(item, "code", None) or {},
                    },
                )
                for item in code_models
            ],
            "plots": _json_safe(plots),
            "runs": _json_safe(runs),
            "widget_kinds": sorted(PANEL_WIDGET_KINDS),
            "widget_sizes": sorted(PANEL_WIDGET_SIZES),
        }
    )


@app.get("/panel/dashboards", response_class=JSONResponse)
async def panel_list_dashboards(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    dashboards = await get_all_entries(db, PanelDashboardORM)
    dashboards = sorted(dashboards, key=lambda item: getattr(item, "date", datetime.min) or datetime.min, reverse=True)
    return _utils._json_response(
        {
            "status": "ok",
            "dashboards": [_panel_summary(item) for item in dashboards],
        }
    )


@app.post("/panel/dashboards", response_class=JSONResponse)
async def panel_create_dashboard(
    payload: PanelDashboardPayload | None = Body(default=None),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    payload = payload or PanelDashboardPayload()
    dashboard = PanelDashboardORM(**_panel_payload_for_create(payload))
    db.add(dashboard)
    await db.commit()
    await db.refresh(dashboard)
    return _utils._json_response({"status": "created", "dashboard": _panel_dashboard_to_dict(dashboard)}, status_code=201)


@app.get("/panel/dashboards/{dashboard_id}", response_class=JSONResponse)
async def panel_get_dashboard(dashboard_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    dashboard = await get_entry(db, PanelDashboardORM, dashboard_id)
    if not dashboard:
        raise HTTPException(status_code=404, detail="Panel dashboard not found")
    return _utils._json_response({"status": "ok", "dashboard": _panel_dashboard_to_dict(dashboard)})


@app.put("/panel/dashboards/{dashboard_id}", response_class=JSONResponse)
async def panel_update_dashboard(
    dashboard_id: int | str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    dashboard = await get_entry(db, PanelDashboardORM, dashboard_id)
    if not dashboard:
        raise HTTPException(status_code=404, detail="Panel dashboard not found")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, Mapping):
        return _utils._json_error("Panel dashboard payload must be a JSON object.", status_code=400)

    widgets = _coerce_panel_widgets(payload.get("widgets")) if "widgets" in payload else _coerce_panel_widgets(getattr(dashboard, "widgets", None))
    if "name" in payload:
        dashboard.name = str(payload.get("name") or dashboard.name or "Amanaje Panel")
    if "description" in payload:
        dashboard.description = str(payload.get("description") or "")
    if "objective" in payload:
        dashboard.objective = str(payload.get("objective") or "")
    if "tint" in payload:
        dashboard.tint = str(payload.get("tint") or "amanaje")
    if "widgets" in payload:
        dashboard.widgets = widgets
    if "layout" in payload:
        dashboard.layout = _coerce_panel_layout(payload.get("layout"), widgets)
    elif "widgets" in payload:
        dashboard.layout = _coerce_panel_layout(getattr(dashboard, "layout", None), widgets)
    if "panel_metadata" in payload:
        dashboard.panel_metadata = _coerce_panel_dict(payload.get("panel_metadata"))

    dashboard.size = 0.0
    dashboard.object_type = "panel_dashboard"
    dashboard.version = _next_panel_version(getattr(dashboard, "version", None))
    dashboard.history = _utils._append_history(
        getattr(dashboard, "history", None),
        {"operation": "panel_dashboard_updated", "updated_at": datetime.now().isoformat()},
    )

    await db.commit()
    await db.refresh(dashboard)
    return _utils._json_response({"status": "ok", "dashboard": _panel_dashboard_to_dict(dashboard)})


@app.delete("/panel/dashboards/{dashboard_id}", response_class=JSONResponse)
async def panel_delete_dashboard(dashboard_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    dashboard = await get_entry(db, PanelDashboardORM, dashboard_id)
    if not dashboard:
        raise HTTPException(status_code=404, detail="Panel dashboard not found")
    deleted_id = getattr(dashboard, "id", dashboard_id)
    await db.delete(dashboard)
    await db.commit()
    return _utils._json_response({"status": "deleted", "id": deleted_id})


def _sync_store_provider_token_env_vars() -> list[str]:
    registered: list[str] = []
    try:
        providers = store_utils.list_providers(STORE_PROVIDERS_PATH)
    except Exception:
        LOGGER.warning("Unable to load Store providers while syncing Settings environment keys.", exc_info=True)
        return registered

    for provider in providers:
        token_env_var = str(provider.get("token_env_var") or "").strip()
        if not token_env_var:
            continue
        registered.append(register_provider_token_env_var(token_env_var, provider_name=str(provider.get("name") or "")))
    return registered


def _settings_config_payload() -> dict[str, Any]:
    _sync_store_provider_token_env_vars()
    state = _utils.apply_runtime_settings_state()
    return build_settings_response(
        _utils.RUNTIME_CONFIG,
        state=state,
        path=_utils.SETTINGS_STATE_PATH,
    )


@app.get("/settings/config", response_class=JSONResponse)
async def settings_config() -> JSONResponse:
    return _utils._json_response(_settings_config_payload())


@app.post("/settings/config", response_class=JSONResponse)
async def settings_update_config(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
) -> JSONResponse:
    try:
        _sync_store_provider_token_env_vars()
        state = update_settings_state(payload or {}, _utils.SETTINGS_STATE_PATH)
        _utils.apply_runtime_settings_state(state)
        return _utils._json_response(
            build_settings_response(
                _utils.RUNTIME_CONFIG,
                state=state,
                path=_utils.SETTINGS_STATE_PATH,
            )
        )
    except SettingsValidationError as exc:
        return _utils._json_error(str(exc), status_code=400, exc=exc, request=request, errors=exc.errors)


@app.get("/settings/logs", response_class=JSONResponse)
async def settings_logs(
    request: Request,
    source: str = "all",
    tail: int = 160,
    include_docker: bool = True,
) -> JSONResponse:
    try:
        return _utils._json_response(
            collect_log_snapshot(
                log_dir=_utils.LOG_DIR,
                activity_log_path=_utils.ACTIVITY_LOG_PATH,
                compose_file=_utils.REPO_ROOT / "docker-compose.yaml",
                source=source,
                tail=tail,
                include_docker=include_docker,
            )
        )
    except LogSourceValidationError as exc:
        return _utils._json_error(
            str(exc),
            status_code=400,
            exc=exc,
            request=request,
            errors=[{"field": "source", "reason": "unknown_log_source", "value": exc.source}],
        )


def _store_error_response(exc: Exception, request: Request | None = None) -> JSONResponse:
    if isinstance(exc, store_utils.StoreValidationError):
        return _utils._json_error(str(exc), status_code=exc.status_code, exc=exc, request=request, errors=exc.errors)
    LOGGER.exception("Store operation failed.")
    return _utils._json_error(str(exc), status_code=500, exc=exc, request=request)


def _store_live_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    live_payload = payload.get("live") if isinstance(payload.get("live"), Mapping) else {}
    raw_live = payload.get("live")
    live_enabled = bool(
        live_payload.get("enabled")
        or (raw_live if not isinstance(raw_live, Mapping) else False)
        or payload.get("isLive")
        or payload.get("live_dataset")
    )
    interval_value = (
        live_payload.get("interval_seconds")
        or live_payload.get("intervalSeconds")
        or payload.get("refreshIntervalSeconds")
        or payload.get("refresh_interval_seconds")
        or 86400
    )
    try:
        interval_seconds = max(int(interval_value), 60)
    except (TypeError, ValueError):
        interval_seconds = 86400
    return {
        "enabled": live_enabled,
        "interval_seconds": interval_seconds,
        "mode": "manual_refresh",
        "status": "pending_validation" if live_enabled else "snapshot",
    }


def _store_dataset_summary(dataset: Any) -> dict[str, Any]:
    summary = _utils._serialize_dataset_summary(dataset)
    summary["connection_string"] = getattr(dataset, "connection_string", None)
    summary["history"] = getattr(dataset, "history", None)
    return summary


@app.get("/store/providers", response_class=JSONResponse)
async def store_providers() -> JSONResponse:
    try:
        providers = store_utils.list_providers(STORE_PROVIDERS_PATH)
        default_provider = next((provider for provider in providers if provider.get("is_default")), None)
        return _utils._json_response(
            {
                "status": "ok",
                "providers": providers,
                "default_provider_id": (default_provider or {}).get("id"),
            }
        )
    except Exception as exc:
        return _store_error_response(exc)


@app.post("/store/providers", response_class=JSONResponse)
async def store_save_provider(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        provider = store_utils.save_provider(payload if isinstance(payload, Mapping) else {}, STORE_PROVIDERS_PATH)
        registered_env_var = None
        token_env_var = str(provider.get("token_env_var") or "").strip()
        if token_env_var:
            registered_env_var = register_provider_token_env_var(token_env_var, provider_name=str(provider.get("name") or ""))
        return _utils._json_response({"status": "ok", "provider": provider, "settings_env_registered": registered_env_var})
    except Exception as exc:
        return _store_error_response(exc, request)


@app.post("/store/providers/{provider_id}/default", response_class=JSONResponse)
async def store_set_default_provider(provider_id: str, request: Request) -> JSONResponse:
    try:
        provider = store_utils.set_default_provider(provider_id, STORE_PROVIDERS_PATH)
        return _utils._json_response({"status": "ok", "provider": provider, "default_provider_id": provider.get("id")})
    except Exception as exc:
        return _store_error_response(exc, request)


@app.delete("/store/providers/{provider_id}", response_class=JSONResponse)
async def store_delete_provider(provider_id: str, request: Request) -> JSONResponse:
    try:
        deleted = store_utils.delete_provider(provider_id, STORE_PROVIDERS_PATH)
        providers = store_utils.list_providers(STORE_PROVIDERS_PATH)
        default_provider = next((provider for provider in providers if provider.get("is_default")), None)
        return _utils._json_response(
            {
                "status": "deleted",
                "provider": deleted,
                "providers": providers,
                "default_provider_id": (default_provider or {}).get("id"),
            }
        )
    except Exception as exc:
        return _store_error_response(exc, request)


@app.post("/store/providers/{provider_id}/test", response_class=JSONResponse)
async def store_test_provider(provider_id: str, request: Request) -> JSONResponse:
    try:
        provider = store_utils.resolve_provider(provider_id, STORE_PROVIDERS_PATH)
        catalog = await store_utils.list_catalog(provider, page=1)
        detail_checked = False
        first_item = next((item for item in catalog.get("items") or [] if item.get("external_id")), None)
        if first_item is not None:
            await store_utils.get_dataset_detail(provider, str(first_item["external_id"]))
            detail_checked = True
        return _utils._json_response(
            {
                "status": "ok",
                "provider": catalog.get("provider"),
                "sample_count": len(catalog.get("items") or []),
                "detail_checked": detail_checked,
                "message": "Provider connection validated.",
            }
        )
    except Exception as exc:
        return _store_error_response(exc, request)


@app.get("/store/catalog", response_class=JSONResponse)
async def store_catalog(
    provider_id: str,
    query: str = "",
    category: str = "",
    page: int = 1,
) -> JSONResponse:
    try:
        provider = store_utils.resolve_provider(provider_id, STORE_PROVIDERS_PATH)
        catalog = await store_utils.list_catalog(provider, query=query, category=category, page=page)
        return _utils._json_response({"status": "ok", **catalog})
    except Exception as exc:
        return _store_error_response(exc)


@app.get("/store/catalog/{external_id}", response_class=JSONResponse)
async def store_catalog_detail(external_id: str, provider_id: str) -> JSONResponse:
    try:
        provider = store_utils.resolve_provider(provider_id, STORE_PROVIDERS_PATH)
        detail = await store_utils.get_dataset_detail(provider, external_id)
        return _utils._json_response({"status": "ok", "provider": store_utils.redact_provider(provider), "dataset": detail})
    except Exception as exc:
        return _store_error_response(exc)


@app.post("/store/preview", response_class=JSONResponse)
async def store_preview(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        provider = store_utils.resolve_provider(str(payload.get("provider_id") or payload.get("providerId") or ""), STORE_PROVIDERS_PATH)
        result = await store_utils.preview_resource(
            provider,
            str(payload.get("dataset_external_id") or payload.get("datasetExternalId") or ""),
            str(payload.get("resource_external_id") or payload.get("resourceExternalId") or ""),
        )
        return _utils._json_response(result)
    except Exception as exc:
        return _store_error_response(exc, request)


@app.post("/store/materialize", response_class=JSONResponse)
async def store_materialize(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, Mapping):
            raise store_utils.StoreValidationError("Store materialize payload must be an object.")
        provider_id = str(payload.get("provider_id") or payload.get("providerId") or "")
        dataset_external_id = str(payload.get("dataset_external_id") or payload.get("datasetExternalId") or "")
        resource_external_id = str(payload.get("resource_external_id") or payload.get("resourceExternalId") or "")
        provider = store_utils.resolve_provider(provider_id, STORE_PROVIDERS_PATH)
        live = _store_live_metadata(payload)
        validation_result = None
        if live["enabled"]:
            validation_result = await store_utils.validate_resource_connection(provider, dataset_external_id, resource_external_id)
            live = {**live, "status": "validated", "validated_at": validation_result.get("validated_at")}

        dataset_name = str(payload.get("name") or payload.get("datasetName") or "store_dataset").strip() or "store_dataset"
        materialized = await store_utils.materialize_resource_files(
            provider,
            dataset_external_id,
            resource_external_id,
            dataset_name=dataset_name,
            dataset_dir=STORE_DATASET_DIR,
        )
        output_path = Path(materialized["output_path"])
        dataframe = materialized["dataframe"]
        history_entry = {
            "operation": "store_materialize",
            "when": datetime.now().isoformat(),
            "provider_id": provider_id,
            "dataset_external_id": dataset_external_id,
            "resource_external_id": resource_external_id,
            "resource_title": materialized["resource"].get("title"),
            "live": live,
        }
        dataset_payload = {
            "name": dataset_name,
            "description": str(payload.get("description") or materialized["dataset"].get("description") or "Dataset materialized from Store workspace."),
            "object_type": "dataset",
            "size": round(output_path.stat().st_size / (1024 * 1024), 4),
            "path": str(output_path),
            "date": datetime.now(),
            "version": int(payload.get("version") or 1),
            "history": [history_entry],
            "dataset_type": str(payload.get("dataset_type") or payload.get("datasetType") or "dataset"),
            "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "has_features": bool(len(dataframe.columns)),
            "features_list": [str(column) for column in dataframe.columns],
            "connection_string": materialized["connection_string"],
        }
        created_dataset = await create_entry(db, DatasetORM, dataset_payload)
        manifest = store_utils.build_store_manifest(
            dataset_id=int(getattr(created_dataset, "id")),
            dataset_name=dataset_name,
            output_path=output_path,
            connection_string=materialized["connection_string"],
            provider=provider,
            external_dataset=materialized["dataset"],
            resource=materialized["resource"],
            parser_report=materialized["parser_report"],
            live=live,
            manifest_dir=STORE_MANIFEST_DIR,
        )
        updated_history = _utils._append_history(
            getattr(created_dataset, "history", None),
            {
                "operation": "store_manifest_created",
                "when": datetime.now().isoformat(),
                "manifest_path": manifest.get("manifest_path"),
                "connection_string": materialized["connection_string"],
            },
        )
        created_dataset = await update_entry(db, DatasetORM, getattr(created_dataset, "id"), {"history": updated_history})
        return _utils._json_response(
            {
                "status": "ok",
                "dataset": _store_dataset_summary(created_dataset),
                "manifest": manifest,
                "preview_rows": materialized["preview_rows"],
                "columns": materialized["columns"],
                "parser_report": materialized["parser_report"],
                "live_validation": validation_result,
            }
        )
    except Exception as exc:
        return _store_error_response(exc, request)


@app.post("/store/datasets/{dataset_id}/validate-live", response_class=JSONResponse)
async def store_validate_live_dataset(dataset_id: int, request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None:
            raise store_utils.StoreValidationError("Dataset not found.", status_code=404)
        connection = store_utils.parse_connection_string(str(getattr(dataset, "connection_string", "") or ""))
        provider = store_utils.resolve_provider(connection["provider_id"], STORE_PROVIDERS_PATH)
        result = await store_utils.validate_resource_connection(
            provider,
            connection["dataset_external_id"],
            connection["resource_external_id"],
        )
        return _utils._json_response({"status": "ok", "dataset": _store_dataset_summary(dataset), "validation": result})
    except Exception as exc:
        return _store_error_response(exc, request)


@app.post("/store/datasets/{dataset_id}/refresh", response_class=JSONResponse)
async def store_refresh_dataset(dataset_id: int, request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None:
            raise store_utils.StoreValidationError("Dataset not found.", status_code=404)
        connection = store_utils.parse_connection_string(str(getattr(dataset, "connection_string", "") or ""))
        provider = store_utils.resolve_provider(connection["provider_id"], STORE_PROVIDERS_PATH)
        output_path = _utils._resolve_fs_path(getattr(dataset, "path", None), default_parent=_utils.REPO_ROOT)
        if output_path is None:
            output_path = STORE_DATASET_DIR / f"dataset_{dataset_id}.csv"
        refreshed = await store_utils.refresh_resource_file(
            provider,
            connection["dataset_external_id"],
            connection["resource_external_id"],
            output_path=Path(output_path),
        )
        dataframe = refreshed["dataframe"]
        manifest = {}
        try:
            manifest = store_utils.load_store_manifest(dataset, STORE_MANIFEST_DIR)
        except store_utils.StoreValidationError:
            pass
        manifest.update(
            {
                "dataset_id": dataset_id,
                "dataset_name": getattr(dataset, "name", None),
                "snapshot_path": str(output_path),
                "connection_string": getattr(dataset, "connection_string", None),
                "provider": refreshed["provider"],
                "external_dataset": refreshed["dataset"],
                "resource": refreshed["resource"],
                "parser_report": refreshed["parser_report"],
                "updated_at": datetime.now().isoformat(),
            }
        )
        manifest_path = STORE_MANIFEST_DIR / f"dataset_{dataset_id}_store_manifest.json"
        manifest["manifest_path"] = str(manifest_path)
        save_json(manifest, manifest_path)
        history_entry = {
            "operation": "store_refresh",
            "when": datetime.now().isoformat(),
            "provider_id": connection["provider_id"],
            "dataset_external_id": connection["dataset_external_id"],
            "resource_external_id": connection["resource_external_id"],
            "manifest_path": str(manifest_path),
        }
        updated = await update_entry(
            db,
            DatasetORM,
            dataset_id,
            {
                "size": round(Path(output_path).stat().st_size / (1024 * 1024), 4),
                "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
                "has_features": bool(len(dataframe.columns)),
                "features_list": [str(column) for column in dataframe.columns],
                "history": _utils._append_history(getattr(dataset, "history", None), history_entry),
            },
        )
        return _utils._json_response(
            {
                "status": "ok",
                "dataset": _store_dataset_summary(updated),
                "manifest": manifest,
                "preview_rows": refreshed["preview_rows"],
                "columns": refreshed["columns"],
                "parser_report": refreshed["parser_report"],
            }
        )
    except Exception as exc:
        return _store_error_response(exc, request)


@app.get("/upload/support", response_class=JSONResponse)
async def upload_support() -> JSONResponse:
    return _utils._json_response(
        {
            "datasets": _utils.get_dataset_capability_matrix(),
            "models": _utils.get_model_capability_matrix(),
            "assistant_models": {
                ".zip": {
                    "label": "AssistantModel Bundle",
                    "register": True,
                    "inspect": True,
                    "train": "prepare-tokenized-jsonl",
                    "predict": False,
                    "simulate": "external-server",
                    "monitor": True,
                    "notes": [
                        "Requires assistant_model_manifest.json, PyTorch/HF weights, tokenizer assets, and prompt template metadata.",
                        "Generation is served by a separate OpenAI-compatible assistant server.",
                    ],
                }
            },
        }
    )


@app.get("/runtime/accelerators", response_class=JSONResponse)
async def runtime_accelerators() -> JSONResponse:
    return _utils._json_response(_cached_torch_accelerator_status())


@app.get("/runtime/workers", response_class=JSONResponse)
async def runtime_workers() -> JSONResponse:
    return _utils._json_response(get_worker_runtime_status())


def _cached_torch_accelerator_status() -> dict[str, Any]:
    ttl = float(os.getenv("AMANAJE_ACCELERATOR_STATUS_TTL_SECONDS", "15") or "15")
    now = time.monotonic()
    source_id = id(get_torch_accelerator_status)
    cached_payload = ACCELERATOR_STATUS_CACHE.get("payload")
    if (
        cached_payload is not None
        and ACCELERATOR_STATUS_CACHE.get("source_id") == source_id
        and now < float(ACCELERATOR_STATUS_CACHE.get("expires_at") or 0)
    ):
        return dict(cached_payload)
    payload = get_torch_accelerator_status()
    ACCELERATOR_STATUS_CACHE.update({"payload": payload, "expires_at": now + ttl, "source_id": source_id})
    return dict(payload)


@app.get("/examples/catalog", response_class=JSONResponse)
async def examples_catalog(include_payloads: bool = True) -> JSONResponse:
    return _utils._json_response(
        build_example_catalog(routes=app.routes, include_payloads=include_payloads)
    )


@app.get("/examples/catalog/{object_name}", response_class=JSONResponse)
async def examples_catalog_object(object_name: str, include_payloads: bool = True) -> JSONResponse:
    entry = find_example_object(object_name, routes=app.routes, include_payloads=include_payloads)
    if entry is None:
        return _utils._json_error(f"Example object '{object_name}' was not found.", status_code=404)
    return _utils._json_response({"status": "ok", "object": entry})


@app.post("/upload/{operation_id}", response_class=JSONResponse)
async def post_upload(
    operation_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    form = await request.form()
    upload_file = form.get("file")
    normalized_operation = _normalize_upload_operation(str(form.get("operationId") or operation_id))

    if normalized_operation not in ALLOWED_UPLOAD_OPERATIONS:
        return _utils._json_error(
            f"operation_id must be one of: {', '.join(sorted(ALLOWED_UPLOAD_OPERATIONS))}",
            status_code=400,
        )
    if upload_file is None:
        return _utils._json_error("no file provided", status_code=400)

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
            "object_type": "learning_model" if normalized_operation in {MODEL_OPERATION, ASSISTANT_MODEL_OPERATION} else "dataset",
            "size": size_mb,
            "path": str(destination_path),
            "date": now,
            "version": version,
            "history": history,
        }

        assistant_bundle: dict[str, Any] | None = None
        if normalized_operation == ASSISTANT_MODEL_OPERATION:
            extract_slug = "".join(character.lower() if character.isalnum() else "_" for character in str(object_name))[:64]
            extract_slug = "_".join(part for part in extract_slug.split("_") if part) or "assistant_model"
            extract_dir = ASSISTANT_MODEL_DIR / "bundles" / f"{extract_slug}_{uuid.uuid4().hex[:8]}"
            assistant_bundle = inspect_assistant_model_bundle(destination_path, extract_dir=extract_dir)
            payload = build_payload_model(common_payload, form)
            payload["model_type"] = "assistant_model"
            parameters = dict(payload.get("parameters", {}) or {})
            assistant_config = dict(parameters.get("assistant") or {})
            assistant_config.update(assistant_bundle["assistant_parameters"])
            assistant_config.setdefault("provider_type", "openai_compatible")
            assistant_config.setdefault("runtime_kind", "pytorch_hf_server")
            parameters["assistant"] = assistant_config
            parameters["artifact_manifest"] = {
                "artifact_format": "assistant_model_bundle",
                "framework": "pytorch_hf",
                "loader": "external_openai_compatible_assistant_server",
                "load_runtime": False,
                "runtime_kind": assistant_config.get("runtime_kind"),
                "bundle": assistant_bundle,
                "runtime_capabilities": {
                    "register": True,
                    "inspect": True,
                    "train": False,
                    "predict": False,
                    "generate": True,
                    "requires_assistant_server": True,
                },
            }
            payload["parameters"] = parameters
            payload["metrics"] = {
                **dict(payload.get("metrics", {}) or {}),
                "bundle_valid": True,
                "server_health": "not_checked",
            }
            payload["reference_data"] = payload.get("reference_data") or "runtime_artifacts/assistant_datasets/interactions.jsonl"
            payload["input_features"] = payload.get("input_features") or ["prompt", "context_pack", "target_type"]
            payload["output_features"] = payload.get("output_features") or ["workflow_draft"]
            payload["history"] = _utils._append_history(
                history,
                {
                    "operation": "assistant_model_bundle_uploaded",
                    "when": now.isoformat(),
                    "bundle_sha256": assistant_bundle.get("bundle_sha256"),
                    "runtime_kind": assistant_config.get("runtime_kind"),
                    "model_name": assistant_config.get("model_name"),
                    "model_version": assistant_config.get("model_version"),
                    "extracted_dir": assistant_config.get("extracted_dir"),
                },
            )
            orm_model = AssistantORM
        elif normalized_operation == MODEL_OPERATION:
            payload = build_payload_model(common_payload, form)
            artifact_manifest = recover_model_params(destination_path)
            payload["parameters"] = {
                **dict(payload.get("parameters", {}) or {}),
                "artifact_manifest": artifact_manifest,
            }
            orm_model = LearningORM
        else:
            payload, dataframe, parser_report = await build_payload_data(common_payload, form, upload_file)
            payload["history"] = _utils._append_history(
                history,
                {
                    "operation": "profiled",
                    "when": now.isoformat(),
                    "parser_report": parser_report,
                },
            )
            save_json(
                _utils._dataset_analysis_summary(dataframe, str(destination_path)),
                TRAINING_ARTIFACT_DIR / f"dataset_{object_name}_profile.json",
            )
            orm_model = DatasetORM

        created_object = await create_entry(db, orm_model, payload)
        response_payload = {
            "status": "ok",
            "id": getattr(created_object, "id", None),
            "path": str(destination_path),
            "size": size_mb,
            "name": object_name,
        }
        if normalized_operation == DATASET_OPERATION:
            response_payload["shape"] = payload.get("shape")
            response_payload["features_list"] = payload.get("features_list")
            response_payload["parser_report"] = parser_report
            response_payload["column_explorer"] = _utils.build_column_explorer(dataframe)
            response_payload["support_matrix"] = _utils.get_dataset_capability_matrix()
        else:
            response_payload["artifact_manifest"] = payload["parameters"].get("artifact_manifest", {})
            response_payload["runtime_capabilities"] = (
                payload["parameters"].get("artifact_manifest", {}).get("runtime_capabilities", {})
            )
            if normalized_operation == ASSISTANT_MODEL_OPERATION:
                response_payload["assistant_bundle"] = assistant_bundle or {}
                response_payload["bundle_status"] = assistant_bundle.get("status") if assistant_bundle else None
                response_payload["provider_config"] = redact_assistant_provider_config(
                    assistant_model_to_provider_config(created_object)
                )
        _append_activity_log(
            f"Upload completed for operation={normalized_operation} name={object_name}.",
            event_type="registry.upload",
            details=response_payload,
        )
        return _utils._json_response(response_payload)
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Upload failed for operation '%s'.", normalized_operation)
        return _utils._json_error(str(exc), status_code=500)


@app.post("/execute", response_class=JSONResponse)
async def post_execute(request: Request) -> JSONResponse:
    try:
        body = _utils.ExecuteRequest.model_validate(await request.json())
        code = body.code.strip()
    except Exception:
        raw = await request.body()
        code = raw.decode("utf-8").strip() if raw else ""

    if not code:
        return _utils._json_response(
            {"status": "error", "variables": {}, "stdout": "", "stderr": "", "error": "No code provided"},
            status_code=400,
        )

    return _utils._json_response(execute_editor_code(code))


def _coerce_registry_context_ids(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    raw_items = value if isinstance(value, list) else [value]
    ids: list[int] = []
    for item in raw_items:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed not in ids:
            ids.append(parsed)
    return ids[:24]


async def _build_editor_registry_context(
    db: AsyncSession,
    requested_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    requested = dict(requested_context or {})
    dataset_ids = _coerce_registry_context_ids(requested.get("dataset_ids") or requested.get("datasetIds"))
    model_ids = _coerce_registry_context_ids(requested.get("model_ids") or requested.get("modelIds"))
    datasets: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        record = await DatasetModel.read(db, dataset_id)
        if record is not None:
            datasets.append(_utils._serialize_dataset_summary(record))
    for model_id in model_ids:
        record = await LearningModel.read(db, model_id)
        if record is not None:
            models.append(_utils._serialize_model_summary(record))
    return {
        "datasets": datasets,
        "models": models,
        "summary": {
            "dataset_count": len(datasets),
            "model_count": len(models),
            "dataset_ids": [item.get("id") for item in datasets],
            "model_ids": [item.get("id") for item in models],
        },
    }


@app.get("/editor/context", response_class=JSONResponse)
async def editor_context(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    models = await get_all_entries(db, LearningORM)
    return _utils._json_response(
        {
            "status": "ok",
            "datasets": [_utils._serialize_dataset_summary(dataset) for dataset in datasets],
            "models": [_utils._serialize_model_summary(model) for model in models],
        }
    )


@app.post("/execute/jobs", response_class=JSONResponse)
async def post_execute_job(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    registry_context_request: Mapping[str, Any] = {}
    try:
        payload = await request.json()
        body = _utils.ExecuteRequest.model_validate(payload)
        code = body.code.strip()
        registry_context_request = body.registryContext or {}
    except Exception:
        raw = await request.body()
        code = raw.decode("utf-8").strip() if raw else ""

    if not code:
        return _utils._json_error("No code provided", status_code=400)

    registry_context = await _build_editor_registry_context(db, registry_context_request)
    run_entry = _utils.create_run_entry(
        RUN_LEDGER_DIR,
        run_type="editor_execution",
        context={"source": "editor", "gpu_allowed": False, **registry_context["summary"]},
        parameters={"code_length": len(code), "registry_context": registry_context["summary"]},
        status="queued",
    )
    try:
        try:
            enqueue_result = enqueue_editor_execution(
                run_entry["run_id"],
                code=code,
                registry_context=registry_context,
            )
        except TypeError as exc:
            if "registry_context" not in str(exc):
                raise
            enqueue_result = enqueue_editor_execution(run_entry["run_id"], code=code)
    except QueueUnavailableError as exc:
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_entry["run_id"],
            status="failed",
            stage="failed",
            merge={"error": str(exc), "error_code": exc.error_code},
            event_message=str(exc),
        )
        return _queue_unavailable_response(exc, run_id=run_entry["run_id"], ledger=run_entry)

    _utils.update_run_entry(
        RUN_LEDGER_DIR,
        run_entry["run_id"],
        merge={
            "queue": enqueue_result,
            "rerun_payload": {
                "run_type": "editor_execution",
                "code": code,
                "queue_payload": {
                    "code": code,
                    "registry_context": registry_context,
                },
            },
        },
        event_message=f"Editor execution enqueued on {enqueue_result.get('queue')}.",
    )
    return _utils._json_response(
        {
            "status": "accepted",
            "run_id": run_entry["run_id"],
            "job_id": run_entry["run_id"],
            "ledger": {**run_entry, "queue": enqueue_result},
            "queue": enqueue_result,
        },
        status_code=202,
    )

def _infer_generate_target_type(operation_id: str, prompt: str) -> str:
    prompt_text = f"{operation_id} {prompt}".lower()
    if operation_id == MODEL_OPERATION or any(
        token in prompt_text for token in ("model", "onnx", "train", "sklearn", "torch", "joblib")
    ):
        return "model_generation"
    if any(token in prompt_text for token in ("feature", "column", "transform", "fill missing", "rename")):
        return "feature_operations"
    if any(token in prompt_text for token in ("study", "optuna", "hyperparameter", "optimization")):
        return "study"
    return "dataset_generation"


def _legacy_generate_fallback_script(prompt: str, target_type: str) -> str:
    prompt_comment = "\n".join(f"# {line}" for line in prompt.splitlines() if line.strip())
    if target_type == "model_generation":
        return textwrap.dedent(
            f"""
            # Generated model template
            {prompt_comment}

            import base64
            import io
            import joblib
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)

            name = "generated_model"
            description = "Generated from the editor prompt"
            object_type = "learning_model"
            path = "generated/generated_model.joblib"
            version = 1
            model_type = "supervised_model"
            parameters = {{"framework": "sklearn", "estimator_class": "RandomForestRegressor"}}
            metrics = {{"task": "regression"}}
            reference_data = "sample_dataset.csv"
            input_features = ["input"]
            output_features = ["target"]
            is_trained = False
            is_tested = False
            is_deployed = False
            joblib_bytes = base64.b64encode(buffer.read()).decode("utf-8")
            """
        ).strip()

    return textwrap.dedent(
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


@app.post("/generate", response_class=JSONResponse)
async def post_generate(request: Request) -> JSONResponse:
    try:
        payload = validate_input(await request.json(), _utils.GenerateRequest)
    except Exception as exc:
        return _utils._json_error(f"Invalid JSON payload: {exc}", status_code=400)

    prompt = payload.prompt.strip()
    operation_id = payload.operationId.lower()
    if not prompt:
        return _utils._json_error("prompt is required", status_code=400)

    target_type = _infer_generate_target_type(operation_id, prompt)
    reference_context = _assistant_reference_context()
    draft_request = AssistantDraftRequest(
        prompt=prompt,
        workflow_goal="Generate a starter draft for the editor/create workflow.",
        target_type=target_type,
        provider="local",
        context={
            "operationId": operation_id,
            "source": "legacy_generate",
            "assistant_reference_snapshot": reference_context,
        },
        constraints={
            "preserve_legacy_response_shape": True,
            "project": "painel-amanaje",
        },
    )
    provider = get_assistant_provider("local")
    draft = provider.create_draft(draft_request)
    review = review_workflow_draft(draft)
    script = draft.code or _legacy_generate_fallback_script(prompt, target_type)

    assistant_plan = {
        "prompt": prompt,
        "operation_id": operation_id,
        "target_type": target_type,
        "draft_id": draft.draft_id,
        "draft_title": draft.title,
        "summary": draft.summary,
        "review_status": review.status,
        "reference_snapshot": {
            "reference_count": len(reference_context.get("references", []) or []),
            "snapshot_types": sorted(reference_context.keys()),
        },
        "next_actions": draft.next_actions,
        "pages": {
            "upload": {
                "goal": "Turn a natural-language request into dataset metadata, column expectations, and upload notes.",
                "fills": ["objectName", "description", "connectionString", "featuresList"],
            },
            "training": {
                "goal": "Generate runtime parameters, feature selections, and study search-space suggestions from the selected model and dataset.",
                "fills": ["parametersJson", "inputFeatures", "outputFeatures", "studyParamsJson"],
            },
            "editor": {
                "goal": "Draft starter scripts, code comments, and variable templates that match the requested workflow.",
                "fills": ["code", "variables", "metadata"],
            },
            "production": {
                "goal": "Propose monitoring thresholds, simulation presets, and inference narratives for the selected deployed model.",
                "fills": ["monitoring tolerance", "simulation controls", "deployment notes"],
            },
            "registry": {
                "goal": "Generate clean descriptions, storage paths, and structured JSON payloads before the object is saved.",
                "fills": ["description", "objectPath", "parameters", "metrics", "studyParams", "inferenceParams"],
            },
        },
    }

    return _utils._json_response(
        {
            "script": script,
            "assistant_plan": assistant_plan,
            "draft": draft,
            "review": review,
            "references": reference_context,
        }
    )


@app.post("/assistant/sessions", response_class=JSONResponse)
async def assistant_sessions(payload: AssistantSessionRequest = Body(default=AssistantSessionRequest())) -> JSONResponse:
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    run_entry = _utils.create_run_entry(
        RUN_LEDGER_DIR,
        run_type="assistant",
        context={
            "session_id": session_id,
            "title": payload.title,
            "user_id": payload.user_id,
            **dict(payload.context or {}),
        },
        parameters={"scope": "session"},
        status="running",
    )
    run_entry = _utils.update_run_entry(
        RUN_LEDGER_DIR,
        run_entry["run_id"],
        status="running",
        stage="session",
        event_message="Assistant session created.",
    )
    return _utils._json_response(
        {
            "status": "active",
            "session_id": session_id,
            "run_id": run_entry["run_id"],
            "ledger": run_entry,
        }
    )


@app.post("/assistant/jobs/{operation}", response_class=JSONResponse)
async def post_assistant_job(operation: str, request: Request) -> JSONResponse:
    normalized_operation = _normalize_assistant_operation(operation)
    if normalized_operation not in ASSISTANT_BACKGROUND_OPERATIONS:
        return _utils._json_error(
            "Unsupported assistant background operation.",
            status_code=400,
            operation=normalized_operation,
            supported_operations=sorted(ASSISTANT_BACKGROUND_OPERATIONS),
        )
    try:
        payload_data = await request.json()
        if not isinstance(payload_data, Mapping):
            payload_data = {}
    except Exception:
        payload_data = {}

    assistant_model_id = (
        payload_data.get("assistant_model_id")
        or payload_data.get("assistantModelId")
        or payload_data.get("model_id")
    )
    run_entry = _utils.create_run_entry(
        RUN_LEDGER_DIR,
        run_type="assistant_operation",
        context={
            "source": "assistant",
            "operation": normalized_operation,
            "assistant_model_id": assistant_model_id,
        },
        parameters={
            "operation": normalized_operation,
            "payload_keys": sorted(str(key) for key in payload_data.keys()),
            "prompt_length": len(str(payload_data.get("prompt") or "")),
        },
        status="queued",
    )
    try:
        enqueue_result = enqueue_assistant_operation(
            run_entry["run_id"],
            operation=normalized_operation,
            payload_data=payload_data,
        )
    except QueueUnavailableError as exc:
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_entry["run_id"],
            status="failed",
            stage="failed",
            merge={"error": str(exc), "error_code": exc.error_code},
            event_message=str(exc),
        )
        return _queue_unavailable_response(exc, run_id=run_entry["run_id"], ledger=run_entry)

    _utils.update_run_entry(
        RUN_LEDGER_DIR,
        run_entry["run_id"],
        merge={
            "queue": enqueue_result,
            "rerun_payload": {
                "run_type": "assistant_operation",
                "operation": normalized_operation,
                "queue_payload": {
                    "operation": normalized_operation,
                    "payload_data": payload_data,
                },
            },
        },
        event_message=f"Assistant operation enqueued on {enqueue_result.get('queue')}.",
    )
    return _utils._json_response(
        {
            "status": "accepted",
            "run_id": run_entry["run_id"],
            "job_id": run_entry["run_id"],
            "operation": normalized_operation,
            "ledger": {**run_entry, "queue": enqueue_result},
            "queue": enqueue_result,
        },
        status_code=202,
    )


def _record_assistant_training_example(
    *,
    label: str,
    draft: WorkflowDraft,
    review: Any = None,
    approval: Mapping[str, Any] | None = None,
    final_payload: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
    user_edits: Mapping[str, Any] | None = None,
) -> str | None:
    try:
        path = append_assistant_training_example(
            ASSISTANT_DATASET_DIR,
            label=label,
            draft=draft,
            review=review,
            approval=approval,
            final_payload=final_payload,
            outcome=outcome,
            user_edits=user_edits,
        )
        return str(path)
    except Exception:
        LOGGER.debug("Assistant training example persistence failed.", exc_info=True)
        return None


def _assistant_reference_context() -> dict[str, Any]:
    try:
        return build_assistant_reference_snapshot(ASSISTANT_DATASET_DIR, ASSISTANT_EVAL_DIR)
    except Exception:
        LOGGER.debug("Assistant reference snapshot failed.", exc_info=True)
        return {}


async def _assistant_training_registry_records(db: AsyncSession) -> dict[str, list[dict[str, Any]]]:
    async def _records(orm_model: Any) -> list[dict[str, Any]]:
        try:
            records: list[dict[str, Any]] = []
            for record in await get_all_entries(db, orm_model):
                payload = _utils._json_payload(record)
                if isinstance(payload, Mapping):
                    records.append(dict(payload))
                else:
                    LOGGER.debug(
                        "Skipping non-mapping assistant training registry payload from %s: %r",
                        getattr(orm_model, "__name__", orm_model),
                        payload,
                    )
            return records
        except Exception:
            LOGGER.debug("Assistant training registry source failed for %s.", getattr(orm_model, "__name__", orm_model), exc_info=True)
            return []

    datasets = [
        record
        for record in await _records(DatasetORM)
        if str(record.get("dataset_type") or "").lower() != "assistant_training_dataset"
    ]
    learning_models = [
        record
        for record in await _records(LearningORM)
        if str(record.get("model_type") or "").lower() != "assistant_model"
    ]
    return {
        "datasets": datasets,
        "learning_models": learning_models,
        "assistant_models": await _records(AssistantORM),
        "code_models": await _records(CodeORM),
        "inference_models": await _records(InferenceORM),
        "study_models": await _records(StudyORM),
    }


def _persist_assistant_review_artifacts(draft: WorkflowDraft, evaluation: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        context_path = persist_assistant_context_pack(ASSISTANT_DATASET_DIR / "context_packs", draft)
        if context_path is not None:
            paths["context_pack_path"] = str(context_path)
    except Exception:
        LOGGER.debug("Assistant context pack persistence failed.", exc_info=True)
    try:
        eval_path = persist_assistant_eval_result(ASSISTANT_EVAL_DIR, draft, evaluation)
        paths["evaluation_path"] = str(eval_path)
    except Exception:
        LOGGER.debug("Assistant evaluation persistence failed.", exc_info=True)
    try:
        promotion_path = persist_assistant_promotion_report(ASSISTANT_MODEL_DIR / "promotion_reports", draft, evaluation)
        paths["promotion_report_path"] = str(promotion_path)
    except Exception:
        LOGGER.debug("Assistant promotion report persistence failed.", exc_info=True)
    return paths


@app.get("/assistant/alignment/contracts", response_class=JSONResponse)
async def assistant_alignment_contracts_endpoint() -> JSONResponse:
    return _utils._json_response(
        {
            "alignment_contracts": assistant_alignment_contracts(),
            "mlflow_experiment": get_assistant_provider_status().get("mlflow_experiment"),
        }
    )


def _resolve_assistant_reference_pack(payload: AssistantDraftRequest) -> list[dict[str, Any]]:
    context = dict(payload.context or {})
    reference_query = _assistant_reference_query(payload)
    reference_tags = _assistant_reference_tags(payload)
    references = build_selected_reference_pack(
        ASSISTANT_REFERENCE_DIR,
        reference_ids=payload.reference_ids,
        query=reference_query,
        tags=reference_tags,
        limit=int(payload.constraints.get("max_references") or 8),
    )
    inline_references = context.get("content_references")
    if isinstance(inline_references, list):
        for item in inline_references[:8]:
            if isinstance(item, Mapping):
                references.append(item)  # type: ignore[arg-type]
    return [
        reference.model_dump(mode="json") if hasattr(reference, "model_dump") else dict(reference)
        for reference in references
    ]


def _assistant_reference_query(payload: AssistantDraftRequest) -> str:
    context = dict(payload.context or {})
    query_parts = [
        payload.prompt,
        payload.workflow_goal or "",
        str(context.get("reference_query") or ""),
        str(context.get("operationId") or ""),
        str(context.get("registryType") or context.get("modelType") or ""),
    ]
    for key in ("objectName", "datasetId", "modelId", "studyId"):
        if context.get(key):
            query_parts.append(str(context[key]))
    return " ".join(part for part in query_parts if part).strip()


def _assistant_reference_tags(payload: AssistantDraftRequest) -> list[str]:
    context = dict(payload.context or {})
    tags = {str(tag).strip() for tag in context.get("reference_tags") or [] if str(tag).strip()}
    target_type = payload.target_type or ""
    operation_id = str(context.get("operationId") or "").strip()
    registry_type = str(context.get("registryType") or context.get("modelType") or "").strip()
    if target_type:
        tags.add(target_type)
    if operation_id == "models":
        tags.add("model_generation")
        tags.add("model")
    if operation_id == "datasets":
        tags.add("dataset_generation")
        tags.add("dataset")
    if registry_type:
        tags.add(registry_type)
        tags.add(registry_type.lower())
        tags.add("registry")
    return sorted(tags)


def _serialize_assistant_model(model_record: Any) -> dict[str, Any]:
    payload = _utils._json_payload(model_record)
    if not isinstance(payload, Mapping):
        payload = dict(vars(model_record)) if hasattr(model_record, "__dict__") else {"value": payload}
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), Mapping) else {}
    assistant_config = dict((parameters or {}).get("assistant") or {})
    provider_config = assistant_model_to_provider_config(model_record)
    return {
        **payload,
        **{key: value for key, value in assistant_config.items() if value is not None},
        "assistant_config": assistant_config,
        "provider_config": redact_assistant_provider_config(provider_config),
        "provider_name": f"assistant_model_{payload.get('id') or payload.get('name') or 'registry'}",
        "bundle_status": assistant_bundle_status_from_model(payload),
    }


def _serialize_assistant_training_dataset(dataset_record: Any) -> dict[str, Any]:
    payload = _utils._json_payload(dataset_record)
    manifest = {}
    manifest_path = payload.get("connection_string")
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    return {
        **payload,
        "manifest": manifest,
        "dataset_hash": manifest.get("dataset_hash"),
        "canonical_jsonl_path": manifest.get("canonical_jsonl_path"),
        "tabular_csv_path": manifest.get("tabular_csv_path"),
        "tabular_hash": manifest.get("tabular_hash"),
        "materialization_format": manifest.get("materialization_format"),
        "generic_training_target": manifest.get("generic_training_target"),
        "record_count": manifest.get("record_count"),
    }


async def _latest_assistant_training_dataset(db: AsyncSession) -> Any | None:
    datasets = await get_all_entries(db, AssistantTrainingDatasetORM)
    if not datasets:
        return None
    return sorted(
        datasets,
        key=lambda item: (
            str(getattr(item, "date", None) or ""),
            getattr(item, "version", 0) or 0,
            getattr(item, "id", 0) or 0,
        ),
        reverse=True,
    )[0]


async def _upsert_assistant_training_dataset_record(
    db: AsyncSession,
    manifest: Mapping[str, Any],
) -> Any:
    latest = await _latest_assistant_training_dataset(db)
    dataset_path = Path(str(manifest.get("tabular_csv_path") or manifest.get("dataset_path") or ""))
    size_mb = dataset_path.stat().st_size / (1024 * 1024) if dataset_path.exists() else 0
    history_entry = {
        "operation": "assistant_training_dataset_rebuild",
        "when": datetime.now().isoformat(),
        "dataset_hash": manifest.get("dataset_hash"),
        "tabular_hash": manifest.get("tabular_hash"),
        "canonical_jsonl_path": manifest.get("canonical_jsonl_path"),
        "tabular_csv_path": manifest.get("tabular_csv_path"),
        "record_count": manifest.get("record_count"),
    }
    payload = {
        "name": "assistant_training_dataset_latest",
        "description": "Redacted JSONL snapshot for training Painel Amanaje AssistantModel entries.",
        "object_type": "dataset",
        "size": size_mb,
        "path": str(manifest.get("tabular_csv_path") or manifest.get("dataset_path") or ""),
        "date": datetime.now(),
        "version": int((getattr(latest, "version", 0) or 0) + 1) if latest is not None else 1,
        "history": _utils._append_history(getattr(latest, "history", None), history_entry),
        "dataset_type": "assistant_training_dataset",
        "shape": list(manifest.get("shape") or [0, 0]),
        "has_features": True,
        "features_list": list(manifest.get("features_list") or []),
        "connection_string": str(manifest.get("manifest_path") or ""),
    }
    if latest is None:
        return await create_entry(db, AssistantTrainingDatasetORM, payload)
    return await update_entry(db, AssistantTrainingDatasetORM, getattr(latest, "id"), payload)


async def _get_assistant_model_record(db: AsyncSession, assistant_model_id: int | str) -> Any:
    model_record = await get_entry(db, AssistantORM, assistant_model_id)
    if model_record is None:
        raise ValueError(f"Assistant model '{assistant_model_id}' was not found in the registry.")
    return model_record


def _assistant_default_model_id() -> str | None:
    state = load_settings_state(_utils.SETTINGS_STATE_PATH)
    assistant_state = state.get("assistant") if isinstance(state, Mapping) else {}
    if not isinstance(assistant_state, Mapping):
        return None
    value = assistant_state.get("default_model_id")
    return str(value).strip() if value not in (None, "") else None


async def _resolve_default_assistant_model(db: AsyncSession) -> Any | None:
    default_id = _assistant_default_model_id()
    if not default_id:
        return None
    try:
        return await _get_assistant_model_record(db, default_id)
    except ValueError:
        return None


def _assistant_model_runtime_kind(model_record: Any, provider_config: Mapping[str, Any] | None = None) -> str:
    parameters = getattr(model_record, "parameters", {}) if not isinstance(model_record, Mapping) else model_record.get("parameters", {})
    assistant_config = dict((parameters or {}).get("assistant") or {}) if isinstance(parameters, Mapping) else {}
    runtime_config = assistant_config.get("runtime_config") if isinstance(assistant_config.get("runtime_config"), Mapping) else {}
    return str(
        (provider_config or {}).get("runtime_kind")
        or assistant_config.get("runtime_kind")
        or runtime_config.get("runtime_kind")
        or ""
    ).strip()


def _assistant_model_needs_runtime_load(model_record: Any, provider_config: Mapping[str, Any] | None = None) -> bool:
    return _assistant_model_runtime_kind(model_record, provider_config) == "pytorch_hf_server"


def _safe_assistant_path_fragment(value: Any, fallback: str = "assistant_model") -> str:
    text = str(value or fallback).strip().replace("\\", "/")
    text = text.strip("/").replace("/", "__")
    safe = "".join(character.lower() if character.isalnum() else "_" for character in text)
    return "_".join(part for part in safe.split("_") if part) or fallback


def _assistant_route_catalog() -> list[dict[str, str]]:
    return [
        {"group": "Management", "method": "GET", "path": "/assistant", "operation": "Render the Assistant Management workspace."},
        {"group": "Management", "method": "GET", "path": "/assistant/management/overview", "operation": "Return aggregated assistant status, route catalog, safety profiles, and artifact locations."},
        {"group": "Examples", "method": "GET", "path": "/examples/catalog", "operation": "Return five validated samples for every registered object and request schema."},
        {"group": "Examples", "method": "GET", "path": "/examples/catalog/{object_name}", "operation": "Return the five scale-tier samples for one catalog object."},
        {"group": "Sessions", "method": "POST", "path": "/assistant/sessions", "operation": "Create an assistant run session in the run ledger."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/draft", "operation": "Create a reviewed WorkflowDraft from the active provider or selected AssistantModel."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/review", "operation": "Review a WorkflowDraft contract and safety state."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/approve", "operation": "Approve a reviewed draft and record a positive training example."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/feedback", "operation": "Record accepted, corrected, rejected, or unsafe feedback for training curation."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/submit", "operation": "Submit an approved assistant run to the configured workflow handoff."},
        {"group": "Providers", "method": "GET", "path": "/assistant/provider/status", "operation": "Inspect configured providers, active provider, fallback state, and model metadata."},
        {"group": "Providers", "method": "POST", "path": "/assistant/provider/test", "operation": "Run a small WorkflowDraft contract diagnostic against a provider."},
        {"group": "Runtime", "method": "GET", "path": "/assistant/runtime/status", "operation": "Inspect assistant-server health, loaded model, and redacted runtime config."},
        {"group": "Models", "method": "GET", "path": "/assistant/models/list", "operation": "List registered AssistantModel entries available for assistant selection."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/import/huggingface", "operation": "Import a Hugging Face snapshot as a registered AssistantModel using server-side token configuration."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/import/pytorch", "operation": "Import a PyTorch/HF AssistantModel bundle from an uploaded zip or server-visible directory."},
        {"group": "Models", "method": "GET", "path": "/assistant/models/{assistant_model_id}/status", "operation": "Inspect a registered AssistantModel provider configuration."},
        {"group": "Models", "method": "GET", "path": "/assistant/models/{assistant_model_id}/bundle/status", "operation": "Inspect uploaded AssistantModel bundle, tokenizer, prompt, and server metadata."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/runtime/load", "operation": "Load a registered PyTorch/HF AssistantModel into assistant-server."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/runtime/unload", "operation": "Unload the active assistant-server model."},
        {"group": "Models", "method": "GET", "path": "/assistant/models/{assistant_model_id}/runtime/status", "operation": "Inspect runtime status for a registered AssistantModel."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/activate", "operation": "Persist a registered AssistantModel as the global default assistant."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/test", "operation": "Run the provider contract diagnostic for one registered AssistantModel."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/training/curate", "operation": "Curate training records for one AssistantModel."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/training/dataset/attach", "operation": "Attach the latest assistant training dataset snapshot to one AssistantModel."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/training/tokenize", "operation": "Prepare token-counted instruction JSONL examples from the canonical assistant training dataset."},
        {"group": "Datasets", "method": "POST", "path": "/assistant/datasets/rebuild", "operation": "Materialize the redacted assistant training JSONL snapshot."},
        {"group": "Datasets", "method": "GET", "path": "/assistant/datasets/status", "operation": "Inspect assistant training dataset readiness and source snapshot."},
        {"group": "Datasets", "method": "GET", "path": "/assistant/datasets/latest", "operation": "Fetch the latest AssistantTrainingDatasetModel registry record."},
        {"group": "References", "method": "GET", "path": "/assistant/references/status", "operation": "Inspect internal reference repository paths and stored reference count."},
        {"group": "References", "method": "POST", "path": "/assistant/references/sync", "operation": "Hash, redact, summarize, and store internal project references."},
        {"group": "References", "method": "POST", "path": "/assistant/references", "operation": "Create an internal/admin ContentReference record without exposing raw content in normal UI."},
        {"group": "References", "method": "GET", "path": "/assistant/references", "operation": "List compact ContentReference metadata for admin inspection."},
        {"group": "References", "method": "POST", "path": "/assistant/references/search", "operation": "Search compact ContentReference metadata by query, tags, and source type."},
        {"group": "Evals", "method": "POST", "path": "/assistant/evals/run", "operation": "Run the assistant golden eval suite for a provider."},
        {"group": "Evals", "method": "POST", "path": "/assistant/training/curate", "operation": "Curate global assistant training examples."},
        {"group": "Contracts", "method": "GET", "path": "/assistant/alignment/contracts", "operation": "Return output, safety, registry, dependency, and MLflow alignment contracts."},
        {"group": "Contracts", "method": "GET", "path": "/assistant/contracts/workflow-draft.schema", "operation": "Return the WorkflowDraft JSON schema contract."},
        {"group": "Execution", "method": "POST", "path": "/execution/review", "operation": "Review generated code against an assistant safety profile."},
        {"group": "Execution", "method": "POST", "path": "/execution/run", "operation": "Run approved assistant code in the guarded execution namespace."},
        {"group": "MLflow", "method": "GET", "path": "/mlflow/health", "operation": "Inspect MLflow availability and tracking URI health."},
        {"group": "MLflow", "method": "GET", "path": "/mlflow/experiments", "operation": "List MLflow experiments and run summaries for assistant alignment."},
    ]


def _assistant_artifact_locations() -> list[dict[str, Any]]:
    paths = {
        "assistant_models": ASSISTANT_MODEL_DIR,
        "assistant_model_uploads": ASSISTANT_MODEL_DIR / "uploads",
        "assistant_model_bundles": ASSISTANT_MODEL_DIR / "bundles",
        "assistant_model_huggingface": ASSISTANT_MODEL_DIR / "huggingface",
        "assistant_datasets": ASSISTANT_DATASET_DIR,
        "assistant_dataset_snapshots": ASSISTANT_DATASET_DIR / "snapshots",
        "assistant_tokenized_training": ASSISTANT_DATASET_DIR / "tokenized",
        "assistant_context_packs": ASSISTANT_DATASET_DIR / "context_packs",
        "assistant_references": ASSISTANT_REFERENCE_DIR,
        "assistant_evals": ASSISTANT_EVAL_DIR,
        "assistant_promotion_reports": ASSISTANT_MODEL_DIR / "promotion_reports",
        "run_ledger": RUN_LEDGER_DIR,
    }
    return [
        {
            "key": key,
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
        }
        for key, path in paths.items()
    ]


def _assistant_safety_profiles_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, profile in SAFETY_PROFILES.items():
        payload[name] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in dict(profile).items()
        }
    return payload


@app.get("/assistant/management/overview", response_class=JSONResponse)
async def assistant_management_overview(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    errors: dict[str, str] = {}
    provider_status: dict[str, Any] = {}
    assistant_models: list[dict[str, Any]] = []
    dataset_status: dict[str, Any] = {}
    reference_status: dict[str, Any] = {}
    latest_references: list[dict[str, Any]] = []
    mlflow_health: dict[str, Any] = {}
    runtime_state: dict[str, Any] = {}
    default_model_id = _assistant_default_model_id()

    try:
        provider_status = get_assistant_provider_status()
    except Exception as exc:
        LOGGER.exception("Assistant management provider status failed.")
        errors["provider_status"] = str(exc)

    try:
        models = await get_all_entries(db, AssistantORM)
        assistant_models = [_serialize_assistant_model(model) for model in models]
    except Exception as exc:
        LOGGER.exception("Assistant management model list failed.")
        errors["assistant_models"] = str(exc)

    try:
        dataset_record = await _latest_assistant_training_dataset(db)
        dataset_status = {
            "status": "ready" if dataset_record is not None else "missing",
            "latest_dataset": _serialize_assistant_training_dataset(dataset_record) if dataset_record is not None else None,
            "source_snapshot": _assistant_reference_context(),
        }
    except Exception as exc:
        LOGGER.exception("Assistant management dataset status failed.")
        errors["dataset_status"] = str(exc)

    try:
        reference_status = assistant_reference_repository_status(ASSISTANT_REFERENCE_DIR, base_dir=REPO_ROOT)
        latest_references = [
            reference.model_dump(mode="json")
            for reference in list_assistant_references(ASSISTANT_REFERENCE_DIR, limit=12)
        ]
    except Exception as exc:
        LOGGER.exception("Assistant management reference status failed.")
        errors["reference_status"] = str(exc)

    try:
        mlflow_health = _utils._build_mlflow_health_snapshot()
    except Exception as exc:
        LOGGER.exception("Assistant management MLflow health failed.")
        errors["mlflow_health"] = str(exc)

    try:
        runtime_state = assistant_runtime_status(timeout=2.0)
    except Exception as exc:
        LOGGER.exception("Assistant management runtime status failed.")
        errors["runtime_status"] = str(exc)

    return _utils._json_response(
        {
            "status": "ok" if not errors else "degraded",
            "generated_at": datetime.now().isoformat(),
            "provider_status": provider_status,
            "runtime": runtime_state,
            "assistant_defaults": {
                "global_default_model_id": default_model_id,
                "selection_order": [
                    "request_assistant_model_id",
                    "browser_session_model_id",
                    "global_default_model_id",
                    "configured_active_provider",
                    "local_fallback",
                ],
            },
            "assistant_models": {
                "count": len(assistant_models),
                "bundle_ready_count": sum(1 for model in assistant_models if (model.get("bundle_status") or {}).get("ready")),
                "pytorch_bundle_count": sum(
                    1
                    for model in assistant_models
                    if (model.get("assistant_config") or {}).get("runtime_kind") == "pytorch_hf_server"
                ),
                "models": assistant_models,
            },
            "assistant_dataset": dataset_status,
            "references": {
                "repository": reference_status,
                "latest": latest_references,
            },
            "huggingface": {
                "cache_dir": str(ASSISTANT_MODEL_DIR / "huggingface"),
                "token_configured": bool(os.getenv("HUGGINGFACE_HUB_TOKEN")),
                "raw_tokens_returned": False,
            },
            "mlflow": mlflow_health,
            "alignment_contracts": assistant_alignment_contracts(),
            "safety_profiles": _assistant_safety_profiles_payload(),
            "artifact_locations": _assistant_artifact_locations(),
            "route_catalog": _assistant_route_catalog(),
            "errors": errors,
        }
    )


@app.post("/assistant/references", response_class=JSONResponse)
async def assistant_create_reference(payload: AssistantReferenceRequest) -> JSONResponse:
    try:
        reference, metadata_path, text_path = create_assistant_reference(ASSISTANT_REFERENCE_DIR, payload)
        mlflow_tracking = log_assistant_mlflow_event(
            "reference_created",
            evaluation={
                "evaluation_profile": "assistant-reference-ingestion-v1",
                "passed": True,
                "checks": {"valid_workflow_draft": True},
                "alignment_contracts": {
                    "output_alignment": True,
                    "safety_alignment": True,
                    "registry_alignment": True,
                    "dependency_alignment": True,
                    "mlflow_alignment": True,
                },
            },
            artifact_paths={
                "reference_metadata": str(metadata_path),
                "reference_text": str(text_path) if text_path else None,
            },
            extra_params={
                "reference_id": reference.reference_id,
                "source_type": reference.source_type,
                "trust_level": reference.trust_level,
                "content_hash": reference.content_hash,
            },
        )
        return _utils._json_response(
            {
                "status": "created",
                "reference": reference,
                "artifact_paths": {
                    "metadata_path": str(metadata_path),
                    "text_path": str(text_path) if text_path else None,
                },
                "mlflow_tracking": mlflow_tracking,
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant reference creation failed.")
        return _utils._json_error(str(exc), status_code=400)


@app.get("/assistant/references", response_class=JSONResponse)
async def assistant_list_references(query: str = "", limit: int = 20) -> JSONResponse:
    references = list_assistant_references(ASSISTANT_REFERENCE_DIR, query=query, limit=limit)
    return _utils._json_response(
        {
            "status": "ok",
            "references": [reference.model_dump(mode="json") for reference in references],
        }
    )


@app.post("/assistant/references/search", response_class=JSONResponse)
async def assistant_search_references(payload: AssistantReferenceSearchRequest) -> JSONResponse:
    references = list_assistant_references(
        ASSISTANT_REFERENCE_DIR,
        query=payload.query,
        tags=payload.tags,
        source_types=payload.source_types,
        limit=payload.limit,
    )
    return _utils._json_response(
        {
            "status": "ok",
            "references": [reference.model_dump(mode="json") for reference in references],
        }
    )


@app.get("/assistant/references/status", response_class=JSONResponse)
async def assistant_references_status() -> JSONResponse:
    payload = assistant_reference_repository_status(
        ASSISTANT_REFERENCE_DIR,
        base_dir=REPO_ROOT,
    )
    return _utils._json_response({"status": "ok", "repository": payload})


@app.post("/assistant/references/sync", response_class=JSONResponse)
async def assistant_sync_references(max_files: int = 100) -> JSONResponse:
    try:
        result = sync_assistant_reference_repository(
            ASSISTANT_REFERENCE_DIR,
            source_paths=resolve_assistant_reference_source_paths(ASSISTANT_REFERENCE_DIR, base_dir=REPO_ROOT),
            base_dir=REPO_ROOT,
            max_files=max_files,
        )
        mlflow_tracking = log_assistant_mlflow_event(
            "reference_repository_sync",
            evaluation={
                "evaluation_profile": "assistant-reference-sync-v1",
                "passed": True,
                "checks": {"reference_repository_synced": True},
                "alignment_contracts": {
                    "output_alignment": True,
                    "safety_alignment": True,
                    "registry_alignment": True,
                    "dependency_alignment": True,
                    "mlflow_alignment": True,
                },
            },
            artifact_paths=[item.get("metadata_path") for item in result.get("created", [])],
            extra_metrics={
                "reference_sync_created": float(result.get("created_count", 0) or 0),
                "reference_sync_skipped": float(result.get("skipped_count", 0) or 0),
            },
        )
        return _utils._json_response({**result, "mlflow_tracking": mlflow_tracking})
    except Exception as exc:
        LOGGER.exception("Assistant reference repository sync failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/datasets/rebuild", response_class=JSONResponse)
async def assistant_datasets_rebuild(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        manifest = build_assistant_training_dataset_snapshot(
            ASSISTANT_DATASET_DIR,
            ASSISTANT_EVAL_DIR,
            RUN_LEDGER_DIR,
            registry_records=await _assistant_training_registry_records(db),
            reference_dir=ASSISTANT_REFERENCE_DIR,
        )
        dataset_record = await _upsert_assistant_training_dataset_record(db, manifest)
        mlflow_tracking = log_assistant_mlflow_event(
            "assistant_training_dataset_rebuild",
            evaluation={
                "evaluation_profile": "assistant-training-dataset-v1",
                "passed": True,
                "checks": {"record_count": int(manifest.get("record_count", 0) or 0)},
            },
            artifact_paths={
                "dataset": manifest.get("dataset_path"),
                "canonical_jsonl": manifest.get("canonical_jsonl_path"),
                "tabular_csv": manifest.get("tabular_csv_path"),
                "manifest": manifest.get("manifest_path"),
            },
            extra_metrics={
                "assistant_training_dataset_records": float(manifest.get("record_count", 0) or 0),
                "assistant_training_dataset_sources": float(manifest.get("source_file_count", 0) or 0),
            },
        )
        return _utils._json_response(
            {
                "status": "rebuilt",
                "dataset": _serialize_assistant_training_dataset(dataset_record),
                "manifest": manifest,
                "mlflow_tracking": mlflow_tracking,
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant training dataset rebuild failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/assistant/datasets/latest", response_class=JSONResponse)
async def assistant_datasets_latest(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    dataset_record = await _latest_assistant_training_dataset(db)
    if dataset_record is None:
        return _utils._json_error("Assistant training dataset has not been built yet.", status_code=404)
    return _utils._json_response(
        {
            "status": "ok",
            "dataset": _serialize_assistant_training_dataset(dataset_record),
        }
    )


@app.get("/assistant/datasets/status", response_class=JSONResponse)
async def assistant_datasets_status(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    dataset_record = await _latest_assistant_training_dataset(db)
    source_snapshot = _assistant_reference_context()
    return _utils._json_response(
        {
            "status": "ready" if dataset_record is not None else "missing",
            "latest_dataset": _serialize_assistant_training_dataset(dataset_record) if dataset_record is not None else None,
            "source_snapshot": source_snapshot,
        }
    )


@app.get("/assistant/models/list", response_class=JSONResponse)
async def assistant_models_list(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        models = await get_all_entries(db, AssistantORM)
        return _utils._json_response(
            {
                "status": "ok",
                "models": [_serialize_assistant_model(model) for model in models],
                "count": len(models),
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant model list failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/import/huggingface", response_class=JSONResponse)
async def assistant_model_import_huggingface(
    payload: dict[str, Any] = Body(default_factory=dict),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        repo_id = str(payload.get("repo_id") or "").strip()
        if not repo_id or ".." in repo_id or repo_id.startswith(("/", "\\")):
            return _utils._json_error("A valid Hugging Face repo_id is required.", status_code=400)
        revision = str(payload.get("revision") or "").strip() or None
        display_name = str(payload.get("display_name") or payload.get("name") or repo_id.rsplit("/", 1)[-1]).strip()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is not installed in the API environment.") from exc

        import_root = ASSISTANT_MODEL_DIR / "huggingface" / _safe_assistant_path_fragment(repo_id)
        if revision:
            import_root = import_root / _safe_assistant_path_fragment(revision)
        import_root.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(import_root),
            token=os.getenv("HUGGINGFACE_HUB_TOKEN") or None,
        )
        supported_draft_types = payload.get("supported_draft_types")
        if isinstance(supported_draft_types, str):
            supported_draft_types = [item.strip() for item in supported_draft_types.split(",") if item.strip()]
        metadata = {
            "repo_id": repo_id,
            "revision": revision,
            "display_name": display_name,
            "model_name": payload.get("model_name") or display_name,
            "model_version": payload.get("model_version") or revision or "huggingface-import",
            "base_model_name": payload.get("base_model_name") or repo_id,
            "supported_draft_types": supported_draft_types,
            "max_context_tokens": payload.get("max_context_tokens") or 4096,
            "base_url": os.getenv("AMANAJE_SLM_BASE_URL") or "http://assistant-server:8080/v1",
            "generation_config": payload.get("generation_config") if isinstance(payload.get("generation_config"), Mapping) else {},
        }
        bundle = normalize_huggingface_assistant_directory(downloaded_path, metadata)
        assistant_params = dict(bundle.get("assistant_parameters") or {})
        created = await create_entry(
            db,
            AssistantORM,
            {
                "name": display_name,
                "description": payload.get("description")
                or f"Hugging Face AssistantModel imported from {repo_id}.",
                "object_type": "learning_model",
                "size": 0,
                "path": str(downloaded_path),
                "date": datetime.now(),
                "version": int(payload.get("version") or 1),
                "history": [
                    {
                        "operation": "assistant_model_huggingface_imported",
                        "when": datetime.now().isoformat(),
                        "repo_id": repo_id,
                        "revision": revision,
                        "bundle_sha256": bundle.get("bundle_sha256"),
                    }
                ],
                "model_type": "assistant_model",
                "parameters": {
                    "assistant": assistant_params,
                    "huggingface": {
                        "repo_id": repo_id,
                        "revision": revision,
                        "local_dir": str(downloaded_path),
                        "token_configured": bool(os.getenv("HUGGINGFACE_HUB_TOKEN")),
                    },
                    "artifact_manifest": bundle.get("manifest"),
                },
                "metrics": {
                    "status": "imported",
                    "bundle_status": "valid",
                    "contract": "workflow_draft_json",
                },
                "reference_data": assistant_params.get("training_dataset_jsonl_path") or "",
                "input_features": ["prompt", "context_pack", "target_type"],
                "output_features": ["workflow_draft"],
                "is_trained": False,
                "is_tested": False,
                "is_deployed": False,
            },
        )
        return _utils._json_response(
            {
                "status": "imported",
                "assistant_model": _serialize_assistant_model(created),
                "bundle": bundle,
                "huggingface": {
                    "repo_id": repo_id,
                    "revision": revision,
                    "local_dir": str(downloaded_path),
                    "token_configured": bool(os.getenv("HUGGINGFACE_HUB_TOKEN")),
                    "raw_token_returned": False,
                },
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant Hugging Face import failed.")
        return _utils._json_error(str(exc), status_code=400)


@app.post("/assistant/models/import/pytorch", response_class=JSONResponse)
async def assistant_model_import_pytorch(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        form = await request.form()
        upload_file = form.get("file")
        bundle_path_value = str(form.get("bundle_path") or "").strip()
        display_name = str(
            form.get("display_name")
            or form.get("name")
            or getattr(upload_file, "filename", "")
            or Path(bundle_path_value).name
            or "PyTorch AssistantModel"
        ).strip()
        supported_draft_types = form.get("supported_draft_types")
        if isinstance(supported_draft_types, str):
            supported_draft_types = [item.strip() for item in supported_draft_types.split(",") if item.strip()]
        metadata = {
            "display_name": display_name,
            "model_name": form.get("model_name") or display_name,
            "model_version": form.get("model_version") or "pytorch-import",
            "base_model_name": form.get("base_model_name") or display_name,
            "supported_draft_types": supported_draft_types,
            "max_context_tokens": form.get("max_context_tokens") or 4096,
            "base_url": os.getenv("AMANAJE_SLM_BASE_URL") or "http://assistant-server:8080/v1",
            "generation_config": {
                "temperature": float(form.get("temperature") or 0.2),
                "max_new_tokens": int(form.get("max_tokens") or 1600),
            },
        }

        size_mb = 0.0
        import_source = "server_path" if bundle_path_value else "upload"
        if upload_file is not None and getattr(upload_file, "filename", ""):
            filename = getattr(upload_file, "filename", "assistant_model_bundle.zip")
            if Path(filename).suffix.lower() != ".zip":
                return _utils._json_error(
                    "PyTorch AssistantModel imports must be .zip bundles. A raw .pt file is not enough because tokenizer, prompt template, and manifest metadata are required.",
                    status_code=400,
                )
            upload_dir = ASSISTANT_MODEL_DIR / "imports" / "pytorch"
            upload_dir.mkdir(parents=True, exist_ok=True)
            destination_path = upload_dir / filename
            size_mb = await save_file_to_disk(upload_file, str(destination_path))
            extract_slug = _safe_assistant_path_fragment(display_name)
            extract_dir = ASSISTANT_MODEL_DIR / "bundles" / f"{extract_slug}_{uuid.uuid4().hex[:8]}"
            bundle = inspect_assistant_model_bundle(destination_path, extract_dir=extract_dir)
            source_path = str(destination_path)
        elif bundle_path_value:
            candidate = Path(bundle_path_value)
            if not candidate.exists():
                return _utils._json_error(f"PyTorch bundle path does not exist: {candidate}", status_code=404)
            if candidate.is_dir():
                bundle = normalize_huggingface_assistant_directory(candidate, metadata)
                source_path = str(candidate)
            elif candidate.suffix.lower() == ".zip":
                extract_slug = _safe_assistant_path_fragment(display_name)
                extract_dir = ASSISTANT_MODEL_DIR / "bundles" / f"{extract_slug}_{uuid.uuid4().hex[:8]}"
                bundle = inspect_assistant_model_bundle(candidate, extract_dir=extract_dir)
                source_path = str(candidate)
                size_mb = candidate.stat().st_size / (1024 * 1024)
            else:
                return _utils._json_error(
                    "PyTorch AssistantModel imports require a .zip bundle or a directory with assistant_model_manifest.json.",
                    status_code=400,
                )
        else:
            return _utils._json_error("Provide a .zip bundle upload or a server-visible bundle_path.", status_code=400)

        assistant_params = dict(bundle.get("assistant_parameters") or {})
        for key in ("model_name", "model_version", "base_model_name", "supported_draft_types", "max_context_tokens", "base_url", "generation_config"):
            value = metadata.get(key)
            if value not in (None, "", []):
                assistant_params[key] = value
        assistant_params["provider_type"] = "openai_compatible"
        assistant_params["runtime_kind"] = "pytorch_hf_server"

        created = await create_entry(
            db,
            AssistantORM,
            {
                "name": display_name,
                "description": form.get("description") or f"PyTorch AssistantModel imported from {source_path}.",
                "object_type": "learning_model",
                "size": size_mb,
                "path": source_path,
                "date": datetime.now(),
                "version": int(form.get("version") or 1),
                "history": [
                    {
                        "operation": "assistant_model_pytorch_imported",
                        "when": datetime.now().isoformat(),
                        "source": import_source,
                        "source_path": source_path,
                        "bundle_sha256": bundle.get("bundle_sha256"),
                    }
                ],
                "model_type": "assistant_model",
                "parameters": {
                    "assistant": assistant_params,
                    "pytorch": {
                        "source": import_source,
                        "source_path": source_path,
                        "bundle_status": bundle.get("status"),
                    },
                    "artifact_manifest": bundle.get("manifest"),
                },
                "metrics": {
                    "status": "imported",
                    "bundle_status": bundle.get("status"),
                    "contract": "workflow_draft_json",
                },
                "reference_data": assistant_params.get("training_dataset_jsonl_path") or "",
                "input_features": ["prompt", "context_pack", "target_type"],
                "output_features": ["workflow_draft"],
                "is_trained": False,
                "is_tested": False,
                "is_deployed": False,
            },
        )
        return _utils._json_response(
            {
                "status": "imported",
                "assistant_model": _serialize_assistant_model(created),
                "bundle": bundle,
                "pytorch": {
                    "source": import_source,
                    "source_path": source_path,
                    "raw_weights_loaded_in_fastapi": False,
                },
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant PyTorch import failed.")
        return _utils._json_error(str(exc), status_code=400)


@app.get("/assistant/models/{assistant_model_id}/status", response_class=JSONResponse)
async def assistant_model_status(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        provider_config = assistant_model_to_provider_config(model_record)
        return _utils._json_response(
            {
                "status": "ok",
                "assistant_model": _serialize_assistant_model(model_record),
                "provider_config": redact_assistant_provider_config(provider_config),
                "bundle_status": assistant_bundle_status_from_model(model_record),
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model status failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/assistant/models/{assistant_model_id}/bundle/status", response_class=JSONResponse)
async def assistant_model_bundle_status(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        return _utils._json_response(
            {
                "status": "ok",
                "assistant_model": _serialize_assistant_model(model_record),
                "bundle_status": assistant_bundle_status_from_model(model_record),
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model bundle status failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/assistant/runtime/status", response_class=JSONResponse)
async def assistant_global_runtime_status() -> JSONResponse:
    return _utils._json_response(
        {
            "status": "ok",
            "runtime": assistant_runtime_status(),
            "runtime_config": assistant_runtime_config(),
            "global_default_model_id": _assistant_default_model_id(),
        }
    )


@app.get("/assistant/models/{assistant_model_id}/runtime/status", response_class=JSONResponse)
async def assistant_model_runtime_status(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        provider_config = assistant_model_to_provider_config(model_record)
        return _utils._json_response(
            {
                "status": "ok",
                "assistant_model": _serialize_assistant_model(model_record),
                "runtime": assistant_runtime_status(),
                "runtime_kind": _assistant_model_runtime_kind(model_record, provider_config),
                "runtime_load_supported": _assistant_model_needs_runtime_load(model_record, provider_config),
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model runtime status failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/runtime/load", response_class=JSONResponse)
async def assistant_model_runtime_load(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        provider_config = assistant_model_to_provider_config(model_record)
        if not _assistant_model_needs_runtime_load(model_record, provider_config):
            return _utils._json_response(
                {
                    "status": "not_required",
                    "assistant_model": _serialize_assistant_model(model_record),
                    "runtime_kind": _assistant_model_runtime_kind(model_record, provider_config) or "external_or_configured_provider",
                }
            )
        result = ensure_assistant_model_runtime_loaded(model_record, provider_config)
        return _utils._json_response(
            {
                "status": result.get("status"),
                "assistant_model": _serialize_assistant_model(model_record),
                "runtime_load": result,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model runtime load failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/runtime/unload", response_class=JSONResponse)
async def assistant_model_runtime_unload(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        result = unload_assistant_runtime_model()
        return _utils._json_response(
            {
                "status": result.get("status"),
                "assistant_model": _serialize_assistant_model(model_record),
                "runtime_unload": result,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model runtime unload failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/activate", response_class=JSONResponse)
async def assistant_model_activate(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        update_settings_state({"assistant": {"default_model_id": assistant_model_id}}, _utils.SETTINGS_STATE_PATH)
        provider_config = assistant_model_to_provider_config(model_record)
        runtime_load = (
            ensure_assistant_model_runtime_loaded(model_record, provider_config)
            if _assistant_model_needs_runtime_load(model_record, provider_config)
            else {"status": "not_required", "runtime_kind": _assistant_model_runtime_kind(model_record, provider_config)}
        )
        return _utils._json_response(
            {
                "status": "activated",
                "global_default_model_id": str(assistant_model_id),
                "assistant_model": _serialize_assistant_model(model_record),
                "runtime_load": runtime_load,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except SettingsValidationError as exc:
        return _utils._json_response({"status": "error", "detail": str(exc), "errors": exc.errors}, status_code=400)
    except Exception as exc:
        LOGGER.exception("Assistant model activation failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/test", response_class=JSONResponse)
async def assistant_model_test(assistant_model_id: int | str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        result = test_assistant_model_contract(model_record)
        return _utils._json_response(result)
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model diagnostic failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/training/curate", response_class=JSONResponse)
async def assistant_model_curate_training(
    assistant_model_id: int | str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        output_dir = ASSISTANT_DATASET_DIR / "curated" / f"assistant_model_{getattr(model_record, 'id', assistant_model_id)}"
        reference_sync = sync_assistant_reference_repository(
            ASSISTANT_REFERENCE_DIR,
            source_paths=resolve_assistant_reference_source_paths(ASSISTANT_REFERENCE_DIR, base_dir=REPO_ROOT),
            base_dir=REPO_ROOT,
            max_files=1000,
        )
        project_examples = build_example_catalog(routes=app.routes, include_payloads=True)
        manifest = curate_assistant_training_examples(
            ASSISTANT_DATASET_DIR,
            output_dir,
            project_examples=project_examples,
            reference_dir=ASSISTANT_REFERENCE_DIR,
        )
        parameters = dict(getattr(model_record, "parameters", {}) or {})
        assistant_config = dict(parameters.get("assistant") or {})
        assistant_config.update(
            {
                "curated_training_manifest": manifest.get("manifest_path"),
                "curated_training_path": manifest.get("curated_path"),
                "curated_training_record_count": manifest.get("record_count"),
                "curated_training_project_example_records": manifest.get("project_example_records"),
                "curated_training_reference_records": manifest.get("reference_records"),
                "curated_training_interaction_records": manifest.get("interaction_records"),
                "curated_training_ready_for_fine_tuning": manifest.get("ready_for_fine_tuning"),
                "project_examples_version": project_examples.get("version"),
                "project_examples_sample_count": (project_examples.get("summary") or {}).get("sample_count"),
            }
        )
        parameters["assistant"] = assistant_config
        updated = await update_entry(
            db,
            AssistantORM,
            getattr(model_record, "id"),
            {
                "parameters": parameters,
                "reference_data": manifest.get("curated_path") or getattr(model_record, "reference_data", None),
                "history": _utils._append_history(
                    getattr(model_record, "history", None),
                    {
                        "operation": "assistant_model_training_curated",
                        "when": datetime.now().isoformat(),
                        "curated_path": manifest.get("curated_path"),
                        "manifest_path": manifest.get("manifest_path"),
                        "record_count": manifest.get("record_count"),
                        "project_example_records": manifest.get("project_example_records"),
                        "reference_records": manifest.get("reference_records"),
                    },
                ),
            },
        )
        return _utils._json_response(
            {
                "status": "curated",
                "assistant_model": _serialize_assistant_model(updated),
                "manifest": manifest,
                "reference_sync": {
                    "created_count": reference_sync.get("created_count"),
                    "skipped_count": reference_sync.get("skipped_count"),
                    "scanned": reference_sync.get("scanned"),
                },
                "project_examples": {
                    "version": project_examples.get("version"),
                    "sample_count": (project_examples.get("summary") or {}).get("sample_count"),
                    "object_count": (project_examples.get("summary") or {}).get("object_count"),
                },
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model training curation failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/training/tokenize", response_class=JSONResponse)
async def assistant_model_tokenize_training_dataset(
    assistant_model_id: int | str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        parameters = dict(getattr(model_record, "parameters", {}) or {})
        assistant_config = dict(parameters.get("assistant") or {})
        manifest_path = assistant_config.get("training_dataset_manifest")
        dataset_record = None
        if not manifest_path:
            dataset_record = await _latest_assistant_training_dataset(db)
            if dataset_record is None:
                return _utils._json_error(
                    "Assistant training dataset has not been built yet. Run /assistant/datasets/rebuild first.",
                    status_code=404,
                )
            manifest_path = getattr(dataset_record, "connection_string", None)
        if not manifest_path:
            return _utils._json_error("Assistant training dataset manifest is missing.", status_code=404)

        output_dir = ASSISTANT_DATASET_DIR / "tokenized" / f"assistant_model_{getattr(model_record, 'id', assistant_model_id)}"
        manifest = prepare_assistant_model_tokenization(model_record, manifest_path, output_dir)
        assistant_config.update(
            {
                "tokenized_training_manifest": manifest.get("manifest_path"),
                "tokenized_training_examples_path": manifest.get("examples_path"),
                "tokenized_training_examples_hash": manifest.get("examples_hash"),
                "tokenized_training_record_count": manifest.get("record_count"),
                "tokenized_training_total_estimated_tokens": manifest.get("total_estimated_tokens"),
                "tokenized_training_truncation_rate": manifest.get("truncation_rate"),
                "training_dataset_manifest": str(manifest_path),
                "training_dataset_jsonl_path": manifest.get("source_jsonl_path"),
                "training_dataset_hash": manifest.get("dataset_hash"),
            }
        )
        parameters["assistant"] = assistant_config
        updated = await update_entry(
            db,
            AssistantORM,
            getattr(model_record, "id"),
            {
                "parameters": parameters,
                "reference_data": manifest.get("source_jsonl_path") or getattr(model_record, "reference_data", None),
                "history": _utils._append_history(
                    getattr(model_record, "history", None),
                    {
                        "operation": "assistant_training_tokenized",
                        "when": datetime.now().isoformat(),
                        "dataset_hash": manifest.get("dataset_hash"),
                        "examples_hash": manifest.get("examples_hash"),
                        "examples_path": manifest.get("examples_path"),
                        "record_count": manifest.get("record_count"),
                        "truncation_rate": manifest.get("truncation_rate"),
                    },
                ),
            },
        )
        return _utils._json_response(
            {
                "status": "prepared",
                "assistant_model": _serialize_assistant_model(updated),
                "manifest": manifest,
                "dataset": _serialize_assistant_training_dataset(dataset_record) if dataset_record is not None else None,
            }
        )
    except ValueError as exc:
        status_code = 404 if "was not found" in str(exc) else 400
        return _utils._json_error(str(exc), status_code=status_code)
    except Exception as exc:
        LOGGER.exception("Assistant model training tokenization failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/models/{assistant_model_id}/training/dataset/attach", response_class=JSONResponse)
async def assistant_model_attach_training_dataset(
    assistant_model_id: int | str,
    rebuild: bool = False,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        model_record = await _get_assistant_model_record(db, assistant_model_id)
        dataset_record = await _latest_assistant_training_dataset(db)
        if dataset_record is None or rebuild:
            manifest = build_assistant_training_dataset_snapshot(
                ASSISTANT_DATASET_DIR,
                ASSISTANT_EVAL_DIR,
                RUN_LEDGER_DIR,
                registry_records=await _assistant_training_registry_records(db),
                reference_dir=ASSISTANT_REFERENCE_DIR,
            )
            dataset_record = await _upsert_assistant_training_dataset_record(db, manifest)
        dataset_payload = _serialize_assistant_training_dataset(dataset_record)
        parameters = dict(getattr(model_record, "parameters", {}) or {})
        assistant_config = dict(parameters.get("assistant") or {})
        assistant_config.update(
            {
                "training_dataset_id": dataset_payload.get("id"),
                "training_dataset_path": (dataset_payload.get("manifest") or {}).get("canonical_jsonl_path") or dataset_payload.get("path"),
                "training_dataset_jsonl_path": (dataset_payload.get("manifest") or {}).get("canonical_jsonl_path"),
                "training_dataset_csv_path": (dataset_payload.get("manifest") or {}).get("tabular_csv_path") or dataset_payload.get("path"),
                "training_dataset_hash": dataset_payload.get("dataset_hash"),
                "training_dataset_tabular_hash": (dataset_payload.get("manifest") or {}).get("tabular_hash"),
                "training_dataset_manifest": dataset_payload.get("connection_string"),
            }
        )
        parameters["assistant"] = assistant_config
        updated = await update_entry(
            db,
            AssistantORM,
            getattr(model_record, "id"),
            {
                "parameters": parameters,
                "reference_data": (dataset_payload.get("manifest") or {}).get("canonical_jsonl_path") or dataset_payload.get("path"),
                "history": _utils._append_history(
                    getattr(model_record, "history", None),
                    {
                        "operation": "assistant_training_dataset_attached",
                        "when": datetime.now().isoformat(),
                        "dataset_id": dataset_payload.get("id"),
                        "dataset_hash": dataset_payload.get("dataset_hash"),
                        "tabular_hash": (dataset_payload.get("manifest") or {}).get("tabular_hash"),
                        "canonical_jsonl_path": (dataset_payload.get("manifest") or {}).get("canonical_jsonl_path"),
                        "tabular_csv_path": (dataset_payload.get("manifest") or {}).get("tabular_csv_path") or dataset_payload.get("path"),
                    },
                ),
            },
        )
        return _utils._json_response(
            {
                "status": "attached",
                "assistant_model": _serialize_assistant_model(updated),
                "dataset": dataset_payload,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model training dataset attach failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/assistant/provider/status", response_class=JSONResponse)
async def assistant_provider_status(run_eval: bool = False) -> JSONResponse:
    payload = get_assistant_provider_status()
    payload["runtime"] = assistant_runtime_status(timeout=2.0)
    payload["global_default_model_id"] = _assistant_default_model_id()
    if run_eval:
        evaluation = run_assistant_golden_evals(get_assistant_provider)
        artifact_paths: dict[str, str] = {}
        try:
            artifact_paths["evaluation_suite_path"] = str(persist_assistant_eval_suite(ASSISTANT_EVAL_DIR, evaluation))
        except Exception:
            LOGGER.debug("Assistant golden eval persistence failed.", exc_info=True)
        mlflow_tracking = log_assistant_mlflow_event(
            "golden_eval",
            evaluation=evaluation,
            artifact_paths=artifact_paths,
            extra_metrics={
                "golden_eval_cases": len(evaluation.get("results", []) or []),
                "golden_eval_passed": float(bool(evaluation.get("passed"))),
            },
        )
        payload["golden_eval"] = evaluation
        payload["golden_eval_artifacts"] = artifact_paths
        payload["mlflow_tracking"] = mlflow_tracking
    return _utils._json_response(payload)


@app.post("/assistant/provider/test", response_class=JSONResponse)
async def assistant_provider_test(
    provider: str = "auto",
    assistant_model_id: Optional[int | str] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if assistant_model_id not in (None, ""):
            model_record = await _get_assistant_model_record(db, assistant_model_id)
            result = test_assistant_model_contract(model_record)
        else:
            result = test_assistant_provider_contract(provider)
        return _utils._json_response(result)
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Assistant provider diagnostic failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/assistant/contracts/workflow-draft.schema", response_class=JSONResponse)
async def assistant_workflow_draft_schema() -> JSONResponse:
    return _utils._json_response(WorkflowDraft.model_json_schema())


@app.post("/assistant/evals/run", response_class=JSONResponse)
async def assistant_run_evals(provider: str = "amanaje_slm") -> JSONResponse:
    try:
        evaluation = run_assistant_golden_evals(get_assistant_provider, provider=provider)
        artifact_paths: dict[str, str] = {}
        try:
            artifact_paths["evaluation_suite_path"] = str(persist_assistant_eval_suite(ASSISTANT_EVAL_DIR, evaluation))
        except Exception:
            LOGGER.debug("Assistant golden eval persistence failed.", exc_info=True)
        mlflow_tracking = log_assistant_mlflow_event(
            "golden_eval",
            evaluation=evaluation,
            artifact_paths=artifact_paths,
            extra_params={"provider": provider},
            extra_metrics={
                "golden_eval_cases": len(evaluation.get("results", []) or []),
                "golden_eval_passed": float(bool(evaluation.get("passed"))),
            },
        )
        return _utils._json_response(
            {
                "status": "passed" if evaluation.get("passed") else "needs_revision",
                "evaluation": evaluation,
                "artifact_paths": artifact_paths,
                "mlflow_tracking": mlflow_tracking,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Assistant golden eval failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/training/curate", response_class=JSONResponse)
async def assistant_curate_training() -> JSONResponse:
    try:
        reference_sync = sync_assistant_reference_repository(
            ASSISTANT_REFERENCE_DIR,
            source_paths=resolve_assistant_reference_source_paths(ASSISTANT_REFERENCE_DIR, base_dir=REPO_ROOT),
            base_dir=REPO_ROOT,
            max_files=1000,
        )
        project_examples = build_example_catalog(routes=app.routes, include_payloads=True)
        manifest = curate_assistant_training_examples(
            ASSISTANT_DATASET_DIR,
            ASSISTANT_DATASET_DIR / "curated",
            project_examples=project_examples,
            reference_dir=ASSISTANT_REFERENCE_DIR,
        )
        artifact_paths = {
            "curated_path": manifest.get("curated_path"),
            "manifest_path": manifest.get("manifest_path"),
        }
        mlflow_tracking = log_assistant_mlflow_event(
            "training_curation",
            evaluation={"evaluation_profile": "assistant-training-curation-v1", "passed": bool(manifest.get("record_count", 0) >= 0)},
            artifact_paths=artifact_paths,
            extra_metrics={
                "curated_record_count": float(manifest.get("record_count", 0) or 0),
                "curated_project_example_records": float(manifest.get("project_example_records", 0) or 0),
                "curated_reference_records": float(manifest.get("reference_records", 0) or 0),
                "curation_rejected_records": float(manifest.get("rejected_records", 0) or 0),
                "curation_ready_for_fine_tuning": float(bool(manifest.get("ready_for_fine_tuning"))),
            },
        )
        return _utils._json_response(
            {
                "status": "curated",
                "manifest": manifest,
                "artifact_paths": artifact_paths,
                "mlflow_tracking": mlflow_tracking,
                "reference_sync": {
                    "created_count": reference_sync.get("created_count"),
                    "skipped_count": reference_sync.get("skipped_count"),
                    "scanned": reference_sync.get("scanned"),
                },
                "project_examples": {
                    "version": project_examples.get("version"),
                    "sample_count": (project_examples.get("summary") or {}).get("sample_count"),
                    "object_count": (project_examples.get("summary") or {}).get("object_count"),
                },
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant training curation failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/draft", response_class=JSONResponse)
async def assistant_draft(payload: AssistantDraftRequest, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        selected_assistant_model = None
        runtime_load_status: dict[str, Any] | None = None
        resolved_assistant_model_id = payload.assistant_model_id
        if resolved_assistant_model_id in (None, "") and str(payload.provider or "auto").strip().lower() in {"", "auto", "default", "active"}:
            default_model = await _resolve_default_assistant_model(db)
            if default_model is not None:
                selected_assistant_model = default_model
                resolved_assistant_model_id = getattr(default_model, "id", None)
        if selected_assistant_model is None and resolved_assistant_model_id not in (None, ""):
            selected_assistant_model = await _get_assistant_model_record(db, resolved_assistant_model_id)
        if selected_assistant_model is not None:
            provider_config = assistant_model_to_provider_config(selected_assistant_model, overrides=payload.provider_overrides)
            if _assistant_model_needs_runtime_load(selected_assistant_model, provider_config):
                runtime_load_status = ensure_assistant_model_runtime_loaded(selected_assistant_model, provider_config)
            provider = get_assistant_provider_for_model(
                selected_assistant_model,
                overrides=payload.provider_overrides,
            )
        else:
            provider = get_assistant_provider(payload.provider)
        reference_context = _assistant_reference_context()
        selected_references = _resolve_assistant_reference_pack(payload)
        assistant_model_payload = (
            _serialize_assistant_model(selected_assistant_model)
            if selected_assistant_model is not None
            else None
        )
        provider_payload = payload.model_copy(
            update={
                "provider": provider.name,
                "context": {
                    **dict(payload.context or {}),
                    "assistant_model": assistant_model_payload,
                    "assistant_model_id": resolved_assistant_model_id,
                    "assistant_runtime_load": runtime_load_status,
                    "assistant_reference_snapshot": reference_context,
                    "content_references": selected_references,
                }
            },
            deep=True,
        )
        draft = provider.create_draft(provider_payload)
        review = review_workflow_draft(draft)
        evaluation = evaluate_draft_contract(draft, review)
        artifact_paths = _persist_assistant_review_artifacts(draft, evaluation)
        training_example_path = None
        if not review.approved:
            training_example_path = _record_assistant_training_example(
                label="negative",
                draft=draft,
                review=review,
                outcome={"stage": "draft_review", "status": review.status},
            )
        mlflow_tracking = log_assistant_mlflow_event(
            "draft_review",
            draft=draft,
            review=review,
            evaluation=evaluation,
            artifact_paths=artifact_paths,
            training_example_path=training_example_path,
            extra_metrics={
                "assistant_alignment_passed": float(bool(evaluation.get("passed"))),
            },
        )
        run_entry = _utils.create_run_entry(
            RUN_LEDGER_DIR,
            run_type="assistant",
            context={
                "session_id": draft.session_id,
                "draft_id": draft.draft_id,
                "draft_type": draft.draft_type,
                "context_pack_id": draft.context_pack_id,
                "context_pack_hash": draft.context_pack_hash,
                "model_version": draft.model_version,
                "prompt_template_version": draft.prompt_template_version,
                "evaluation_profile": draft.evaluation_profile,
                "requested_reference_ids": list(payload.reference_ids),
                "selected_reference_ids": draft.provider_metadata.get("selected_reference_ids", []),
                "reference_pack_hash": draft.provider_metadata.get("reference_pack_hash"),
                "assistant_model_id": resolved_assistant_model_id,
                "assistant_runtime_load": runtime_load_status,
            },
            parameters={
                "prompt": payload.prompt,
                "provider": provider.name,
                "requested_provider": payload.provider,
                "assistant_model_id": resolved_assistant_model_id,
                "assistant_model_name": assistant_model_payload.get("name") if assistant_model_payload else None,
                "target_type": payload.target_type,
                "safety_profile": draft.execution_profile,
                "reference_count": len(selected_references),
                "reference_selection": "automatic",
            },
            status="queued",
        )
        run_entry = _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_entry["run_id"],
            status="completed",
            stage="reviewed" if review.approved else "needs_revision",
            merge={
                "result": {
                    "draft": _utils._json_payload(draft),
                    "review": _utils._json_payload(review),
                    "evaluation": evaluation,
                    "artifact_paths": artifact_paths,
                    "training_example_path": training_example_path,
                    "mlflow_tracking": mlflow_tracking,
                    "selected_references": selected_references,
                },
                "metrics": {
                    "assistant_latency_ms": draft.provider_metadata.get("latency_ms"),
                    "assistant_validation_passed": bool(review.approved),
                    "assistant_alignment_passed": bool(evaluation.get("passed")),
                    "assistant_fallback_used": bool(draft.provider_metadata.get("fallback_provider")),
                },
            },
            event_message="Assistant draft created and reviewed.",
        )
        return _utils._json_response(
            {
                "status": "reviewed" if review.approved else "needs_revision",
                "run_id": run_entry["run_id"],
                "draft": draft,
                "review": review,
                "evaluation": evaluation,
                "artifact_paths": artifact_paths,
                "mlflow_tracking": mlflow_tracking,
                "runtime_load": runtime_load_status,
                "ledger": run_entry,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Assistant draft failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/review", response_class=JSONResponse)
async def assistant_review(payload: AssistantReviewRequest) -> JSONResponse:
    try:
        review = review_workflow_draft(payload.draft)
        return _utils._json_response(
            {
                "status": review.status,
                "approved": review.approved,
                "review": review,
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant review failed.")
        return _utils._json_error(str(exc), status_code=400)


@app.post("/assistant/feedback", response_class=JSONResponse)
async def assistant_feedback(payload: AssistantFeedbackRequest) -> JSONResponse:
    try:
        review = review_workflow_draft(payload.draft)
        normalized_label = (payload.label or "negative").strip().lower()
        if normalized_label not in {"positive", "corrected", "negative", "unsafe", "rejected"}:
            normalized_label = "negative"
        training_example_path = _record_assistant_training_example(
            label=normalized_label,
            draft=payload.draft,
            review=review,
            outcome={
                "stage": "feedback",
                "status": review.status,
                "reason": payload.reason,
                "notes": payload.notes,
            },
            user_edits=payload.user_edits,
        )
        evaluation = evaluate_draft_contract(payload.draft, review)
        mlflow_tracking = log_assistant_mlflow_event(
            "feedback",
            draft=payload.draft,
            review=review,
            evaluation=evaluation,
            training_example_path=training_example_path,
            extra_params={"feedback_label": normalized_label, "feedback_reason": payload.reason},
            extra_metrics={
                "assistant_feedback_positive": float(normalized_label in {"positive", "corrected"}),
                "assistant_feedback_negative": float(normalized_label in {"negative", "unsafe", "rejected"}),
            },
        )
        updated_entry = None
        if payload.run_id:
            existing_entry = _utils.read_run_entry(RUN_LEDGER_DIR, payload.run_id)
            if existing_entry is not None:
                updated_entry = _utils.update_run_entry(
                    RUN_LEDGER_DIR,
                    payload.run_id,
                    status=existing_entry.get("status", "completed"),
                    stage=existing_entry.get("stage", "feedback"),
                    merge={
                        "result": {
                            "feedback": {
                                "label": normalized_label,
                                "reason": payload.reason,
                                "notes": payload.notes,
                                "training_example_path": training_example_path,
                                "mlflow_tracking": mlflow_tracking,
                            }
                        }
                    },
                    event_message="Assistant feedback recorded.",
                )
        return _utils._json_response(
            {
                "status": "recorded",
                "label": normalized_label,
                "training_example_path": training_example_path,
                "evaluation": evaluation,
                "mlflow_tracking": mlflow_tracking,
                "ledger": updated_entry,
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant feedback failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/approve", response_class=JSONResponse)
async def assistant_approve(payload: AssistantApprovalRequest) -> JSONResponse:
    try:
        review = review_workflow_draft(payload.draft)
        if not review.approved:
            return _utils._json_error(
                "Draft cannot be approved until required review actions are resolved.",
                status_code=400,
                review=review,
            )

        run_id = payload.run_id
        if run_id:
            existing_entry = _utils.read_run_entry(RUN_LEDGER_DIR, run_id)
            if existing_entry is None:
                return _utils._json_error("Assistant run not found", status_code=404)
        else:
            run_entry = _utils.create_run_entry(
                RUN_LEDGER_DIR,
                run_type="assistant",
                context={
                    "session_id": payload.draft.session_id,
                    "draft_id": payload.draft.draft_id,
                    "draft_type": payload.draft.draft_type,
                },
                parameters={"scope": "approval"},
                status="queued",
            )
            run_id = str(run_entry["run_id"])

        approval = {
            "approval_id": f"approval_{uuid.uuid4().hex[:12]}",
            "reviewer": payload.reviewer,
            "notes": payload.notes,
            "approved_at": datetime.now().isoformat(),
        }
        training_example_path = _record_assistant_training_example(
            label="positive",
            draft=payload.draft,
            review=review,
            approval=approval,
            final_payload={"action": "approved"},
            outcome={"stage": "approved", "status": review.status},
            user_edits=payload.user_edits,
        )
        evaluation = evaluate_draft_contract(payload.draft, review)
        mlflow_tracking = log_assistant_mlflow_event(
            "approval",
            draft=payload.draft,
            review=review,
            evaluation=evaluation,
            training_example_path=training_example_path,
            extra_metrics={
                "assistant_approved": 1.0,
                "assistant_alignment_passed": float(bool(evaluation.get("passed"))),
            },
        )
        updated_entry = _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="completed",
            stage="approved",
            merge={
                "result": {
                    "draft": _utils._json_payload(payload.draft),
                    "review": _utils._json_payload(review),
                    "approval": approval,
                    "training_example_path": training_example_path,
                    "mlflow_tracking": mlflow_tracking,
                }
            },
            event_message="Assistant draft approved.",
        )
        return _utils._json_response(
            {
                "status": "approved",
                "run_id": run_id,
                "approval": approval,
                "review": review,
                "evaluation": evaluation,
                "mlflow_tracking": mlflow_tracking,
                "ledger": updated_entry,
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant approval failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/submit", response_class=JSONResponse)
async def assistant_submit(
    payload: AssistantSubmitRequest,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    run_entry = _utils.read_run_entry(RUN_LEDGER_DIR, payload.run_id)
    if run_entry is None:
        return _utils._json_error("Assistant run not found", status_code=404)

    result = dict(run_entry.get("result", {}) or {})
    if not result.get("approval"):
        return _utils._json_error("Assistant run must be approved before submission.", status_code=400)

    try:
        draft = WorkflowDraft.model_validate(result.get("draft"))
    except Exception as exc:
        return _utils._json_error(f"Assistant run does not contain a valid draft: {exc}", status_code=400)

    try:
        submission = await _submit_assistant_draft(draft, payload, db)
        review = review_workflow_draft(draft)
        evaluation = evaluate_draft_contract(draft, review)
        mlflow_tracking = log_assistant_mlflow_event(
            "submission",
            draft=draft,
            review=review,
            evaluation=evaluation,
            extra_params={"submit_action": payload.action},
            extra_metrics={"assistant_submitted": 1.0},
        )
        updated_entry = _utils.update_run_entry(
            RUN_LEDGER_DIR,
            payload.run_id,
            status="completed",
            stage="submitted",
            merge={"result": {"submission": submission, "submission_mlflow_tracking": mlflow_tracking}},
            event_message=f"Assistant draft submitted with action={payload.action}.",
        )
        return _utils._json_response(
            {
                "status": "submitted",
                "run_id": payload.run_id,
                "draft_type": draft.draft_type,
                "action": payload.action,
                "submission": submission,
                "mlflow_tracking": mlflow_tracking,
                "ledger": updated_entry,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Assistant submission failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/execution/review", response_class=JSONResponse)
async def execution_review(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    code = str(payload.get("code") or "")
    profile = str(payload.get("profile") or "dataset_generation")
    safety = review_code_safety(code, profile=profile)
    return _utils._json_response(
        {
            "status": "approved" if safety.ok else "needs_revision",
            "approved": safety.ok,
            "safety": safety,
        },
        status_code=200 if safety.ok else 400,
    )


@app.post("/execution/run", response_class=JSONResponse)
async def execution_run(payload: ExecutionRunRequest) -> JSONResponse:
    if not payload.approved:
        return _utils._json_error("approved=true is required before guarded execution.", status_code=400)

    safety = review_code_safety(payload.code, profile=payload.profile)
    if not safety.ok:
        return _utils._json_error(
            "Code did not pass assistant safety review.",
            status_code=400,
            safety=safety,
        )

    run_entry = _utils.create_run_entry(
        RUN_LEDGER_DIR,
        run_type="assistant",
        context={
            "draft_id": payload.draft_id,
            "profile": payload.profile,
            **dict(payload.context or {}),
        },
        parameters={"scope": "execution_run"},
        status="running",
    )

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    namespace = _build_guarded_execution_namespace(payload.profile)
    error_message = None
    try:
        exec(compile(payload.code, "<assistant-execution>", "exec"), namespace, namespace)
    except Exception:
        error_message = traceback.format_exc()
    finally:
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    extracted_variables = _utils._extract_variables(namespace)
    output_variables = {
        name: value
        for name, value in extracted_variables.items()
        if name in APPROVED_EXECUTION_OUTPUT_FIELDS
    }
    materialized_outputs = {
        name: payload.get("raw_value", payload.get("value"))
        for name, payload in output_variables.items()
        if "raw_value" in payload or "value" in payload
    }
    metadata = dict(materialized_outputs)
    defined_names = [name for name in _utils._extract_defined_names(payload.code) if name in APPROVED_EXECUTION_OUTPUT_FIELDS]
    status = "success" if error_message is None else "error"
    mlflow_tracking = log_assistant_mlflow_event(
        "guarded_execution",
        evaluation={
            "evaluation_profile": "assistant-guarded-execution-v1",
            "passed": error_message is None,
            "checks": {"valid_workflow_draft": True},
            "alignment_contracts": {
                "output_alignment": error_message is None,
                "safety_alignment": safety.ok,
                "registry_alignment": True,
                "dependency_alignment": safety.ok,
                "mlflow_alignment": True,
            },
        },
        extra_params={
            "draft_id": payload.draft_id,
            "profile": payload.profile,
            "context_pack_hash": dict(payload.context or {}).get("context_pack_hash"),
        },
        extra_metrics={
            "assistant_execution_success": float(error_message is None),
            "assistant_output_count": float(len(output_variables)),
        },
    )

    updated_entry = _utils.update_run_entry(
        RUN_LEDGER_DIR,
        run_entry["run_id"],
        status="completed" if error_message is None else "failed",
        stage="executed" if error_message is None else "failed",
        merge={
            "result": {
                "status": status,
                "metadata_keys": sorted(metadata),
                "defined_names": defined_names,
                "safety": _utils._json_payload(safety),
                "context_pack_hash": dict(payload.context or {}).get("context_pack_hash"),
                "mlflow_tracking": mlflow_tracking,
            },
            "metrics": {
                "assistant_execution_success": error_message is None,
                "assistant_output_count": len(output_variables),
            },
            "error": error_message,
        },
        event_message="Assistant guarded execution completed." if error_message is None else "Assistant guarded execution failed.",
    )

    return _utils._json_response(
        {
            "status": status,
            "run_id": run_entry["run_id"],
            "profile": payload.profile,
            "variables": output_variables,
            "metadata": metadata,
            "materialized_outputs": materialized_outputs,
            "summary": {
                "variable_count": len(output_variables),
                "defined_name_count": len(defined_names),
                "defined_names": defined_names,
                "has_error": error_message is not None,
            },
            "stdout": stdout_output,
            "stderr": stderr_output,
            "error": error_message,
            "safety": safety,
            "mlflow_tracking": mlflow_tracking,
            "ledger": updated_entry,
        },
        status_code=200 if error_message is None else 400,
    )


def _build_guarded_execution_namespace(profile: str) -> dict[str, Any]:
    allowed_imports = set(SAFETY_PROFILES.get(profile, SAFETY_PROFILES["dataset_generation"])["allowed_imports"])

    def _safe_import(
        name: str,
        globals_: Optional[dict[str, Any]] = None,
        locals_: Optional[dict[str, Any]] = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        root = str(name or "").split(".", 1)[0]
        if level != 0 or root not in allowed_imports:
            raise ImportError(f"Import '{name}' is not allowed for assistant profile '{profile}'.")
        return __import__(name, globals_, locals_, fromlist, level)

    safe_builtins = {**SAFE_EXECUTION_BUILTINS, "__import__": _safe_import}
    return {"__builtins__": safe_builtins}


async def _submit_assistant_draft(
    draft: WorkflowDraft,
    payload: AssistantSubmitRequest,
    db: AsyncSession,
) -> dict[str, Any]:
    action = (payload.action or "prepare").strip().lower()
    if action == "prepare":
        return _assistant_prepare_payload(draft)
    if action == "feature_preview":
        return await _assistant_feature_preview(draft, payload, db)
    if action == "training":
        return await _assistant_training_submit(draft, payload, db)
    raise ValueError("action must be one of: prepare, feature_preview, training")


def _assistant_prepare_payload(draft: WorkflowDraft) -> dict[str, Any]:
    next_endpoint_by_type = {
        "dataset_generation": "/execution/review",
        "feature_operations": "/features/preview",
        "model_generation": "/execution/review",
        "registry_object": "/registry",
        "study": "/studies/{study_id}/optimize",
        "training_run": "/training/{model_id}",
    }
    return {
        "action": "prepare",
        "next_endpoint": next_endpoint_by_type.get(draft.draft_type),
        "draft": _utils._json_payload(draft),
        "message": "Draft is approved and ready for an explicit workflow action.",
    }


async def _assistant_feature_preview(
    draft: WorkflowDraft,
    payload: AssistantSubmitRequest,
    db: AsyncSession,
) -> dict[str, Any]:
    if draft.draft_type != "feature_operations":
        raise ValueError("feature_preview action requires a feature_operations draft.")

    request_payload = {
        **dict(draft.context or {}),
        **dict(payload.payload or {}),
        "transforms": list(payload.payload.get("transforms") or draft.operations),
    }
    dataset_id = request_payload.get("datasetId") or request_payload.get("dataset_id")
    if dataset_id in (None, "", 0, "0"):
        raise ValueError("datasetId is required for feature_preview submission.")

    dataset = await DatasetModel.read(db, int(dataset_id))
    if dataset is None:
        raise ValueError("Dataset not found")

    dataframe = _utils._load_dataset_frame_from_record(dataset)
    operations = _utils._coerce_feature_operations_from_payload(request_payload)
    transformed, transform_summary = _utils.apply_feature_operations(dataframe, operations)
    preview = _utils._build_feature_workspace_payload(
        dataset,
        transformed,
        preview=transform_summary,
        operations=operations,
    )
    return {
        "action": "feature_preview",
        "dataset_id": int(dataset_id),
        "operations": operations,
        "preview": preview,
    }


async def _assistant_training_submit(
    draft: WorkflowDraft,
    payload: AssistantSubmitRequest,
    db: AsyncSession,
) -> dict[str, Any]:
    if draft.draft_type != "training_run":
        raise ValueError("training action requires a training_run draft.")

    training_request = {
        **dict(draft.training_request or {}),
        **dict(payload.payload or {}),
    }
    model_id = training_request.get("modelId") or training_request.get("model_id")
    dataset_id = training_request.get("datasetId") or training_request.get("dataset_id")
    if model_id in (None, "", 0, "0") or dataset_id in (None, "", 0, "0"):
        raise ValueError("modelId and datasetId are required for training submission.")

    request_payload = TrainingRequest.model_validate(
        {
            "datasetId": int(dataset_id),
            "inputType": training_request.get("inputType") or "Assistant",
            "studyId": training_request.get("studyId"),
            "parameters": dict(training_request.get("parameters") or {}),
            "inferenceId": training_request.get("inferenceId"),
        }
    )
    response = await post_training(int(model_id), payload=request_payload, db=db)
    response_payload = json.loads(response.body.decode("utf-8"))
    if response.status_code >= 400:
        raise ValueError(response_payload.get("detail") or "Training submission failed.")
    return {
        "action": "training",
        "model_id": int(model_id),
        "dataset_id": int(dataset_id),
        "training": response_payload,
    }


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in (None, "", 0, "0"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_watch_context_id(
    payload_data: Mapping[str, Any] | None,
    *,
    model_id: Any = None,
    dataset_id: Any = None,
    inference_id: Any = None,
) -> str:
    payload_data = payload_data or {}
    requested_watch_context_id = payload_data.get("watchContextId")
    if requested_watch_context_id not in (None, ""):
        return str(requested_watch_context_id)
    return _utils._watch_context_id(model_id=model_id, dataset_id=dataset_id, inference_id=inference_id)


def _update_run_progress(
    run_id: str,
    *,
    stage: str,
    message: str,
    status: Optional[str] = None,
    counters: Optional[Mapping[str, Any]] = None,
) -> None:
    _utils.update_run_entry(
        RUN_LEDGER_DIR,
        run_id,
        status=status,
        stage=stage,
        event_message=message,
        merge={
            "progress": {
                "current_stage": stage,
                "latest_event": message,
                "latest_timestamp": datetime.now().isoformat(),
                "counters": dict(counters or {}),
            }
        },
    )


def _redact_traceback_text(value: str) -> str:
    return SECRET_TRACEBACK_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", str(value or ""))


def _record_run_exception(run_id: str, exc: BaseException, *, stage: str = "failed") -> dict[str, Any]:
    traceback_text = _redact_traceback_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    failure = {
        "type": exc.__class__.__name__,
        "message": _redact_traceback_text(str(exc)),
        "traceback": traceback_text,
        "worker_name": os.getenv("AMANAJE_EFFECTIVE_WORKER_NAME") or os.getenv("AMANAJE_WORKER_NAME"),
        "queue": os.getenv("AMANAJE_WORKER_QUEUE"),
        "job_id": run_id,
        "failed_at": datetime.now().isoformat(),
    }
    try:
        _utils.append_run_terminal_line(RUN_LEDGER_DIR, run_id, message=traceback_text, stream="stderr", stage=stage)
    except Exception:
        LOGGER.exception("Unable to append failure traceback to run terminal for %s.", run_id)
    return failure


async def _execute_training_run(
    run_id: str,
    *,
    model_id: int,
    payload_data: Mapping[str, Any],
) -> None:
    try:
        _update_run_progress(
            run_id,
            status="running",
            stage="resolving_context",
            message="Resolving model, dataset, and optional study context.",
        )
        validated_payload = validate_input(dict(payload_data), _utils.TrainingRequest)
        async with _utils.AsyncSessionLocal() as db:
            learning_model = await LearningModel.read(db, model_id)
            dataset_model = await DatasetModel.read(db, validated_payload.datasetId)
            if learning_model is None:
                raise ValueError("Learning model not found")
            if dataset_model is None:
                raise ValueError("Dataset not found")

            study_record = None
            if validated_payload.studyId not in (None, "", 0, "0"):
                study_record = await StudyModel.read(db, validated_payload.studyId)
                if study_record is None:
                    raise ValueError("StudyModel not found")

            merged_parameters = _utils._merge_training_parameters(learning_model, validated_payload, study_record)
            merged_parameters["training_mode"] = str(merged_parameters.get("training_mode") or "transfer").strip().lower()
            if merged_parameters["training_mode"] not in {"transfer", "fresh"}:
                merged_parameters["training_mode"] = "transfer"
            base_model_record = learning_model
            base_model_id = _coerce_optional_int(merged_parameters.get("base_model_id")) or model_id
            if base_model_id != model_id:
                base_model_record = await LearningModel.read(db, base_model_id)
                if base_model_record is None:
                    raise ValueError("Base LearningModel not found for transfer training.")
            if merged_parameters["training_mode"] == "transfer":
                base_parameters = dict(getattr(base_model_record, "parameters", {}) or {})
                base_lineage = dict(base_parameters.get("training_lineage") or {})
                merged_parameters.setdefault("base_model_id", getattr(base_model_record, "id", base_model_id))
                merged_parameters.setdefault("base_artifact_path", getattr(base_model_record, "path", None))
                merged_parameters.setdefault("base_run_id", base_lineage.get("latest_run_id"))
                merged_parameters.setdefault("transfer_strategy", "auto")
            validate_required_keys(merged_parameters, ["framework"])
            preflight = _utils._build_training_preflight(dataset_model, learning_model, merged_parameters)
            if not preflight.get("ok", False):
                raise ValueError(
                    f"Missing required columns: {preflight.get('missing_columns', [])}. "
                    f"Available columns: {preflight.get('available_columns', [])}"
                )
            _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                stage="training",
                merge={
                    "context": {
                        "model_id": model_id,
                        "dataset_id": validated_payload.datasetId,
                        "study_id": _coerce_optional_int(validated_payload.studyId),
                    },
                    "parameters": merged_parameters,
                },
                event_message="Training workflow started.",
            )
            _update_run_progress(
                run_id,
                stage="validating_schema",
                message="Training preflight completed. Required dataset columns are available.",
            )

            workflow = await asyncio.to_thread(
                lambda: _utils._run_training_workflow(
                    learning_model,
                    dataset_model,
                    merged_parameters,
                    job_id=run_id,
                    progress_callback=lambda event: _update_run_progress(
                        run_id,
                        stage=str(event.get("stage") or "training"),
                        message=str(event.get("message") or "Training workflow updated."),
                        counters=event.get("counters") if isinstance(event.get("counters"), Mapping) else None,
                    ),
                )
            )
            result = workflow["result"]
            prepared: _utils.PreparedDataset = workflow["prepared"]
            monitoring = workflow["monitoring"]
            artifacts = workflow["artifacts"]
            transfer_learning = dict(workflow.get("transfer_learning") or {})
            previous_parameters = dict(getattr(learning_model, "parameters", {}) or {})
            previous_lineage = dict(previous_parameters.get("training_lineage") or {})
            try:
                next_iteration = int(previous_lineage.get("iteration") or 0) + 1
            except (TypeError, ValueError):
                next_iteration = 1
            training_lineage = {
                "iteration": next_iteration,
                "latest_run_id": run_id,
                "base_run_id": transfer_learning.get("base_run_id"),
                "base_model_id": transfer_learning.get("base_model_id"),
                "base_artifact_path": transfer_learning.get("base_artifact_path"),
                "previous_artifact_path": getattr(learning_model, "path", None),
                "new_artifact_path": artifacts.get("model_artifact_path"),
                "transfer_status": transfer_learning.get("transfer_status"),
                "transfer_strategy": transfer_learning.get("transfer_strategy"),
                "training_mode": transfer_learning.get("training_mode"),
                "dataset_snapshot_hash": transfer_learning.get("dataset_snapshot_hash"),
                "mlflow_run_id": result.run_id,
            }

            updated_model = await update_entry(
                db,
                LearningORM,
                model_id,
                {
                    "parameters": {
                        **previous_parameters,
                        **result.parameters,
                        **merged_parameters,
                        "artifact_manifest": artifacts.get("artifact_manifest", {}),
                        "training_lineage": training_lineage,
                    },
                    "metrics": result.metrics,
                    "reference_data": getattr(dataset_model, "name", None),
                    "input_features": prepared.input_features,
                    "output_features": prepared.output_features or ([prepared.output_feature] if prepared.output_feature else []),
                    "is_trained": True,
                    "is_tested": True,
                    "path": artifacts["model_artifact_path"] or getattr(learning_model, "path", None),
                    "history": _utils._append_history(
                        getattr(learning_model, "history", None),
                        {
                            "operation": "training",
                            "when": datetime.now().isoformat(),
                            "dataset_id": validated_payload.datasetId,
                            "framework": result.framework,
                            "job_id": run_id,
                            "run_ledger_id": run_id,
                            "mlflow_run_id": result.run_id,
                            "artifact_path": artifacts.get("model_artifact_path"),
                            "transfer_learning": transfer_learning,
                            "training_lineage": training_lineage,
                        },
                    ),
                },
            )
            inference_record = await _upsert_inference_record(
                db,
                model_record=updated_model,
                dataset_record=dataset_model,
                input_features=prepared.input_features,
                output_features=prepared.output_features or ([prepared.output_feature] if prepared.output_feature else []),
                inference_updates={
                    "status": "trained",
                    "framework": result.framework,
                    "latest_metrics": result.metrics,
                    "monitoring": monitoring,
                    "artifacts": artifacts,
                    "artifact_manifest": artifacts.get("artifact_manifest", {}),
                    "last_job_id": run_id,
                    "last_trained_at": datetime.now().isoformat(),
                    "simulation_defaults": {
                        "steps": 12,
                        "strategy": "carry-forward",
                        "amplitude": 0.05,
                    },
                },
                history_entry={
                    "operation": "training_sync",
                    "when": datetime.now().isoformat(),
                    "job_id": run_id,
                    "model_id": model_id,
                    "dataset_id": validated_payload.datasetId,
                },
                artifact_path=artifacts["model_artifact_path"] or getattr(updated_model, "path", None),
            )

            if study_record is not None:
                await update_entry(
                    db,
                    StudyORM,
                    study_record.id,
                    {
                        "history": _utils._append_history(
                            getattr(study_record, "history", None),
                            {
                                "operation": "training_run",
                                "when": datetime.now().isoformat(),
                                "job_id": run_id,
                                "model_id": model_id,
                                "dataset_id": validated_payload.datasetId,
                            },
                        )
                    },
                )

            _update_run_progress(
                run_id,
                stage="syncing_inference",
                message="Persisting run outputs to the registry and inference pair.",
            )
            result_payload = {
                "model_id": model_id,
                "dataset_id": validated_payload.datasetId,
                "input_type": validated_payload.inputType,
                "study_id": validated_payload.studyId,
                "framework": result.framework,
                "metrics": result.metrics,
                "parameters": merged_parameters,
                "monitoring": monitoring,
                "artifacts": artifacts,
                "plots": _list_plot_artifacts(job_id=run_id),
                "transfer_learning": transfer_learning,
                "training_lineage": training_lineage,
                "mlflow_run_id": result.run_id,
                "model": _utils._serialize_model_summary(updated_model),
                "inference": _serialize_inference_summary(inference_record),
            }
            _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                status="completed",
                stage="completed",
                merge={
                    "metrics": result.metrics,
                    "artifacts": artifacts,
                    "plots": _list_plot_artifacts(job_id=run_id),
                    "result": result_payload,
                    "mlflow_run_id": result.run_id,
                    "progress": {
                        "counters": {
                            "mlflow_run_id": result.run_id,
                            "metric_count": len(result.metrics or {}),
                        }
                    },
                },
                event_message="Training workflow completed.",
            )
            _append_activity_log(
                f"Completed training job {run_id} for model_id={model_id} dataset_id={validated_payload.datasetId}.",
                event_type="training.completed",
                details={
                    "job_id": run_id,
                    "model_id": model_id,
                    "dataset_id": validated_payload.datasetId,
                    "framework": result.framework,
                    "metrics": result.metrics,
                },
            )
            log_to_mlflow(LOGGER, f"Completed training job {run_id} for model {model_id}.")
    except Exception as exc:
        LOGGER.exception("Training failed for model_id=%s.", model_id)
        failure = _record_run_exception(run_id, exc)
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage="failed",
            merge={"error": str(exc), "failure": failure},
            event_message=str(exc),
        )
    finally:
        BACKGROUND_RUN_TASKS.pop(run_id, None)


async def _execute_study_run(
    run_id: str,
    *,
    study_id: int,
    payload_data: Mapping[str, Any],
) -> None:
    try:
        _update_run_progress(
            run_id,
            status="running",
            stage="resolving_context",
            message="Resolving study, model, and dataset context.",
        )
        request_payload = validate_input(dict(payload_data), _utils.StudyOptimizationRequest)
        async with _utils.AsyncSessionLocal() as db:
            study_record = await StudyModel.read(db, study_id)
            if study_record is None:
                raise ValueError("StudyModel not found")
            learning_model = await LearningModel.read(db, int(study_record.learning_model_id))
            dataset_model = await DatasetModel.read(db, int(study_record.dataset_id))
            if learning_model is None or dataset_model is None:
                raise ValueError("The selected study is not linked to a valid model and dataset.")
            preflight = _utils._build_training_preflight(
                dataset_model,
                learning_model,
                {
                    **dict(getattr(learning_model, "parameters", {}) or {}),
                    **dict(getattr(study_record, "study_params", {}) or {}),
                },
            )
            if not preflight.get("ok", False):
                raise ValueError(
                    f"Missing required columns: {preflight.get('missing_columns', [])}. "
                    f"Available columns: {preflight.get('available_columns', [])}"
                )
            _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                merge={
                    "context": {
                        "study_id": study_id,
                        "model_id": getattr(study_record, "learning_model_id", None),
                        "dataset_id": getattr(study_record, "dataset_id", None),
                    },
                    "parameters": request_payload.model_dump(),
                },
                event_message="Study context resolved.",
            )
            _update_run_progress(
                run_id,
                stage="validating_schema",
                message="Study preflight completed. Required dataset columns are available.",
            )
            result_payload = await _utils._optimize_study(
                study_record,
                db,
                request_payload,
                progress_callback=lambda event: _update_run_progress(
                    run_id,
                    stage=str(event.get("stage") or "optimizing"),
                    message=str(event.get("message") or "Study optimization updated."),
                    counters=event.get("counters") if isinstance(event.get("counters"), Mapping) else {
                        key: value
                        for key, value in {
                            "trial_number": event.get("trial_number"),
                            "completed_trials": event.get("completed_trials"),
                            "total_trials": event.get("total_trials"),
                            "best_value": event.get("best_value"),
                        }.items()
                        if value is not None
                    },
                ),
            )
            _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                status="completed",
                stage="completed",
                merge={
                    "result": result_payload,
                    "progress": {
                        "counters": {
                            "completed_trials": dict(result_payload.get("summary", {}) or {}).get("completed_trials"),
                            "n_trials": dict(result_payload.get("summary", {}) or {}).get("n_trials"),
                        }
                    },
                },
                event_message="Study optimization completed.",
            )
    except Exception as exc:
        LOGGER.exception("Optuna optimization failed for study_id=%s.", study_id)
        failure = _record_run_exception(run_id, exc)
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage="failed",
            merge={
                "error": str(exc),
                "failure": failure,
                "progress": {"latest_event": str(exc)},
            },
            event_message=str(exc),
        )
    finally:
        BACKGROUND_RUN_TASKS.pop(run_id, None)


@app.post("/training/{model_id}", response_class=JSONResponse)
async def post_training(
    model_id: int,
    payload: TrainingRequest,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        validated_payload = validate_input(payload.model_dump(), _utils.TrainingRequest)
        payload_dict = validated_payload.model_dump()
        if validated_payload.datasetId <= 0:
            return _utils._json_error("datasetId must be a positive integer", status_code=400)
        learning_model = await LearningModel.read(db, model_id)
        dataset_model = await DatasetModel.read(db, validated_payload.datasetId)
        if learning_model is None:
            return _utils._json_error("Learning model not found", status_code=404)
        if dataset_model is None:
            return _utils._json_error("Dataset not found", status_code=404)
        study_record = None
        if validated_payload.studyId not in (None, "", 0, "0"):
            study_record = await StudyModel.read(db, validated_payload.studyId)
            if study_record is None:
                return _utils._json_error("StudyModel not found", status_code=404)
        merged_parameters = _utils._merge_training_parameters(learning_model, validated_payload, study_record)
        preflight = _utils._build_training_preflight(dataset_model, learning_model, merged_parameters)
        if not preflight.get("ok", False):
            return _utils._json_error(
                "The selected dataset does not contain all required training columns.",
                status_code=400,
                missing_columns=preflight.get("missing_columns", []),
                available_columns=preflight.get("available_columns", []),
                recommended_pairing=preflight.get("recommended_pairing", {}),
            )
        run_entry = _utils.create_run_entry(
            RUN_LEDGER_DIR,
            run_type="training",
            context={
                "model_id": model_id,
                "dataset_id": validated_payload.datasetId,
                "study_id": _coerce_optional_int(validated_payload.studyId),
                "inference_id": _coerce_optional_int(payload_dict.get("inferenceId")),
            },
            parameters=merged_parameters,
            status="queued",
        )
        try:
            enqueue_result = enqueue_training_run(
                run_entry["run_id"],
                model_id=model_id,
                payload_data=validated_payload.model_dump(),
                prefer_gpu=_prefers_gpu_queue(merged_parameters),
            )
        except QueueUnavailableError as exc:
            _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_entry["run_id"],
                status="failed",
                stage="failed",
                merge={"error": str(exc), "error_code": exc.error_code},
                event_message=str(exc),
            )
            return _queue_unavailable_response(exc, run_id=run_entry["run_id"], ledger=run_entry)
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_entry["run_id"],
            merge={
                "queue": enqueue_result,
                "rerun_payload": {
                    "run_type": "training",
                    "model_id": model_id,
                    "payload_data": validated_payload.model_dump(),
                    "parameters": merged_parameters,
                },
            },
            event_message=f"Training run enqueued on {enqueue_result.get('queue')}.",
        )
        return _utils._json_response(
            {
                "status": "accepted",
                "run_id": run_entry["run_id"],
                "job_id": run_entry["run_id"],
                "model_id": model_id,
                "dataset_id": validated_payload.datasetId,
                "study_id": validated_payload.studyId,
                "ledger": {**run_entry, "queue": enqueue_result},
                "queue": enqueue_result,
            },
            status_code=202,
        )
    except ValueError as exc:
        LOGGER.exception("Training failed for model_id=%s.", model_id)
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Training failed for model_id=%s.", model_id)
        return _utils._json_error(str(exc), status_code=500)


@app.post("/studies/{study_id}/optimize", response_class=JSONResponse)
async def post_optimize_study(
    study_id: int,
    payload: StudyOptimizationRequest = Body(default=_utils.StudyOptimizationRequest()),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        study_record = await StudyModel.read(db, study_id)
        if study_record is None:
            return _utils._json_error("StudyModel not found", status_code=404)
        learning_model = await LearningModel.read(db, int(study_record.learning_model_id))
        dataset_model = await DatasetModel.read(db, int(study_record.dataset_id))
        if learning_model is None or dataset_model is None:
            return _utils._json_error("The selected study is not linked to a valid model and dataset.", status_code=400)
        study_queue_parameters = {
            **dict(getattr(learning_model, "parameters", {}) or {}),
            **dict(getattr(study_record, "study_params", {}) or {}),
            **dict(payload.model_dump() or {}),
        }
        preflight = _utils._build_training_preflight(
            dataset_model,
            learning_model,
            study_queue_parameters,
        )
        if not preflight.get("ok", False):
            return _utils._json_error(
                "The selected study references columns that are missing from the current dataset.",
                status_code=400,
                missing_columns=preflight.get("missing_columns", []),
                available_columns=preflight.get("available_columns", []),
                recommended_pairing=preflight.get("recommended_pairing", {}),
            )
        run_entry = _utils.create_run_entry(
            RUN_LEDGER_DIR,
            run_type="study",
            context={
                "study_id": study_id,
                "model_id": getattr(study_record, "learning_model_id", None),
                "dataset_id": getattr(study_record, "dataset_id", None),
                "inference_id": _coerce_optional_int(payload.model_dump().get("inferenceId")),
            },
            parameters=payload.model_dump(),
            status="queued",
        )
        try:
            enqueue_result = enqueue_study_run(
                run_entry["run_id"],
                study_id=study_id,
                payload_data=payload.model_dump(),
                prefer_gpu=_prefers_gpu_queue(study_queue_parameters),
            )
        except QueueUnavailableError as exc:
            _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_entry["run_id"],
                status="failed",
                stage="failed",
                merge={"error": str(exc), "error_code": exc.error_code},
                event_message=str(exc),
            )
            return _queue_unavailable_response(exc, run_id=run_entry["run_id"], ledger=run_entry)
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_entry["run_id"],
            merge={
                "queue": enqueue_result,
                "rerun_payload": {
                    "run_type": "study",
                    "study_id": study_id,
                    "payload_data": payload.model_dump(),
                    "parameters": study_queue_parameters,
                },
            },
            event_message=f"Study run enqueued on {enqueue_result.get('queue')}.",
        )
        return _utils._json_response(
            {
                "status": "accepted",
                "run_id": run_entry["run_id"],
                "study_id": study_id,
                "ledger": {**run_entry, "queue": enqueue_result},
                "queue": enqueue_result,
            },
            status_code=202,
        )
    except ValueError as exc:
        LOGGER.exception("Optuna optimization failed for study_id=%s.", study_id)
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Optuna optimization failed for study_id=%s.", study_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/runs/list", response_class=JSONResponse)
async def runs_list(
    run_type: Optional[str] = None,
    status: Optional[str] = None,
    active_only: bool = False,
    limit: Optional[int] = None,
    run_ids: Optional[str] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    study_id: Optional[int] = None,
    inference_id: Optional[int] = None,
    view: str = "detail",
) -> JSONResponse:
    parsed_run_ids = [
        item.strip()
        for item in str(run_ids or "").split(",")
        if item.strip()
    ]
    runs = _utils.list_run_entries(
        RUN_LEDGER_DIR,
        run_type=run_type,
        status=status,
        active_only=active_only,
        run_ids=parsed_run_ids,
        model_id=model_id,
        dataset_id=dataset_id,
        study_id=study_id,
        inference_id=inference_id,
        limit=limit,
    )
    if str(view or "").strip().lower() == "summary":
        runs = [_utils.summarize_run_entry(run) for run in runs]
    return _utils._json_response(
        {
            "runs": runs,
            "view": "summary" if str(view or "").strip().lower() == "summary" else "detail",
        }
    )


@app.get("/runs/get/{run_id}", response_class=JSONResponse)
async def runs_get(run_id: str) -> JSONResponse:
    payload = _utils.read_run_entry(RUN_LEDGER_DIR, run_id)
    if payload is None:
        return _utils._json_error("Run not found", status_code=404)
    return _utils._json_response(payload)


def _is_active_run_status(status: Any) -> bool:
    return str(status or "queued") in {"queued", "running", "paused", "cancel_requested"}


def _run_lookup_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, Mapping):
        for child_value in value.values():
            values.extend(_run_lookup_values(child_value))
    elif isinstance(value, list):
        for child_value in value:
            values.extend(_run_lookup_values(child_value))
    elif value not in (None, ""):
        values.append(str(value).strip().lower())
    return values


def _operation_compute_target(run: Mapping[str, Any]) -> tuple[str, str]:
    queue_payload = dict(run.get("queue") or {})
    queue_name = str(queue_payload.get("queue") or queue_payload.get("name") or "").lower()
    if "gpu" in queue_name or "cuda" in queue_name:
        return "gpu", f"queue:{queue_name}"
    if "cpu" in queue_name:
        return "cpu", f"queue:{queue_name}"

    parameters = dict(run.get("parameters") or {})
    result = dict(run.get("result") or {})
    context = dict(run.get("context") or {})
    lookup_groups = [
        ("parameters", parameters),
        ("result", result),
        ("context", context),
        ("rerun_payload", dict(run.get("rerun_payload") or {})),
    ]
    for source, payload in lookup_groups:
        for value in _run_lookup_values(payload):
            if value in {"cuda", "gpu"} or "cuda:" in value or value.startswith("gpu"):
                return "gpu", f"{source}:device"
            if value == "cpu":
                return "cpu", f"{source}:device"

    if context.get("prefer_gpu") or context.get("gpu_allowed") is True:
        return "gpu", "context:gpu"
    return "cpu", "default:cpu"


def _run_rerun_capability(run: Mapping[str, Any]) -> tuple[bool, str]:
    run_type = str(run.get("run_type") or "").strip().lower()
    rerun_payload = dict(run.get("rerun_payload") or {})
    context = dict(run.get("context") or {})
    if rerun_payload:
        if run_type not in {"training", "study", "editor_execution", "assistant_operation"}:
            return False, "This run type does not support re-run."
        if run_type == "editor_execution" and not (rerun_payload.get("code") or dict(rerun_payload.get("queue_payload") or {}).get("code")):
            return False, "Editor rerun payload is missing code."
        if run_type == "assistant_operation" and not (rerun_payload.get("operation") or dict(rerun_payload.get("queue_payload") or {}).get("operation")):
            return False, "Assistant rerun payload is missing operation."
        return True, "Payload-backed rerun is available."
    if run_type == "training" and context.get("model_id") and context.get("dataset_id"):
        return True, "Training context can be reconstructed."
    if run_type == "study" and context.get("study_id"):
        return True, "Study context can be reconstructed."
    return False, "This historical run does not include enough payload to re-run."


def _operation_action_capabilities(run: Mapping[str, Any]) -> dict[str, Any]:
    status = str(run.get("status") or "queued")
    active = _is_active_run_status(status)
    can_rerun, rerun_reason = _run_rerun_capability(run)
    return {
        "pause": status in {"queued", "running"},
        "resume": status == "paused",
        "cancel": status in {"queued", "running", "paused", "cancel_requested"},
        "rerun": (not active) and can_rerun,
        "analyze": True,
        "rerun_reason": rerun_reason,
    }


def _operation_title(run: Mapping[str, Any]) -> str:
    run_type = str(run.get("run_type") or "operation").replace("_", " ")
    context = dict(run.get("context") or {})
    parts = []
    if context.get("operation"):
        parts.append(str(context["operation"]).replace("_", " "))
    if context.get("model_id"):
        parts.append(f"model {context['model_id']}")
    if context.get("dataset_id"):
        parts.append(f"dataset {context['dataset_id']}")
    if context.get("study_id"):
        parts.append(f"study {context['study_id']}")
    return f"{run_type}{' - ' + ' - '.join(parts) if parts else ''}"


def _enrich_operation_run(
    run: Mapping[str, Any],
    *,
    view: str = "summary",
    queue_positions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    full_payload = _json_safe(dict(run))
    payload = full_payload if str(view or "").lower() == "detail" else _utils.summarize_run_entry(full_payload)
    compute_target, compute_reason = _operation_compute_target(full_payload)
    run_id = str(full_payload.get("run_id") or "")
    queue_position = dict((queue_positions or {}).get(run_id) or {})
    terminal_summary = dict(payload.get("terminal_summary") or {})
    terminal_summary.setdefault("line_count", len(_utils.build_run_terminal_lines(full_payload)))
    terminal_summary["next_index"] = dict(full_payload.get("terminal") or {}).get("next_index")
    payload.update(
        {
            "title": _operation_title(payload),
            "compute_target": compute_target,
            "compute_reason": compute_reason,
            "actions": _operation_action_capabilities(full_payload),
            "terminal_summary": terminal_summary,
            "queue_diagnostics": queue_position,
        }
    )
    return payload


def _run_matches_operation_query(run: Mapping[str, Any], query: str) -> bool:
    if not query:
        return True
    progress = run.get("progress", {}) if isinstance(run.get("progress"), Mapping) else {}
    failure = run.get("failure", {}) if isinstance(run.get("failure"), Mapping) else {}
    haystack = " ".join(
        [
            str(run.get("run_id") or ""),
            str(run.get("run_type") or ""),
            str(run.get("title") or ""),
            str(run.get("status") or ""),
            str(run.get("stage") or ""),
            str(run.get("compute_reason") or ""),
            str(progress.get("latest_event") or ""),
            str(run.get("error") or ""),
            str(failure.get("message") or ""),
        ]
    ).lower()
    return query in haystack


def _merge_runs_by_id(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for run in group:
            run_id = str(run.get("run_id") or "")
            if run_id:
                merged[run_id] = run
    return sorted(
        merged.values(),
        key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    )


def _copy_rerun_payload(source_payload: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_type = str(source_payload.get("run_type") or "").strip().lower()
    context = dict(source_payload.get("context") or {})
    parameters = dict(source_payload.get("parameters") or {})
    rerun_payload = dict(source_payload.get("rerun_payload") or {})
    if run_type == "editor_execution":
        queue_payload = dict(rerun_payload.get("queue_payload") or rerun_payload)
        code = str(queue_payload.get("code") or "")
        if not code:
            raise TypeError("Run type 'editor_execution' does not support re-run without stored code.")
        registry_context = dict(queue_payload.get("registry_context") or {})
        rerun_context = {
            "source": context.get("source") or "editor",
            "gpu_allowed": False,
            **{key: value for key, value in context.items() if key.endswith("_ids") or key in {"dataset_ids", "model_ids"}},
        }
        return run_type, rerun_context, parameters, {"code": code, "registry_context": registry_context}
    if run_type == "assistant_operation":
        queue_payload = dict(rerun_payload.get("queue_payload") or rerun_payload)
        operation = str(queue_payload.get("operation") or context.get("operation") or "").strip()
        payload_data = dict(queue_payload.get("payload_data") or queue_payload.get("payload") or {})
        if not operation:
            raise TypeError("Run type 'assistant_operation' does not support re-run without stored operation.")
        return run_type, {
            "source": "assistant",
            "operation": operation,
            "assistant_model_id": context.get("assistant_model_id"),
        }, parameters, {"operation": operation, "payload_data": payload_data}
    if run_type == "training":
        model_id = _coerce_optional_int(context.get("model_id"))
        dataset_id = _coerce_optional_int(context.get("dataset_id"))
        if not model_id or not dataset_id:
            raise ValueError("Training run cannot be re-run without model_id and dataset_id in its ledger context.")
        queue_payload: dict[str, Any] = {
            "datasetId": dataset_id,
            "inputType": source_payload.get("input_type") or parameters.get("inputType") or "Re-run",
            "parameters": parameters,
            "studyId": _coerce_optional_int(context.get("study_id")),
        }
        inference_id = _coerce_optional_int(context.get("inference_id"))
        if inference_id:
            queue_payload["inferenceId"] = inference_id
        return run_type, {"model_id": model_id, "dataset_id": dataset_id, "study_id": queue_payload.get("studyId"), "inference_id": inference_id}, parameters, queue_payload
    if run_type == "study":
        study_id = _coerce_optional_int(context.get("study_id"))
        if not study_id:
            raise ValueError("Study run cannot be re-run without study_id in its ledger context.")
        queue_payload = dict(parameters)
        inference_id = _coerce_optional_int(context.get("inference_id"))
        if inference_id and "inferenceId" not in queue_payload:
            queue_payload["inferenceId"] = inference_id
        return run_type, {
            "study_id": study_id,
            "model_id": _coerce_optional_int(context.get("model_id")),
            "dataset_id": _coerce_optional_int(context.get("dataset_id")),
            "inference_id": inference_id,
        }, parameters, queue_payload
    raise TypeError(f"Run type '{run_type or 'unknown'}' does not support re-run.")


def _rerun_run(source_run_id: str, source_payload: Mapping[str, Any]) -> JSONResponse:
    status = str(source_payload.get("status") or "queued")
    if _is_active_run_status(status):
        return _utils._json_error(
            f"Run is still {status}; wait until it finishes before re-running it.",
            status_code=400,
            run_id=source_run_id,
            ledger=source_payload,
        )
    try:
        run_type, context, parameters, queue_payload = _copy_rerun_payload(source_payload)
    except TypeError as exc:
        return _utils._json_error(str(exc), status_code=400, run_id=source_run_id, ledger=source_payload)
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400, run_id=source_run_id, ledger=source_payload)

    rerun_context = {
        **context,
        "rerun_of": source_run_id,
        "rerun_requested_at": datetime.now().isoformat(),
    }
    rerun_parameters = {
        **parameters,
        "rerun_of": source_run_id,
    }
    run_entry = _utils.create_run_entry(
        RUN_LEDGER_DIR,
        run_type=run_type,
        context=rerun_context,
        parameters=rerun_parameters,
        status="queued",
    )
    try:
        if run_type == "training":
            enqueue_result = enqueue_training_run(
                run_entry["run_id"],
                model_id=int(context["model_id"]),
                payload_data=queue_payload,
                prefer_gpu=_prefers_gpu_queue(parameters),
            )
        elif run_type == "study":
            enqueue_result = enqueue_study_run(
                run_entry["run_id"],
                study_id=int(context["study_id"]),
                payload_data=queue_payload,
                prefer_gpu=_prefers_gpu_queue(parameters),
            )
        elif run_type == "editor_execution":
            try:
                enqueue_result = enqueue_editor_execution(
                    run_entry["run_id"],
                    code=str(queue_payload["code"]),
                    registry_context=dict(queue_payload.get("registry_context") or {}),
                )
            except TypeError as exc:
                if "registry_context" not in str(exc):
                    raise
                enqueue_result = enqueue_editor_execution(run_entry["run_id"], code=str(queue_payload["code"]))
        else:
            enqueue_result = enqueue_assistant_operation(
                run_entry["run_id"],
                operation=str(queue_payload["operation"]),
                payload_data=dict(queue_payload.get("payload_data") or {}),
            )
    except QueueUnavailableError as exc:
        failed = _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_entry["run_id"],
            status="failed",
            stage="failed",
            merge={"error": str(exc), "error_code": exc.error_code, "source_run_id": source_run_id},
            event_message=str(exc),
        )
        return _queue_unavailable_response(exc, run_id=run_entry["run_id"], ledger=failed)

    updated = _utils.update_run_entry(
        RUN_LEDGER_DIR,
        run_entry["run_id"],
        merge={
            "queue": enqueue_result,
            "source_run_id": source_run_id,
            "rerun_payload": {
                "source_run_id": source_run_id,
                "queue_payload": queue_payload,
            },
        },
        event_message=f"Re-run of {source_run_id} enqueued on {enqueue_result.get('queue')}.",
    )
    return _utils._json_response(
        {
            "status": "accepted",
            "run_id": run_entry["run_id"],
            "source_run_id": source_run_id,
            "ledger": updated,
            "queue": enqueue_result,
        },
        status_code=202,
    )


@app.post("/runs/{run_id}/{action}", response_class=JSONResponse)
async def runs_control(run_id: str, action: str) -> JSONResponse:
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"pause", "resume", "cancel", "rerun"}:
        return _utils._json_error("Unsupported run action", status_code=404)

    payload = _utils.read_run_entry(RUN_LEDGER_DIR, run_id)
    if payload is None:
        return _utils._json_error("Run not found", status_code=404)

    if normalized_action == "rerun":
        return _rerun_run(run_id, payload)

    status = str(payload.get("status") or "queued")
    terminal_statuses = {"completed", "failed", "cancelled"}
    if status in terminal_statuses:
        return _utils._json_error(f"Run is already {status}.", status_code=400, run_id=run_id, ledger=payload)

    queue_result: dict[str, Any] | None = None
    control_state = dict(payload.get("control") or {})
    if normalized_action == "pause":
        if status not in {"queued", "running"}:
            return _utils._json_error(f"Run cannot be paused from status {status}.", status_code=400, run_id=run_id)
        control_state.update({"pause_requested": True, "paused_at": datetime.now().isoformat()})
        updated = _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="paused",
            stage="paused",
            merge={"control": control_state},
            event_message="Pause requested for this run.",
        )
    elif normalized_action == "resume":
        if status != "paused":
            return _utils._json_error(f"Run cannot be resumed from status {status}.", status_code=400, run_id=run_id)
        control_state.update({"pause_requested": False, "resumed_at": datetime.now().isoformat()})
        updated = _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="queued",
            stage="resume_requested",
            merge={"control": control_state},
            event_message="Resume requested for this run.",
        )
    else:
        try:
            queue_result = cancel_rq_job(run_id)
        except QueueUnavailableError as exc:
            queue_result = {"backend": "rq", "cancelled": False, "error": str(exc), "error_code": exc.error_code}
        control_state.update({"cancel_requested": True, "cancel_requested_at": datetime.now().isoformat()})
        if status in {"queued", "paused"} or (queue_result or {}).get("cancelled"):
            updated = _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                status="cancelled",
                stage="cancelled",
                merge={"control": control_state, "queue": queue_result or {}},
                event_message="Run cancelled.",
            )
        else:
            updated = _utils.update_run_entry(
                RUN_LEDGER_DIR,
                run_id,
                status="cancel_requested",
                stage="cancellation_requested",
                merge={"control": control_state, "queue": queue_result or {}},
                event_message="Cooperative cancellation requested for this run.",
            )

    return _utils._json_response(
        {
            "status": updated.get("status"),
            "action": normalized_action,
            "run_id": run_id,
            "ledger": updated,
            "queue": queue_result,
        }
    )


@app.get("/operations/queues", response_class=JSONResponse)
async def operations_queues(limit: int = 50) -> JSONResponse:
    return _utils._json_response(get_queue_runtime_snapshot(limit=max(1, min(int(limit or 50), 200))))


@app.get("/operations/summary", response_class=JSONResponse)
async def operations_summary(limit: int = 12, run_ids: Optional[str] = None) -> JSONResponse:
    result_limit = max(1, min(int(limit or 12), 50))
    parsed_run_ids = [
        item.strip()
        for item in str(run_ids or "").split(",")
        if item.strip()
    ]
    recent_runs = _utils.list_run_entries(RUN_LEDGER_DIR, limit=max(result_limit * 4, 40))
    watched_runs = _utils.list_run_entries(RUN_LEDGER_DIR, run_ids=parsed_run_ids, limit=50) if parsed_run_ids else []
    active_runs = [run for run in recent_runs if _is_active_run_status(run.get("status"))]
    recent_failures = [
        run
        for run in recent_runs
        if str(run.get("status") or "") in {"failed", "cancelled"} or run.get("failure")
    ]
    queue_runtime = get_queue_runtime_snapshot(limit=25)
    queue_positions = dict(queue_runtime.get("positions") or {})
    runs = [
        _enrich_operation_run(run, view="summary", queue_positions=queue_positions)
        for run in _merge_runs_by_id(active_runs, recent_failures[:result_limit], recent_runs[:result_limit], watched_runs)
    ][: max(result_limit, len(parsed_run_ids))]
    worker_runtime = get_worker_runtime_status()
    return _utils._json_response(
        {
            "status": "ok",
            "runs": runs,
            "active_runs": [
                _enrich_operation_run(run, view="summary", queue_positions=queue_positions)
                for run in active_runs[:result_limit]
            ],
            "recent_failures": [
                _enrich_operation_run(run, view="summary", queue_positions=queue_positions)
                for run in recent_failures[:result_limit]
            ],
            "worker_runtime": worker_runtime,
            "queue_runtime": queue_runtime,
            "summary": {
                "run_count": len(runs),
                "active_runs": len(active_runs),
                "recent_failures": len(recent_failures),
                "queued_jobs": dict(queue_runtime.get("summary") or {}).get("queued_jobs", 0),
                "live_workers": dict(worker_runtime.get("summary") or {}).get("live_workers", 0),
                "stale_workers": dict(worker_runtime.get("summary") or {}).get("stale_workers", 0),
            },
        }
    )


@app.get("/operations/runs", response_class=JSONResponse)
async def operations_runs(
    run_type: Optional[str] = None,
    status: Optional[str] = None,
    active_only: bool = False,
    limit: Optional[int] = 80,
    run_ids: Optional[str] = None,
    q: Optional[str] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    study_id: Optional[int] = None,
    inference_id: Optional[int] = None,
    view: str = "summary",
) -> JSONResponse:
    parsed_run_ids = [
        item.strip()
        for item in str(run_ids or "").split(",")
        if item.strip()
    ]
    queue_runtime = get_queue_runtime_snapshot(limit=100)
    queue_positions = dict(queue_runtime.get("positions") or {})
    normalized_view = "detail" if str(view or "").strip().lower() == "detail" else "summary"
    runs = [
        _enrich_operation_run(run, view=normalized_view, queue_positions=queue_positions)
        for run in _utils.list_run_entries(
            RUN_LEDGER_DIR,
            run_type=run_type,
            status=status,
            active_only=active_only,
            run_ids=parsed_run_ids,
            model_id=model_id,
            dataset_id=dataset_id,
            study_id=study_id,
            inference_id=inference_id,
            limit=limit,
        )
    ]
    query = str(q or "").strip().lower()
    if query:
        runs = [run for run in runs if _run_matches_operation_query(run, query)]
    groups = {
        "cpu": [run for run in runs if run.get("compute_target") != "gpu"],
        "gpu": [run for run in runs if run.get("compute_target") == "gpu"],
    }
    return _utils._json_response(
        {
            "status": "ok",
            "view": normalized_view,
            "runs": runs,
            "groups": groups,
            "worker_runtime": get_worker_runtime_status(),
            "queue_runtime": queue_runtime,
            "accelerators": _cached_torch_accelerator_status(),
        }
    )


@app.get("/operations/runs/{run_id}/terminal", response_class=JSONResponse)
async def operations_run_terminal(run_id: str, cursor: int = 0, tail: int = 300) -> JSONResponse:
    payload = _utils.read_run_entry(RUN_LEDGER_DIR, run_id)
    if payload is None:
        return _utils._json_error("Run not found", status_code=404)
    tail_limit = max(1, min(int(tail or 300), 1000))
    lines = _utils.build_run_terminal_lines(payload)
    if cursor > 0:
        selected = [line for line in lines if int(line.get("index", 0)) >= cursor]
    else:
        selected = lines[-tail_limit:]
    if len(selected) > tail_limit:
        selected = selected[-tail_limit:]
    next_cursor = (max((int(line.get("index", 0)) for line in lines), default=-1) + 1)
    return _utils._json_response(
        {
            "status": "ok",
            "run_id": run_id,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "lines": selected,
            "run": _enrich_operation_run(payload, view="summary"),
        }
    )


async def _safe_registry_analysis(db: AsyncSession, registry_type: str, item_id: Optional[int]) -> dict[str, Any] | None:
    if item_id is None:
        return None
    try:
        return await _build_registry_analysis(db, registry_type, item_id)
    except Exception as exc:
        return {"status": "error", "detail": str(exc), "registry_type": registry_type, "item_id": item_id}


@app.get("/operations/runs/{run_id}/analysis", response_class=JSONResponse)
async def operations_run_analysis(run_id: str, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    payload = _utils.read_run_entry(RUN_LEDGER_DIR, run_id)
    if payload is None:
        return _utils._json_error("Run not found", status_code=404)
    context = dict(payload.get("context") or {})
    result = dict(payload.get("result") or {})
    model_id = _coerce_optional_int(context.get("model_id") or result.get("model_id"))
    dataset_id = _coerce_optional_int(context.get("dataset_id") or result.get("dataset_id"))
    study_id = _coerce_optional_int(context.get("study_id") or result.get("study_id"))
    inference_id = _coerce_optional_int(context.get("inference_id") or result.get("inference_id"))

    inference_analysis = await _safe_registry_analysis(db, "inferencemodel", inference_id)
    production_context = {
        "state": {
            key: PRODUCTION_STATE.get(key)
            for key in ("running", "status", "active_watch_context_id", "model_id", "dataset_id", "inference_id", "started_at", "stopped_at")
        },
        "watchlist": [
            value
            for value in dict(PRODUCTION_STATE.get("watchlist") or {}).values()
            if not isinstance(value, Mapping)
            or model_id in (None, _coerce_optional_int(value.get("model_id")))
            or dataset_id in (None, _coerce_optional_int(value.get("dataset_id")))
            or inference_id in (None, _coerce_optional_int(value.get("inference_id")))
        ],
    }
    plots = _list_plot_artifacts(
        job_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        inference_id=inference_id,
    )
    return _utils._json_response(
        {
            "status": "ok",
            "run": _enrich_operation_run(payload, view="detail"),
            "analysis": {
                "model": await _safe_registry_analysis(db, "learningmodel", model_id),
                "dataset": await _safe_registry_analysis(db, "datasetmodel", dataset_id),
                "study": await _safe_registry_analysis(db, "studymodel", study_id),
                "inference": inference_analysis,
                "production": production_context,
                "plots": plots,
                "metrics": dict(payload.get("metrics") or {}),
                "artifacts": dict(payload.get("artifacts") or {}),
                "raw": payload,
            },
        }
    )


@app.get("/plots/artifacts", response_class=JSONResponse)
async def plots_artifacts(
    job_id: Optional[str] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    inference_id: Optional[int] = None,
    kind: Optional[str] = None,
) -> JSONResponse:
    plots = _list_plot_artifacts(
        job_id=job_id,
        model_id=model_id,
        dataset_id=dataset_id,
        inference_id=inference_id,
        kind=kind,
    )
    return _utils._json_response({"status": "ok", "count": len(plots), "plots": plots, "artifacts": plots})


@app.get("/plots/artifacts/{plot_id}/file", response_class=FileResponse)
async def plots_artifact_file(plot_id: str) -> FileResponse:
    try:
        artifact_path = _resolve_plot_artifact_path(plot_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    return FileResponse(
        str(artifact_path),
        filename=artifact_path.name,
        media_type=media_types.get(artifact_path.suffix.lower(), "application/octet-stream"),
    )


# Feature extraction starts from registered dataset files and can materialize transformed views
# back to PostgreSQL through /features/materialize.
@app.get("/features", response_class=JSONResponse)
async def list_features(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return _utils._json_response([_utils._serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/features/extract", response_class=JSONResponse)
async def extract_feature_candidates(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None:
            return _utils._json_error("Dataset not found", status_code=404)
        dataframe = _utils._load_dataset_frame_from_record(dataset)
        summary = _utils._build_feature_extraction_summary(dataframe)
        manifest = _build_extraction_manifest(dataset, dataframe)
        workspace_payload = _utils._build_feature_workspace_payload(dataset, dataframe, preview={"summary": summary})
        artifact_path = TRAINING_ARTIFACT_DIR / f"dataset_{dataset_id}_features.json"
        save_json({"summary": summary, "manifest": manifest}, artifact_path)
        await update_entry(
            db,
            DatasetORM,
            dataset_id,
            {
                "history": _utils._append_history(
                    getattr(dataset, "history", None),
                    {
                        "operation": "extract_profile",
                        "when": datetime.now().isoformat(),
                        "artifact_path": str(artifact_path),
                    },
                )
            },
        )
        return _utils._json_response(
            {
                "dataset_id": dataset_id,
                "dataset_name": getattr(dataset, "name", None),
                "summary": summary,
                "manifest": manifest,
                "preview_rows": workspace_payload.get("preview_rows", []),
                "column_explorer": workspace_payload.get("column_explorer", []),
                "metric_cards": workspace_payload.get("metric_cards", []),
                "plots": workspace_payload.get("plots", []),
                "transform_suggestions": workspace_payload.get("transform_suggestions", []),
            }
        )
    except Exception as exc:
        LOGGER.exception("Feature extraction failed for dataset_id=%s.", dataset_id)
        return _utils._json_error(str(exc), status_code=500)


@app.post("/features/preview", response_class=JSONResponse)
async def preview_feature_workspace(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload = await request.json()
        dataset_id = int(payload.get("datasetId"))
        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None:
            return _utils._json_error("Dataset not found", status_code=404)
        dataframe = _utils._load_dataset_frame_from_record(dataset)
        secondary_dataset = None
        secondary_dataframe = None
        secondary_dataset_id = _coerce_optional_int(payload.get("secondaryDatasetId") or payload.get("secondary_dataset_id"))
        if secondary_dataset_id is not None:
            secondary_dataset = await DatasetModel.read(db, secondary_dataset_id)
            if secondary_dataset is None:
                return _utils._json_error("Secondary dataset not found", status_code=404)
            secondary_dataframe = _utils._load_dataset_frame_from_record(secondary_dataset)
        preview_row_limit = _utils._coerce_preview_row_limit(payload.get("previewRows") or payload.get("limitRows") or 20)
        if _utils._is_sql_feature_payload(payload):
            sql_query = _utils._feature_sql_query_from_payload(payload)
            sql_kwargs: dict[str, Any] = {"preview_rows": preview_row_limit}
            if secondary_dataframe is not None:
                sql_kwargs["secondary_dataframe"] = secondary_dataframe
            transformed, transform_summary = await _utils.execute_feature_sql_query(db, dataframe, sql_query, **sql_kwargs)
            operations = _utils.build_feature_sql_operation_metadata(sql_query, has_secondary=secondary_dataframe is not None)
        else:
            operations = _utils._coerce_feature_operations_from_payload(payload)
            transformed, transform_summary = _utils.apply_feature_operations(dataframe, operations)
        workspace_payload = _utils._build_feature_workspace_payload(
            dataset,
            transformed,
            preview=transform_summary,
            operations=operations,
            preview_row_limit=preview_row_limit,
        )
        if secondary_dataset is not None:
            workspace_payload["secondary_dataset"] = _utils._serialize_dataset_summary(secondary_dataset)
        return _utils._json_response(workspace_payload)
    except ValueError as exc:
        LOGGER.warning("Feature preview validation failed: %s", exc)
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Feature preview failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/features/materialize", response_class=JSONResponse)
async def materialize_feature_workspace(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload = await request.json()
        dataset_id = int(payload.get("datasetId"))
        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None:
            return _utils._json_error("Dataset not found", status_code=404)

        dataframe = _utils._load_dataset_frame_from_record(dataset)
        sql_mode = _utils._is_sql_feature_payload(payload)
        secondary_dataset = None
        secondary_dataframe = None
        secondary_dataset_id = _coerce_optional_int(payload.get("secondaryDatasetId") or payload.get("secondary_dataset_id"))
        if secondary_dataset_id is not None:
            secondary_dataset = await DatasetModel.read(db, secondary_dataset_id)
            if secondary_dataset is None:
                return _utils._json_error("Secondary dataset not found", status_code=404)
            secondary_dataframe = _utils._load_dataset_frame_from_record(secondary_dataset)
        if sql_mode:
            sql_query = _utils._feature_sql_query_from_payload(payload)
            sql_kwargs: dict[str, Any] = {}
            if secondary_dataframe is not None:
                sql_kwargs["secondary_dataframe"] = secondary_dataframe
            transformed, transform_summary = await _utils.execute_feature_sql_query(db, dataframe, sql_query, **sql_kwargs)
            operations = _utils.build_feature_sql_operation_metadata(sql_query, has_secondary=secondary_dataframe is not None)
        else:
            operations = _utils._coerce_feature_operations_from_payload(payload)
            transformed, transform_summary = _utils.apply_feature_operations(dataframe, operations)
        transformed_name = str(payload.get("name") or f"{getattr(dataset, 'name', 'dataset')}_feature_view")
        table_name = str(
            payload.get("tableName")
            or f"feature_{dataset_id}_{_normalize_registry_type(transformed_name)}"
        )
        output_path = FEATURE_ARTIFACT_DIR / f"{_normalize_registry_type(transformed_name)}.csv"
        transformed.to_csv(output_path, index=False, encoding="utf-8")

        postgres_result = {"table_name": table_name, "materialized": False}
        bind = getattr(db, "bind", None)
        if bind is not None:
            try:
                async with bind.begin() as conn:
                    await conn.run_sync(
                        lambda sync_conn: transformed.to_sql(
                            table_name,
                            con=sync_conn,
                            if_exists="replace",
                            index=False,
                        )
                    )
                postgres_result["materialized"] = True
            except Exception as exc:
                postgres_result["warning"] = str(exc)

        history_entry = {
            "operation": "feature_materialize",
            "when": datetime.now().isoformat(),
            "source_dataset_id": dataset_id,
            "secondary_dataset_id": secondary_dataset_id,
            "mode": "sql" if sql_mode else "structured",
            "table_name": table_name,
            "materialized_to_postgres": postgres_result["materialized"],
        }
        if sql_mode:
            history_entry.update(
                {
                    "source_relation": operations.get("source_relation"),
                    "sql_hash": operations.get("sql_hash"),
                }
            )

        dataset_payload = {
            "name": transformed_name,
            "description": str(payload.get("description") or f"Feature workspace materialization derived from dataset #{dataset_id}."),
            "object_type": "dataset",
            "size": round(output_path.stat().st_size / (1024 * 1024), 4),
            "path": str(output_path),
            "date": datetime.now(),
            "version": 1,
            "history": [history_entry],
            "dataset_type": "dataset",
            "shape": [int(transformed.shape[0]), int(transformed.shape[1])],
            "has_features": bool(len(transformed.columns)),
            "features_list": [str(column) for column in transformed.columns],
            "connection_string": str(payload.get("connectionString") or f"table:{table_name}"),
        }
        created_dataset = await create_entry(db, DatasetORM, dataset_payload)
        analysis = _utils._dataset_analysis_summary(transformed, str(output_path))
        save_json(analysis, FEATURE_ARTIFACT_DIR / f"{_normalize_registry_type(transformed_name)}_analysis.json")
        workspace_payload = _utils._build_feature_workspace_payload(
            created_dataset,
            transformed,
            preview=transform_summary,
            operations=operations,
        )
        return _utils._json_response(
            {
                "status": "ok",
                "dataset": _utils._serialize_dataset_summary(created_dataset),
                "preview": transform_summary,
                "analysis": analysis,
                "postgres": postgres_result,
                "preview_rows": workspace_payload.get("preview_rows", []),
                "column_explorer": workspace_payload.get("column_explorer", []),
                "metric_cards": workspace_payload.get("metric_cards", []),
                "plots": workspace_payload.get("plots", []),
                "transform_suggestions": workspace_payload.get("transform_suggestions", []),
                "operations": workspace_payload.get("operations", {}),
                "secondary_dataset": _utils._serialize_dataset_summary(secondary_dataset) if secondary_dataset is not None else None,
            }
        )
    except ValueError as exc:
        LOGGER.warning("Feature materialization validation failed: %s", exc)
        return _utils._json_error(str(exc), status_code=400)
    except Exception as exc:
        LOGGER.exception("Feature materialization failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/list/dataset", response_class=JSONResponse)
async def list_dataset(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return _utils._json_response([_utils._serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/list/model", response_class=JSONResponse)
async def list_model(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    models = await get_all_entries(db, LearningORM)
    return _utils._json_response([_utils._serialize_model_summary(model) for model in models])


@app.get("/analysis/data", response_class=JSONResponse)
async def analysis_dataset(
    dataset_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if dataset_id is None:
            datasets = await get_all_entries(db, DatasetORM)
            return _utils._json_response(
                {
                    "available_datasets": [
                        {"id": dataset.id, "name": dataset.name, "shape": dataset.shape}
                        for dataset in datasets
                    ]
                }
            )

        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None or not getattr(dataset, "path", None):
            return _utils._json_error("Dataset not found", status_code=404)

        df_path = str(_utils._resolve_fs_path(dataset.path, default_parent=REPO_ROOT))
        df = _utils._load_dataset_frame_from_record(dataset)
        summary = _utils._dataset_analysis_summary(df, df_path)
        plots = _build_dataset_plots(summary, dataframe=df)
        artifacts = _list_plot_artifacts(dataset_id=dataset_id)
        existing_plot_ids = {plot.get("id") for plot in plots}
        plots.extend([plot for plot in artifacts if plot.get("id") not in existing_plot_ids])
        save_json(summary, TRAINING_ARTIFACT_DIR / f"dataset_{dataset_id}_analysis.json")
        return _utils._json_response(
            {
                "dataset_id": dataset_id,
                "registry_type": "DatasetModel",
                "item_id": dataset_id,
                "summary": summary,
                "plots": plots,
                "artifacts": artifacts,
                "history": getattr(dataset, "history", None) or [],
            }
        )
    except Exception as exc:
        LOGGER.exception("Dataset analysis failed for dataset_id=%s.", dataset_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/analysis/model", response_class=JSONResponse)
async def analysis_model(
    model_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if model_id is None:
            models = await get_all_entries(db, LearningORM)
            return _utils._json_response(
                {
                    "available_models": [
                        {"id": model.id, "name": model.name, "type": model.model_type, "is_trained": model.is_trained}
                        for model in models
                    ]
                }
            )

        model = await LearningModel.read(db, model_id)
        if model is None:
            return _utils._json_error("Model not found", status_code=404)
        summary = _utils._augment_model_analysis(model)
        plots = _build_model_plots(summary)
        artifacts = _list_plot_artifacts(model_id=model_id)
        existing_plot_ids = {plot.get("id") for plot in plots}
        plots.extend([plot for plot in artifacts if plot.get("id") not in existing_plot_ids])
        return _utils._json_response(
            {
                "model_id": model_id,
                "registry_type": "LearningModel",
                "item_id": model_id,
                "summary": summary,
                "plots": plots,
                "artifacts": artifacts,
                "history": getattr(model, "history", None) or [],
            }
        )
    except Exception as exc:
        LOGGER.exception("Model analysis failed for model_id=%s.", model_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/analysis/object", response_class=JSONResponse)
async def analysis_object(
    registry_type: str,
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        return _utils._json_response(await _build_registry_analysis(db, registry_type, item_id))
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Object analysis failed for registry_type=%s item_id=%s.", registry_type, item_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/registry/dependencies/{registry_type}/{item_id}", response_class=JSONResponse)
async def registry_dependencies(
    registry_type: str,
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        binding = _get_registry_binding(registry_type)
        dependency_report = await get_entry_dependencies(db, binding["orm"], item_id)
        if not dependency_report.get("exists"):
            return _utils._json_error(f"{binding['label']} not found", status_code=404)
        return _utils._json_response(
            {
                "registry_type": binding["label"],
                "item_id": item_id,
                "dependency_report": dependency_report,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=400)


def _mlflow_public_url() -> str:
    tracking_uri = str(_utils.RUNTIME_CONFIG.get("mlflow", {}).get("tracking_uri") or "").strip()
    if tracking_uri.startswith(("http://", "https://")) and "://mlflow:" not in tracking_uri:
        return tracking_uri.rstrip("/")
    return "http://localhost:5000"


@app.get("/mlflow", response_class=RedirectResponse)
async def mlflow_redirect() -> RedirectResponse:
    return RedirectResponse(url=_mlflow_public_url())


@app.get("/mlflow/ui", response_class=RedirectResponse)
async def mlflow_ui_redirect() -> RedirectResponse:
    return RedirectResponse(url=_mlflow_public_url())


@app.get("/mlflow/experiments", response_class=JSONResponse)
async def mlflow_list_experiments() -> JSONResponse:
    health = _utils._build_mlflow_health_snapshot()
    try:
        return _utils._json_response({"experiments": list_experiments(), "health": health})
    except Exception as exc:
        LOGGER.exception("Unable to list MLflow experiments.")
        return _utils._json_response({"experiments": [], "health": {**health, "status": "degraded", "warnings": [*health.get("warnings", []), str(exc)]}})


@app.get("/mlflow/experiments/{exp_id}", response_class=JSONResponse)
async def mlflow_get_experiment(exp_id: str, limit: int = 20, offset: int = 0) -> JSONResponse:
    health = _utils._build_mlflow_health_snapshot()
    try:
        experiment = get_experiment_summary(exp_id, max_runs=max(1, min(limit, 100)), offset=max(0, offset))
        return _utils._json_response(
            {
                "experiment": experiment,
                "limit": max(1, min(limit, 100)),
                "offset": max(0, offset),
                "has_more": dict(experiment.get("run_summary", {}) or {}).get("has_more", False),
                "health": health,
            }
        )
    except Exception as exc:
        LOGGER.exception("Unable to load MLflow experiment '%s'.", exp_id)
        return _utils._json_response({"experiment": None, "health": {**health, "status": "degraded", "warnings": [*health.get("warnings", []), str(exc)]}})


@app.get("/mlflow/health", response_class=JSONResponse)
async def mlflow_health() -> JSONResponse:
    return _utils._json_response(_utils._build_mlflow_health_snapshot())



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
    control_plan = {
        "pre_training": [
            "Lock the expected input_features and output_features before the export target is chosen.",
            "Keep framework-specific hyperparameters in the runtime payload so the same study can prepare both training and ONNX export.",
        ],
        "post_training": [
            "Validate the persisted artifact path and compare the exported metadata against the registered model parameters.",
            "Generate a serving stub only after metrics and monitoring thresholds are accepted.",
        ],
        "recommended_flow": "Prepare ONNX metadata before training to freeze the schema, then validate the artifact after training to confirm the exported graph matches the trained model.",
    }
    return _utils._json_response({"status": "prepared", "payload": prepared_payload, "controls": control_plan})


@app.get("/onnx/validate/{model_id}", response_class=JSONResponse)
async def onnx_validate(model_id: int, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_record = await LearningModel.read(db, model_id)
        if model_record is None:
            return _utils._json_error("Model not found", status_code=404)

        path = _utils._resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
        if path is None or not path.exists():
            fallback = _utils._resolve_fs_path(PROJECT_ROOT / "model.onnx")
            path = fallback if fallback and fallback.exists() else path
        if path is None or not path.exists():
            return _utils._json_error("No ONNX artifact was found for this model.", status_code=404)

        return _utils._json_response({"status": "validated", "model_id": model_id, "path": str(path), "metadata": recover_model_params(path)})
    except Exception as exc:
        LOGGER.exception("ONNX validation failed for model_id=%s.", model_id)
        return _utils._json_error(str(exc), status_code=500)


async def _onnx_netron_status_payload(model_id: int, db: AsyncSession) -> dict[str, Any] | None:
    model_record = await LearningModel.read(db, model_id)
    if model_record is None:
        return None

    path = _utils._resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
    artifact_manifest = recover_model_params(path) if path is not None and path.exists() and path.is_file() else {}
    return build_netron_status(
        model_id=model_id,
        model_name=getattr(model_record, "name", None) or getattr(model_record, "model_name", None),
        artifact_path=path,
        artifact_manifest=artifact_manifest if isinstance(artifact_manifest, Mapping) else {},
    )


@app.get("/onnx/netron/{model_id}/status", response_class=JSONResponse)
async def onnx_netron_status(model_id: int, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    payload = await _onnx_netron_status_payload(model_id, db)
    if payload is None:
        return _utils._json_error("Model not found", status_code=404)
    return _utils._json_response(payload)


@app.post("/onnx/netron/{model_id}/open", response_class=JSONResponse)
async def onnx_netron_open(model_id: int, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    model_record = await LearningModel.read(db, model_id)
    if model_record is None:
        return _utils._json_error("Model not found", status_code=404)

    path = _utils._resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
    artifact_manifest = recover_model_params(path) if path is not None and path.exists() and path.is_file() else {}
    status_payload = build_netron_status(
        model_id=model_id,
        model_name=getattr(model_record, "name", None) or getattr(model_record, "model_name", None),
        artifact_path=path,
        artifact_manifest=artifact_manifest if isinstance(artifact_manifest, Mapping) else {},
    )
    if not status_payload.get("viewable") or path is None:
        return _utils._json_response(status_payload)

    try:
        return _utils._json_response(start_netron_viewer(artifact_path=path, status_payload=status_payload))
    except NetronUnavailableError as exc:
        return _utils._json_response(
            {
                **status_payload,
                "status": "unavailable",
                "viewable": False,
                "viewer_active": False,
                "viewer_url": None,
                "warnings": [*status_payload.get("warnings", []), str(exc)],
            }
        )
    except Exception as exc:
        LOGGER.exception("Unable to start Netron for model_id=%s.", model_id)
        return _utils._json_error(str(exc), status_code=500)


@app.post("/onnx/netron/stop", response_class=JSONResponse)
async def onnx_netron_stop() -> JSONResponse:
    try:
        return _utils._json_response(stop_netron_viewer())
    except NetronUnavailableError as exc:
        return _utils._json_response({"status": "unavailable", "viewer_active": False, "warnings": [str(exc)]})



@app.get("/production/status", response_class=JSONResponse)
async def production_status() -> JSONResponse:
    payload = {**PRODUCTION_STATE, "activity_log": _read_activity_log(limit=30)}
    return _utils._json_response(payload)


@app.post("/production/start", response_class=JSONResponse)
async def production_start(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}
        payload = validate_input(payload_data, _utils.ProductionStartRequest)
        model_id = payload.modelId
        dataset_id = payload.datasetId
        inference_id = payload_data.get("inferenceId")
        schedule_payload = payload_data.get("schedule") if isinstance(payload_data.get("schedule"), Mapping) else {}
        watch_schedule = {
            "enabled": bool(schedule_payload.get("enabled", False)),
            "interval_seconds": max(30, int(schedule_payload.get("interval_seconds") or schedule_payload.get("intervalSeconds") or 300)),
            "drift_threshold": float(schedule_payload.get("drift_threshold") or schedule_payload.get("driftThreshold") or 0.15),
            "degradation_threshold": float(schedule_payload.get("degradation_threshold") or schedule_payload.get("degradationThreshold") or 0.1),
            "auto_retrain": bool(schedule_payload.get("auto_retrain", schedule_payload.get("autoRetrain", False))),
        }

        if model_id is None and inference_id in (None, "", 0, "0"):
            models = await get_all_entries(db, LearningORM)
            trained_models = [model for model in models if getattr(model, "is_trained", False)]
            if not trained_models:
                return _utils._json_error("No trained model is available for production.", status_code=404)
            trained_models.sort(key=lambda model: getattr(model, "id", 0), reverse=True)
            model_id = trained_models[0].id

        inference_record, model_record, dataset_record = await _resolve_runtime_context(
            db,
            model_id=model_id,
            dataset_id=dataset_id,
            inference_id=int(inference_id) if inference_id not in (None, "", 0, "0") else None,
        )
        if model_record is None:
            return _utils._json_error("Model not found", status_code=404)

        if dataset_record is None and dataset_id is None:
            datasets = await get_all_entries(db, DatasetORM)
            if datasets:
                datasets.sort(key=lambda dataset: getattr(dataset, "id", 0), reverse=True)
                dataset_record = datasets[0]

        if dataset_record is None:
            return _utils._json_error("Dataset not found", status_code=404)

        dataset_id = int(getattr(dataset_record, "id"))
        model_id = int(getattr(model_record, "id"))
        runtime_probe = None
        artifact_path = _utils._resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
        if artifact_path is not None and artifact_path.exists():
            runtime_probe = _utils.load_runtime_artifact(
                artifact_path,
                parameters=dict(getattr(model_record, "parameters", {}) or {}),
            )

        deployment_summary = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "inference_id": getattr(inference_record, "id", None),
            "model_name": getattr(model_record, "name", None),
            "dataset_name": getattr(dataset_record, "name", None),
            "artifact_path": getattr(model_record, "path", None),
            "started_at": datetime.now().isoformat(),
            "status": "active",
            "schedule": watch_schedule,
        }
        if getattr(model_record, "path", None):
            try:
                resolved_artifact_path = _utils._resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
                deployment_summary["artifact_metadata"] = recover_model_params(resolved_artifact_path or getattr(model_record, "path"))
            except Exception:
                LOGGER.exception("Unable to recover deployment artifact metadata.")
        if runtime_probe is not None:
            deployment_summary["artifact_manifest"] = runtime_probe.get("manifest", {})

        save_deployment_summary(deployment_summary, str(DEPLOYMENT_ARTIFACT_DIR / f"deployment_{model_id}.json"))
        inference_record = await _upsert_inference_record(
            db,
            model_record=model_record,
            dataset_record=dataset_record,
            input_features=_utils._coerce_feature_list(getattr(model_record, "input_features", None)),
            output_features=_utils._coerce_feature_list(getattr(model_record, "output_features", None)),
            inference_updates={
                "status": "active",
                "framework": dict(getattr(model_record, "parameters", {}) or {}).get("framework"),
                "latest_metrics": getattr(model_record, "metrics", {}) or {},
                "deployment": deployment_summary,
                "artifact_manifest": deployment_summary.get("artifact_metadata", {}),
                "simulation_defaults": {
                    "steps": 12,
                    "strategy": "carry-forward",
                    "amplitude": 0.05,
                },
            },
            history_entry={
                "operation": "production_start",
                "when": deployment_summary["started_at"],
                "model_id": model_id,
                "dataset_id": dataset_id,
            },
            artifact_path=getattr(model_record, "path", None),
        )
        deployment_summary["inference_id"] = getattr(inference_record, "id", None)

        PRODUCTION_STATE.update(
            {
                "running": True,
                "status": "active",
                "active_watch_context_id": _resolve_watch_context_id(
                    payload_data,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    inference_id=getattr(inference_record, "id", None),
                ),
                "inference_id": getattr(inference_record, "id", None),
                "model_id": model_id,
                "dataset_id": dataset_id,
                "started_at": deployment_summary["started_at"],
                "stopped_at": None,
                "deployment": deployment_summary,
                "metrics": _utils._normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
                "active_model": _utils._serialize_model_summary(model_record),
                "active_inference": _serialize_inference_summary(inference_record),
            }
        )
        watch_id = str(PRODUCTION_STATE.get("active_watch_context_id"))
        deployment_summary["watch_context_id"] = watch_id
        PRODUCTION_STATE.setdefault("watchlist", {})[watch_id] = {
            "watch_context_id": watch_id,
            "status": "active",
            "model_id": model_id,
            "dataset_id": dataset_id,
            "inference_id": getattr(inference_record, "id", None),
            "deployment": deployment_summary,
            "schedule": watch_schedule,
            "active_model": _utils._serialize_model_summary(model_record),
            "active_inference": _serialize_inference_summary(inference_record),
            "updated_at": datetime.now().isoformat(),
        }
        _append_activity_log(
            f"Production started with model_id={model_id} dataset_id={dataset_id}.",
            event_type="production.started",
            details=deployment_summary,
        )

        await update_entry(
            db,
            LearningORM,
            model_id,
            {
                "is_deployed": True,
                "history": _utils._append_history(
                    getattr(model_record, "history", None),
                    {"operation": "production_start", "when": deployment_summary["started_at"], "dataset_id": dataset_id},
                ),
            },
        )
        return _utils._json_response(
            {
                "status": "started",
                "watch_context_id": watch_id,
                **PRODUCTION_STATE,
                "activity_log": _read_activity_log(limit=30),
            }
        )
    except ValueError as exc:
        extra_payload = {}
        if isinstance(exc, _utils.RuntimeArtifactDependencyError):
            extra_payload = exc.to_payload()
        return _utils._json_error(
            str(exc),
            status_code=400,
            watch_context_id=_resolve_watch_context_id(payload_data),
            **extra_payload,
        )
    except Exception as exc:
        LOGGER.exception("Production start failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/production/stop", response_class=JSONResponse)
async def production_stop(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        model_id = PRODUCTION_STATE.get("model_id")
        dataset_id = PRODUCTION_STATE.get("dataset_id")
        inference_id = PRODUCTION_STATE.get("inference_id")
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
                        "history": _utils._append_history(
                            getattr(model_record, "history", None),
                            {"operation": "production_stop", "when": stopped_at},
                        ),
                    },
                )
        inference_record = await _find_inference_record(
            db,
            inference_id=int(inference_id) if inference_id not in (None, "", 0, "0") else None,
            model_id=int(model_id) if model_id is not None else None,
            dataset_id=int(dataset_id) if dataset_id is not None else None,
        )
        if inference_record is not None and model_id is not None and dataset_id is not None:
            inference_record = await _upsert_inference_record(
                db,
                model_record=model_record or await LearningModel.read(db, int(model_id)),
                dataset_record=await DatasetModel.read(db, int(dataset_id)),
                inference_updates={
                    "status": "inactive",
                    "stopped_at": stopped_at,
                },
                history_entry={"operation": "production_stop", "when": stopped_at},
                artifact_path=getattr(inference_record, "path", None),
            )
            watch_id = _utils._watch_context_id(
                model_id=model_id,
                dataset_id=dataset_id,
                inference_id=getattr(inference_record, "id", None),
            )
            PRODUCTION_STATE.setdefault("watchlist", {}).setdefault(watch_id, {})
            PRODUCTION_STATE["watchlist"][watch_id].update(
                {
                    "watch_context_id": watch_id,
                    "status": "inactive",
                    "updated_at": stopped_at,
                    "stopped_at": stopped_at,
                }
            )
        PRODUCTION_STATE.update(
            {
                "running": False,
                "status": "inactive",
                "active_watch_context_id": watch_id if inference_record is not None else PRODUCTION_STATE.get("active_watch_context_id"),
                "stopped_at": stopped_at,
                "simulation": {},
            }
        )
        if inference_record is not None:
            PRODUCTION_STATE["active_inference"] = _serialize_inference_summary(inference_record)
        _append_activity_log("Production stopped.", event_type="production.stopped", details={"when": stopped_at})
        return _utils._json_response({"status": "stopped", **PRODUCTION_STATE, "activity_log": _read_activity_log(limit=30)})
    except Exception as exc:
        LOGGER.exception("Production stop failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/production/monitor", response_class=JSONResponse)
async def production_monitor(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}
        payload = validate_input(payload_data, _utils.ProductionMonitorRequest)
        inference_id = payload_data.get("inferenceId") or PRODUCTION_STATE.get("inference_id")
        inference_record, model_record, dataset_record = await _resolve_runtime_context(
            db,
            model_id=payload.modelId or PRODUCTION_STATE.get("model_id"),
            dataset_id=payload.datasetId or PRODUCTION_STATE.get("dataset_id"),
            inference_id=int(inference_id) if inference_id not in (None, "", 0, "0") else None,
        )
        if model_record is None or dataset_record is None:
            return _utils._json_error("Select a model and dataset before monitoring production.", status_code=400)

        model_id = int(getattr(model_record, "id"))
        dataset_id = int(getattr(dataset_record, "id"))
        snapshot = await _utils._build_live_monitoring_snapshot(db, model_id, dataset_id, tolerance=float(payload.tolerance))
        inference_record = await _upsert_inference_record(
            db,
            model_record=model_record,
            dataset_record=dataset_record,
            input_features=snapshot.get("input_features") or _utils._coerce_feature_list(getattr(model_record, "input_features", None)),
            output_features=_utils._coerce_feature_list(getattr(model_record, "output_features", None)),
            inference_updates={
                "status": "active" if PRODUCTION_STATE.get("running") else "monitored",
                "latest_metrics": snapshot.get("metrics", {}),
                "monitoring": snapshot,
            },
            history_entry={
                "operation": "monitoring_refresh",
                "when": datetime.now().isoformat(),
                "tolerance": float(payload.tolerance),
            },
            artifact_path=getattr(model_record, "path", None),
        )
        PRODUCTION_STATE["monitoring"] = snapshot
        PRODUCTION_STATE["metrics"] = snapshot["metrics"]
        PRODUCTION_STATE["model_id"] = model_id
        PRODUCTION_STATE["dataset_id"] = dataset_id
        PRODUCTION_STATE["active_model"] = _utils._serialize_model_summary(model_record)
        PRODUCTION_STATE["inference_id"] = getattr(inference_record, "id", None)
        PRODUCTION_STATE["active_inference"] = _serialize_inference_summary(inference_record)
        watch_id = _resolve_watch_context_id(
            payload_data,
            model_id=model_id,
            dataset_id=dataset_id,
            inference_id=getattr(inference_record, "id", None),
        )
        PRODUCTION_STATE["active_watch_context_id"] = watch_id
        PRODUCTION_STATE.setdefault("watchlist", {}).setdefault(watch_id, {})
        PRODUCTION_STATE["watchlist"][watch_id].update(
            {
                "watch_context_id": watch_id,
                "status": "active" if PRODUCTION_STATE.get("running") else "monitored",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "inference_id": getattr(inference_record, "id", None),
                "monitoring": snapshot,
                "updated_at": datetime.now().isoformat(),
            }
        )
        _append_activity_log(
            f"Production monitoring refreshed for model_id={model_id} dataset_id={dataset_id}.",
            event_type="production.monitoring",
            details={"model_id": model_id, "dataset_id": dataset_id, "tolerance": float(payload.tolerance)},
        )
        return _utils._json_response(
            {
                "status": "ok",
                "watch_context_id": watch_id,
                "monitoring": snapshot,
                "production": {**PRODUCTION_STATE, "activity_log": _read_activity_log(limit=30)},
            }
        )
    except ValueError as exc:
        extra_payload = {}
        if isinstance(exc, _utils.RuntimeArtifactDependencyError):
            extra_payload = exc.to_payload()
        return _utils._json_error(
            str(exc),
            status_code=400,
            watch_context_id=_resolve_watch_context_id(payload_data),
            **extra_payload,
        )
    except Exception as exc:
        LOGGER.exception("Production monitoring failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/production/retrain", response_class=JSONResponse)
async def production_retrain(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}
        payload = validate_input(payload_data, _utils.ProductionMonitorRequest)
        inference_id = payload_data.get("inferenceId") or PRODUCTION_STATE.get("inference_id")
        inference_record, model_record, dataset_record = await _resolve_runtime_context(
            db,
            model_id=payload.modelId or PRODUCTION_STATE.get("model_id"),
            dataset_id=payload.datasetId or PRODUCTION_STATE.get("dataset_id"),
            inference_id=int(inference_id) if inference_id not in (None, "", 0, "0") else None,
        )
        if model_record is None or dataset_record is None:
            return _utils._json_error("Select a model and dataset before retraining.", status_code=400)

        model_id = int(getattr(model_record, "id"))
        dataset_id = int(getattr(dataset_record, "id"))
        retraining_result = await _utils._trigger_retraining_from_monitoring(
            db,
            model_id,
            dataset_id,
            tolerance=float(payload.tolerance),
        )
        inference_record = await _upsert_inference_record(
            db,
            model_record=model_record,
            dataset_record=dataset_record,
            input_features=_utils._coerce_feature_list(getattr(model_record, "input_features", None)),
            output_features=_utils._coerce_feature_list(getattr(model_record, "output_features", None)),
            inference_updates={
                "status": "retrained",
                "latest_retraining": retraining_result,
            },
            history_entry={
                "operation": "retraining",
                "when": datetime.now().isoformat(),
                "tolerance": float(payload.tolerance),
            },
            artifact_path=getattr(model_record, "path", None),
        )
        PRODUCTION_STATE["inference_id"] = getattr(inference_record, "id", None)
        PRODUCTION_STATE["model_id"] = model_id
        PRODUCTION_STATE["dataset_id"] = dataset_id
        PRODUCTION_STATE["active_model"] = _utils._serialize_model_summary(model_record)
        PRODUCTION_STATE["active_inference"] = _serialize_inference_summary(inference_record)
        watch_id = _resolve_watch_context_id(
            payload_data,
            model_id=model_id,
            dataset_id=dataset_id,
            inference_id=getattr(inference_record, "id", None),
        )
        PRODUCTION_STATE["active_watch_context_id"] = watch_id
        PRODUCTION_STATE.setdefault("watchlist", {}).setdefault(watch_id, {})
        PRODUCTION_STATE["watchlist"][watch_id].update(
            {
                "watch_context_id": watch_id,
                "status": "retrained",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "inference_id": getattr(inference_record, "id", None),
                "latest_retraining": retraining_result,
                "updated_at": datetime.now().isoformat(),
            }
        )
        _append_activity_log(
            f"Retraining flow executed for model_id={model_id} dataset_id={dataset_id}.",
            event_type="production.retraining",
            details={"model_id": model_id, "dataset_id": dataset_id, "tolerance": float(payload.tolerance)},
        )
        return _utils._json_response(
            {
                "status": "ok",
                "watch_context_id": watch_id,
                "retraining": retraining_result,
                "production": {**PRODUCTION_STATE, "activity_log": _read_activity_log(limit=30)},
            }
        )
    except ValueError as exc:
        extra_payload = {}
        if isinstance(exc, _utils.RuntimeArtifactDependencyError):
            extra_payload = exc.to_payload()
        return _utils._json_error(
            str(exc),
            status_code=400,
            watch_context_id=_resolve_watch_context_id(payload_data),
            **extra_payload,
        )
    except Exception as exc:
        LOGGER.exception("Production retraining failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/production/history", response_class=JSONResponse)
async def production_history(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        inference_records = await get_all_entries(db, InferenceORM)
        inference_records = sorted(inference_records, key=lambda record: getattr(record, "id", 0), reverse=True)
        runs = [
            {
                **_serialize_inference_summary(record),
                "plots": _build_inference_plots(_serialize_inference_summary(record)),
            }
            for record in inference_records
        ]
        return _utils._json_response({"runs": runs, "activity_log": _read_activity_log(limit=200)})
    except Exception as exc:
        LOGGER.exception("Unable to build the production history.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/production/simulation/context", response_class=JSONResponse)
async def production_simulation_context(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    query = request.query_params
    payload_data = {
        "watchContextId": query.get("watchContextId") or query.get("watch_context_id"),
        "inferenceId": query.get("inferenceId") or query.get("inference_id"),
        "modelId": query.get("modelId") or query.get("model_id"),
        "datasetId": query.get("datasetId") or query.get("dataset_id"),
        "baselineSource": query.get("baselineSource") or query.get("baseline_source") or "last",
    }
    try:
        inference_id = payload_data.get("inferenceId") or PRODUCTION_STATE.get("inference_id")
        inference_record, model_record, dataset_record = await _resolve_runtime_context(
            db,
            model_id=payload_data.get("modelId") or PRODUCTION_STATE.get("model_id"),
            dataset_id=payload_data.get("datasetId") or PRODUCTION_STATE.get("dataset_id"),
            inference_id=int(inference_id) if inference_id not in (None, "", 0, "0") else None,
        )
        if model_record is None or dataset_record is None:
            return _utils._json_error("Select an inference pair, model, and dataset before loading simulation context.", status_code=400)

        try:
            prepared, runtime_result = _prepare_runtime_execution(model_record, dataset_record)
            context = _utils._build_simulation_context(
                model_record=model_record,
                dataset_record=dataset_record,
                inference_record=inference_record,
                prepared=prepared,
                runtime_result=runtime_result,
                baseline_source=str(payload_data.get("baselineSource") or "last"),
            )
        except Exception as runtime_exc:
            context = _utils._build_unavailable_simulation_context(
                model_record=model_record,
                dataset_record=dataset_record,
                inference_record=inference_record,
                runtime_error=runtime_exc,
            )
        watch_id = _resolve_watch_context_id(
            payload_data,
            model_id=getattr(model_record, "id", None),
            dataset_id=getattr(dataset_record, "id", None),
            inference_id=getattr(inference_record, "id", None),
        )
        return _utils._json_response(
            {
                "status": "ok",
                "watch_context_id": watch_id,
                "context": context,
            }
        )
    except ValueError as exc:
        extra_payload = {}
        if isinstance(exc, _utils.RuntimeArtifactDependencyError):
            extra_payload = exc.to_payload()
        return _utils._json_error(
            str(exc),
            status_code=400,
            watch_context_id=_resolve_watch_context_id(payload_data),
            **extra_payload,
        )
    except Exception as exc:
        LOGGER.exception("Unable to build production simulation context.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/production/simulate", response_class=JSONResponse)
async def production_simulate(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    payload_data: dict[str, Any] = {}
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}

        inference_id = payload_data.get("inferenceId") or PRODUCTION_STATE.get("inference_id")
        inference_record, model_record, dataset_record = await _resolve_runtime_context(
            db,
            model_id=payload_data.get("modelId") or PRODUCTION_STATE.get("model_id"),
            dataset_id=payload_data.get("datasetId") or PRODUCTION_STATE.get("dataset_id"),
            inference_id=int(inference_id) if inference_id not in (None, "", 0, "0") else None,
        )
        if model_record is None or dataset_record is None:
            return _utils._json_error("Select an inference pair, model, and dataset before simulating production.", status_code=400)

        prepared, runtime_result = _prepare_runtime_execution(model_record, dataset_record)
        simulation_summary = _utils._run_simulation_workbench(
            payload_data,
            prepared=prepared,
            runtime_result=runtime_result,
            model_record=model_record,
            dataset_record=dataset_record,
            inference_record=inference_record,
        )
        PRODUCTION_STATE["simulation"] = simulation_summary
        output_features = simulation_summary.get("output_features") or []
        output_feature = simulation_summary.get("output_feature") or "prediction"
        primary_summary = simulation_summary.get("prediction_summary") or {}
        inference_updates = {
            "status": "simulated",
            "latest_simulation": simulation_summary,
        }
        if payload_data.get("saveDefaults"):
            simulation_profile = dict(simulation_summary.get("simulation_profile") or {})
            if simulation_summary.get("mode"):
                simulation_profile["default_mode"] = simulation_summary.get("mode")
            inference_updates["simulation_defaults"] = {
                "steps": simulation_summary.get("steps"),
                "trend": simulation_summary.get("trend"),
                "amplitude": simulation_summary.get("amplitude"),
                "baseline_source": simulation_summary.get("baseline_source"),
                "family": simulation_summary.get("family"),
                "mode": simulation_summary.get("mode"),
                "scenario_count": len(simulation_summary.get("scenarios") or []),
                "sensitivity_enabled": bool((simulation_summary.get("sensitivity") or {}).get("enabled")),
            }
            inference_updates["simulation_profile"] = simulation_profile

        inference_record = await _upsert_inference_record(
            db,
            model_record=model_record,
            dataset_record=dataset_record,
            input_features=_utils._simulation_feature_columns(prepared),
            output_features=output_features,
            inference_updates=inference_updates,
            history_entry={
                "operation": "simulation",
                "when": datetime.now().isoformat(),
                "steps": simulation_summary.get("steps"),
                "trend": simulation_summary.get("trend"),
                "amplitude": simulation_summary.get("amplitude"),
                "scenario_count": len(simulation_summary.get("scenarios") or []),
            },
            artifact_path=getattr(model_record, "path", None),
        )
        PRODUCTION_STATE["inference_id"] = getattr(inference_record, "id", None)
        PRODUCTION_STATE["model_id"] = getattr(model_record, "id", None)
        PRODUCTION_STATE["dataset_id"] = getattr(dataset_record, "id", None)
        PRODUCTION_STATE["active_model"] = _utils._serialize_model_summary(model_record)
        PRODUCTION_STATE["active_inference"] = _serialize_inference_summary(inference_record)
        watch_id = _resolve_watch_context_id(
            payload_data,
            model_id=getattr(model_record, "id", None),
            dataset_id=getattr(dataset_record, "id", None),
            inference_id=getattr(inference_record, "id", None),
        )
        PRODUCTION_STATE["active_watch_context_id"] = watch_id
        PRODUCTION_STATE.setdefault("watchlist", {}).setdefault(watch_id, {})
        PRODUCTION_STATE["watchlist"][watch_id].update(
            {
                "watch_context_id": watch_id,
                "status": "simulated",
                "model_id": getattr(model_record, "id", None),
                "dataset_id": getattr(dataset_record, "id", None),
                "inference_id": getattr(inference_record, "id", None),
                "latest_simulation": simulation_summary,
                "updated_at": datetime.now().isoformat(),
            }
        )
        _append_activity_log(
            f"Simulation executed for model_id={getattr(model_record, 'id', None)} dataset_id={getattr(dataset_record, 'id', None)}.",
            event_type="production.simulation",
            details={
                "steps": simulation_summary.get("steps"),
                "trend": simulation_summary.get("trend"),
                "amplitude": simulation_summary.get("amplitude"),
                "scenario_count": len(simulation_summary.get("scenarios") or []),
            },
        )
        return _utils._json_response(
            {
                "status": "ok",
                "watch_context_id": watch_id,
                "output_features": output_features,
                "output_feature": output_feature,
                "prediction_summary": primary_summary,
                "simulation": simulation_summary,
                "production": {**PRODUCTION_STATE, "activity_log": _read_activity_log(limit=30)},
            }
        )
    except ValueError as exc:
        extra_payload = {}
        if isinstance(exc, _utils.RuntimeArtifactDependencyError):
            extra_payload = exc.to_payload()
        return _utils._json_error(
            str(exc),
            status_code=400,
            watch_context_id=_resolve_watch_context_id(payload_data),
            **extra_payload,
        )
    except Exception as exc:
        LOGGER.exception("Production simulation failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/registry/download/{registry_type}/{item_id}", response_class=FileResponse)
async def registry_download(
    registry_type: str,
    item_id: int,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    try:
        binding = _get_registry_binding(registry_type)
        record = await binding["model"].read(db, item_id)
        if record is None:
            raise ValueError(f"{binding['label']} not found")

        path = _utils._resolve_fs_path(getattr(record, "path", None), default_parent=REPO_ROOT)
        if path is not None and path.exists() and path.is_file():
            return FileResponse(str(path), filename=path.name, media_type="application/octet-stream")

        export_path = EXPORT_ARTIFACT_DIR / f"{_normalize_registry_type(registry_type)}_{item_id}.json"
        save_json(await _build_registry_analysis(db, registry_type, item_id), export_path)
        return FileResponse(str(export_path), filename=export_path.name, media_type="application/json")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
