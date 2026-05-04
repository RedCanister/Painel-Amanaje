from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import traceback
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from fastapi import Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
from app.models.model_orm import AssistantORM, AssistantTrainingDatasetORM, CodeORM, DatasetORM, InferenceORM, LearningORM, StudyORM
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
from app.utils.io import save_json
from app.utils.logging import log_to_mlflow
from app.utils.mlflow_utils import get_experiment_summary, list_experiments
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
RUN_LEDGER_DIR = _utils.RUN_LEDGER_DIR
ASSISTANT_MODEL_DIR = _utils.ASSISTANT_MODEL_DIR
ASSISTANT_DATASET_DIR = _utils.ASSISTANT_DATASET_DIR
ASSISTANT_EVAL_DIR = _utils.ASSISTANT_EVAL_DIR
ASSISTANT_REFERENCE_DIR = ASSISTANT_DATASET_DIR / "references"
PRODUCTION_STATE = _utils.PRODUCTION_STATE
DATASET_OPERATION = _utils.DATASET_OPERATION
MODEL_OPERATION = _utils.MODEL_OPERATION
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
_find_inference_record = _utils._find_inference_record
_build_extraction_manifest = _utils._build_extraction_manifest
_build_registry_analysis = _utils._build_registry_analysis
_get_registry_binding = _utils._get_registry_binding
_json_safe = _utils._json_safe
_normalize_registry_type = _utils._normalize_registry_type
_resolve_runtime_context = _utils._resolve_runtime_context
_prepare_runtime_execution = _utils._prepare_runtime_execution

BACKGROUND_RUN_TASKS: dict[str, asyncio.Task[Any]] = {}

APPROVED_EXECUTION_OUTPUT_FIELDS = {
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


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request) -> HTMLResponse:
    return _render_page("base_template.html", request, )


@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request) -> HTMLResponse:
    return _render_page("base_red.html", request, )


@app.get("/create", response_class=HTMLResponse)
async def page_create(request: Request) -> HTMLResponse:
    return _render_page("base_create.html", request, )


@app.get("/feature", response_class=HTMLResponse)
async def page_feature(request: Request) -> HTMLResponse:
    return _render_page("base_feature.html", request, )


@app.get("/training", response_class=HTMLResponse)
async def page_training(request: Request) -> HTMLResponse:
    return _render_page("base_green.html", request, )

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


@app.get("/registry", response_class=HTMLResponse)
async def page_registry(request: Request) -> HTMLResponse:
    return _render_page("base_purple.html", request, )


@app.get("/assistant", response_class=HTMLResponse)
async def page_assistant(request: Request) -> HTMLResponse:
    return _render_page("base_assistant.html", request, )


@app.get("/upload/support", response_class=JSONResponse)
async def upload_support() -> JSONResponse:
    return _utils._json_response(
        {
            "datasets": _utils.get_dataset_capability_matrix(),
            "models": _utils.get_model_capability_matrix(),
        }
    )


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
            "object_type": "learning_model" if normalized_operation == MODEL_OPERATION else "dataset",
            "size": size_mb,
            "path": str(destination_path),
            "date": now,
            "version": version,
            "history": history,
        }

        if normalized_operation == MODEL_OPERATION:
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
        extracted_variables = _utils._extract_variables(namespace)
    except Exception:
        error_message = traceback.format_exc()
    finally:
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    editor_metadata = _utils._extract_editor_metadata(extracted_variables)
    normalized_document = _utils._normalize_code_document(code=code, variables=extracted_variables, metadata=editor_metadata)
    defined_names = _utils._extract_defined_names(code)

    return _utils._json_response(
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

# TODO - /Generate should return a random draft made from all the available references within the constraints of painel-amanaje 
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

    prompt_comment = "\n".join(f"# {line}" for line in prompt.splitlines() if line.strip())
    is_model_request = operation_id == MODEL_OPERATION or any(
        token in prompt.lower() for token in ("model", "onnx", "train", "sklearn", "torch")
    )

    if is_model_request:
        script = textwrap.dedent(
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

    assistant_plan = {
        "prompt": prompt,
        "operation_id": operation_id,
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

    return _utils._json_response({"script": script, "assistant_plan": assistant_plan})


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
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), Mapping) else {}
    assistant_config = dict((parameters or {}).get("assistant") or {})
    provider_config = assistant_model_to_provider_config(model_record)
    return {
        **payload,
        **{key: value for key, value in assistant_config.items() if value is not None},
        "assistant_config": assistant_config,
        "provider_config": redact_assistant_provider_config(provider_config),
        "provider_name": f"assistant_model_{payload.get('id') or payload.get('name') or 'registry'}",
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


def _assistant_route_catalog() -> list[dict[str, str]]:
    return [
        {"group": "Management", "method": "GET", "path": "/assistant", "operation": "Render the Assistant Management workspace."},
        {"group": "Management", "method": "GET", "path": "/assistant/management/overview", "operation": "Return aggregated assistant status, route catalog, safety profiles, and artifact locations."},
        {"group": "Sessions", "method": "POST", "path": "/assistant/sessions", "operation": "Create an assistant run session in the run ledger."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/draft", "operation": "Create a reviewed WorkflowDraft from the active provider or selected AssistantModel."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/review", "operation": "Review a WorkflowDraft contract and safety state."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/approve", "operation": "Approve a reviewed draft and record a positive training example."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/feedback", "operation": "Record accepted, corrected, rejected, or unsafe feedback for training curation."},
        {"group": "Drafts", "method": "POST", "path": "/assistant/submit", "operation": "Submit an approved assistant run to the configured workflow handoff."},
        {"group": "Providers", "method": "GET", "path": "/assistant/provider/status", "operation": "Inspect configured providers, active provider, fallback state, and model metadata."},
        {"group": "Providers", "method": "POST", "path": "/assistant/provider/test", "operation": "Run a small WorkflowDraft contract diagnostic against a provider."},
        {"group": "Models", "method": "GET", "path": "/assistant/models/list", "operation": "List registered AssistantModel entries available for assistant selection."},
        {"group": "Models", "method": "GET", "path": "/assistant/models/{assistant_model_id}/status", "operation": "Inspect a registered AssistantModel provider configuration."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/test", "operation": "Run the provider contract diagnostic for one registered AssistantModel."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/training/curate", "operation": "Curate training records for one AssistantModel."},
        {"group": "Models", "method": "POST", "path": "/assistant/models/{assistant_model_id}/training/dataset/attach", "operation": "Attach the latest assistant training dataset snapshot to one AssistantModel."},
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
        "assistant_datasets": ASSISTANT_DATASET_DIR,
        "assistant_dataset_snapshots": ASSISTANT_DATASET_DIR / "snapshots",
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

    return _utils._json_response(
        {
            "status": "ok" if not errors else "degraded",
            "generated_at": datetime.now().isoformat(),
            "provider_status": provider_status,
            "assistant_models": {
                "count": len(assistant_models),
                "models": assistant_models,
            },
            "assistant_dataset": dataset_status,
            "references": {
                "repository": reference_status,
                "latest": latest_references,
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
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model status failed.")
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
        manifest = curate_assistant_training_examples(ASSISTANT_DATASET_DIR, output_dir)
        return _utils._json_response(
            {
                "status": "curated",
                "assistant_model": _serialize_assistant_model(model_record),
                "manifest": manifest,
            }
        )
    except ValueError as exc:
        return _utils._json_error(str(exc), status_code=404)
    except Exception as exc:
        LOGGER.exception("Assistant model training curation failed.")
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
        manifest = curate_assistant_training_examples(ASSISTANT_DATASET_DIR, ASSISTANT_DATASET_DIR / "curated")
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
            }
        )
    except Exception as exc:
        LOGGER.exception("Assistant training curation failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.post("/assistant/draft", response_class=JSONResponse)
async def assistant_draft(payload: AssistantDraftRequest, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        selected_assistant_model = None
        if payload.assistant_model_id not in (None, ""):
            selected_assistant_model = await _get_assistant_model_record(db, payload.assistant_model_id)
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
                    "assistant_model_id": payload.assistant_model_id,
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
                "assistant_model_id": payload.assistant_model_id,
            },
            parameters={
                "prompt": payload.prompt,
                "provider": provider.name,
                "requested_provider": payload.provider,
                "assistant_model_id": payload.assistant_model_id,
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
                    "output_features": [prepared.output_feature],
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
                output_features=[prepared.output_feature],
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
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage="failed",
            merge={"error": str(exc)},
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
        _utils.update_run_entry(
            RUN_LEDGER_DIR,
            run_id,
            status="failed",
            stage="failed",
            merge={
                "error": str(exc),
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
        task = asyncio.create_task(
            _execute_training_run(
                run_entry["run_id"],
                model_id=model_id,
                payload_data=validated_payload.model_dump(),
            )
        )
        BACKGROUND_RUN_TASKS[run_entry["run_id"]] = task
        return _utils._json_response(
            {
                "status": "accepted",
                "run_id": run_entry["run_id"],
                "job_id": run_entry["run_id"],
                "model_id": model_id,
                "dataset_id": validated_payload.datasetId,
                "study_id": validated_payload.studyId,
                "ledger": run_entry,
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
        preflight = _utils._build_training_preflight(
            dataset_model,
            learning_model,
            {
                **dict(getattr(learning_model, "parameters", {}) or {}),
                **dict(getattr(study_record, "study_params", {}) or {}),
            },
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
        task = asyncio.create_task(
            _execute_study_run(
                run_entry["run_id"],
                study_id=study_id,
                payload_data=payload.model_dump(),
            )
        )
        BACKGROUND_RUN_TASKS[run_entry["run_id"]] = task
        return _utils._json_response(
            {
                "status": "accepted",
                "run_id": run_entry["run_id"],
                "study_id": study_id,
                "ledger": run_entry,
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
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    study_id: Optional[int] = None,
    inference_id: Optional[int] = None,
) -> JSONResponse:
    return _utils._json_response(
        {
            "runs": _utils.list_run_entries(
                RUN_LEDGER_DIR,
                run_type=run_type,
                model_id=model_id,
                dataset_id=dataset_id,
                study_id=study_id,
                inference_id=inference_id,
            )
        }
    )


@app.get("/runs/get/{run_id}", response_class=JSONResponse)
async def runs_get(run_id: str) -> JSONResponse:
    payload = _utils.read_run_entry(RUN_LEDGER_DIR, run_id)
    if payload is None:
        return _utils._json_error("Run not found", status_code=404)
    return _utils._json_response(payload)


# TODO - The feature extraction process should be the upload of a dataset stored in file to a postgres table
# where the user can manipulate the individual features of a dataset within any possible transformation.
# Like OneHotEnconding, Numpy transformations, queries and operations 
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
        operations = _utils._coerce_feature_operations_from_payload(payload)
        transformed, transform_summary = _utils.apply_feature_operations(dataframe, operations)
        workspace_payload = _utils._build_feature_workspace_payload(
            dataset,
            transformed,
            preview=transform_summary,
            operations=operations,
        )
        return _utils._json_response(workspace_payload)
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

        dataset_payload = {
            "name": transformed_name,
            "description": str(payload.get("description") or f"Feature workspace materialization derived from dataset #{dataset_id}."),
            "object_type": "dataset",
            "size": round(output_path.stat().st_size / (1024 * 1024), 4),
            "path": str(output_path),
            "date": datetime.now(),
            "version": 1,
            "history": [
                {
                    "operation": "feature_materialize",
                    "when": datetime.now().isoformat(),
                    "source_dataset_id": dataset_id,
                    "table_name": table_name,
                    "materialized_to_postgres": postgres_result["materialized"],
                }
            ],
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
            }
        )
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


# TODO - The analysis route is to be used to view all the graphs and information of a model or dataset in one page.
# This need to be the place where the user can view all the plots generated over the lifetime of each object.
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
        save_json(summary, TRAINING_ARTIFACT_DIR / f"dataset_{dataset_id}_analysis.json")
        return _utils._json_response({"dataset_id": dataset_id, "summary": summary, "plots": _build_dataset_plots(summary)})
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
        return _utils._json_response({"model_id": model_id, "summary": summary, "plots": _build_model_plots(summary)})
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


# TODO - The MLflow route should lead directly for MLflow :5000 for analysis
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


@app.post("/production/simulate", response_class=JSONResponse)
async def production_simulate(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    try:
        payload_data = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                payload_data = await request.json()
            except Exception:
                payload_data = {}

        steps = max(1, min(int(payload_data.get("steps", 12) or 12), 120))
        amplitude = float(payload_data.get("amplitude", 0.05) or 0.05)
        trend = str(payload_data.get("trend", "up") or "up").strip().lower()
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
        if prepared.feature_frame.empty:
            return _utils._json_error("The selected dataset does not contain rows for simulation.", status_code=400)

        baseline = prepared.feature_frame.tail(1).iloc[0].copy()
        scenario = payload_data.get("scenario", {}) if isinstance(payload_data.get("scenario", {}), Mapping) else {}
        direction = -1.0 if trend == "down" else 1.0
        simulated_rows = []
        for step_index in range(1, steps + 1):
            row = baseline.copy()
            for column_name in row.index:
                base_value = scenario.get(column_name, row[column_name])
                try:
                    numeric_value = float(base_value)
                    row[column_name] = numeric_value * (1.0 + direction * amplitude * (step_index / max(steps, 1)))
                except (TypeError, ValueError):
                    row[column_name] = base_value
            simulated_rows.append(row.copy())

        simulation_frame = pd.DataFrame(simulated_rows)
        predictions = _utils._predict_with_result(runtime_result, simulation_frame, prepared.task_type)
        label_mapping = {value: key for key, value in prepared.label_mapping.items()}

        def _row_to_json_payload(row: pd.Series) -> dict[str, Any]:
            payload = {}
            for key, value in row.to_dict().items():
                try:
                    payload[key] = round(float(value), 6)
                except (TypeError, ValueError):
                    payload[key] = value
            return _json_safe(payload)

        prediction_series = []
        for step_index, prediction in enumerate(np.asarray(predictions).reshape(-1), start=1):
            if prepared.task_type == "classification":
                rendered_prediction = label_mapping.get(int(prediction), int(prediction))
            else:
                rendered_prediction = float(prediction)
            prediction_series.append(
                {
                    "step": step_index,
                    "prediction": rendered_prediction,
                    "inputs": _row_to_json_payload(simulation_frame.iloc[step_index - 1]),
                }
            )

        numeric_prediction_series = [
            (item["step"], item["prediction"])
            for item in prediction_series
            if isinstance(item["prediction"], (int, float))
        ]
        final_row = simulation_frame.tail(1).iloc[0]
        drift_series = []
        for column_name in simulation_frame.columns[:8]:
            try:
                drift_series.append((column_name, abs(float(final_row[column_name]) - float(baseline[column_name]))))
            except (TypeError, ValueError):
                continue
        simulation_summary = {
            "model_id": getattr(model_record, "id", None),
            "dataset_id": getattr(dataset_record, "id", None),
            "inference_id": getattr(inference_record, "id", None),
            "task_type": prepared.task_type,
            "steps": steps,
            "trend": trend,
            "amplitude": amplitude,
            "baseline": _row_to_json_payload(baseline),
            "series": prediction_series,
            "plots": [
                _build_line_plot(
                    "Simulated Prediction Path",
                    numeric_prediction_series,
                    description="Future-step predictions generated from the current production pair.",
                ),
                _build_bar_plot(
                    "Feature Drift Applied",
                    drift_series,
                    description="Absolute change applied between the baseline row and the final simulated step.",
                ),
            ],
        }
        simulation_summary["plots"] = [plot for plot in simulation_summary["plots"] if plot["series"]]
        PRODUCTION_STATE["simulation"] = simulation_summary

        inference_record = await _upsert_inference_record(
            db,
            model_record=model_record,
            dataset_record=dataset_record,
            input_features=list(simulation_frame.columns),
            output_features=_utils._coerce_feature_list(getattr(model_record, "output_features", None)),
            inference_updates={
                "status": "simulated",
                "latest_simulation": simulation_summary,
            },
            history_entry={
                "operation": "simulation",
                "when": datetime.now().isoformat(),
                "steps": steps,
                "trend": trend,
                "amplitude": amplitude,
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
            details={"steps": steps, "trend": trend, "amplitude": amplitude},
        )
        return _utils._json_response(
            {
                "status": "ok",
                "watch_context_id": watch_id,
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
