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
import app.utils.main_utils as _utils
from app.utils.main_utils import (
    TrainingRequest, 
    StudyOptimizationRequest, 
    OnnxPrepareRequest
)

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
RUNTIME_DIR = ensure_dir(PROJECT_ROOT / "runtime_artifacts")

CONFIG_SNAPSHOT_DIR = ensure_dir(RUNTIME_DIR / "config")
TRAINING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "training")
MONITORING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "monitoring")
DEPLOYMENT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "deployment")
OPTUNA_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "optuna")
PLOT_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "plots")
SERVING_ARTIFACT_DIR = ensure_dir(RUNTIME_DIR / "serving")

LOG_DIR = ensure_dir(PROJECT_ROOT / "logs")
MLRUNS_DIR = ensure_dir(PROJECT_ROOT / "mlruns")

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

RUNTIME_CONFIG = _utils._build_runtime_config()
configure_tracking(
    tracking_uri=RUNTIME_CONFIG["mlflow"]["tracking_uri"],
    registry_uri=RUNTIME_CONFIG["mlflow"]["registry_uri"],
)


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

# TODO - Remember to include InferenceModel ORM pair in the update.
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
    return _utils._render_page("base_template.html", request, )


@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request) -> HTMLResponse:
    return _utils._render_page("base_red.html", request, )


@app.get("/create", response_class=HTMLResponse)
async def page_create(request: Request) -> HTMLResponse:
    return _utils._render_page("base_red copy.html", request, )


@app.get("/training", response_class=HTMLResponse)
async def page_training(request: Request) -> HTMLResponse:
    return _utils._render_page("base_green.html", request, )

#
@app.get("/optimization", response_class=HTMLResponse)
async def page_optimization(request: Request) -> HTMLResponse:
    return _utils._render_page("base_green.html", request, )


@app.get("/editor", response_class=HTMLResponse)
async def page_editor(request: Request) -> HTMLResponse:
    return _utils._render_page("base_editor.html", request, )


@app.get("/production", response_class=HTMLResponse)
async def page_production(request: Request) -> HTMLResponse:
    return _utils._render_page("base_blue.html", request, )


@app.get("/registry", response_class=HTMLResponse)
async def page_registry(request: Request) -> HTMLResponse:
    return _utils._render_page("base_purple.html", request, )


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
            orm_model = LearningORM
        else:
            payload, dataframe = await build_payload_data(common_payload, form, upload_file)
            payload["history"] = _utils._append_history(history, {"operation": "profiled", "when": now.isoformat()})
            save_json(
                _utils._dataset_analysis_summary(dataframe, str(destination_path)),
                TRAINING_ARTIFACT_DIR / f"dataset_{object_name}_profile.json",
            )
            orm_model = DatasetORM

        created_object = await create_entry(db, orm_model, payload)
        return _utils._json_response({"status": "ok", "id": getattr(created_object, "id", None), "path": str(destination_path)})
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


# TODO - Write a plan on how to implement a LLM all over the project, a use for each page just to fill in the form fields or code scripts with generated text 
# that matches the user's input.
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

    return _utils._json_response({"script": script})


@app.post("/training/{model_id}", response_class=JSONResponse)
async def post_training(
    model_id: int,
    payload: TrainingRequest,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        validated_payload = validate_input(payload.model_dump(), _utils.TrainingRequest)
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
        validate_required_keys(merged_parameters, ["framework"])

        job_id = f"train_{model_id}_{validated_payload.datasetId}_{uuid.uuid4().hex[:8]}"
        workflow = _utils._run_training_workflow(learning_model, dataset_model, merged_parameters, job_id=job_id)
        result: TrainingResult = workflow["result"]
        prepared: _utils.PreparedDataset = workflow["prepared"]
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
                "history": _utils._append_history(
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
                    "history": _utils._append_history(
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

        return _utils._json_response(
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
                "model": _utils._serialize_model_summary(updated_model),
            }
        )
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
        return _utils._json_response({"status": "completed", **(await _utils._optimize_study(study_record, db, payload))})
    except Exception as exc:
        LOGGER.exception("Optuna optimization failed for study_id=%s.", study_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/features", response_class=JSONResponse)
async def list_features(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return _utils._json_response([_utils._serialize_dataset_summary(dataset) for dataset in datasets])


# TODO - The extraction process is how the user can import .csvs into the postgres as a single table fully described among it's many columns and rows
# instead of having only the file as reference, it uses the file as a reference to create a table that is equal to the pandas DataFrame created from it.
# I should expect to be able to edit, transform, format and analyze extracted tables from a .csv
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
        save_json(summary, TRAINING_ARTIFACT_DIR / f"dataset_{dataset_id}_features.json")
        return _utils._json_response({"dataset_id": dataset_id, "dataset_name": getattr(dataset, "name", None), "summary": summary})
    except Exception as exc:
        LOGGER.exception("Feature extraction failed for dataset_id=%s.", dataset_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/list/dataset", response_class=JSONResponse)
async def list_dataset(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return _utils._json_response([_utils._serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/list/model", response_class=JSONResponse)
async def list_model(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    models = await get_all_entries(db, LearningORM)
    return _utils._json_response([_utils._serialize_model_summary(model) for model in models])


# TODO - I would like for the analysis to be how the user can see any form of plot automatically generated based on the selected object, for any object.
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
        return _utils._json_response({"dataset_id": dataset_id, "summary": summary})
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
        return _utils._json_response({"model_id": model_id, "summary": _utils._augment_model_analysis(model)})
    except Exception as exc:
        LOGGER.exception("Model analysis failed for model_id=%s.", model_id)
        return _utils._json_error(str(exc), status_code=500)


@app.get("/mlflow/experiments", response_class=JSONResponse)
async def mlflow_list_experiments() -> JSONResponse:
    try:
        return _utils._json_response(list_experiments())
    except Exception as exc:
        LOGGER.exception("Unable to list MLflow experiments.")
        return _utils._json_error(str(exc), status_code=500)


@app.get("/mlflow/experiments/{exp_id}", response_class=JSONResponse)
async def mlflow_get_experiment(exp_id: str) -> JSONResponse:
    try:
        return _utils._json_response(get_experiment_summary(exp_id))
    except Exception as exc:
        LOGGER.exception("Unable to load MLflow experiment '%s'.", exp_id)
        return _utils._json_error(str(exc), status_code=500)


# TODO - The ONNX implementation of painel amanaje should be of being able to give the user move control after their pytorch model's after being done,
# be it before or after it's training. Make a suggestion on it
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
    return _utils._json_response({"status": "prepared", "payload": prepared_payload})


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
    return _utils._json_response(PRODUCTION_STATE)

# TODO - The production of model's in the painel amanaje should be about recording real time inference of continuously updated or simulated data of beyond the scope of training.
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

        if model_id is None:
            models = await get_all_entries(db, LearningORM)
            trained_models = [model for model in models if getattr(model, "is_trained", False)]
            if not trained_models:
                return _utils._json_error("No trained model is available for production.", status_code=404)
            trained_models.sort(key=lambda model: getattr(model, "id", 0), reverse=True)
            model_id = trained_models[0].id

        model_record = await LearningModel.read(db, model_id)
        if model_record is None:
            return _utils._json_error("Model not found", status_code=404)

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
                resolved_artifact_path = _utils._resolve_fs_path(getattr(model_record, "path", None), default_parent=REPO_ROOT)
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
                "metrics": _utils._normalize_numeric_metrics(getattr(model_record, "metrics", {}) or {}),
                "active_model": _utils._serialize_model_summary(model_record),
            }
        )
        _utils._append_production_log(f"Production started with model_id={model_id} dataset_id={dataset_id}.")

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
        return _utils._json_response({"status": "started", **PRODUCTION_STATE})
    except Exception as exc:
        LOGGER.exception("Production start failed.")
        return _utils._json_error(str(exc), status_code=500)


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
                        "history": _utils._append_history(
                            getattr(model_record, "history", None),
                            {"operation": "production_stop", "when": stopped_at},
                        ),
                    },
                )
        PRODUCTION_STATE.update({"running": False, "status": "inactive", "stopped_at": stopped_at})
        _utils._append_production_log("Production stopped.")
        return _utils._json_response({"status": "stopped", **PRODUCTION_STATE})
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
        model_id = payload.modelId or PRODUCTION_STATE.get("model_id")
        dataset_id = payload.datasetId or PRODUCTION_STATE.get("dataset_id")
        if model_id is None or dataset_id is None:
            return _utils._json_error("Select a model and dataset before monitoring production.", status_code=400)

        snapshot = await _utils._build_live_monitoring_snapshot(db, int(model_id), int(dataset_id), tolerance=float(payload.tolerance))
        PRODUCTION_STATE["monitoring"] = snapshot
        PRODUCTION_STATE["metrics"] = snapshot["metrics"]
        _utils._append_production_log(f"Production monitoring refreshed for model_id={model_id} dataset_id={dataset_id}.")
        return _utils._json_response({"status": "ok", "monitoring": snapshot, "production": PRODUCTION_STATE})
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
        model_id = payload.modelId or PRODUCTION_STATE.get("model_id")
        dataset_id = payload.datasetId or PRODUCTION_STATE.get("dataset_id")
        if model_id is None or dataset_id is None:
            return _utils._json_error("Select a model and dataset before retraining.", status_code=400)

        retraining_result = await _utils._trigger_retraining_from_monitoring(
            db,
            int(model_id),
            int(dataset_id),
            tolerance=float(payload.tolerance),
        )
        _utils._append_production_log(f"Retraining flow executed for model_id={model_id} dataset_id={dataset_id}.")
        return _utils._json_response({"status": "ok", "retraining": retraining_result})
    except Exception as exc:
        LOGGER.exception("Production retraining failed.")
        return _utils._json_error(str(exc), status_code=500)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return _utils._json_response({"status": "error", "detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(_: Request, exc: ResponseValidationError) -> JSONResponse:
    LOGGER.exception("Response validation failed.", exc_info=exc)
    return _utils._json_response({"status": "error", "detail": str(exc)}, status_code=500)


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception", exc_info=exc)
    error_detail = str(exc) if os.getenv("DEBUG") else "An error occurred"
    return _utils._json_response({"detail": error_detail}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
