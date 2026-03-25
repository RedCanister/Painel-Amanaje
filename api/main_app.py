import ast
import base64
import json
import logging
import os
import sys
import textwrap
import traceback
import tracemalloc
import uuid
import optuna

from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_session import get_db, init_models
from app.database.db_utils import create_entry, get_all_entries
from app.models.model_objects import CodeModel, DatasetModel, LearningModel, StudyModel
from app.models.model_orm import CodeORM, DatasetORM, LearningORM
from app.models.model_registry import ModelRegistry
from app.utils.utils import (
    build_payload_data,
    build_payload_model,
    get_upload_dir,
    save_file_to_disk,
)

tracemalloc.start()

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

DATASET_OPERATION = "datasets"
MODEL_OPERATION = "models"
ALLOWED_UPLOAD_OPERATIONS = {DATASET_OPERATION, MODEL_OPERATION}


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
    studyId: Optional[str] = None


app = FastAPI(title="Painel Amanaje API", version="0.2.0")

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _register_models() -> None:
    registry_pairs = (
        (DatasetModel, DatasetORM),
        (LearningModel, LearningORM),
        (CodeModel, CodeORM),
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


def _render_page(request: Request, template_name: str, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(template_name, {"request": request, **context})


def _json_error(detail: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"status": "error", "detail": detail}, status_code=status_code)


def _serialize_dataset_summary(dataset: Any) -> dict[str, Any]:
    return {
        "id": getattr(dataset, "id", None),
        "name": getattr(dataset, "name", None),
        "description": getattr(dataset, "description", None),
        "dataset_type": getattr(dataset, "dataset_type", None),
        "shape": getattr(dataset, "shape", None),
        "size": getattr(dataset, "size", None),
    }


def _serialize_model_summary(model: Any) -> dict[str, Any]:
    return {
        "id": getattr(model, "id", None),
        "name": getattr(model, "name", None),
        "description": getattr(model, "description", None),
        "model_type": getattr(model, "model_type", None),
        "is_trained": getattr(model, "is_trained", False),
        "metrics": getattr(model, "metrics", {}) or {},
        "size": getattr(model, "size", None),
    }


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
    "onnx_bytes",
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
        "code": {
            "script": code,
            "language": "python",
            "line_count": len(code.splitlines()),
        },
    }


def _dataset_analysis_summary(df: pd.DataFrame, df_path: str) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}

    for column_name in df.columns:
        column = df[column_name]
        if pd.api.types.is_numeric_dtype(column):
            stats[column_name] = {
                "type": "numeric",
                "mean": None if column.empty else float(column.mean()),
                "std": None if column.empty else float(column.std()),
                "min": None if column.empty else float(column.min()),
                "max": None if column.empty else float(column.max()),
                "null_count": int(column.isnull().sum()),
            }
            continue

        mode = column.mode(dropna=True)
        stats[column_name] = {
            "type": "categorical",
            "unique_values": int(column.nunique(dropna=True)),
            "null_count": int(column.isnull().sum()),
            "mode": None if mode.empty else str(mode.iloc[0]),
        }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": int(df.isnull().sum().sum()),
        "stats": stats,
        "path": df_path,
    }


_register_models()
_include_registry_routers()


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await init_models()
    except Exception:
        LOGGER.exception("Database initialization failed during startup")


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request) -> HTMLResponse:
    return _render_page(request, "base_template.html")


@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request) -> HTMLResponse:
    return _render_page(request, "base_red.html")


@app.get("/create", response_class=HTMLResponse)
async def page_create(request: Request) -> HTMLResponse:
    return _render_page(request, "base_red copy.html")


@app.get("/training", response_class=HTMLResponse)
async def page_training(request: Request) -> HTMLResponse:
    return _render_page(request, "base_green.html")


@app.get("/editor", response_class=HTMLResponse)
async def page_editor(request: Request) -> HTMLResponse:
    return _render_page(request, "base_editor.html")


@app.get("/optimization", response_class=HTMLResponse)
async def page_optimization(request: Request) -> HTMLResponse:
    return _render_page(request, "base_test.html")


@app.get("/production", response_class=HTMLResponse)
async def page_production(request: Request) -> HTMLResponse:
    return _render_page(request, "base_blue.html")


@app.get("/registry", response_class=HTMLResponse)
async def page_registry(request: Request) -> HTMLResponse:
    return _render_page(request, "base_purple.html")


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
            history = (
                json.loads(raw_history)
                if raw_history
                else [{"operation": "upload", "when": now.isoformat()}]
            )
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
            "path": form.get("path") or str(destination_path),
            "date": now,
            "version": version,
            "history": history,
        }

        if normalized_operation == MODEL_OPERATION:
            payload = build_payload_model(common_payload, form)
            orm_model = LearningORM
        else:
            payload, _ = await build_payload_data(common_payload, form, upload_file)
            orm_model = DatasetORM

        created_object = await create_entry(db, orm_model, payload)
        return JSONResponse(
            {
                "status": "ok",
                "id": getattr(created_object, "id", None),
                "path": str(destination_path),
            }
        )
    except Exception as exc:
        LOGGER.exception("Upload failed for operation '%s'", normalized_operation)
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
        return JSONResponse(
            {
                "status": "error",
                "variables": {},
                "stdout": "",
                "stderr": "",
                "error": "No code provided",
            },
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
    normalized_document = _normalize_code_document(
        code=code,
        variables=extracted_variables,
        metadata=editor_metadata,
    )
    defined_names = _extract_defined_names(code)

    return JSONResponse(
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
        body = GenerateRequest.model_validate(await request.json())
    except json.JSONDecodeError:
        return _json_error("Invalid JSON payload", status_code=400)

    prompt = body.prompt.strip()
    operation_id = body.operationId.lower()

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
            path = "generated/generated_model.onnx"
            version = 1
            model_type = "supervised"
            parameters = {{"prompt": {prompt!r}}}
            metrics = {{"task": "define_me"}}
            reference_data = "sample_dataset.csv"
            input_features = ["input"]
            output_features = ["output"]
            is_trained = False
            is_tested = False
            is_deployed = False
            onnx_bytes = ""
            """
        ).strip()
    else:
        script = textwrap.dedent(
            f"""
            # Generated dataset template
            {prompt_comment}

            import pandas as pd

            df = pd.DataFrame({{
                "value": [1, 2, 3],
                "label": ["a", "b", "c"]
            }})

            csv_text = df.to_csv(index=False)

            name = "generated_dataset"
            description = "Generated from the editor prompt"
            object_type = "dataset"
            path = "generated/generated_dataset.csv"
            version = 1
            dataset_type = "csv"
            connection_string = ""
            """
        ).strip()

    return JSONResponse({"script": script})


@app.post("/training/{model_id}", response_class=JSONResponse)
async def post_training(
    model_id: int,
    payload: TrainingRequest,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    _ = db

    model_id = model_id
    dataset_id = payload.dataset_id
    model_type = payload.model_type
    study_id = payload.studyId
    parameters = payload.runtimeParameters
    framework = payload.parameters.framework
    input_features = payload.parameters.input_features
    output_features = payload.parameters.output_features

    learning_model = LearningModel.read(_, model_id)
    dataset_model = DatasetModel.read(_, dataset_id)
    study_model = StudyModel.read(_, study_id)
    parameters = study_model.study_params


    if study_model.objective == "torch":
        study_objective = study_model.objective_torch()
    if study_model.objective == "sklearn":
        study_objective = study_model.objective_sklearn()


    study = optuna.create_study(
            direction = parameters.direction,
            pruner = parameters.pruner,
        )
    study.optimize(study_objective, n_trials=parameters.n_trials)



    try:
        if payload.datasetId <= 0:
            return _json_error("datasetId must be a positive integer", status_code=400)

        job_id = f"train_{model_id}_{payload.datasetId}_{uuid.uuid4().hex[:8]}"
        return JSONResponse(
            {
                "status": "submitted",
                "job_id": job_id,
                "model_id": model_id,
                "dataset_id": payload.datasetId,
                "input_type": payload.inputType,
                "study_id": payload.studyId,
                "estimated_time": "30 minutes",
            }
        )
    except Exception as exc:
        LOGGER.exception("Training submission failed for model_id=%s", model_id)
        return _json_error(str(exc), status_code=500)


@app.get("/features", response_class=JSONResponse)
async def list_features(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return JSONResponse([_serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/list/dataset", response_class=JSONResponse)
async def list_dataset(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    datasets = await get_all_entries(db, DatasetORM)
    return JSONResponse([_serialize_dataset_summary(dataset) for dataset in datasets])


@app.get("/list/model", response_class=JSONResponse)
async def list_model(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    models = await get_all_entries(db, LearningORM)
    return JSONResponse([_serialize_model_summary(model) for model in models])


@app.get("/analysis/data", response_class=JSONResponse)
async def analysis_dataset(
    dataset_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if dataset_id is None:
            datasets = await get_all_entries(db, DatasetORM)
            return JSONResponse(
                {
                    "available_datasets": [
                        {
                            "id": dataset.id,
                            "name": dataset.name,
                            "shape": dataset.shape,
                        }
                        for dataset in datasets
                    ]
                }
            )

        dataset = await DatasetModel.read(db, dataset_id)
        if dataset is None or not getattr(dataset, "path", None):
            return JSONResponse({"error": "Dataset not found"}, status_code=404)

        df_path = str(dataset.path)
        df = pd.read_csv(df_path)
        return JSONResponse(
            {
                "dataset_id": dataset_id,
                "summary": _dataset_analysis_summary(df, df_path),
            }
        )
    except Exception as exc:
        LOGGER.exception("Dataset analysis failed for dataset_id=%s", dataset_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/analysis/model", response_class=JSONResponse)
async def analysis_model(
    model_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    try:
        if model_id is None:
            models = await get_all_entries(db, LearningORM)
            return JSONResponse(
                {
                    "available_models": [
                        {
                            "id": model.id,
                            "name": model.name,
                            "type": model.model_type,
                            "is_trained": model.is_trained,
                        }
                        for model in models
                    ]
                }
            )

        model = await LearningModel.read(db, model_id)
        if model is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)

        return JSONResponse(
            {
                "model_id": model_id,
                "summary": {
                    "name": model.name,
                    "type": model.model_type,
                    "is_trained": model.is_trained,
                    "is_tested": getattr(model, "is_tested", False),
                    "is_deployed": getattr(model, "is_deployed", False),
                    "metrics": getattr(model, "metrics", {}) or {},
                    "parameters": getattr(model, "parameters", {}) or {},
                    "description": model.description,
                    "size": model.size,
                },
            }
        )
    except Exception as exc:
        LOGGER.exception("Model analysis failed for model_id=%s", model_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/mlflow/experiments", response_class=JSONResponse)
async def mlflow_list_experiments() -> JSONResponse:
    return JSONResponse([{"id": "exp_1", "name": "Example Experiment"}])


@app.get("/mlflow/experiments/{exp_id}", response_class=JSONResponse)
async def mlflow_get_experiment(exp_id: str) -> JSONResponse:
    return JSONResponse(
        {
            "id": exp_id,
            "name": "Example Experiment",
            "artifact_location": "/mlruns/1",
            "tags": {},
        }
    )


@app.get("/airflow/dags", response_class=JSONResponse)
async def airflow_list_dags() -> JSONResponse:
    return JSONResponse([{"id": "example_dag", "name": "Example DAG"}])


@app.post("/airflow/dags/{dag_id}/trigger", response_class=JSONResponse)
async def airflow_trigger_dag(dag_id: str) -> JSONResponse:
    return JSONResponse({"status": "triggered", "dag_id": dag_id})


@app.post("/production/start", response_class=JSONResponse)
async def production_start() -> JSONResponse:
    return JSONResponse({"status": "started"})


@app.post("/production/stop", response_class=JSONResponse)
async def production_stop() -> JSONResponse:
    return JSONResponse({"status": "stopped"})


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"status": "error", "detail": exc.detail},
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception", exc_info=exc)
    error_detail = str(exc) if os.getenv("DEBUG") else "An error occurred"
    return JSONResponse({"detail": error_detail}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
