import os
import tracemalloc
import aiofiles
import asyncio
import traceback
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_objects import DatasetModel, LearningModel
from app.models.model_orm import DatasetORM, LearningORM
from app.models.model_registry import ModelRegistry

from app.database.db_utils import create_entry, get_all_entries
from app.database.db_session import get_db, init_models, engine
from app.utils.utils import get_file_size

# Start tracemalloc for debugging memory if needed
tracemalloc.start()

app = FastAPI(title="Painel Amanajé API", version="0.1")

# Templates and static
cur_dir = os.path.dirname(os.path.abspath(__file__))
# repo_root points to the repository root (one level up from api/)
repo_root = os.path.abspath(os.path.join(cur_dir, ".."))
static_dir = os.path.join(cur_dir, "static")
templates_dir = os.path.join(cur_dir, "templates")

# Ensure static directory exists locally (templates are already present in repo)
os.makedirs(static_dir, exist_ok=True)

# Mount static files and configure Jinja2 templates. Using absolute paths keeps
# behaviour consistent when running inside Docker or on-host.
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.on_event("startup")
async def startup_event():

    # Register pydantic <-> ORM pairs
    ModelRegistry.register_model(DatasetModel, DatasetORM)
    ModelRegistry.register_model(LearningModel, LearningORM)

    # Generating FastAPI routes for objects in model registry
    model_routers = ModelRegistry.generate_all_routers()
    for router in model_routers:
        app.include_router(router)

    # Create DB tables if not present
    try:
        await init_models()
        print("Initialized....")
    except Exception as e:
        # In some dev environments init_models may require DB available; print and continue
        print("init_models error (db may not be reachable during dev):", e)
        print(f"Error string (strerror): {traceback.format_tb(e.__traceback__)}")
        

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render index with navigation to other pages"""
    return templates.TemplateResponse("base_template.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request):
    # Render upload page. The upload POST endpoint accepts multipart/form-data
    # and decides where to store the uploaded file based on operationId. The
    # form in `base_red.html` includes operationId, datasetName and description.
    return templates.TemplateResponse("base_red.html", {"request": request})


# NOTE - What if the user sends more than one file at a time?
@app.post("/upload/{operation_id}", response_class=JSONResponse)
async def post_upload(operation_id: str,
                      request: Request,
                      db: AsyncSession = Depends(get_db)):
    """Receive uploaded file, store it under data/<datasets|models>, and create a DB entry.

    This endpoint accepts multipart/form-data. The form fields expected by the
    current templates are: operationId, datasetName, description and file (UploadFile).

    Operation mapping:
      - 1 => dataset (stored under data/datasets)
      - 2 => model   (stored under models/)

    For scaling: payloads are kept flexible and the ORM creation will choose
    DatasetORM vs LearningORM depending on operation_id.
    """

    form = await request.form()
    upload_file = form.get('file')  # starlette UploadFile
    # Fallback when JS posts without the file key
    if upload_file is None:
        return JSONResponse({"status": "error", "detail": "no file provided"}, status_code=400)

    # Extract form fields (templates provide these names)
    object_name = form.get('objectName') or getattr(upload_file, 'filename', 'unnamed')
    description = form.get('description') or ''

    # Normalize operation id (prefer path param, but accept form override)
    try:
        operation_id = str(form.get('operationId') or operation_id)
    except Exception:
        operation_id = str(operation_id)

    # Determine storage directory based on operation_id
    if operation_id == "models":
        upload_dir = os.path.join(repo_root, 'models')
    elif operation_id == "datasets":
        # default dataset
        upload_dir = os.path.join(repo_root, 'data', 'datasets')

    os.makedirs(upload_dir, exist_ok=True)

    filename = getattr(upload_file, 'filename', 'uploaded.bin')
    dest_path = os.path.join(upload_dir, filename)

    # Save file to disk asynchronously
    try:
        contents = await upload_file.read()
        async with aiofiles.open(dest_path, 'wb') as out_f:
            await out_f.write(contents)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": f"write failed: {e}"}, status_code=500)

    # Compute file size (MB) from bytes written -- fast and accurate per-file
    size_mb = round(len(contents) / (1024 * 1024), 4)
    now = datetime.now()

    # Build common payload (ObjectORM shape)
    common_payload = {
        "name": object_name,
        "description": description or f"Uploaded file {filename}",
        "object_type": "dataset" if operation_id != "models" else "learning_model",
        "size": size_mb,
        "path": dest_path,
        "date": now,
        "version": 1,
        "history": [{"operation": "upload", "when": now.isoformat()}],
    }

    # Add type-specific options
    if operation_id == "models":
        # Learning model metadata
        payload = {
            **common_payload,
            "model_type": form.get('modelType') or 'unknown',
            "parameters": {},
            "metrics": {},
            "reference_data": form.get('referenceData') or None,
            "input_features": form.get('inputFeatures') or None, # To be updated in training
            "output_features": form.get('outputFeatures') or None, # To be updated in training
            "is_trained": False, # After being trained once
            "is_tested": False, # After being studied or validated
            "is_deployed": False, # After being deploy, or removed from it
        }
        try:
            obj = await create_entry(db, LearningORM, payload)
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
    elif operation_id == "datasets":
        # Dataset metadata
        payload = {
            **common_payload,
            "dataset_type": form.get('datasetType') or None,
            "has_features": False, # To be updated in a @app.put
            "features_list": None, # Only added features can be shown here. 
            "connection_string": None, # TODO - A page to receive and update data on a dataset. 
        }
        try:
            obj = await create_entry(db, DatasetORM, payload)
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "id": getattr(obj, "id", None), "path": dest_path})


@app.get("/features", response_class=HTMLResponse)
async def features_page(request: Request, db: AsyncSession = Depends(get_db)):
    # For simplicity render a page showing JSON list of datasets
    objs = await get_all_entries(db, DatasetORM)
    list_data = [obj.__dict__ for obj in objs]
    return templates.TemplateResponse("base_green.html", {"request": request, "datasets": list_data})


@app.post("/training/{operation_id}", response_class=JSONResponse)
async def post_training(operation_id: int,
                        request: Request,
                        db: AsyncSession = Depends(get_db)):
    """Start a training job (lightweight orchestration placeholder).

    This endpoint accepts a JSON payload (the template JS posts a JSONified
    FormData). The function records a LearningORM entry and returns the id.
    In future this will enqueue a worker job (Celery/RQ/Kubernetes Job).
    """
    payload_json = {}
    try:
        payload_json = await request.json()
    except Exception:
        # If no JSON was provided, keep payload empty
        payload_json = {}

    now = datetime.now()

    # Merge incoming parameters into the stored payload for traceability.
    payload = {
        "name": payload_json.get('name') or f"training-{operation_id}",
        "description": payload_json.get('description') or f"Training job {operation_id}",
        "object_type": "learning_model",
        "model_type": payload_json.get('modelType') or 'training_job',
        "size": 0.0,
        "path": "",
        "date": now,
        "version": 1,
        "history": [{"operation": "training_requested", "when": now.isoformat()}],
        "parameters": payload_json.get('parameters') or {k: v for k, v in payload_json.items() if k not in ['name', 'description', 'modelType']},
        "metrics": {},
        "reference_data": payload_json.get('datasetId') or None,
        "input_features": payload_json.get('inputFeatures') or None,
        "output_features": payload_json.get('outputFeatures') or None,
        "is_trained": False,
        "is_tested": False,
        "is_deployed": False,
    }

    try:
        obj = await create_entry(db, LearningORM, payload)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    # In a real system we'd enqueue a background worker here and return a job id.
    return JSONResponse({"status": "ok", "id": getattr(obj, "id", None)})

# NOTE - @app.get('/features)-list_features returns a JSON of the chosen object as a list
# NOTE - Validate and test. If there's an incompatible object
# with a different __tablename__, /features will not open
@app.get('/features', response_class=JSONResponse)
async def list_features(db: AsyncSession = Depends(get_db)):
    """Return a JSON list of datasets/features for UI population."""
    objs = await get_all_entries(db, DatasetORM)
    return JSONResponse([{
        'id': getattr(o, 'id', None),
        'name': getattr(o, 'name', None),
        'description': getattr(o, 'description', None),
        'dataset_type': getattr(o, 'dataset_type', None),
        'size': getattr(o, 'size', None),
    } for o in objs])


@app.get('/analysis', response_class=JSONResponse)
async def analysis(dataset_id: int = None):
    """Return a small analysis placeholder for the selected dataset.

    In the future this will proxy to a Dash/Plotly service or generate server-side
    plots. For now return synthetic stats to wire the UI.
    """
    # Minimal placeholder response
    return JSONResponse({
        'dataset_id': dataset_id,
        'summary': {
            'rows': 1000,
            'columns': 12,
            'missing_values': 5
        }
    })


@app.get('/airflow/dags', response_class=JSONResponse)
async def airflow_list_dags():
    """Return a minimal DAG list. Replace with actual Airflow API integration later."""
    return JSONResponse([{"id": "example_dag", "name": "Example DAG"}])


@app.post('/airflow/dags/{dag_id}/trigger', response_class=JSONResponse)
async def airflow_trigger_dag(dag_id: str):
    # In a real deploy we would call Airflow's REST API; here we return a stubbed response.
    return JSONResponse({"status": "triggered", "dag_id": dag_id})


@app.get('/mlflow/experiments', response_class=JSONResponse)
async def mlflow_list_experiments():
    return JSONResponse([{"id": "exp_1", "name": "Example Experiment"}])


@app.get('/mlflow/experiments/{exp_id}', response_class=JSONResponse)
async def mlflow_get_experiment(exp_id: str):
    return JSONResponse({"id": exp_id, "name": "Example Experiment", "artifact_location": "/mlruns/1", "tags": {}})


@app.get("/production", response_class=HTMLResponse)
async def get_production(request: Request):
    return templates.TemplateResponse("base_blue.html", {"request": request})


@app.post('/production/start', response_class=JSONResponse)
async def production_start():
    # Start/activate production model — in production this may hit an orchestrator
    return JSONResponse({"status": "started"})


@app.post('/production/stop', response_class=JSONResponse)
async def production_stop():
    return JSONResponse({"status": "stopped"})


# Generic exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"detail": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
