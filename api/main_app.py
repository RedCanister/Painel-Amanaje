import os, sys, json
from io import StringIO
import traceback
import tracemalloc
import aiofiles

import asyncio
import traceback, subprocess, tempfile, os, joblib, sys, uuid
from datetime import datetime
from typing import List

import pandas as pd

from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_objects import DatasetModel, LearningModel, ObjectModel
from app.models.model_orm import DatasetORM, LearningORM, ObjectORM
from app.models.model_registry import ModelRegistry

from app.database.db_utils import create_entry, get_all_entries
from app.database.db_session import get_db, init_models, engine
from app.utils.utils import get_file_size, debug_type, save_file_to_disk

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
    ModelRegistry.register_orm_pair(name="code",
                                    fields= {
                                        "variables": dict
                                    },
                                    base_orm=ObjectORM,
                                    base_pydantic=ObjectModel,
                                    table_name="code")

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
    """
    Home page / Dashboard
    
    CHANGED: Now properly returns base_template.html with request context
    This serves as the main entry point with navigation to all sections
    
    Returns:
    - HTML response with navigation links to:
      - Data Operations (Red): /upload, /features, /analysis
      - Model Operations (Green): /training, /optimization, /onnx
      - MLOps (Blue): /production, /airflow, /mlflow
    """
    return templates.TemplateResponse("base_template.html", {"request": request})



@app.get("/upload", response_class=HTMLResponse)
async def page_upload(request: Request):
    """
    Render the upload page
    
    CHANGED: Enhanced documentation
    
    Serves the base_red.html template which provides:
    - File upload interface with drag-and-drop support
    - Form toggle between datasets and models
    - Upload history display
    - Feature management interface
    
    The form will POST to /upload/{operation_id} with:
    - file (UploadFile)
    - objectName
    - description
    - operation_id ('datasets' or 'models')
    - type-specific fields (datasetType, connectionString, etc.)
    """
    return templates.TemplateResponse("base_red.html", {"request": request})


# NOTE - What if the user sends more than one file at a time?
@app.post("/upload/{operation_id}", response_class=JSONResponse)
async def post_upload(operation_id: str,
                      request: Request,
                      db: AsyncSession = Depends(get_db)):
    """
    Receive uploaded file, store it under data/<datasets|models>, and create a DB entry.

    CHANGED: Refactored to reduce cognitive complexity by extracting helper functions
    
    This endpoint accepts multipart/form-data. The form fields expected are:
    - file: the uploaded file (UploadFile)
    - operationId: 'datasets' or 'models'
    - objectName: display name for the object
    - description: optional description
    - type-specific fields depending on operation_id

    Operation mapping:
      - 'datasets' => DatasetORM (stored under data/datasets)
      - 'models' => LearningORM (stored under models/)

    Returns:
    - status: 'ok' or 'error'
    - id: database record ID
    - path: file storage path on disk
    """
    
    # CHANGED: Extract and validate form data
    form = await request.form()
    upload_file = form.get('file')

    #debug_type(upload_file)
    
    if upload_file is None:
        return JSONResponse({"status": "error", "detail": "no file provided"}, status_code=400)

    # Extract common fields
    object_name = form.get('objectName') or getattr(upload_file, 'filename', 'unnamed')
    description = form.get('description') or ''
    operation_id = str(form.get('operationId') or operation_id)

    # CHANGED: Helper function to determine storage path
    def get_upload_dir(op_id: str) -> str:
        if op_id == "models":
            return os.path.join(repo_root, 'models')
        else:  # Default to datasets
            return os.path.join(repo_root, 'data', 'datasets')

    # CHANGED: Helper function to build operation-specific payload
    # To consider extensions: .pkl, .onnx, .ph, etc...
    def build_payload_model(common: dict, form_data, ) -> dict:
        """Build type-specific payload based on operation_id"""

        return {
            **common,
            "model_type": form_data.get('modelType') or 'learning_model',
            "parameters": {},
            "metrics": {},
            "reference_data": form_data.get('referenceData'),
            "input_features": form_data.get('inputFeatures'),
            "output_features": form_data.get('outputFeatures'),
            "is_trained": False,
            "is_tested": False,
            "is_deployed": False,
        }
        
    async def build_payload_data(common: dict, form_data, uploaded_file) -> tuple:
        
        from io import StringIO
        debug_type(uploaded_file)
        await uploaded_file.seek(0)
        content = await uploaded_file.read()

        s = StringIO(content.decode('UTF-8'))

        print("s", s)

        df = pd.read_csv(s, header=0)

        debug_type(df)

        payload = {
            **common,
            "dataset_type": form_data.get('datasetType') or 'dataset',
            "shape": list(df.shape) or None,
            "has_features": False,
            "features_list": df.columns.to_list() or None,
            "connection_string": form_data.get('connectionString'),
        }

        return payload, df

    
    try:
        # Save file to disk
        upload_dir = get_upload_dir(operation_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = getattr(upload_file, 'filename', 'uploaded.bin')
        dest_path = os.path.join(upload_dir, filename)
        
        size_mb = await save_file_to_disk(upload_file, dest_path)
        now = datetime.now()

        # Build payloads and save to database
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

        if operation_id == "models":
            payload = build_payload_model(common_payload, form)
        if operation_id == "datasets":
            payload, df = await build_payload_data(common_payload, form, upload_file)
            debug_type(df)
        
        orm_model = LearningORM if operation_id == "models" else DatasetORM
        obj = await create_entry(db, orm_model, payload)

        return JSONResponse({"status": "ok", "id": getattr(obj, "id", None), "path": dest_path})

    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# CHANGED: base_test.html aligned with /create endpoint; frontend posts code JSON to /create
# CHANGED: post_create supports optional 'save' flag to persist code as LearningORM entry
# NOTE - The /create endpoint serves as another measure for the user to input a file
# in the amanaje app. Instead of uploading a file, the user can write his own code, but
# I must have a proper way to parse and save the model created in the code.
@app.post("/execute", response_class=JSONResponse)
async def post_execute(request: Request,
                      db: AsyncSession = Depends(get_db)):
    """
    CHANGED: Simplified endpoint signature - removed unused operation_id path parameter
    
    This was causing 422 errors because FastAPI expected:
    POST /execute/{operation_id}
    
    But frontend was sending:
    POST /execute
    
    Now accepts JSON body with:
    {
        "code": "python code here",
        "save": false,  # optional
        "objectName": "model_name"  # optional if save=true
    }
    
    Returns: {
        "status": "success" or "error",
        "variables": {...},
        "stdout": "...",
        "stderr": "...",
        "error": null or error_message
    }
    """
    
    try:  
        # CHANGED: Parse JSON body safely
        try:
            data = await request.json()
            code = data.get("code", "").strip()
        except Exception:
            # Fallback to raw text body if not JSON
            raw = await request.body()
            code = raw.decode("utf-8") if raw else None

        if not code:
            return JSONResponse({
                "status": "error",
                "variables": {},
                "stdout": "",
                "stderr": "",
                "error": "No code provided"
            }, status_code=400)
        
        # CHANGED: Create namespace with common libraries
        namespace = {
            '__builtins__': __builtins__,
            'np': __import__('numpy'),
            'pd': __import__('pandas'),
            'ox': __import__('onnx') 
        }

        # CHANGED: Capture stdout/stderr during execution
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        extracted_variables = {}
        error_message = None

        try:
            # CHANGED: Execute user code in isolated namespace
            exec(code, namespace)

            # CHANGED: Extract non-private variables from namespace
            for name, variable in namespace.items():
                if not name.startswith('_') and not isinstance(variable, type(sys)):
                    try:
                        var_type = type(variable).__name__
                        var_repr = repr(variable)

                        # CHANGED: Truncate long representations
                        if len(var_repr) > 150:
                            var_repr = var_repr[:147] + '...'

                        extracted_variables[name] = {
                            'type': var_type,
                            'value': var_repr
                        }

                    except Exception as e:
                        extracted_variables[name] = {
                            'type': type(variable).__name__,
                            'value': f'<Error serializing: {str(e)}>'
                        }

        except Exception as e:
            # CHANGED: Capture full traceback for debugging
            error_message = traceback.format_exc()

        finally:
            # CHANGED: Always restore stdout/stderr
            stdout_output = sys.stdout.getvalue()
            stderr_output = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # CHANGED: Return structured response
        return JSONResponse({
            "status": "success" if not error_message else "error",
            "variables": extracted_variables,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "error": error_message
        })

    except json.JSONDecodeError:
        return JSONResponse({
            "status": "error",
            "variables": {},
            "stdout": "",
            "stderr": "",
            "error": "Invalid JSON payload"
        }, status_code=400)

    except Exception as e:
        # CHANGED: Catch-all for unexpected errors
        return JSONResponse({
            "status": "error",
            "variables": {},
            "stdout": "",
            "stderr": "",
            "error": str(e)
        }, status_code=500)

@app.get("/features", response_class=HTMLResponse)
async def features_page(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Render features management page
    
    CHANGED: Now properly returns base_green.html template
    
    Provides interface for:
    - Model selection and training configuration
    - Hyperparameter specification (manual or study-based)
    - Model optimization via Optuna
    - ONNX export and validation
    """
    objs = await get_all_entries(db, DatasetORM)
    list_data = [obj.__dict__ for obj in objs]
    return templates.TemplateResponse("base_green.html", {"request": request, "datasets": list_data})


@app.post("/training/{operation_id}", response_class=JSONResponse)
async def post_training(operation_id: int,
                        request: Request,
                        db: AsyncSession = Depends(get_db)):
    """
    Start a training job
    
    CHANGED: Enhanced documentation and error handling
    
    This endpoint accepts a JSON payload (posted by base_green.html form).
    The payload includes:
    - name: training job identifier
    - modelType: type of model being trained
    - datasetId: reference to training dataset
    - parameters: hyperparameters for this run
    - inputFeatures: input column names
    - outputFeatures: target column names
    - inputType: 'Manual' or 'Study' (Optuna)
    - studyId: (if inputType='Study') reference to Optuna study
    
    Workflow:
    1. Accept training configuration from frontend
    2. Store as LearningORM record with history entry
    3. Return job ID for tracking
    4. (Future) enqueue to Celery/Kubernetes for execution
    
    Returns:
    - status: 'ok' or 'error'
    - id: database training record ID
    """

    payload_json = {}
    try:
        payload_json = await request.json()
    except Exception:
        # If no JSON provided, use empty dict
        payload_json = {}

    now = datetime.now()

    # CHANGED: Extract and merge incoming parameters
    # Parameters become model.parameters dict; other fields go to appropriate ORM columns
    payload = {
        "name": payload_json.get('name') or f"training-{operation_id}",
        "description": payload_json.get('description') or f"Training job {operation_id}",
        "object_type": "learning_model",
        "model_type": payload_json.get('modelType') or 'training_job',
        "size": 0.0,  # Will be updated after training completes
        "path": "",   # Will be updated after model is saved
        "date": now,
        "version": 1,
        "history": [{"operation": "training_requested", "when": now.isoformat(), "input_type": payload_json.get('inputType')}],
        # CHANGED: Hyperparameters from form become parameters dict
        "parameters": payload_json.get('parameters') or {
            k: v for k, v in payload_json.items() 
            if k not in ['name', 'description', 'modelType', 'datasetId']
        },
        "metrics": {},  # Will be populated during training
        "reference_data": payload_json.get('datasetId'),  # Reference to training dataset
        "input_features": payload_json.get('inputFeatures'),
        "output_features": payload_json.get('outputFeatures'),
        "is_trained": False,
        "is_tested": False,
        "is_deployed": False,
    }

    try:
        obj = await create_entry(db, LearningORM, payload)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    # CHANGED: In real deployment, would enqueue background job here:
    # - Could use Celery: celery_app.send_task('tasks.train_model', args=[obj.id])
    # - Could use Kubernetes: job = create_kubernetes_job(model_id=obj.id, ...)
    # - Could use Airflow: airflow_client.trigger_dag('train_model_dag', conf={'model_id': obj.id})
    
    return JSONResponse({"status": "ok", "id": getattr(obj, "id", None)})

# NOTE - @app.get('/features)-list_features returns a JSON of the chosen object as a list
# NOTE - Validate and test. If there's an incompatible object
# with a different __tablename__, /features will not open
@app.get('/features', response_class=JSONResponse)
async def list_features(db: AsyncSession = Depends(get_db)):
    """
    Return JSON list of available datasets/features
    
    CHANGED: Enhanced documentation and error handling
    
    This endpoint is called by:
    - base_red.html: populate #datasetSelect dropdown
    - base_green.html: populate #datasetId dropdown for training
    - Analysis interface: to choose which dataset to analyze
    
    Returns:
    - Array of dataset objects with: id, name, description, dataset_type, size
    
    Scaling considerations:
    - Returns only publicly available datasets (future: add permission checks)
    - Could implement pagination for large datasets (limit/offset params)
    - Could add filtering by dataset_type
    """
    objs = await get_all_entries(db, DatasetORM)
    return JSONResponse([{
        'id': getattr(o, 'id', None),
        'name': getattr(o, 'name', None),
        'description': getattr(o, 'description', None),
        'dataset_type': getattr(o, 'dataset_type', None),
        'size': getattr(o, 'size', None),
    } for o in objs])


@app.get('/analysis', response_class=JSONResponse)
async def analysis(dataset_id: int = None, db: AsyncSession = Depends(get_db)):
    """
    Return analysis data for selected dataset
    
    CHANGED: Enhanced with detailed documentation
    
    Query parameters:
    - dataset_id: (optional) ID of dataset to analyze
    
    Future enhancements:
    - Compute actual statistics: min, max, mean, std for numeric columns
    - Detect missing values and data types
    - Return correlation matrix for feature relationships
    - Generate visualizations via Plotly/Dash service
    - Support multi-column analysis and custom aggregations
    
    Current implementation:
    - Returns placeholder stats for demo purposes
    - Ready for backend integration with pandas/numpy
    
    Returns:
    - dataset_id: requested dataset ID
    - summary: dict with row count, column count, missing values stats
    """
    # CHANGED: Minimal placeholder response
    # FUTURE: Implement real analysis by uncommenting and integrating pandas:
    # 1. Fetch dataset from database by ID
    # 2. Read file (CSV, Parquet, etc.) into pandas DataFrame
    # 3. Compute statistics: min, max, mean, std, quantiles
    # 4. Detect data types and missing value patterns
    # 5. Return correlation matrix for feature relationships
    # 6. (Optional) Generate interactive Plotly/Dash visualizations
    # 7. Cache results for repeated queries
    
    dataset = await DatasetModel.read(db, dataset_id)
    debug_type(dataset)

    df_path = dataset.path
    df = pd.read_csv(df_path)
    debug_type(df)


    return JSONResponse({
        'dataset_id': dataset_id,
        'summary': {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_values': 5,
            'path': df_path
        }
    })

@app.get('/optimization', response_class=HTMLResponse)
async def code_prompt_test(request: Request,):
    return templates.TemplateResponse("base_test copy.html", {"request": request, })

@app.get("/registry", response_class=HTMLResponse)
async def model_registry_view(request: Request):
    """
    Model Registry CRUD Interface
    
    CHANGED: Implemented the TODO from model_registry.py to provide base_purple.html
    
    Serves base_purple.html which provides a comprehensive interface for:
    - CREATE: Register new datasets or learning models with custom fields
    - READ: View all entries in grid or table format
    - UPDATE: Edit existing model entries via modal forms
    - DELETE: Remove entries with confirmation
    - SEARCH: Find and view detailed information about specific entries
    
    This template utilizes the auto-generated CRUD routes from ModelRegistry.generate_router()
    which creates endpoints like:
    - /dataset/create, /dataset/list, /dataset/get/{id}, /dataset/update/{id}, /dataset/delete/{id}
    - /learning/create, /learning/list, /learning/get/{id}, /learning/update/{id}, /learning/delete/{id}
    
    The interface is fully generic and works with any models registered via:
    ModelRegistry.register_model(PydanticModel, ORMModel)
    
    Users can easily manage the entire data model ecosystem without writing queries.
    """
    return templates.TemplateResponse("base_purple.html", {"request": request})

@app.get('/airflow/dags', response_class=JSONResponse)
async def airflow_list_dags():
    """
    Return list of available Airflow DAGs
    
    CHANGED: Added documentation for integration pattern
    
    This endpoint is called by base_blue.html to populate DAG selector.
    
    Production implementation should:
    - Call Airflow REST API: requests.get('http://airflow:8080/api/v1/dags')
    - Handle authentication (Kerberos, Basic Auth, OAuth2)
    - Filter by user permissions
    - Cache results with TTL (e.g., 5 minute cache)
    - Return only ENABLED dags (paused_at=null)
    
    Response format:
    - Array of DAG objects with: id, name, owner, last_run, is_active
    
    Current: Returns stub data for UI testing
    """
    # CHANGED: Stub response - replace with Airflow API call in production
    return JSONResponse([
        {"id": "example_dag", "name": "Example DAG"},
        # In production, populate from:
        # response = requests.get(
        #     'http://airflow-scheduler:8080/api/v1/dags',
        #     auth=HTTPBasicAuth(user, pwd)
        # )
        # return response.json()['dags']
    ])


@app.post('/airflow/dags/{dag_id}/trigger', response_class=JSONResponse)
async def airflow_trigger_dag(dag_id: str):
    """
    Trigger execution of an Airflow DAG
    
    CHANGED: Added documentation for production integration
    
    Path parameters:
    - dag_id: ID of DAG to trigger
    
    Production implementation should:
    - Make POST request to Airflow REST API
    - Pass configuration parameters if needed (from request body)
    - Poll DAG run status for completion
    - Return run_id for tracking progress
    - Handle rate limiting and error cases
    
    Use cases:
    - Trigger data processing pipeline (ETL)
    - Trigger model retraining when new data arrives
    - Trigger feature engineering workflows
    
    Current: Returns stub response
    """
    # CHANGED: Stub response - replace with Airflow API call in production
    # In production:
    # conf = await request.json() if request.method == 'POST' else {}
    # response = requests.post(
    #     f'http://airflow-scheduler:8080/api/v1/dags/{dag_id}/dagRuns',
    #     json={'conf': conf},
    #     auth=HTTPBasicAuth(user, pwd)
    # )
    # return response.json()
    
    return JSONResponse({"status": "triggered", "dag_id": dag_id})


@app.get('/mlflow/experiments', response_class=JSONResponse)
async def mlflow_list_experiments():
    """
    Return list of MLflow experiments
    
    CHANGED: Added documentation for MLflow integration
    
    This endpoint is called by base_blue.html to populate experiment selector.
    
    MLflow experiments group related model training runs:
    - Different hyperparameter combinations
    - Different datasets
    - Different algorithms (all variants tracked together)
    
    Production implementation should:
    - Call MLflow REST API: requests.get('http://mlflow:5000/api/2.0/mlflow/experiments/search')
    - Filter by user/project if using MLflow Projects
    - Return with basic metadata (id, name, creation_time, artifact_location)
    - Current: stubbed for UI development
    
    Returns:
    - Array of experiments with: id, name, artifact_location, created_time
    """
    # CHANGED: Stub response - replace with MLflow API call in production
    # from mlflow.tracking import MlflowClient
    # client = MlflowClient('http://mlflow:5000')
    # experiments = client.search_experiments()
    # return [{'id': e.experiment_id, 'name': e.name} for e in experiments]
    
    return JSONResponse([{"id": "exp_1", "name": "Example Experiment"}])


@app.get('/mlflow/experiments/{exp_id}', response_class=JSONResponse)
async def mlflow_get_experiment(exp_id: str):
    """
    Get detailed information about an MLflow experiment
    
    CHANGED: Added documentation for experiment details retrieval
    
    Path parameters:
    - exp_id: Experiment ID to retrieve
    
    Returns:
    - Experiment metadata: id, name, artifact_location, tags, created_time
    - Could extend to include best metrics across all runs in experiment
    - Could include list of runs and their parameters/metrics
    
    Production implementation:
    - Call MLflow API for experiment details
    - List all runs in experiment with best metrics
    - Retrieve artifacts (models, plots, etc.)
    - Return formatted comparison view
    
    Current: Returns stub experiment data
    """
    # CHANGED: Stub response - replace with MLflow API call in production
    # from mlflow.tracking import MlflowClient
    # client = MlflowClient('http://mlflow:5000')
    # exp = client.get_experiment(exp_id)
    # runs = client.search_runs(experiment_ids=[exp_id])
    # return {
    #     'id': exp.experiment_id,
    #     'name': exp.name,
    #     'artifact_location': exp.artifact_location,
    #     'tags': exp.tags,
    #     'runs': [...list of runs with metrics...]
    # }
    
    return JSONResponse({
        "id": exp_id,
        "name": "Example Experiment",
        "artifact_location": "/mlruns/1",
        "tags": {}
    })


@app.get("/production", response_class=HTMLResponse)
async def get_production(request: Request):
    """
    Render production monitoring dashboard
    
    CHANGED: Enhanced documentation
    
    Serves base_blue.html template with:
    - Production model health metrics
    - Airflow DAG orchestration interface
    - MLflow experiment tracking
    - Production controls (start/stop/restart)
    """
    return templates.TemplateResponse("base_blue.html", {"request": request})


@app.post('/production/start', response_class=JSONResponse)
async def production_start():
    """
    Start production model serving
    
    CHANGED: Added comprehensive documentation
    
    This endpoint triggers model server startup.
    
    Production implementation should:
    - Load latest model from model registry
    - Start TorchServe/TensorFlow Serving/custom server
    - Perform health checks (test inference request)
    - Warm up GPU/accelerators if applicable
    - Update load balancer to route traffic
    - Log event to audit trail
    
    Returns:
    - status: 'started' or 'error'
    - message: reason if error occurs
    - server_info: (optional) endpoint, version, etc.
    
    Current: Stub implementation for UI testing
    """
    # CHANGED: Stub response
    # Production implementation might:
    # 1. Check if server already running (idempotent)
    # 2. Load model: model = mlflow.pytorch.load_model(model_uri)
    # 3. Start server: torch.jit.script(model).serve()
    # 4. Verify health: requests.get('http://localhost:8080/health')
    # 5. Update metrics: prometheus_client.set_gauge('model_serving_active', 1)
    
    return JSONResponse({"status": "started"})


@app.post('/production/stop', response_class=JSONResponse)
async def production_stop():
    """
    Stop production model serving
    
    CHANGED: Added comprehensive documentation
    
    This endpoint triggers graceful model server shutdown.
    
    Production implementation should:
    - Stop accepting new requests (drain pool)
    - Wait for in-flight requests to complete (with timeout)
    - Save final metrics to database
    - Stop model server process
    - Remove from load balancer
    - Log event to audit trail
    - (Optional) Archive logs for this serving session
    
    Returns:
    - status: 'stopped' or 'error'
    - uptime: (optional) how long the server was running
    - requests_processed: (optional) total requests served
    
    Current: Stub implementation for UI testing
    """
    # CHANGED: Stub response
    # Production implementation might:
    # 1. Set "shutdown" flag to reject new requests
    # 2. Wait for in-flight requests: for i in range(60): if no requests, break; sleep(1)
    # 3. Shutdown server: torch_server.shutdown()
    # 4. Update metrics: prometheus_client.set_gauge('model_serving_active', 0)
    # 5. Save stats: db.record_serving_session(requests, uptime, errors)
    
    return JSONResponse({"status": "stopped"})


# Generic exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    
    CHANGED: Added comprehensive documentation
    
    Catches all exceptions not handled by specific exception handlers.
    
    Behavior:
    - Log full traceback for debugging
    - Return JSON error response to client
    - Sanitize error message to avoid leaking internal details
    - Set appropriate HTTP status code (500 for server errors)
    - (Optional) Send alert if critical errors occur
    
    Returns:
    - detail: error message (sanitized in production)
    - request_id: (optional) for tracking in logs
    - timestamp: when error occurred
    
    In production, should:
    - Log to centralized log aggregation (ELK, Splunk, etc.)
    - Send alerts for specific error types
    - Return generic error message to client (don't expose internals)
    - Track error patterns to identify bugs
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # CHANGED: Return generic error message in production
    # In development, might return str(exc) for debugging
    error_detail = str(exc) if os.getenv("DEBUG") else "An error occurred"
    
    return JSONResponse(
        {"detail": error_detail},
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
