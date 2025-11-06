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
static_dir = os.path.join(cur_dir, "static")
templates_dir = os.path.join(cur_dir, "templates")

# Ensure static and templates directories exist (templates exist already in repo)
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.on_event("startup")
async def startup_event():

    # Register pydantic <-> ORM pairs
    ModelRegistry.register_model(DatasetModel, DatasetORM)
    ModelRegistry.register_model(LearningModel, LearningORM)

    # Create DB tables if not present
    try:
        await init_models()
        print("Initialized....")
    except Exception as e:
        # In some dev environments init_models may require DB available; print and continue
        print("init_models error (db may not be reachable during dev):", e)
        print(f"Error string (strerror): {traceback.format_tb(e.__traceback__)}")


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    """Render index with navigation to other pages"""
    return templates.TemplateResponse("base_template.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def get_upload(request: Request):
    # TODO - Needs to accept both LearningModel and DatasetModel as pydantic forms for 
    # the user to input data
    return templates.TemplateResponse("base_red.html", {"request": request})


@app.post("/upload/{operation_id}", response_class=JSONResponse)
async def post_upload(operation_id: int,
                      file: UploadFile = File(...),
                      db: AsyncSession = Depends(get_db)):
    
    """Receive uploaded file, store it under data/uploads, and create a DB entry."""    

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Base directory:", base_dir)
    
    # TODO - When operation_id == 1, the post function will send the file to the datasets folder.
    # When operation_id == 2, the post function will send the file to the models folder. 
    upload_dir = os.path.join(base_dir, "..", "data", "datasets")
    print("Base directory:", upload_dir)
    
    file_path = os.path.abspath(upload_dir)
    print("Base directory:", file_path)

    os.makedirs(file_path, exist_ok=True)

    #os.chmod(base_dir, 0o755)
    #os.chmod(file_path, 0o755)

    # TODO - Everytime the user uploads a file, the data he passes on serves to create a folder if that is the first upload
    # TODO - Every subsequent file in the same description goes to that folder.
    # Save file
    
    async with aiofiles.open(file_path + "/" + file.filename, "wb") as f:
        contents = await file.read()
        await f.write(contents)

    # compute sizes
    _, size_mb = get_file_size(file_path)

    file_name = file.filename
    now = datetime.now()

    # Time Series data:          Ticker - Interval - Source 
    # Classification data:       Object - DataType - Source   
    
    # TODO - The dict variables "name", "description", "object_type", "connection_string"
    # all need to come in from the formData in base_red.html
    # Build a payload that maps to the DatasetORM column names
    object_payload = {
        "name": file_name,
        "description": f"Uploaded file {file_name}",
        "object_type": "dataset",
        "size": size_mb,
        "path": file_path + "/" + file_name,
        "date": now,
        "version": 1,
        "history": [{"operation": "upload", "when": now.isoformat()}],
    }

    # TODO - In the interface, the user can choose if the file he is uploading is
    # a model or a dataset. When this post function is requested, operation_id == 1 will
    # return the dataset form, and the operation_id == 2 will return de learning_model form.
    # The payload represents the extra formulary that comtains the specific data for each
    # object type.

    #if operation_id == 1:
    option_payload = {
            "has_features": False,
            "features_list": None,
            "connection_string": None,
        }
    
    # TODO - Make new form for the parameters on the uploaded dataset, or extract from file
    # If pre-made download __dict__ from template class
    
    # elif operation_id == 2:
    option_payload2 = {
            "model_type": "model_type",
            "parameters": "class __dict__",
            "metrics": ["MAE", "Precision", "etc",],
            "reference_data": "Related dataset",
            "input_features": "input_features",
            "output_features": "output_features",
            "is_trained": False,
            "is_tested": False,
            "is_deployed": False,
        }

    try:
        obj = await create_entry(db, DatasetORM, object_payload | option_payload)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "id": getattr(obj, "id", None)})


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request, db: AsyncSession = Depends(get_db)):
    # For simplicity render a page showing JSON list of datasets
    objs = await get_all_entries(db, DatasetORM)
    list_data = [obj.__dict__ for obj in objs]
    return templates.TemplateResponse("base_green.html", {"request": request, "datasets": list_data})


@app.post("/training/{operation_id}", response_class=JSONResponse)
async def post_training(operation_id: int,
                        db: AsyncSession = Depends(get_db)):
    # Placeholder for training orchestration — create a simple LearningORM entry to register training request
    now = datetime.now()

    payload = {
        "name": f"training-{operation_id}",
        "description": f"Training job {operation_id}",
        "object_type": "object",
        "model_type": "training_job",
        "size": 0.0,
        "path": "",
        "date": now,
        "version": 1,
        "history": [{"operation": "training_requested", "when": now.isoformat()}],
        
        "parameters": {},
        "metrics": {},
        "reference_data": None,
        "input_features": None,
        "output_features": None,
        "is_trained": False,
        "is_tested": False,
        "is_deployed": False,
    }

    try:
        obj = await create_entry(db, LearningORM, payload)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    # Kick off asynchronous training task here if desired (omitted)
    return JSONResponse({"status": "ok", "id": getattr(obj, "id", None)})


@app.get("/production", response_class=HTMLResponse)
async def get_production(request: Request):
    return templates.TemplateResponse("base_blue.html", {"request": request})


# Generic exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse({"detail": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
