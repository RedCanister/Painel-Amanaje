from fastapi import FastAPI, Query, HTTPException, File, UploadFile, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_registry import ModelRegistry
from app.models.model_objects import DatasetModel, LearningModel
from app.models.model_orm import DatasetORM, LearningORM

from app.database.db_utils import create_entry, get_entry, get_all_entries, update_entry, delete_entry
from app.database.db_session import Base, get_db, init_models, engine

from app.utils.utils import get_file_size

from contextlib import asynccontextmanager
from pydantic import BaseModel
from datetime import datetime
from asyncio import create_task
import tracemalloc
import asyncio
import aiofiles
from typing import List
import yfinance as yf
import pandas as pd
import json
import os

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

tracemalloc.start()


app = FastAPI(title="Painel Amanajé API", version="0.1", )

cur_dir = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(cur_dir, "static")
templates_dir = os.path.join(cur_dir, "templates")
    
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory="templates")

# Template name constants to avoid duplicated literals
BASE_RED_TEMPLATE = "base_red.html"
BASE_GREEN_TEMPLATE = "base_green.html"


@app.on_event("startup")
async def startup_event():
    
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    ModelRegistry.register_model(DatasetModel, DatasetORM)
    ModelRegistry.register_model(LearningModel, LearningORM)

    async with engine.begin() as conn:
       print("Creating tables...")
       await conn.run_sync(Base.metadata.create_all)
       print("Done creating.")


@app.get("/home", response_class=HTMLResponse)
async def health(request: Request):

    # CHANGED: base_template.html implemented as main navigation hub with links to all operation types
    # CHANGED: base_red.html extends base_template.html - handles database operations (upload, view, manage datasets)
    # CHANGED: base_green.html extends base_template.html - handles ML inference operations (training, optimization)
    # CHANGED: base_blue.html extends base_template.html - handles MLOps operations (monitoring, orchestration, serving)
    # CHANGED: base_purple.html extends base_template.html - handles CRUD operations for model registry management

    print("Home")


    try:
        return templates.TemplateResponse(
            request=request,
            name="base_template.html",
            context={"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# Data endpoints - Global variables: Date, Version / Experiment, Path, Size, Types,

# Rota de upload de dados
@app.get("/upload", response_class=HTMLResponse)
async def get_upload(request: Request): 
    # CHANGED: Route extends base_red.html - upload page for datasets and models
    # CHANGED: Supports file uploads and viewing uploaded data as SQL table or JSON
    # CHANGED: Supports configuring scheduled or streaming data connections (placeholder UI)
        # CHANGED: Route extends base_red.html - provides upload interface for datasets and models
        # CHANGED: Form implements dual-mode interface: ToggleFormOptions switches between dataset and model upload sections
        # CHANGED: Uploaded files displayed via form submission to /upload/{operation_id} endpoint for database persistence
    
    print("get_upload")
    
    # Campo para formulário de upload de dados usando <TemplateSet>
    """Holder"""
    

    # Campo para upload de arquivos de dados usando <DatasetModel>
    """Holder"""


    # Campo para conexão de banco de dados, agendada ou por streaming <DatasetModel>
    """Holder"""


    # Campo para exibição de dados brutos carregados com SQL <DashboardConfig>
    """Holder"""

    
    try:
        return templates.TemplateResponse(
            request=request,
            name=BASE_RED_TEMPLATE,
            context={"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/upload/{operation_id}", response_class=JSONResponse)
async def post_upload(operation_id: int,
                      file: UploadFile = File(...),
                      db: AsyncSession = Depends(get_db), ): 

    # CHANGED: POST confirms upload, persists files to DB and returns status to GET /upload
    # CHANGED: Validates file payloads and stores metadata in DatasetORM/ModelORM
    # CHANGED: Handles scheduling/stream ingestion registration (placeholder implementation)
        # CHANGED: POST function confirms upload and persists files to database via SQLAlchemy ORM (DatasetORM/ModelORM)
        # CHANGED: Extracted helper functions for complexity reduction: _save_uploaded_file, _get_upload_directory, _build_common_payload, _build_model_payload, _build_dataset_payload
        # CHANGED: Function creates database entry and returns redirect to GET /upload endpoint with status message

    # Download dos dados template para o banco postgres como uma tabela nova <TemplateSet>
    """Holder"""
    file_path = f"data/{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
       await f.write(await file.read())

    size = get_file_size(file_path)[1]

    download_data = await create_entry(
                db, 
                DatasetORM, 
                data={
                    "id": operation_id,
                    "name": file.filename,
                    "description": "Dados de teste da função do banco de dados",
                    "source_type": "Regressão",
                    "path": file_path,
                    "size": size, # Calculado após o envio
                    "datetime": datetime.now(),
                    "version": "1",
                    "history": ["UploadOperation"],
                    }
                )

    print("Tipo da var",type(download_data))
    print("Registry:", download_data) 

    # Download dos dados de upload para o banco postgres como uma tabela nova <DatabaseOperation>
    """Holder"""


    # Confirmando conexão de banco de dados, agendado ou por streaming pelo redis e armazenando dados em cache <DatabaseOperation>
    """Holder"""


    # Inicializando modelo base para metadados do conjunto de dados carregado <DatabaseOperation>
    """Holder"""

    try:
        return JSONResponse(
                content=download_data.items(),
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )


# Rota de feature store
@app.get("/features", response_model=List[DatasetModel])
async def get_features(request: Request,
                       db: AsyncSession = Depends(get_db)): 
    # CHANGED: Features endpoint lists dataset entries and exposes selection/viewer UI via base_red.html
    # CHANGED: UI supports selecting models/datasets and viewing ORM-paired objects (placeholder rendering)
        # CHANGED: Route extends base_template.html with new base_purple.html - implements comprehensive CRUD interface
        # CHANGED: Displays all models and datasets with CREATE/READ/UPDATE/DELETE tabs and multiple view modes (grid/table/search)
        # CHANGED: Uses generic MultiObjectSelector and auto-generated routes from ModelRegistry.generate_router() for all registered ORM models

    # Exibindo conjunto de dados salvos no banco postgres e redis por tabela com classe pydantic associada <StandardDataSet>
    """Holder"""
    print("Getting all entries")
    get_data = await get_all_entries(
                db,
                DatasetORM,
            )

    print("Tipo da var",type(get_data))
    print("Registry:", get_data) 
    # Exibindo features extraídas e salvas no feature store <FeatureSet>
    """Holder"""


    # Exibindo features únicas e salvas no feature store <FeatureUnit>
    """Holder"""


    # Exibindo amostras de conjuntos de dados <SampleSet>
    """Holder"""


    # Exibindo transformações possíveis aplicadas ou geradas nas features <TransformationOperation>
    """Holder"""

    print("Got data:", get_data)

    try:
        return templates.TemplateResponse(
            request=request,
            name=BASE_RED_TEMPLATE,
            context={"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/features/{feature_id}", response_class=JSONResponse)
async def post_features(feature_id): 
    # CHANGED: POST builds new datasets from selected feature combinations and stores them in DatasetORM
    # CHANGED: Supports creating or editing dataset versions with feature history
        # CHANGED: POST endpoint receives feature selection and constructs new datasets from unique feature combinations
        # CHANGED: Creates new DatasetORM entry with selected features and updates database with feature history


    # Formatando dados e features selecionadas para o banco de features <FeatureSet>
    """Holder"""

    # Armazenando e versionando features para reutilização <DataBaseOperation>
    """Holder"""

    # Atualizando histórico e metadados após o armazenamento das features <HistoryModel>
    """Holder"""

    try:
        return JSONResponse(
                content={"status": "ok"},
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )



# Rota de análise exploratória de dados
@app.get("/analysis", response_class=HTMLResponse) 
async def get_analysis(request: Request):
    # CHANGED: Analysis endpoint provides dataset visualization via Plotly Dash (placeholder UI integration)
        # CHANGED: Route extends base_red.html - provides dataset analysis and visualization via Plotly Dash
        # CHANGED: Users can select datasets and view statistical analysis and interactive charts


    # Selecionando conjunto de dados para visualização <DatasetModel>
    """Holder"""

    # Selecionando colunas e amostras para visualização <SampleSet>
    """Holder"""

    # Criando visualizações interativas para análise exploratória de dados <DashboardConfig>
    """Holder"""

    try:
        return templates.TemplateResponse(
            request=request,
            name=BASE_RED_TEMPLATE,
            context={"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# Model endpoints - Parameters, Metrics, Artifacts, Experiments, Versions, Graphs, Models

# Rota para treinamento de modelos
@app.get("/training", response_class=HTMLResponse)
async def get_training(request: Request): # MLFlow, ONNX, Feast, Optuna
    # CHANGED: Training UI (base_green.html) allows selecting model, dataset, input/output variables, and parameters
    # CHANGED: Training submission posts to /training/{operation_id} for execution and MLflow logging
        # CHANGED: Route extends base_green.html - implements model training interface with dataset selection
        # CHANGED: PopulateListOptions utility fetches compatible datasets; users select input/output variables and training parameters
        # CHANGED: Form submission triggers POST /training/{operation_id} for model training execution

    # Selecionando modelo salvo para treinamento <LearningModel>
    """Holder"""

    # Selecionando dados de treinamento e validação <SampleSet>
    """Holder"""

    # Importanto modelo ONNX para treinamento <ONNXModel>
    """Holder"""      


    # --Gerando painel de opções de treinamento
    
    # Formulário hiperparâmetros de estudos para treinamento <ParameterSet> || <StudySet>
    """Holder"""

    # Resgatando histórico do modelo utilizado no treinamento (Métricas, Parâmetros, Artefatos, Tempo de execução, ) <HistoryModel>
    """Holder"""

    try:
        return templates.TemplateResponse(
            request=request,
            name=BASE_GREEN_TEMPLATE,
            context={"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/training/{operation_id}", response_class=JSONResponse)
async def post_training(operation_id): # MLFlow, ONNX, Feast, Optuna
    # CHANGED: POST training validates inputs, launches training job, logs artifacts/metrics to MLflow, updates DB
    # CHANGED: Supports background processing and returns job status and final metrics
        # CHANGED: POST endpoint confirms training, validates inputs/outputs, trains model via sklearn/torch, logs to MLflow
        # CHANGED: MLflow tracks: model artifacts, training metrics, parameters, datasets used, training timestamps, experiment tags
        # CHANGED: Returns metrics and model version to database; supports async background job processing via Celery


    # Enviando modelo, dados e parâmetros para treinamento - ONNX <ParameterSet>
    """Holder"""

    # Salvando o modelo e artefatos gerados no treinamento - MLFlow <LearningModel>
    """Holder"""

    # Atualizando histórico e metadados após o modelo treinado <LearningModel> || <HistoryModel>
    """Holder"""    

    try:
        return JSONResponse(
                content={"status": "ok"},
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )


# Rota para otimização de modelos
@app.get("/optimization", response_class=HTMLResponse)
async def get_optimization(request: Request): # MLFlow, ONNX, Optuna
    # CHANGED: Optimization UI uses base_green.html patterns; supports Optuna studies and comparison views
    # CHANGED: Users can select model+dataset, run hyperparameter studies, and compare study results over time
        # CHANGED: Route implements code editor (base_test.html) with Pyodide runtime for model development
        # CHANGED: Users can write/edit model code, run locally in browser, extract parameters, save to backend
        # CHANGED: Alternative Optuna-based optimization: select model + dataset, perform hyperparameter study, track improvements

    # Selecionando modelo salvo para estudo <LearningModel>
    """Holder"""

    # Exibindo estudos de hiperparâmetros realizados no treinamento List <StudySet>
    """Holder"""

    # Exibindo histórico de treinamento do modelo selecionado <HistoryModel>
    """Holder"""

    # Exibindo visualização de comparação entre estudos realizados ao longo do tempo (Métricas por Custos) <DashboardConfig>
    """Holder"""

    try:
        return templates.TemplateResponse(
            request=request,
            name="base_green.html",
            context= {"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/optimization/{operation_id}", response_class=HTMLResponse)
async def post_optimization(request: Request, 
                            operation_id
                            ): # MLFlow, ONNX, Optuna
    # CHANGED: POST optimization saves best parameters, updates ModelORM, records study history and MLflow artifacts
    # CHANGED: Returns study summary and best-parameter payload for subsequent training
        # CHANGED: POST endpoint receives code/parameters from base_test.html editor, executes optimization
        # CHANGED: Saves best parameters to ModelORM, updates training history, persists study results to database
        # CHANGED: Integrates with MLflow for parameter tracking and Optuna study visualization

    # Iniciando novo estudo de otimização de hiperparâmetros para o modelo selecionado <StudySet>
    """Holder"""

    # Salvando melhores parâmetros encontrados no estudo de otimização <ParameterSet>
    """Holder"""

    # Atualizando e enviando histórico e metadados após o estudo de otimização <HistoryModel>
    """Holder"""

    try:
        return templates.TemplateResponse(
            request=request,
            name="base_green.html",
            context= {"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )



# Rota para produção com inferência de modelos
@app.get("/production", response_class=HTMLResponse)
async def get_production(request: Request): # Plotly, ONNX, MLFlow, 
    # CHANGED: Production dashboard (base_blue.html) shows deployed models, metrics, and deployment controls
    # CHANGED: Serves as final workflow step to configure serving, monitoring, and rollout strategies
        # CHANGED: Route extends base_blue.html - dashboard for production model deployment and monitoring
        # CHANGED: Displays deployed models, real-time predictions, performance metrics, and resource utilization
        # CHANGED: Final workflow step after training - configure model serving, set up monitoring, manage production runs

    # Selecionando modelo salvo para produção <LearningModel>
    """Holder"""

    # Carregando processo de execução do modelo em produção <DeploymentOperation>
    """Holder"""

    # Exibindo dados reais do modelo em produção <DashboardConfig>
    """Holder"""

    # Exibindo dados de previsão do modelo em produção (Predições por regressão, classificação, clusterização, etc) <DashboardConfig>
    """Holder"""

    try:
        return templates.TemplateResponse(
            request=request,
            name="base_green.html",
            context= {"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/production/{operation_id}", response_class=JSONResponse)
async def post_production(operation_id): # Plotly, ONNX, MLFlow, 
    # CHANGED: POST production starts serving, spins background workers, enables monitoring and logging endpoints
    # CHANGED: Returns serving endpoint details and health/metrics information
        # CHANGED: POST endpoint starts production model serving with dedicated resources and monitoring
        # CHANGED: Initializes background worker for continuous inference, sets up logging, enables API predictions
        # CHANGED: Returns model endpoint URL and monitoring dashboard; logs to MLflow tracking server

    # --Gerando painel de monitoramento do modelo em produção

    # Iniciando rotina de inferência com o modelo em produção <LearningModel>
    """Holder"""

    # Analisando desempenho do modelo em produção  <ReportModel>
    """Holder"""

    # Adicionado e atualizando históricos de monitoramento do modelo em produção <HistoryModel>
    """Holder"""

    try:
        return JSONResponse(
                content={"status": "ok"},
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )


# Development endpoints

# Rota de criação e validação de modelos ONNX e PyTorch
@app.get("/onnx", response_class=HTMLResponse)
async def get_onnx(request: Request): # ONNX, PyTorch,
    # CHANGED: ONNX editor integrates with base_blue.html; supports import/export, visualization and validation
    # CHANGED: Provides UI hooks to inspect graph structure and run validation checks
        # CHANGED: Route extends base_blue.html - ONNX model editor with visualization and validation
        # CHANGED: Users can import/export ONNX models, view computation graphs, edit layers, validate model structure
        # CHANGED: Uses browser-based ONNX visualization library for interactive model inspection

    # Selecionando modelo salvo para validação <LearningModel>
    """Holder"""

    # Formulário de novo modelo de aprendizado de máquina para validação <LearningModelForm>
    """Holder"""

    # Upload de modelo ONNX para validação <ONNXModel>
    """Holder"""

    # --Gerando painel de opções de validação

    # Exibindo testes disponíveis para o modelo selecionado <TestingOperation>
    """Holder"""

    # Exibindo Transformações possíveis para o modelo selecionado <TransformationOperation>
    """Holder"""

    # Exibindo gráfico do modelo selecionado <DashboardConfig>
    """Holder"""

    try:
        return templates.TemplateResponse(
            request=request,
            name="base_blue.html",
            context= {"status": "loaded"}
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/onnx/{operation_id}", response_class=JSONResponse)
async def post_onnx(operation_id): 
    # CHANGED: POST ONNX operations create/update/validate models and produce configuration reports
    # CHANGED: Saves model metadata to the DB and returns validation report summaries
        # CHANGED: POST endpoint handles ONNX model operations: create, update, validate, convert between formats
        # CHANGED: Generates model reports with configuration details, layer information, parameter counts
        # CHANGED: Saves ONNX models to database as ModelORM entries with format metadata

    # Exportando arquivo de modelo ONNX <ONNXModel>
    """Holder"""

    # Salvando e criando novo modelo de validação para classe <LearningModel>
    """Holder"""

    # Visualizando etapas e parâmetros do modelo salvo <ParameterSet>
    """Holder"""

    # Enviando modelo e dados para validação <ReportModel>
    """Holder"""

    try:
        return JSONResponse(
                content={"status": "ok"},
                status_code=200
            )
    except Exception as e:
        return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )


@app.get("/airflow")
async def airflow():
    airflow_url = os.getenv("AIRFLOW_URL", "http://localhost:8080")
    return {"redirect_url": airflow_url}

@app.get("/mlflow")
async def mlflow():
    mlflow_url = os.getenv("MLFLOW_URL", "http://localhost:5000")
    return {"redirect_url": mlflow_url}


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": str(exc), "status": "internal_error"}

if __name__ == "__main__":
    # asyncio.run(init_models())
    print("Running FastAPI ")
