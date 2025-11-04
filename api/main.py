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

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

#     ModelRegistry.register_model(DatasetModel, DatasetORM)
#     ModelRegistry.register_model(LearningModel, LearningORM)

#     print("DatasetModel registered:", ModelRegistry.get_orm(DatasetModel))
#     print("LearningModel registered:", ModelRegistry.get_orm(LearningModel))

#     registry = ModelRegistry._registry
#     print("Tipo da var", type(registry))
#     print("Registry:", registry) 

#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)


app = FastAPI(title="Painel Amanajé API", version="0.1", )

cur_dir = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(cur_dir, "static")
templates_dir = os.path.join(cur_dir, "templates")
    

#app.mount("/static", StaticFiles(directory=static_dir), name="static")
#templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    ModelRegistry.register_model(DatasetModel, DatasetORM)
    ModelRegistry.register_model(LearningModel, LearningORM)

    print("DatasetModel registered:", ModelRegistry.get_orm(DatasetModel))
    print("LearningModel registered:", ModelRegistry.get_orm(LearningModel))

    registry = ModelRegistry._registry
    print("Tipo da var", type(registry))
    print("Registry:", registry) 

    async with engine.begin() as conn:
       print("Creating tables...")
       await conn.run_sync(Base.metadata.create_all)
       print("Done creating.")


@app.get("/home", response_class=HTMLResponse)
async def health(request: Request):

    # TODO - base_template.html é a base e é sobre acessar todas as outras rotas para toda a aplicação

    # TODO - base_red.html extende de base_template.html e é sobre realizar todas as ações de banco de dados

    # TODO - base_green.html extende de base_template.html e é sobre realizar todas as ações de inferência de ML

    # TODO - base_blue.html extende de base_template.html e é sobre realizar todas as ações de MLOPs

    print("Home")

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

# Data endpoints - Global variables: Date, Version / Experiment, Path, Size, Types,

# Rota de upload de dados
@app.get("/upload", response_class=HTMLResponse)
async def get_upload(request: Request): 
    # TODO - Essa rota extende de base_red.html e é utilizada para fazer upload de dados, sejam datasets ou modelos

    # TODO - Na mesma página, o usuário deve ter a opção de fazer o upload de um conjunto de dados, de um arquivo de modelo de aprendizado, e visualizá-lo em uma tabela SQL ou JSON

    # TODO - Também, ele deve ter a opção de carregar dados de uma conexão agendada ou por streaming
    
    print("get_upload")
    
    # Campo para formulário de upload de dados usando <TemplateSet>
    """Holder"""
    

    # Campo para upload de arquivos de dados usando <DatasetModel>
    """Holder"""


    # Campo para conexão de banco de dados, agendada ou por streaming <DatasetModel>
    """Holder"""


    # Campo para exibição de dados brutos carregados com SQL <DashboardConfig>
    """Holder"""

    
    return {"status": "ok"}

@app.post("/upload/{operation_id}")
async def post_upload(operation_id: int,
                      file: UploadFile = File(...),
                      db: AsyncSession = Depends(get_db), ): 

    # TODO - Essa rota é uma função post e depois de confirmar o upload dos dados ela retorna para get_upload()

    # TODO - Essa rota deve confirmar e realizar a transferência dos arquivos enviados pelo usuário para o banco de dados

    # TODO - Também deve confirmar o agendamento ou conexão com o banco de dados e atualizar todos os registros da operação

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
async def get_features(db: AsyncSession = Depends(get_db)): 
    # TODO - Essa rota extende de base_red.html e é utilizada para exibir todos os dados da aplicação, por par de classe orm

    # TODO - A interface deve permitir a seleção e visualização de todos os modelos com os objetos associados, tendo como referência o ObjectORM

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
        return JSONResponse(
            content=get_data,
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/features/{feature_id}")
async def post_features(feature_id): 
    # TODO - Essa rota é uma função post e ela serve para realizar a construção de novos datasets a partir de features únicas

    # TODO - Ela deve servir para confirmar a criação ou edição de conjuntos de dados existentes em conjuntos novos 


    # Formatando dados e features selecionadas para o banco de features <FeatureSet>
    """Holder"""

    # Armazenando e versionando features para reutilização <DataBaseOperation>
    """Holder"""

    # Atualizando histórico e metadados após o armazenamento das features <HistoryModel>
    """Holder"""

    return feature_id



# Rota de análise exploratória de dados
@app.get("/analysis") 
async def get_analysis():
    # TODO - Essa rota extende de base_red.html e é utilizada para criar visualizações de datasets utilizando plotly dash para selecionar os dados e exibir os gráficos


    # Selecionando conjunto de dados para visualização <DatasetModel>
    """Holder"""

    # Selecionando colunas e amostras para visualização <SampleSet>
    """Holder"""

    # Criando visualizações interativas para análise exploratória de dados <DashboardConfig>
    """Holder"""

    return {"status": "ok"}



# Model endpoints - Parameters, Metrics, Artifacts, Experiments, Versions, Graphs, Models

# Rota para treinamento de modelos
@app.get("/training")
async def get_training(): # MLFlow, ONNX, Feast, Optuna
    # TODO - Essa rota extende de base_green.html e é utilizada para treinar modelos com dados salvos no banco de dados

    # TODO - O usuário deve conseguir selecionar um modelo salvo, um conjunto de dados compatível, selecionar as variáveis de entrada e saída, os parâmetros e as configurações do treinamento

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

    return {"status": "ok"}

@app.post("/training/{operation_id}")
async def post_training(operation_id): # MLFlow, ONNX, Feast, Optuna
    # TODO - Essa rota é uma função post e ela serve para confirmar o treinamento, os dados, as métricas, os parâmetros e retornar á get_training

    # TODO - Essa rota deve carregar todos os dados, validar as entradas e saídas do treinamento, dar retorno ativo sobre o processo e atualizar o banco de dados com o resultado caso haja sucesso


    # Enviando modelo, dados e parâmetros para treinamento - ONNX <ParameterSet>
    """Holder"""

    # Salvando o modelo e artefatos gerados no treinamento - MLFlow <LearningModel>
    """Holder"""

    # Atualizando histórico e metadados após o modelo treinado <LearningModel> || <HistoryModel>
    """Holder"""    

    return {"status": "ok"}


# Rota para otimização de modelos
@app.get("/optimization")
async def get_optimization(): # MLFlow, ONNX, Optuna
    # TODO - Essa rota extende de base_green.html e é utilizada para otimizar modelos com estudos do optuna

    # TODO - O usuário deve ser capaz de selecionar um modelo e um conjunto de dados para treinar com a biblioteca optuna. 

    # TODO - Ele deve ser capaz de fazer um comparação da mudança ao longo do tempo sobre os parâmetros e estudos do modelo

    # Selecionando modelo salvo para estudo <LearningModel>
    """Holder"""

    # Exibindo estudos de hiperparâmetros realizados no treinamento List <StudySet>
    """Holder"""

    # Exibindo histórico de treinamento do modelo selecionado <HistoryModel>
    """Holder"""

    # Exibindo visualização de comparação entre estudos realizados ao longo do tempo (Métricas por Custos) <DashboardConfig>
    """Holder"""

    return {"status": "ok"}

@app.get("/optimization/{operation_id}")
async def post_optimization(operation_id): # MLFlow, ONNX, Optuna
    # TODO - Essa rota é uma função post e ela serve para confirmar o estudo, salvar os melhores parâmetros e atualizar os modelos e os dados
    
    # TODO - Ela pode seguir a mesma lógica de envio do treinamento de um modelo para o processamento, mas deve retornar os parâmetros do estudo para o banco de dados

    # Iniciando novo estudo de otimização de hiperparâmetros para o modelo selecionado <StudySet>
    """Holder"""

    # Salvando melhores parâmetros encontrados no estudo de otimização <ParameterSet>
    """Holder"""

    # Atualizando e enviando histórico e metadados após o estudo de otimização <HistoryModel>
    """Holder"""

    return {"status": "ok"}


# Rota para produção com inferência de modelos
@app.get("/production")
async def get_production(): # Plotly, ONNX, MLFlow, 
    # TODO - Essa rota extende de base_green.html e é utilizada para ver modelos implementados em um ambiente de produção

    # TODO - Ela deve servir como a última página no workflow do usuário, servindo para colocar modelos com configurações e condições reais após o treinamento

    # Selecionando modelo salvo para produção <LearningModel>
    """Holder"""

    # Carregando processo de execução do modelo em produção <DeploymentOperation>
    """Holder"""

    # Exibindo dados reais do modelo em produção <DashboardConfig>
    """Holder"""

    # Exibindo dados de previsão do modelo em produção (Predições por regressão, classificação, clusterização, etc) <DashboardConfig>
    """Holder"""

    return {"status": "ok"}

@app.post("/production/{operation_id}")
async def post_production(operation_id): # Plotly, ONNX, MLFlow, 
    # TODO - Essa rota é uma função post e ela serve para abrir uma rota e recursos dedicados para um modelo que realiza inferência a nível de produção

    # TODO - Essa rota deve retornar o modelo, os dados e os recursos para um processo recorrente que permite que o modelo realize inferência constante sobre as variáveis alvo, com produção de relatório

    # --Gerando painel de monitoramento do modelo em produção

    # Iniciando rotina de inferência com o modelo em produção <LearningModel>
    """Holder"""

    # Analisando desempenho do modelo em produção  <ReportModel>
    """Holder"""

    # Adicionado e atualizando históricos de monitoramento do modelo em produção <HistoryModel>
    """Holder"""

    return {"status": "ok"}


# Development endpoints

# Rota de criação e validação de modelos ONNX e PyTorch
@app.get("/onnx")
async def get_onnx(): # ONNX, PyTorch,
    # TODO - Essa rota extende de base_blue.html e é utilizada para criar, visualizar, importar e exportar modelos onnx

    # TODO - O usuário deve ser capaz de editar modelos onnx e de visualizar o gráfico resultante do formato. A rota deve ser usada pra realizar a validação dos modelos

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

    return {"status": "ok"}

@app.post("/onnx/{operation_id}")
async def post_onnx(operation_id): 
    # TODO - Essa rota é uma função post e ela serve para confirmar a criação, atualização e operação sobre modelos onnx

    # TODO - Essa rota deve confirmar o envio dos dados e produzir um relatório sobre as configurações e parâmetros dos modelos onnx que estiverem salvos no banco de dados

    # Exportando arquivo de modelo ONNX <ONNXModel>
    """Holder"""

    # Salvando e criando novo modelo de validação para classe <LearningModel>
    """Holder"""

    # Visualizando etapas e parâmetros do modelo salvo <ParameterSet>
    """Holder"""

    # Enviando modelo e dados para validação <ReportModel>
    """Holder"""

    return {"status": "ok"}


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