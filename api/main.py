from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import yfinance as yf

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

app = FastAPI(title="Painel Amanajé API", version="0.1")

class QuoteResponse(BaseModel):
    ticker: str
    last_close: float
    recent: dict


@app.get("/home")
async def health():
    return {"status": "ok"}


# Data endpoints

# Rota de upload de dados
@app.post("/upload")
async def upload(): # Airflow, Pandas, Postgres, Redis
    # Carregando dados de um função template de bibliotecas python para análise
    """Holder"""

    # Fazendo upload de um conjunto de dados para análise
    """Holder"""

    # Extraindo dados de uma conexão de banco de dados, agendada ou por streaming
    """Holder"""

    # Exibindo dados brutos carregados com SQL
    """Holder"""

    return {"status": "ok"}

# Rota de feature store
@app.get("/features")
async def features(): # Feast, Pandas, Postgres, Redis
    # Selecionando e armazenando novo conjunto de features a partir de conjunto de dados
    """Holder"""

    # Armazenando e versionando features para reutilização
    """Holder"""

    # Criando novo conjunto de features a partir de dados brutos
    """Holder"""

    return {"status": "ok"}

# Rota de análise exploratória de dados
@app.get("/analysis") # Plotly, Feast
async def analysis():
    # Selecionando conjunto de dados para visualização
    """Holder"""

    # Selecionando colunas e amostras para visualização
    """Holder"""

    # Criando visualizações interativas para análise exploratória de dados
    """Holder"""

    return {"status": "ok"}

#

# Model endpoints
@app.post("/training")
async def training(): # MLFlow, ONNX, Feast, Optuna
    # Selecionando modelo salvo para treinamento
    """Holder"""

    # Carregando dados de treinamento e validação
    """Holder"""

    # Importanto modelo ONNX para treinamento
    """Holder"""      


    # --Gerando painel de opções de treinamento
    
    # Selecionando hiperparâmetros de estudos para treinamento
    """Holder"""

    # Verificando histórico do modelo utilizado no treinamento (Métricas, Parâmetros, Artefatos, Tempo de execução, )
    """Holder"""

    # Recebendo parâmetros manualmente para treinamento
    """Holder"""

    # Enviando modelo, dados e parâmetros para treinamento - 
    """Holder"""

    # Salvando o modelo e artefatos gerados no treinamento - MLFlow
    """Holder"""


    return {"status": "ok"}

@app.get("/optimization")
async def optimization(): # MLFlow, ONNX, Optuna
    # Selecionando modelo salvo para estudo
    """Holder"""

    # Exibindo estudos de hiperparâmetros realizados no treinamento
    """Holder"""

    # Exibindo histórico de treinamento do modelo selecionado
    """Holder"""

    # Exibindo visualização de comparação entre estudos realizados ao longo do tempo (Métricas por Custos)
    """Holder"""

    return {"status": "ok"}

@app.get("/production")
async def production(): # Plotly, ONNX, MLFlow, 
    # Selecionando modelo salvo para produção
    """Holder"""

    # Carregando processo de execução do modelo em produção
    """Holder"""

    # Realizando inferência com o modelo em produção
    """Holder"""

    # --Gerando painel de monitoramento do modelo em produção


    # Visualizando dados reais do modelo em produção
    """Holder"""

    # Visualizando dados de inferência do modelo em produção (Predições por regressão, classificação, clusterização, etc)
    """Holder"""

    return {"status": "ok"}


# Development endpoints
@app.post("/onnx")
async def onnx(): # ONNX, PyTorch,
    # Selecionando modelo salvo para validação
    """Holder"""

    # Criando novo modelo de validação
    """Holder"""

    # Importando modelo ONNX para validação
    """Holder"""


    # --Gerando painel de opções de validação

    # Visualizando gráfico de modelo ONNX
    """Holder"""

    # Verificando histórico do modelo selecionado
    """Holder"""

    # Visualizando etapas e parâmetros do modelo salvo
    """Holder"""

    # Enviando modelo e dados para validação
    """Holder"""

    return {"status": "ok"}

@app.get("/airflow")
async def airflow():
    # Redirecionando para o painel do Apache Airflow
    """Holder"""

    return {"status": "ok"}

@app.get("/mlflow")
async def mlflow():
    # Redirecionando para o painel do MLFlow
    """Holder"""

    return {"status": "ok"}


@app.get("/quote", response_model=QuoteResponse)
async def get_quote(ticker: str = Query(..., min_length=1)):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d")
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data for ticker")
        recent = hist.tail(5)[["Open", "High", "Low", "Close"]].round(4).to_dict(orient="index")
        last_close = float(hist["Close"].iloc[-1])
        return {"ticker": ticker.upper(), "last_close": last_close, "recent": recent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mlflow/log")
async def mlflow_log(metric_name: str = "example_metric", value: float = 1.0):
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=501, detail="mlflow not available in this container")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        mlflow.log_metric(metric_name, float(value))
    return {"logged": True, "metric": metric_name, "value": value}