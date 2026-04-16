FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN mkdir -p /mlruns

ENV MLFLOW_ALLOWED_HOSTS="mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,10.*,192.168.*,*.svc,*.svc.cluster.local" \
    MLFLOW_ARTIFACT_ROOT="/mlruns" \
    MLFLOW_BACKEND_STORE_URI="sqlite:////mlruns/mlflow.db"

VOLUME ["/mlruns"]

EXPOSE 5000

CMD mlflow server --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" --serve-artifacts --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" --host 0.0.0.0 --port 5000 --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}"
