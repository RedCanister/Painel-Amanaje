FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt /app/requirements.txt

RUN python -m venv /opt/venv && \
/opt/venv/bin/pip install --upgrade pip && \
/opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS final

ENV PATH="/opt/venv/bin:$PATH" \
    MLFLOW_LOCAL_DIR=/mlruns \
    MLFLOW_TMP_DIR=/app/.mlflow_tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv/

COPY --link . /app

RUN useradd --uid 1000 --create-home appuser || true
RUN mkdir -p /app/.mlflow_tmp /app/logs /app/runtime_artifacts /app/mlruns /data/datasets /models /mlruns && \
    chown -R 1000:1000 /app /data /models /mlruns
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8000"]
