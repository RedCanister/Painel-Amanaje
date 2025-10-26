FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN useradd --uid 1000 --create-home dashuser || true
USER dashuser

EXPOSE 8050
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:server", "--bind", "0.0.0.0:8050", "--workers", "2"]