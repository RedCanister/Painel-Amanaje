# Painel Amanaje

Painel Amanaje is a local-first ML/MLOps platform built around a FastAPI control plane, HTML dashboards, model and dataset registries, training utilities, MLflow tracking, ONNX preparation, and production monitoring workflows.

The project started as a stock forecasting effort around NVIDIA market data and Temporal Fusion Transformer ideas, but the current repository has evolved into a broader experimentation and operations workspace for datasets, models, studies, and deployment flows.

## Current status

- Active API entrypoint: `api/main_app.py`
- Current FastAPI app version: `0.3.0`
- Main local stack: PostgreSQL, Redis, Airflow, MLflow, FastAPI, pgAdmin
- Optional sidecars in the repo: Dash dashboard, Feast example repo, MkDocs docs, Kubernetes/Kubeflow design notes
- Major cleanup and alignment work from the project reports has already been applied:
  - unified upload workflow
  - metadata-driven create flow
  - reusable frontend form utilities
  - console/runtime bug fixes
  - registry CRUD improvements
  - test-covered `v2` registry refactor in progress

## What is implemented today

### UI routes

| Route | Purpose |
| --- | --- |
| `/` | Main navigation page |
| `/upload` | Upload datasets and models |
| `/create` | Metadata-first editor and object creation flow |
| `/training` | Training configuration and execution UI |
| `/optimization` | Optimization page currently rendered from `base_green.html` |
| `/editor` | Code editor page |
| `/production` | Production monitoring and deployment controls |
| `/registry` | Registry CRUD interface |

### Active API capabilities

- `POST /upload/{operation_id}` uploads datasets or models through a unified handler.
- `POST /execute` runs editor code server-side and returns variables, metadata, stdout, and stderr.
- `POST /generate` creates starter dataset or model scripts from prompts.
- `POST /training/{model_id}` runs training, writes artifacts, updates the registry, and logs metrics.
- `POST /studies/{study_id}/optimize` runs Optuna-backed study optimization.
- `GET /features`, `GET /list/dataset`, and `GET /list/model` expose dataset and model summaries.
- `GET /features/extract`, `GET /analysis/data`, and `GET /analysis/model` generate feature and analysis summaries.
- `POST /onnx/prepare` and `GET /onnx/validate/{model_id}` support ONNX preparation and validation flows.
- `GET /production/status`, `POST /production/start`, `POST /production/stop`, `POST /production/monitor`, and `POST /production/retrain` manage production state.
- `GET /mlflow/experiments` and `GET /mlflow/experiments/{exp_id}` expose MLflow experiment information.

### Auto-generated CRUD routes

`api/main_app.py` mounts registry-generated CRUD routes from `api/app/models/model_registry.py` for:

- `datasetmodel`
- `learningmodel`
- `codemodel`
- `studymodel`

Each resource gets `create`, `get`, `list`, `update`, and `delete` endpoints.

## Architecture snapshot

- FastAPI + Jinja templates power the web UI and JSON endpoints.
- Async SQLAlchemy + PostgreSQL store registry metadata.
- Runtime artifacts are written under `api/runtime_artifacts/` for training, monitoring, deployment, plots, serving, and config snapshots.
- MLflow is used for experiment tracking and model-related artifact logging.
- Optuna utilities support study optimization.
- scikit-learn and PyTorch training helpers live in `api/app/utils/training.py`.
- Production monitoring, drift checks, and retraining helpers live in `api/app/utils/monitoring.py` and `api/app/utils/retrain_utils.py`.

## Services in `docker-compose.yaml`

| Service | Port | Notes |
| --- | --- | --- |
| `api` | `8000` | Main FastAPI app |
| `mlflow` | `5000` | MLflow tracking server |
| `airflow-webserver` | `8080` | Airflow UI/API |
| `postgres` | `5432` | Metadata database |
| `pgadmin` | `5050` | Optional DB admin UI |
| `redis` | internal | Broker/cache for Airflow Celery setup |

Notes:

- The compose file currently enables `airflow-init` and `airflow-webserver`.
- `airflow-scheduler`, `airflow-worker`, `airflow-triggerer`, and the Dash `dashboard` service are present but commented out.
- Feast is included as a sample repository under `features/`, not as an active compose service.

## Repository map

```text
api/
  main_app.py              # active FastAPI application
  main.py                  # older legacy API entrypoint
  templates/               # HTML pages
  static/js/               # frontend helpers
  app/models/              # current mounted registry + ORM models
  app/models/v2/           # stricter registry schemas and router refactor
  app/utils/               # training, MLflow, Optuna, deployment, monitoring
  runtime_artifacts/       # generated outputs

dashboard/                 # optional Dash market viewer
features/                  # Feast quickstart/example repo
data/                      # datasets used by the app
models/                    # model artifacts
mlflow-server/             # MLflow server container
airflow/                   # Airflow-related dependencies
painel-amanaje/            # MkDocs docs folder
docker-compose.yaml        # local multi-service stack
```

## Quick start

### Option 1: Full local stack with Docker Compose

1. Make sure Docker and Docker Compose are installed.
2. Make sure a root `.env` file exists with the database and MLflow settings you want to use.
3. Start the stack:

```powershell
docker compose up --build
```

4. Open the main services:

- FastAPI app: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`
- pgAdmin: `http://localhost:5050`

### Option 2: Run only the API locally

The API expects a reachable PostgreSQL instance. By default it looks for:

- `POSTGRES_HOST=postgres`
- `POSTGRES_PORT=5432`
- `POSTGRES_DB=airflow`
- `POSTGRES_USER=airflow`
- `POSTGRES_PASSWORD=airflow`

To run only the API:

```powershell
cd api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main_app:app --reload --host 0.0.0.0 --port 8000
```

## Metadata-driven create workflow

The editor-based creation flow described in `METADATA_VARIABLES_GUIDE.md` is part of the current project direction.

Define metadata at the top of a script:

```python
name = "sales_2024"
description = "Cleaned sales dataset"
dataset_type = "csv"
connection_string = ""
```

For models, use fields such as:

```python
name = "customer_churn_model"
description = "Baseline classifier"
model_type = "sklearn"
```

The flow is:

1. Write code in the editor.
2. Extract variables and metadata.
3. Review the generated metadata panel.
4. Create the dataset or model entry through the unified upload path.

## Testing

Focused tests currently exist for the newer registry layer:

```powershell
cd api
pytest tests/test_model_registry_v2.py
```

The `v2` registry code lives in `api/app/models/v2/` and is more strictly validated than the currently mounted registry routes.

## Important implementation notes

- `api/main_app.py` is the current source of truth for active routes and runtime behavior.
- `api/main.py` is an older implementation that still exists in the repo for reference.
- Some historical markdown notes mention earlier route layouts such as `/airflow/*` stubs; use `api/main_app.py` when in doubt.
- `base_test.html` still exists in the repo, but the current `/optimization` route renders `base_green.html`.
- The Dash app in `dashboard/app.py` is a lightweight stock price viewer and is not enabled by default in `docker-compose.yaml`.
- The Feast folder is a quickstart/example integration, not a fully wired part of the main API flow yet.
- The MkDocs site under `painel-amanaje/` is mostly scaffolded today, but the Kubernetes note is useful and up to date.

## Related project notes

- [ALIGNMENT_COMPLETION_REPORT.md](ALIGNMENT_COMPLETION_REPORT.md) - major template, endpoint, and validation alignment work
- [PROJECT_UPDATE_SUMMARY.md](PROJECT_UPDATE_SUMMARY.md) - endpoint consolidation and API fixes
- [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - reusable form utility work
- [CONSOLE_ERRORS_FIXED.md](CONSOLE_ERRORS_FIXED.md) - console and runtime fixes
- [METADATA_VARIABLES_GUIDE.md](METADATA_VARIABLES_GUIDE.md) - editor metadata workflow
- [features/README.md](features/README.md) - Feast example repo notes
- [painel-amanaje/docs/kubernetes-training-stack.md](painel-amanaje/docs/kubernetes-training-stack.md) - Kubernetes and Kubeflow target architecture

## Where to start in the codebase

- API app: `api/main_app.py`
- Model registry: `api/app/models/model_registry.py`
- Registry refactor: `api/app/models/v2/model_registry.py`
- Training helpers: `api/app/utils/training.py`
- MLflow helpers: `api/app/utils/mlflow_utils.py`
- Monitoring/retraining: `api/app/utils/monitoring.py`, `api/app/utils/retrain_utils.py`
- Frontend helpers: `api/static/js/form-utilities.js`, `api/static/js/editor-utilities.js`, `api/static/js/create-page.js`

## Roadmap themes already visible in the repo

- tighten the registry around the `v2` schemas and routes
- deepen Airflow orchestration beyond the current infrastructure layer
- make the Dash, Feast, and MkDocs pieces first-class instead of sidecars
- continue moving from placeholder or legacy templates toward a single consistent UI flow
- expand cloud-native deployment around the Kubernetes and Kubeflow design documented in `painel-amanaje/docs/kubernetes-training-stack.md`
