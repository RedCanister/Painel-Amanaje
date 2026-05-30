<img width="1480" height="412" alt="painel-amanaje-banner" src="https://github.com/user-attachments/assets/a972310c-c5ff-4589-8b53-6d2d48419840" />

# Painel Amanaje

Painel Amanaje is a local-first MLOps workspace for building, training, tracking, and operating machine learning workflows from one place.

It combines a FastAPI control plane, a multi-page HTML interface, registry-backed dataset and model management, MLflow experiment tracking, Optuna studies, ONNX export helpers, and production monitoring flows.

## Why this repo is useful

- Upload datasets and model artifacts through a unified workflow
- Create metadata-rich assets from code in the built-in editor
- Train scikit-learn and PyTorch models from the UI or API
- Track experiments and artifacts with MLflow
- Run Optuna optimization studies
- Prepare models for ONNX validation and deployment workflows
- Monitor production state, simulations, drift signals, and retraining actions

## What is live today

### Main app

- Active entrypoint: `api/main_app.py`
- FastAPI app version: `0.7.0`
- Core stack: FastAPI, PostgreSQL, Redis, Airflow, MLflow, SQLAlchemy, Jinja templates
- Optional sidecars in the repo: Kubernetes 

### UI pages

| Route | What it does |
| --- | --- |
| `/` | Home navigation |
| `/upload` | Upload datasets and model files |
| `/create` | Metadata-first creation flow |
| `/feature` | Feature analysis and extraction workspace |
| `/training` | Training configuration and execution |
| `/production` | Deployment, monitoring, simulation, retraining |
| `/registry` | CRUD and asset browsing |
| `/assistant` | Code editor and code generation |
| `/editor` | Code editor and code generation |
| `/settings` | Code editor and code generation |

### Registry resources

The current app mounts generated CRUD routes for:

- `datasetmodel`
- `learningmodel`
- `assistantmodel`
- `inferencemodel`
- `studymodel`
- `codemodel`

Each resource gets `create`, `get`, `list`, `update`, and `delete` endpoints.

## Quick start

### Option 1: Run the full local stack

This is the fastest way to see the project as intended.

```powershell
docker compose up --build
```

Open:

- App: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`
- pgAdmin: `http://localhost:5050:80`

### Option 2: Run only the API

The API expects a reachable PostgreSQL instance. Default settings in the repo point to:

- `POSTGRES_HOST=postgres`
- `POSTGRES_PORT=5432`
- `POSTGRES_DB=airflow`
- `POSTGRES_USER=airflow`
- `POSTGRES_PASSWORD=airflow`

Run locally:

```powershell
cd api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main_app:app --reload --host 0.0.0.0 --port 8000
```

### Node.js prerequisite for frontend tooling

Node.js LTS is now expected on developer machines for frontend-oriented tooling and any future Node-based checks.

- Recommended install target: system-wide Windows Node.js LTS
- Validate with: `node --version` and `npm --version`
- If Node was just installed and the commands do not resolve in your current terminal, close and reopen the shell so the updated `PATH` is picked up
- The repo does not currently require a top-level `package.json`, so this is environment enablement for tooling rather than an app runtime dependency

<img width="1517" height="1063" alt="painel-amanaje-registry" src="https://github.com/user-attachments/assets/5562699c-0e2b-4b1a-8c67-de270ee889d7" />

## Local stack

The main `docker-compose.yaml` currently brings up:

| Service | Port | Notes |
| --- | --- | --- |
| `api` | `8000` | FastAPI app |
| `mlflow` | `5000` | Tracking server |
| `airflow-webserver` | `8080` | Airflow UI/API |
| `postgres` | `5432` | Metadata database |
| `pgadmin` | `5050` | Database admin UI |
| `redis` | internal | Broker/cache |

Notes:

- `airflow-init` is part of the startup flow
- `airflow-scheduler`, `airflow-worker`, `airflow-triggerer`, and `dashboard` are present but commented out
- `features/` is an example integration area, not an active compose service

## Architecture at a glance

```text
api/
  main_app.py              active FastAPI app
  templates/               Jinja UI pages
  static/                  frontend assets
  app/models/              registry and ORM models
  app/models/v2/           stricter registry refactor
  app/utils/               training, MLflow, Optuna, monitoring, deployment
  runtime_artifacts/       generated outputs

dashboard/                 optional Dash viewer
features/                  Feast example workspace
data/                      dataset storage
models/                    model storage
mlflow-server/             MLflow container assets
airflow/                   Airflow dependencies
docker-compose.yaml        full local stack
```


<img width="1478" height="928" alt="painel-amanaje-creator" src="https://github.com/user-attachments/assets/ba8c4a98-9477-4755-9244-5234188c987c" />

## Main workflows

### 1. Upload

Use `/upload` to register datasets and model artifacts through the same backend flow. Dataset uploads also generate profiling metadata and feature summaries.

### 2. Create from code

Use `/create` and `/editor` together to turn Python snippets into structured assets. The app can extract variables, metadata, and starter object definitions from code.

Example dataset metadata:

```python
name = "sales_2024"
description = "Cleaned sales dataset"
dataset_type = "csv"
connection_string = ""
```

Example model metadata:

```python
name = "customer_churn_model"
description = "Baseline classifier"
model_type = "sklearn"
```

### 3. Train and optimize

Use `/training` to run model training and the study tab for study-driven tuning. Current dependencies include:

- `fastapi==0.135.1`
- `uvicorn==0.41.0`
- `mlflow~=3.10.0`
- `optuna==4.7.0`
- `scikit-learn==1.7.2`
- `torch==2.9.0`

<img width="1447" height="1432" alt="painel-amanaje-training" src="https://github.com/user-attachments/assets/87f0dac8-52a1-447f-8f94-ad3fa82bb547" />


### 4. Operate in production

Use `/production` to:

- start a model-dataset runtime pair
- inspect health and monitoring signals
- run simulation scenarios
- trigger retraining flows
- review production history and activity logs

<img width="1493" height="1140" alt="painel-amanaje-production" src="https://github.com/user-attachments/assets/7b5e46b9-ab18-4218-8049-d35c7df91501" />


## Testing

The repo includes focused API and runtime tests under `api/tests/`.

Run the main test suite from the API folder:

```powershell
cd api
pytest
```

If you want the lighter test dependency set first:

```powershell
pip install -r requirements-test.txt
```

## Manual signoff

The repository also includes a human-run signoff package:

- Checklist: `docs/manual-signoff-checklist.md`
- Report template: `docs/manual-signoff-report-template.md`
- Fixture builder: `scripts/build_manual_test_fixtures.py`
- Prep helper: `scripts/prepare_manual_signoff.ps1`

## Where to start in the code

- App entrypoint: `api/main_app.py`
- Shared app setup: `api/app/utils/main_utils.py`
- Registry routes: `api/app/models/model_registry.py`
- Registry refactor: `api/app/models/v2/model_registry.py`
- Database utilities: `api/app/database/db_utils.py`
- Training pipeline: `api/app/utils/training.py`
- MLflow helpers: `api/app/utils/mlflow_utils.py`
- Monitoring and retraining: `api/app/utils/monitoring.py`, `api/app/utils/retrain_utils.py`

## Project notes

- [ALIGNMENT_COMPLETION_REPORT.md](reports/ALIGNMENT_COMPLETION_REPORT.md)
- [PROJECT_UPDATE_SUMMARY.md](reports/PROJECT_UPDATE_SUMMARY.md)
- [CHANGES_SUMMARY.md](reports/CHANGES_SUMMARY.md)


## Current direction

(2025-01)
The repository has clearly moved beyond its original NVIDIA forecasting prototype into a broader ML operations workspace. The next visible themes in the codebase are stronger registry validation, deeper orchestration, cleaner frontend consistency, and more production-ready deployment paths.

(2026-05)
Considering a slow start to the development of Painel Amanajé, due to a long planning phase, the project is now ready to be shared, but it still needs work in the areas of front-end efficiency and paralellism, but also on the implementation of the Assistant SLM, that still needs a proper model to work on instead of standard placeholder answers to everything. 

## Future Plans

Next up on the development is:
1. Finish building up the Assistant SLM to it can attend any possible user.
2. Introduce parallelism, GPU use and multithreading
3. Improving the logging system that can capture all system actions
4. Implement a JSON to DataFrame Plot generator that can bring life to all the data in the project.
5. Integratea and upgrade the postgresql instance to a live instance that can capture real-time data.
6. Introduce the use of Airflow DAGs that can configure entire automatic pipelines.
7. Implement ONNX in the project to facilitate the use of ML models.
8. Build up the Simulation and Prediction system so it can attend to needs of high-level users.
9. Refine the UI and UX elements with the help of testers.
10. Make a public server of painel-amanaje that can be shared with non-developers, with hosting and multi-worker options.
11. Add support to more file types, ML libraries, frameworks, algorithms, workflows, etc...
