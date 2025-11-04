## Quick orientation for AI coding agents

This repository is an Airflow-based ML/data platform with a small API/dashboard. Focus on being precise, conservative, and referencing files shown in the repo.

Key components
- Airflow orchestrator: docker-compose-based dev environment in `docker-compose.yaml`. DAGs live in `dags/` and are mounted into containers via the compose file.
- Configuration: container/runtime Airflow settings are in `config/airflow.cfg` (used by the docker-compose stack).
- Python dependencies: pinned in `requirements.txt` (Airflow 3.x, FastAPI, Torch, TFT libs).
- Plugins: local Airflow plugins live in `plugins/` and are mounted into the Airflow image by compose.
- API / dashboard: code folders `api/` and `dashboard/` (may be empty in the workspace snapshot). The web UI is served by the Airflow api-server (see docker-compose).

What to assume
- Development runs inside Docker Compose by default. Commands we suggest should work from project root and can be run in PowerShell on Windows. If you need to run code locally outside containers, note the heavy dependency set in `requirements.txt`.
- Secrets and credentials are provided via environment variables or an `.env` (not checked in). Do not attempt to hardcode credentials; use placeholders and reference `.env` or docker-compose variables.

Common tasks & exact commands (project root)
- Start dev Airflow stack (read `docker-compose.yaml` before changing):
  - docker compose up --build
  - For Windows PowerShell, prepend any environment exports by creating a `.env` file and then run `docker compose up --build`.
- Validate docker-compose: `docker compose config` (useful to check env interpolation)
- Install Python deps locally (if needed): `python -m pip install -r requirements.txt` — heavy; prefer using containers for reproducibility.

Project-specific patterns to follow
- DAGs: keep top-level DAG definitions in `dags/`. Avoid heavy computation at import time. Use sensors/operators that are compatible with CeleryExecutor (this repo uses Celery + Redis/Postgres per `docker-compose.yaml`).
- Plugins: place reusable hooks/operators in `plugins/` to be discovered by Airflow. The compose mounts this folder to `/opt/airflow/plugins` in containers; follow existing plugin layout (module with Python entry-points).
- Configuration edits: prefer environment-variable overrides (docker-compose `.env`) rather than editing `config/airflow.cfg` directly. `airflow-init` in compose will initialize DB and copy config into containers.

```markdown
## Quick orientation for AI coding agents

This repo is an Airflow-based ML/data platform with a small API and dashboard. Development is intended to run inside the Docker Compose stack in the repository root.

Key components (what to open first)
- `docker-compose.yaml` — single dev stack (Airflow scheduler, worker, redis, postgres, api, etc.).
- `dags/` — project DAGs (mount into Airflow containers). Add new DAGs here and keep top-level import-safe.
- `plugins/` — local Airflow plugins (operators/hooks) mounted to `/opt/airflow/plugins`.
- `config/airflow.cfg` — runtime Airflow settings (prefer env overrides via `.env`).
- `api/` and `dashboard/` — small services; `api/main.py` and `dashboard/app.py` are the likely entrypoints.
- `requirements.txt` and per-service requirement files: `airflow/requirements.txt`, `api/requirements.txt`, `dashboard/requirements.txt`, `mlflow-server/requirements.txt`.

Quick, copy-paste dev commands (PowerShell)
- Start full dev environment (recommended):
  - docker compose up --build
- Validate compose interpolation:
  - docker compose config
- Run API locally (optional):
  - uvicorn api.main:app --reload --port 8000
  (useful for quick endpoint checks without the full stack)
- Install Python deps locally (heavy):
  - python -m pip install -r requirements.txt

Project conventions and actionable rules (do these)
- DAGs: keep heavy work out of module import. Define operators/tasks inside the DAG or factory functions. Example: create `dags/my_new_dag.py` and define tasks inside the DAG context.
- Plugins: put reusable operators and hooks in `plugins/` (module with Python entrypoints). Restart scheduler/workers after changes.
- Config edits: prefer environment-variable overrides (place values in `.env` used by docker compose) rather than editing `config/airflow.cfg` directly. The compose `airflow-init` step initializes DB/config.

Integration points to be aware of
- Executor: CeleryExecutor with Redis broker and Postgres metadata DB (see `docker-compose.yaml`). Workers run as `airflow-worker` and `celery worker` processes.
- API server: `airflow-apiserver` exposes FastAPI endpoints (mapped to port 8080 in compose). Use it to trigger DAGs or inspect state.
- MLflow: `mlflow-server/` and `mlruns/` are present; MLflow artifacts may be produced by DAGs or notebooks. `mlflow-server` has its own Dockerfile and requirements.

Repo-specific examples and files to inspect
- Example feature repo: `features/feature_repo/` (contains `feature_store.yaml` and `test_workflow.py`) — use this as a template for feature store workflows.
- Tests: `features/feature_repo/test_workflow.py` is an example test; run `pytest` targeting that file to validate changes locally.
- Logs: `logs/` and `logs/dag_processor/` contain scheduler/dag parsing logs from local runs.

Common pitfalls seen here
- Windows file ownership: mounted volumes can create root-owned files inside containers. Set `AIRFLOW_UID` in `.env` (mentioned in compose comments) to avoid permission friction.
- Long dependency installs: `requirements.txt` installs are heavy (Airflow/Torch). Prefer running inside the compose-built containers or build a new image for CI.

If you need clarification
- Ask which service you plan to change (DAG, plugin, API, dashboard, mlflow) and whether you run via `docker compose` or on-host. Also confirm `.env` value sources / secrets handling.

``` 
