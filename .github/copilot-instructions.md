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

Integration & runtime notes
- Executor: CeleryExecutor with Redis broker and Postgres metadata DB. Workers run `celery worker` in compose services (`airflow-worker`).
- API server: `airflow-apiserver` exposes FastAPI endpoints on port 8080 (mapped in compose). Use it to trigger runs or inspect DAGs.
- Logs & volumes: `dags/`, `logs/`, `plugins/` and `config/` are mounted into containers; editing them locally updates containers (watch for uid/permission issues on non-Linux hosts).

Examples to reference in edits
- To add a DAG: create `dags/my_new_dag.py` with only import-safe code at top-level and tasks/operators defined inside functions or DAG context; don't run heavy training steps during import.
- To add an operator plugin: add `plugins/my_op.py` exporting a class deriving from `airflow.models.BaseOperator` (or using TaskFlow decorators) and restart `airflow-scheduler`/`airflow-worker`.

Testing, debugging & troubleshooting
- To inspect scheduler logs: check `logs/dag_processor/` or use `docker compose logs airflow-scheduler`.
- If DAGs aren't discovered, ensure `dags/` is mounted and `dags_are_paused_at_creation` or `load_examples` settings in `config/airflow.cfg` are not hiding the new DAGs.
- Common Windows Docker pitfall: file ownership/permission issues. Use `.env` to set `AIRFLOW_UID` per the docker-compose comments if you see root-owned files.

When editing code
- Keep changes descriptive, stable and runnable inside the compose environment. Prefer adding small unit tests where possible. If touching requirements, note the long install time; prefer creating a new container image for CI rather than `pip install` on container start.
- Preserve backwards compatibility of DAGs/operators: changing DAG ids, task ids, or serialization shape may affect running/serialized DAGs in the DB.

Files to inspect first for context
- `docker-compose.yaml`, `requirements.txt`, `config/airflow.cfg`, `dags/`, `plugins/`, `api/`, `dashboard/`, `README.md`.

If anything is unclear
- Ask the human: local dev workflow (do they run via compose or on-host?), expected CI/CD, secrets management (where `.env` values come from), and whether the `api/` and `dashboard/` folders are active services.

End of file.
