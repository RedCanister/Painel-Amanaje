# Domain-Aligned Revision Audit

Date: 2026-05-11

## Procedure

1. Reproduced the persistent Redis/RQ failure inside the running API container.
   - Local `.venv` import passed with `redis 6.4.0` and `rq 2.8.0`.
   - Running API container initially failed with `ModuleNotFoundError: No module named 'redis'`.

2. Rebuilt and recreated the API execution contexts.
   - Rebuilt `api` and `api-worker-cpu` images.
   - Recreated both containers.
   - Verified imports inside both containers: `redis 6.4.0`, `rq 2.8.0`.
   - Verified `redis` service is healthy and `api-worker-cpu` is listening on `amanaje:default`.

3. Hardened project test runners.
   - Fixed `scripts/run_full_stack_tests.ps1` parameter ordering.
   - Added `PYTHONPATH` setup, repo-local pytest temp directories, and explicit native exit-code propagation to fast, perf, and full-stack runners.
   - Added `temp/pytest-*/` to `.gitignore`.

4. Completed automated verification.
   - `scripts/run_fast_tests.ps1`: `153 passed, 30 deselected`.
   - `scripts/run_perf_tests.ps1`: `3 passed, 180 deselected`.
   - `scripts/run_full_stack_tests.ps1`: `54 passed, 129 deselected`.
   - Explicit Playwright E2E: `27 passed`.
   - `git diff --check` for touched files: passed, with line-ending warnings only.

5. Performed live API queue smoke tests.
   - `POST /execute/jobs` accepted an RQ job and worker completed it.
     - Run: `editor_execution_eb52890ba558`
     - Final status: `completed`
     - Output: `queue-ok`
   - `POST /training/2` accepted an RQ job and worker completed it.
     - Run: `training_9d25d5505bc3`
     - Final status: `completed`
   - `POST /studies/3/optimize` accepted an RQ job and worker completed it.
     - Run: `study_bae6212ccf15`
     - Final status: `completed`

6. Verified Visualization domain live routes.
   - `GET /visualization`: HTTP 200, renders Visualization Gallery.
   - `GET /plot`: HTTP 200, renders Visualization Gallery.
   - `GET /plots/artifacts`: status `ok`, count `40`.

## Domain Coverage

- Upload: Playwright route/layout/support-matrix coverage passed.
- Create: Playwright route/layout/support-matrix coverage passed.
- Feature: Playwright route/layout/assistant-toggle coverage passed; live `/features` fetch no longer causes E2E teardown failures.
- Training: queue dependency, structured queue error tests, full-stack route tests, and live RQ training run passed.
- Production: Playwright route/layout and full-stack regression coverage passed.
- Visualization: `/visualization`, `/plot`, local Plotly fallback, and artifact API coverage passed.
- Registry: route contract and full-stack regression coverage passed.
- Assistant: management and route coverage passed; runtime status endpoint responds with structured availability state.
- Settings: logs/debug UI coverage passed.

## Remaining Gaps

- `assistant-server` service is not currently running in the compose stack. `GET /assistant/runtime/status` responds successfully, but its nested runtime payload reports `status: unavailable` with `Name or service not known` for `http://assistant-server:8080`.
- Starting `assistant-server` was attempted, but the escalation/usage gate rejected the Docker Compose action. This is the only remaining environment-side item from the requested coverage.

---

# Full Live CPU-Stack Verification

Date: 2026-05-13

## Procedure

1. Rechecked the current dirty worktree and preserved all existing runtime artifacts.
2. Verified Redis/RQ availability:
   - Local `.venv`: `redis 6.4.0`, `rq 2.8.0`.
   - Recreated `api` container: `redis 6.4.0`, `rq 2.8.0`.
   - Recreated `api-worker-cpu` container: `redis 6.4.0`, `rq 2.8.0`.
3. Rebuilt and recreated the requested CPU-stack execution services:
   - `docker compose build api api-worker-cpu dashboard`
   - `docker compose up -d --no-deps --force-recreate api api-worker-cpu dashboard`
4. Cross-checked `docs/worker-hosts.md` against `/runtime/workers`:
   - Backend: `rq`
   - Redis URL: `redis://redis:6379/0`
   - Configured queues: `amanaje:default`, `amanaje:gpu`
   - Active worker: `amanaje-compose-cpu-1`
   - Worker queue: `amanaje:default`
   - Worker state: `idle`
   - Active workers: `1`
   - Queued jobs after preflight: `0`
5. Fixed one stale Assistant UI coverage assertion:
   - The Assistant page now queues draft/eval/reference operations through `/assistant/jobs/{operation}`.
   - `api/tests/test_assistant_workflow.py` now checks the queued operation contract instead of requiring direct `/assistant/draft`, `/assistant/evals/run`, and `/assistant/references/sync` strings in the page script.

## Test Results

- Fast suite: `161 passed, 30 deselected, 3 warnings`.
- Perf suite: `3 passed, 188 deselected, 3 warnings`.
- Full-stack suite: `54 passed, 137 deselected, 3 warnings`.
- Explicit Playwright E2E: `27 passed`.
- `git diff --check`: passed; line-ending warnings only.
- Final service state:
  - `api`: up on `0.0.0.0:8000`
  - `api-worker-cpu`: up and healthy
  - `dashboard`: up on `0.0.0.0:8050`
  - `mlflow`: up on `0.0.0.0:5000`
  - `redis`: healthy
  - `postgres`: healthy

## Live Queue Smokes

- `POST /execute/jobs`
  - Run: `editor_execution_36e4373e2f00`
  - Queue: `amanaje:default`
  - Backend: `rq`
  - Final status: `completed`
  - Output: `full-live-queue-ok`
- `POST /training/2`
  - Run: `training_6c82dd774e87`
  - Queue: `amanaje:default`
  - Backend: `rq`
  - Final status: `completed`
- `POST /studies/3/optimize`
  - Run: `study_f35c6f6f7026`
  - Queue: `amanaje:default`
  - Backend: `rq`
  - Final status: `completed`

## Live Route Smokes

- `GET /visualization`: HTTP 200.
- `GET /plot`: HTTP 200.
- `GET /plots/artifacts`: HTTP 200.
- `GET /settings/logs?include_docker=false&limit=20`: HTTP 200.
- `GET /runtime/accelerators`: HTTP 200.
- `GET /production/status`: HTTP 200.
- `GET /assistant/management/overview`: HTTP 200.
- `GET /runtime/workers`: HTTP 200.

## Generated Artifacts Preserved

- Run ledgers:
  - `api/runtime_artifacts/runs/editor_execution_36e4373e2f00.json`
  - `api/runtime_artifacts/runs/training_6c82dd774e87.json`
  - `api/runtime_artifacts/runs/study_f35c6f6f7026.json`
- Training artifacts:
  - `api/runtime_artifacts/training/training_6c82dd774e87/training_summary.json`
  - `api/runtime_artifacts/training/training_6c82dd774e87/model_spec.json`
  - `api/runtime_artifacts/training/training_6c82dd774e87/model_pytorch.pt`
  - `api/runtime_artifacts/serving/training_6c82dd774e87_service.py`
- Visualization and monitoring artifacts:
  - `api/runtime_artifacts/plots/training_6c82dd774e87/metric_comparison.png`
  - `api/runtime_artifacts/plots/training_6c82dd774e87/loss_curve.png`
  - `api/runtime_artifacts/monitoring/training_6c82dd774e87/monitoring_summary.json`
- MLflow run:
  - `runs:/efa09d9bd95a425791195c5f1d1bae1b/model`

## Remaining Gaps

- Assistant-server and GPU worker were intentionally out of scope for this CPU-stack run.
- The worktree remains heavily dirty from prior project work and preserved runtime artifacts. The final `git status --short` also still reports the pre-existing `airflow/logs/dag_processor/latest` directory warning.
