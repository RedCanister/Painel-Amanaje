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

---

# Whole-Repo Stability And Distribution Revision

Date: 2026-05-14

## Procedure

1. Treated visible prompt history, this report, current dirty diff, and repo-local evidence as the audit source.
   - External chat history remains unavailable and is an explicit verification limit.
   - Reviewed current modified/new source surfaces first, then adjacent route, model registry, template, static asset, worker, dashboard, compose, and docs contracts.
2. Hardened the newest implementation surfaces.
   - Panel workflow: added safe layout/version coercion, object-type alias normalization, route-contract coverage, Playwright CRUD coverage, and malformed layout live-smoke verification.
   - Create workspace: converted the saved scripts panel into a real collapsible themed `<details>` section and removed stale TODO comments.
   - Dashboard extensions: added direct unit coverage for disabled state, missing directory reporting, valid manifest loading, allowlist behavior, disabled manifest override, and invalid manifest rejection.
   - Test infrastructure: changed fast/perf/full-stack runners to use unique repo-local basetemp directories to avoid Windows locked-directory setup errors.
3. Rebuilt and recreated distribution services.
   - `docker compose build api api-worker-cpu dashboard assistant-server`: passed.
   - `docker compose up -d --force-recreate api api-worker-cpu dashboard assistant-server`: passed.
   - `docker compose --profile gpu up -d api-worker-gpu`: passed.
4. Verified Redis/RQ in all requested contexts.
   - Local `.venv`: `redis 6.4.0`, `rq 2.8.0`.
   - API container: `redis 6.4.0`, `rq 2.8.0`.
   - CPU worker container: `redis 6.4.0`, `rq 2.8.0`.
   - GPU worker container: `redis 6.4.0`, `rq 2.8.0`.
5. Cross-checked runtime worker behavior and docs.
   - `/runtime/workers`: HTTP 200, backend `rq`, queues `amanaje:default` and `amanaje:gpu`.
   - Active workers after GPU profile start: `2`.
   - CPU worker: `amanaje-compose-cpu-1`, queue `amanaje:default`, idle, `successful_job_count: 4`.
   - GPU worker: `amanaje-compose-gpu-1`, queue `amanaje:gpu`, idle.
   - `docs/worker-hosts.md` remains aligned with queue names, host contract variables, scaling guidance, and worker foreground behavior.

## Test Results

- Targeted preflight:
  - Core `py_compile`: passed.
  - Panel route/helper tests: `3 passed`.
  - Dashboard extension tests: `5 passed`.
  - Model/ORM/template tests: `7 passed`.
- Automated suites:
  - `scripts/run_fast_tests.ps1`: `168 passed, 54 deselected`.
  - `scripts/run_perf_tests.ps1`: `3 passed, 219 deselected`.
  - `scripts/run_full_stack_tests.ps1`: `78 passed, 144 deselected`.
  - Explicit Playwright E2E: `51 passed`.
- Final source compile:
  - Broad `compileall api dashboard scripts` timed out while walking generated artifacts.
  - Source-focused `compileall api dashboard scripts -x "runtime_artifacts|__pycache__|temp"` passed.
- `git diff --check`: passed with line-ending warnings only.
- Known non-failing warnings:
  - `.pytest_cache` remains unavailable to pytest with Windows `Access denied`.
  - Docker Compose warns `POSTGRES_PASSWORD` is unset and defaults to blank.
  - `git status` still warns about `airflow/logs/dag_processor/latest`.

## Live Smokes

- `POST /execute/jobs`
  - Run: `editor_execution_ae7d3c363e93`
  - Queue: `amanaje:default`
  - Backend: `rq`
  - Final status: `completed`
  - Output: `amanaje live smoke`
- `POST /training/2`
  - Run: `training_fe360af3eb9c`
  - Queue: `amanaje:default`
  - Backend: `rq`
  - Final status: `completed`
  - MLflow run: `c19b8b6e16004cfab92bac07974d7fab`
  - Resolved device: `cpu`
- `POST /studies/3/optimize`
  - Run: `study_b341b4e57eb6`
  - Queue: `amanaje:default`
  - Backend: `rq`
  - Final status: `completed`
  - Trials: `1`
- Panel API CRUD:
  - `POST /panel/dashboards`: HTTP 201, created temporary panel id `36`.
  - `GET /panel/dashboards/36`: HTTP 200.
  - Malformed layout values were coerced to `version: 1`, `columns: 12`.
  - `PUT /panel/dashboards/36`: HTTP 200.
  - `DELETE /panel/dashboards/36`: HTTP 200.
- API route smokes:
  - `/visualization`, `/plot`, `/plots/artifacts`, `/settings/logs`, `/runtime/accelerators`, `/runtime/workers`, `/production/status`, `/assistant/management/overview`, `/examples/catalog`: HTTP 200.
- Dashboard service smokes:
  - `/`, `/embed`, `/extensions`, `/extensions.json`: HTTP 200.
  - `/extensions.json`: `enabled: false`, `errors: []`.
- Assistant-server smokes:
  - `/health`: HTTP 200, `status: unavailable`, no active model loaded.
  - `/admin/models/status` without token: HTTP 401 as expected.
  - `/admin/models/status` with compose token: HTTP 200, `status: unavailable`, no active model loaded.
- GPU profile:
  - `api-worker-gpu`: up and healthy on `amanaje:gpu`.
  - `torch 2.9.0+cu128` is present in the GPU worker.
  - `torch.cuda.is_available(): False`, `device_count: 0`.

## Domain Coverage

- Panel: page route, API context, CRUD, model/ORM registration, DB table bootstrap, widget save/reload/delete, malformed layout fallback, and responsive E2E passed.
- Upload: route, support matrix, styled panels, viewport E2E, and backend support tests passed.
- Create: route, assistant/script controls, saved-script collapsible panel, support matrix, styled panels, viewport E2E, and TODO compliance passed.
- Feature: route, feature controls, assistant toggle, styled panels, viewport E2E, and feature materialization regressions passed.
- Training: route, queue submission, structured queue errors, live CPU training, run ledger, artifacts, MLflow logging, and performance tests passed.
- Optimization: study route and live one-trial Optuna run passed.
- ONNX: route, context controls, viewport E2E, and existing artifact-readiness coverage passed.
- Production: route, status API, simulation context, controls, viewport E2E, and full-stack regressions passed.
- Visualization: `/visualization`, `/plot`, local fallback, artifact API, dashboard root/embed, and extension status endpoint passed.
- Registry: route contract, model registry mappings, dependency checks, and registry page E2E passed.
- Assistant: management overview, provider/runtime endpoints, assistant-server health, and token-gated admin status behavior passed; no model is currently loaded.
- Settings: config, logs, debug capture, safe environment settings, runtime accelerators, and worker status passed.
- Runtime/Workers: Redis/RQ dependencies, CPU worker, GPU queue worker, queue summaries, and host contract docs passed.
- Docker/Distribution: image rebuilds, service recreation, CPU stack, assistant-server, dashboard, and GPU worker profile startup passed.
- Docs/Reports: this report updated; external chat history remains unavailable.

## Generated Artifacts Preserved

- Run ledgers:
  - `api/runtime_artifacts/runs/editor_execution_ae7d3c363e93.json`
  - `api/runtime_artifacts/runs/training_fe360af3eb9c.json`
  - `api/runtime_artifacts/runs/study_b341b4e57eb6.json`
- Training artifacts:
  - `api/runtime_artifacts/training/training_fe360af3eb9c/training_summary.json`
  - `api/runtime_artifacts/training/training_fe360af3eb9c/model_spec.json`
  - `api/runtime_artifacts/training/training_fe360af3eb9c/model_pytorch.pt`
- Visualization artifacts:
  - `api/runtime_artifacts/plots/training_fe360af3eb9c/loss_curve.png`
  - `api/runtime_artifacts/plots/training_fe360af3eb9c/metric_comparison.png`
- Study artifacts overwritten/updated by the live study smoke:
  - `api/runtime_artifacts/optuna/study_3/study_summary.json`
  - `api/runtime_artifacts/optuna/study_3/best_params.json`
  - `api/runtime_artifacts/optuna/study_3/trials_summary.json`
  - `api/runtime_artifacts/optuna/study_3/optimization_history.png`
- E2E visual check screenshots already present and preserved under:
  - `temp/atlas-restore-check/`
- Additional full-stack/E2E artifacts preserved from this pass:
  - `api/runtime_artifacts/runs/assistant_operation_323550cf6aa6.json`
  - `api/runtime_artifacts/assistant_evals/assistant_golden_eval_20260514213540.json`
  - `api/runtime_artifacts/mlflow_models/pytorch_6ada80093c194563aa96a288990c1540/`
  - `api/runtime_artifacts/monitoring/training_fe360af3eb9c/`
  - `api/runtime_artifacts/serving/training_fe360af3eb9c_service.py`
  - `mlruns/3/c19b8b6e16004cfab92bac07974d7fab/`
  - `mlruns/8/0c780fb0aa464145b371237bbb52451b/`
  - `mlruns/8/10a03ce7833b443591a587ae6b51a93c/`
  - `mlruns/9/5685f47c60e242b98463fc2322666271/`
  - `mlflow-server/mlflow.db`

## Remaining Gaps

- External chat history cannot be inspected from this environment; compliance is limited to visible prompts, local reports, and repo evidence.
- Assistant-server is running and healthy, but no assistant model is loaded. Runtime model-serving behavior remains unavailable until a valid bundle is loaded under `/app/runtime_artifacts/assistant_models/bundles`.
- GPU worker queue wiring is healthy, but this host exposes no CUDA device to PyTorch. CUDA-accelerated training remains an environment/hardware blocker.
- `.pytest_cache` cannot be written because Windows denies access to the cache directory. Unique basetemp directories keep suites passing, but the cache warning remains.
- Compose still warns that `POSTGRES_PASSWORD` is unset. This is acceptable for the current local stack but should be resolved before packaged distribution.
- The worktree remains intentionally dirty with preserved source changes, generated runtime artifacts, and prior screenshots.
