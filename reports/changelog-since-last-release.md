# Changelog Since Last Release

Generated: 2026-05-30

## Release Baseline

- Last release tag: `Release`
- Tagged commit: `356c43c` on 2026-05-05
- Tagged commit message: `Making the Readme shine!`
- Current commit reviewed: `c13113a` on `local_install`
- Commit range: `Release..HEAD`
- Committed delta: 764 files changed, 361,208 insertions, 3,882 deletions
- Current working tree delta after `HEAD`: 16 modified tracked files, plus new runtime run/deployment artifacts

Most of the raw line count is generated runtime state, MLflow runs, screenshots, datasets, and preserved verification artifacts. The source-facing change is a broad local-first beta expansion of Painel Amanaje from a basic MLOps app into a multi-surface ML engineer workspace.

## Commit Inventory

| Commit | Date | Message |
| --- | --- | --- |
| `6fd94d7` | 2026-05-12 | Rearrange images in README.md |
| `7f34352` | 2026-05-14 | Latest version to stabilize |
| `7c5a86e` | 2026-05-15 | Stable 0.7.2 |
| `ae4f2e0` | 2026-05-20 | Halfway improvements |
| `f3c803b` | 2026-05-26 | Update 0.7.3 |
| `c13113a` | 2026-05-30 | Merge branch `local_install` into `local_install` |

## Highlights

### Added

- Added new first-class workspace routes and templates for `/panel`, `/store`, `/onnx`, `/operations`, `/visualization`, `/plot`, and `/settings`.
- Added a dashboard-style Panel workspace with saved dashboards, visual tiles, Plotly/Dash workspace embedding, artifact palettes, dashboard CRUD, and route/context tests.
- Added Store provider management, catalog browsing, preview, materialization, live validation, refresh flows, and provider persistence utilities.
- Added Operations as a run/queue evidence surface with queue summaries, run listings, terminal output, analysis views, retry actions, and Redis/RQ worker visibility.
- Added runtime worker support with Redis/RQ operation queues, CPU/GPU queue routing, worker health checks, run ledgers, and Docker Compose service wiring.
- Added accelerator detection through `/runtime/accelerators`, including torch/CUDA status and CPU fallback reporting.
- Added a plot registry and visualization gallery with plot artifact discovery, legacy plot migration support, artifact file serving, and local fallback behavior.
- Added ONNX and Netron readiness routes for prepare, validate, viewer status, viewer open, and viewer stop workflows.
- Added an assistant model management surface with reference packs, dataset rebuilds, evals, model import, bundle status, runtime status, load/unload, activation, testing, curation, tokenization, and dataset attachment endpoints.
- Added a PyTorch/Hugging Face assistant-server sidecar with OpenAI-compatible chat-completion behavior, admin model status, load/unload controls, and example runtime config.
- Added an example catalog generator and `/examples/catalog` API to expose sample payloads and route proof for registry/API workflows.
- Added dashboard extension loading with manifest validation, disabled-state handling, a model quality overview example extension, and `/extensions`/`/extensions.json` coverage.
- Added docs and reports for worker hosts, visualization extensions, ML engineer hand-design direction, cohesion benchmarking, release completeness, and project visualization.

### Changed

- Reworked the main FastAPI app into a much broader control plane spanning Upload, Create, Feature, Store, Training, ONNX, Optimization, Editor, Production, Operations, Visualization, Registry, Assistant, Settings, and Panel.
- Expanded the navigation and shared UI styling so the app presents as a local-first ML workspace rather than isolated template demos.
- Reworked Upload and Create flows around richer metadata, examples, support matrices, and reusable frontend utilities.
- Expanded Feature workflows with preview/materialization paths and stronger dataset/feature summaries.
- Extended Training to support queued execution, run ledgers, MLflow/artifact logging, sklearn and PyTorch flows, Optuna study jobs, metrics, plots, and serving artifacts.
- Hardened sklearn estimator handling, including multi-output wrappers, task-family inference, metric compatibility, parameter filtering, and estimator class instantiation.
- Expanded Production into status, start/stop, monitoring, retraining, history, simulation context, and structured prediction review flows.
- Reframed production simulation toward `Prediction Review` terminology with scenario comparison, changed features, prediction paths, plot specifications, and cleaner default controls.
- Made Assistant model selection stricter by surfacing provider configuration, bundle readiness, runtime load status, fallback usage, and draft validation state.
- Reduced Panel context payload weight by summarizing large lists, metrics, parameters, plot artifacts, and run entries.
- Updated frontend scripts for Assistant, Panel, Operations, Settings, editor utilities, form utilities, JSON explorers, and shared UI helpers.
- Updated Docker Compose for API worker services, assistant-server, dashboard, Redis/RQ, and related environment wiring.
- Updated README content and imagery for the local-first MLOps workspace posture.

### Fixed

- Fixed Redis/RQ dependency gaps across local/API/worker contexts and added verification reports for queue health.
- Fixed queue execution flows for editor jobs, training jobs, study optimization, and assistant operations.
- Fixed production simulation failures so runtime/model dependency errors return structured user-facing `400` responses instead of generic server errors.
- Fixed assistant selected-model behavior so runtime failure blocks silent fallback for selected AssistantModel runs.
- Fixed stale Assistant UI coverage to expect queued assistant operations.
- Fixed Panel save, reload, delete, malformed-layout coercion, widget mode rendering, and dashboard selector behavior.
- Fixed route and E2E coverage for Store, Operations, ONNX/Netron, Panel, Visualization, Assistant, Settings, and dashboard extension surfaces.
- Fixed test runner behavior on Windows by using repo-local pytest temp directories and preserving native exit codes.
- Fixed `git diff --check` issues in touched files, with remaining notices limited to line-ending warnings in the local worktree.

### Testing And Verification Evidence

Latest recorded verification in `reports/domain-aligned-revision-audit.md` shows:

- `scripts/run_fast_tests.ps1`: 218 passed, 69 deselected
- `scripts/run_perf_tests.ps1`: 3 passed, 284 deselected
- `scripts/run_full_stack_tests.ps1`: 100 passed, 187 deselected
- Explicit Playwright E2E: 66 passed
- Targeted production dependency regression passed
- `git diff --check` passed with line-ending warnings only
- Live services verified for API, CPU worker, GPU worker registration, Redis, dashboard, MLflow, Postgres, and assistant-server

This changelog was generated from git history, local diffs, and existing reports. The full suites were not rerun while drafting this file.

## Current Local Changes After HEAD

The worktree has additional uncommitted changes beyond `c13113a`. These should be reviewed before tagging the next release:

- Panel context now caps and summarizes plots, runs, metrics, parameters, and feature lists to make `/panel/context` lighter.
- Panel UI copy has shifted from "Panel" to "Dashboard Workspace", with Plotly Dash workspace as the primary tile type.
- Panel save/delete selector state now updates immediately while still refreshing from the backend.
- Production simulation UI is now presented as "Prediction Review"; raw scenario JSON is hidden in Advanced Payload by default.
- `/production/simulate` now returns structured `simulation_runtime_error` payloads and includes `primary_result`, `series`, `scenario_comparison`, `changed_features`, and `plot_specs` aliases.
- Assistant UI now renders explicit runtime gates for provider configured, bundle ready, runtime loaded, and draft validated.
- Selected AssistantModel draft generation now refuses unavailable runtimes and rejects fallback output when a selected model was requested.
- Assistant provider config now defaults selected model fallback to disabled.
- Legacy plot artifact listing supports a limit to avoid expensive traversal.
- Training estimator class construction is handled before clone paths, preserving constructor parameter filtering.
- E2E tests now use longer route readiness timeouts and assert the updated dashboard/prediction wording.
- Assistant workflow tests now cover selected-model runtime failure blocking fallback.

## Generated And Runtime Artifacts

The range since `Release` adds or updates a large amount of generated state:

- Assistant datasets, evals, imported assistant model metadata, and promotion reports under `api/runtime_artifacts/assistant_*`.
- Run ledgers for editor, assistant, training, and study jobs under `api/runtime_artifacts/runs`.
- Training, serving, monitoring, plot, deployment, ONNX, Store, and MLflow artifacts.
- MLflow run directories and `mlflow-server/mlflow.db`.
- E2E and UI audit screenshots under `temp/`.
- Test temp artifacts under `api/tests/.tmp`.

These artifacts are useful as verification evidence, but they remain a release-cleanliness concern. Before a packaged release, decide which artifacts become fixtures, which are documented evidence, and which should be ignored or removed from source control.

## Known Gaps Before Next Release

- README/config still report app version `0.7.0` while commit history includes `Stable 0.7.2` and `Update 0.7.3`; version metadata should be reconciled.
- The worktree is dirty with both source changes and runtime artifacts.
- Assistant-server is healthy, but selected Gemma-style runtime generation still fails without a validated draft in the latest recorded live run.
- GPU worker registration is healthy, but the host reports no CUDA device to PyTorch.
- Some live training/study run ledgers remain stale after heavy CPU smoke attempts.
- `.pytest_cache` still hits Windows access-denied warnings, though the configured temp dirs keep suites passing.
- Docker Compose still warns that `POSTGRES_PASSWORD` is unset for local distribution.
- External chat-history recovery remains blocked until raw transcript files are added to `reports/chat-input/`.

## Release Note Draft

Painel Amanaje has grown into a much more complete local-first MLOps workspace. Since the `Release` tag, the app added dashboard composition, Store/provider workflows, Operations run evidence, queue-backed execution, richer training and production flows, plot and visualization artifact management, ONNX/Netron readiness, assistant model management, a local assistant runtime sidecar, dashboard extensions, and broad automated coverage. The next release should focus on cleaning generated artifacts, reconciling version metadata, validating assistant runtime readiness, and preserving the passing test posture on a clean checkout.
