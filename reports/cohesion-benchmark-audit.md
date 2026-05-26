# Cohesion Benchmark Audit And Roadmap

Date: 2026-05-21

## Summary

Painel Amanaje has moved from a broad generated ML workspace into a working local-first MLOps control plane. The strongest improvements are operational: Redis/RQ queueing, run ledgers, MLflow-backed training artifacts, Panel dashboard persistence, dashboard extension safety, assistant management endpoints, Store/Operations additions, and wider test coverage.

The main weakness is no longer "does this page exist?" It is cohesion. The app has many capable domains, but they do not yet share one stable object context, comparison model, readiness model, or product hierarchy. The result is an app that can do many things, but still asks the user to assemble the meaning across pages.

The best next direction is to make Panel, Visualization, Training, Production, Registry, Store, and Operations orbit the same active context: dataset, model, study, run, inference pair, artifact, environment, and assistant review.

## Evidence Collected

- Current route inventory shows active pages for `/`, `/panel`, `/upload`, `/create`, `/feature`, `/store`, `/training`, `/onnx`, `/optimization`, `/editor`, `/production`, `/operations`, `/visualization`, `/plot`, `/registry`, `/assistant`, and `/settings`.
- Public API surface now includes Panel CRUD, Store providers/catalog/materialization, runtime workers, queued execution/training/studies, Operations run analysis/terminal, plot artifacts, ONNX Netron endpoints, Production monitoring/simulation, Registry dependencies/downloads, and Assistant runtime/model/eval endpoints.
- Template inventory includes mature pages plus legacy copies: `base_red copy.html` and `base_test copy.html`.
- Static JS inventory now includes domain scripts for assistant, create, operations, panel, settings, UI utilities, editor utilities, form utilities, and global helpers.
- Current worktree is heavily dirty with source changes, runtime artifacts, assistant model imports, Store/Operations/Netron additions, MLflow DB changes, and generated logs. This is useful evidence, but it is not distribution-clean.
- `git diff --check` passed with line-ending warnings only.
- `/runtime/workers` returned HTTP 200 with RQ backend information.
- Direct `docker compose ps` was not used in this completed pass because Docker socket access required elevated permission in the prior attempt; runtime health endpoints were used instead.

## Test Results

- Fast suite: `195 passed, 60 deselected, 4 warnings`.
- Targeted Store/Netron/dashboard-extension suite: `16 passed, 4 warnings`.
- Dashboard service E2E smoke: `1 passed, 1 warning`.
- Explicit Playwright E2E: `54 passed, 3 failed, 2 warnings`.
- Current E2E failures:
  - `/registry` route timed out waiting for `domcontentloaded` in one route smoke.
  - Panel save/reload/delete timed out waiting for the `POST /panel/dashboards` response.
  - `/upload` styled-panel test timed out waiting for `networkidle`.
- Timing checks after the E2E run:
  - `GET /registry`: about 2.8 seconds.
  - `GET /upload`: about 2.9 seconds.
  - `GET /panel/dashboards`: about 4.7 seconds.
  - `GET /panel/context`: about 15.5 seconds.
- Persistent warnings:
  - `.pytest_cache` cannot be written because Windows denies access.
  - Pydantic class `Config` deprecation.
  - FastAPI `on_event` deprecation.
  - `git status` still reports the `airflow/logs/dag_processor/latest` directory warning.

## Benchmark Map

| Painel domain | Better working comparison | Lesson to borrow |
| --- | --- | --- |
| Panel/workspace | [JupyterLab workspaces](https://jupyterlab.readthedocs.io/en/stable/user/workspaces.html), [Grafana dashboards](https://grafana.com/docs/grafana/latest/visualizations/dashboards/) | Persist layout and context as first-class workspace state, and make panels query/transform known data sources rather than isolated widgets. |
| Visualization/dashboarding | [Grafana dashboards](https://grafana.com/docs/grafana/latest/visualizations/dashboards/), [Apache Superset](https://superset.apache.org/user-docs/intro/), [Plotly Dash](https://dash.plotly.com/) | Treat plots as decision surfaces with data-source contracts, not just embedded chart output. |
| Training/experiments | [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [W&B run comparison](https://docs.wandb.ai/models/runs/compare-runs) | Make run comparison, model/dataset-linked metrics, pinned baselines, and candidate deltas first-class. |
| Feature/Store | [Feast feature views](https://docs.feast.dev/getting-started/concepts/feature-view) | Separate feature definitions, materialization, online/offline availability, and schema contracts. |
| Production/monitoring | [Evidently monitoring](https://docs.evidentlyai.com/docs/platform/monitoring_overview), Grafana | Turn monitoring artifacts into persisted report runs and dashboard health signals with thresholds. |
| Registry/lineage | MLflow model search, [W&B Registry](https://docs.wandb.ai/models/registry), [Dagster assets](https://docs.dagster.io/) | Move from CRUD-first objects to lineage, readiness, asset health, and dependency navigation. |
| Assistant | [LangSmith](https://docs.langchain.com/langsmith/home), [OpenAI evals](https://platform.openai.com/docs/guides/evaluation-best-practices) | Treat assistant output as traceable, evaluated, reviewable work with datasets and graders, not just generated text. |
| Dashboard extensions | [JupyterLab extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html) | Keep extension loading safe, but add lifecycle: discovery, metadata, compatibility, trust, and failure isolation. |

## Domain Findings

| Domain | Cohesion score | Improved | Faltered | Does not fit yet |
| --- | --- | --- | --- | --- |
| Panel/workspace | 3/5 | Real route, saved dashboard CRUD, widgets, filters, run/edit modes, context hydration, and E2E coverage exist. | Panel context is slow enough to destabilize E2E. It also remains a separate cockpit rather than the shared state center for other pages. | It cannot yet be the product's home cockpit until active object context is shared across Training, Production, Visualization, Registry, Store, and Operations. |

| Upload/Create | 3/5 | Support matrices, assistant-assisted creation, editor execution, and JSON UI helpers are stronger than the original upload-only flow. | `/upload` still has stale behavior such as "Feature extraction is coming soon" in the legacy template, and E2E can time out on `networkidle`. | Upload should become an ingestion utility under Store/Create, not a separate conceptual domain competing with them. |

| Store/Feature | 3/5 | New `/store` route, provider catalog APIs, preview/materialize paths, and Store utility tests strengthen the data domain. | Feature and Store are adjacent but not yet one feature-store model. | It should not claim Feast-like semantics until it defines feature views, online/offline stores, materialization history, and freshness/readiness contracts. |

| Training/Optimization | 4/5 | RQ jobs, structured run ledgers, MLflow artifacts, study optimization, and fast/full-stack coverage have improved substantially. | Comparison is still spread across run history, plots, and MLflow rather than a single baseline/candidate workflow. | GPU acceleration is wired at queue level but cannot be treated as available when CUDA is absent. |

| ONNX/Netron | 3/5 | ONNX preparation and new Netron viewer endpoints give model portability a more inspectable path. | ONNX is still a side route, not part of a promotion gate tied to model readiness. | Netron viewing should be folded into Registry/Promotion evidence before being presented as a distribution-ready deployment step. |

| Visualization/Dashboard | 3/5 | `/visualization`, `/plot`, Dash `/embed`, `/extensions`, and `/extensions.json` exist with fallback behavior and tests. | API visualization, Dash service, Panel iframes, and plot artifacts overlap without one source of truth for plot intent and comparison state. | Dashboard extensions are safe to disable, but not yet a mature plugin ecosystem. |

| Registry/Lineage | 3/5 | Polymorphic DB fixes, object aliases, dependency routes, downloads, and CRUD coverage make the registry more reliable. | Registry remains CRUD-first. Relationship navigation is present but not the primary experience. | It cannot carry product cohesion until each object exposes readiness, lineage, key artifacts, linked runs, and next actions. |

| Production/Operations | 3/5 | Production status, simulation, monitoring, retraining, history, and new `/operations` run terminal/analysis endpoints show real operational shape. | Monitoring is still artifact/log oriented, and Operations appears as another surface rather than the common run lens. | It cannot yet stand as a production control plane without service lifecycle, health thresholds, audit trails, and rollback semantics. |

| Assistant | 3/5 | Assistant management, sessions, jobs, references, evals, model imports, Hugging Face artifacts, and runtime status endpoints are broad. | Assistant-server currently reports unavailable when no valid model bundle is loaded; generated output quality depends on missing model/runtime readiness. | It should remain an inspector/reviewer until loaded models, eval datasets, graders, and trace dashboards are stable. |

| Runtime/Workers | 4/5 | Redis/RQ dependencies, queue routing, worker status endpoint, CPU/GPU queues, and operation ledgers are much stronger. | Docker inspection remains environment-sensitive, and distribution state is obscured by dirty runtime artifacts. | GPU worker wiring is not equivalent to CUDA-backed training until the host exposes devices to PyTorch. |

| Docs/Distribution | 2/5 | Reports preserve a useful audit trail, and README explains the local-first MLOps ambition. | README still says dashboard is commented out even though dashboard is now active; older reports overstate production readiness compared to current gaps. | The repo is not distribution-clean until generated artifacts, stale docs, copied templates, secrets/defaults, and environment warnings are separated from source release state. |

## Cross-Cutting Cohesion Gaps

1. No global context contract.
   The app needs one persisted context object for dataset, model, study, run, inference pair, artifact, device, and environment. Today each page has its own selectors and partial state.

2. No shared comparison model.
   Training, Visualization, Panel, Production, and Registry all need baseline/candidate semantics. W&B-style pinned baselines and MLflow model/dataset-linked metrics are the clearest references.

3. Plot intent is not authoritative enough.
   Visualization has moved in the right direction, but plot intent should be carried in artifact metadata, registry links, Panel widgets, and Operations analysis.

4. Registry is not yet the product spine.
   Registry objects should expose lineage and readiness: trainable, comparable, deployable, monitorable, exportable, assistant-reviewed.

5. Operational health is split.
   Runtime workers, Operations runs, Production monitoring, MLflow, and dashboard service health are separate surfaces. They should form one operational evidence layer.

6. Distribution is mixed with runtime state.
   Generated artifacts, MLflow DB updates, assistant imports, logs, and dirty runtime files are valuable local evidence but should not sit in the same distribution posture as app source.

7. Legacy surfaces remain visible in source.
   `base_red copy.html` and `base_test copy.html` are harmless if ignored, but they create ambiguity for future maintainers and audits.

## Prioritized Roadmap

### 1. Create A Global Context Contract

- Add one typed context payload shared by Panel, Store, Feature, Training, Visualization, Production, Operations, Registry, and Assistant.
- Persist it locally and server-side when possible.
- Expose active context in the shell header or compact context rail.
- Acceptance criteria:
  - Selecting a dataset filters compatible models, studies, runs, plots, and inference pairs across pages.
  - Changing context in one domain is visible after navigating to another domain.
  - E2E covers context persistence across at least Panel, Training, Visualization, and Production.

### 2. Make Comparison First-Class

- Add baseline/candidate slots to run, plot, model, and production views.
- Store comparison state in the global context.
- Surface metric deltas, parameter diffs, artifact diffs, and readiness decisions.
- Acceptance criteria:
  - A completed run can be pinned as baseline.
  - A new run can be pinned as candidate.
  - Visualization and Panel render the same baseline/candidate meaning.
  - Registry shows which model/run is current, candidate, or archived.

### 3. Promote Visualization Into A Decision Layer

- Treat every plot as answering a question: learning, overfitting, drift, residuals, feature quality, production health, registry trace.
- Carry plot intent through artifact metadata and `/plots/artifacts`.
- Merge local fallback, Dash embeds, and Panel plot widgets around the same plot spec.
- Acceptance criteria:
  - Plot cards show intent, source object, generated time, artifact path, and related run/model/dataset.
  - Plot inspector can pin baseline/candidate plots.
  - Dashboard offline state remains graceful.

### 4. Make Registry The Lineage Backbone

- Add relationship-first detail views for dataset -> model -> study -> run -> inference pair -> production monitor -> assistant review.
- Add readiness badges per object.
- Connect Store, Training, ONNX, Production, and Assistant actions back to registry objects.
- Acceptance criteria:
  - Every object type shows linked upstream/downstream objects.
  - Registry detail pages expose next valid actions.
  - Dependency routes are covered by E2E, not only unit tests.

### 5. Reframe Store And Feature As A Feature-Readiness Domain

- Rename or structure Store/Feature around feature definitions, materialization, validation, and freshness.
- Borrow Feast's separation between feature views, entities, sources, and stores without over-claiming full Feast parity.
- Acceptance criteria:
  - Store providers and feature views are represented as distinct concepts.
  - Feature materialization records freshness and source lineage.
  - Training preflight consumes feature readiness rather than raw dataset availability alone.

### 6. Stabilize Operations As The Runtime Evidence Layer

- Make `/operations` the shared run history, terminal, logs, and analysis view for all queued work.
- Link each operation to source context, artifacts, MLflow run, worker queue, and downstream registry changes.
- Acceptance criteria:
  - Execute, training, study, assistant, ONNX, Store, and production jobs all appear in Operations.
  - Operations offers retry, inspect payload, open logs, and open artifacts where applicable.
  - Long-running endpoint latency is visible and testable.

### 7. Keep Assistant As A Reviewer Until Runtime Is Real

- Keep assistant actions reviewable, reversible, and evidence-backed.
- Require eval dataset and grader status before presenting assistant output as reliable.
- Borrow LangSmith-style trace/eval organization and OpenAI-style datasets/graders.
- Acceptance criteria:
  - Assistant management clearly separates provider health, model bundle readiness, eval readiness, and draft quality.
  - Missing model bundles produce explicit "unavailable" state, not product ambiguity.
  - Assistant artifacts link to Registry and Operations.

### 8. Prepare A Distribution-Clean Profile

- Separate source, generated runtime artifacts, local model imports, screenshots, logs, and MLflow DB state.
- Update README route/service tables for dashboard, Store, Operations, assistant-server, CPU worker, and GPU profile reality.
- Remove or archive copied legacy templates.
- Replace insecure or ambiguous defaults before release.
- Acceptance criteria:
  - A clean checkout can run the documented CPU stack.
  - Runtime artifacts are either ignored, fixture-scoped, or documented as sample data.
  - README no longer conflicts with compose services or active routes.

## Current "Cannot Fit Yet" List

- Assistant cannot fit as a trustworthy autonomous builder until a valid model bundle is loaded and eval quality is visible.
- GPU worker cannot fit as an acceleration promise until CUDA is available to PyTorch in the runtime environment.
- Feature workflow cannot fit as a feature store until it has feature-view contracts, freshness, online/offline semantics, and source lineage.
- Dashboard extensions cannot fit as a plugin ecosystem until there is lifecycle/version/trust/compatibility management.
- Registry cannot fit as the product backbone while it is primarily CRUD and not relationship/readiness first.
- Production cannot fit as a production control plane until service lifecycle, rollback, thresholds, and audit behavior are explicit.
- Distribution cannot fit as release-ready while source, generated artifacts, local DB state, imported assistant models, and stale docs are mixed.
- E2E cannot fit as a confidence gate while common first-load flows still hit timeout failures under Playwright.

## Immediate Fix Backlog

1. Investigate and optimize `/panel/context`; current observed response time was about 15.5 seconds.
2. Fix E2E readiness waits for Panel, Upload, and Registry, but only after confirming whether the root issue is test waiting strategy, slow app endpoints, or page-side network activity.
3. Update README route and compose-service documentation to include dashboard, Store, Operations, assistant-server, CPU worker, and GPU profile status.
4. Move or delete legacy copied templates after confirming they are not referenced.
5. Add a route/domain table test for `/store`, `/operations`, and ONNX Netron endpoints.
6. Add a source-of-truth object relationship schema for Registry and Panel context.
7. Add distribution profile documentation that distinguishes checked-in fixtures from generated runtime evidence.

## Source Notes

External benchmark facts were checked against official documentation: MLflow Tracking, W&B run comparison and Registry, Feast feature views, Evidently monitoring, Grafana dashboards, Apache Superset, Plotly Dash, JupyterLab workspaces/extensions, LangSmith, Dagster, and OpenAI evaluation guidance. These sources were used as design references, not as strict parity requirements.

