# Next-Release Completeness Roadmap

Date: 2026-05-22
Release posture: polished local-first beta

## Evidence Policy

This roadmap is a living completeness ledger for ideas that were deferred while Painel Amanaje focused on becoming functional. It uses two evidence classes:

- Repo-backed evidence: existing reports, docs, README, tests, templates, TODO/deferred-language searches, and current git status.
- Transcript-backed evidence: only raw chat transcript files placed in `reports/chat-input/`.

Current transcript status: pending. `reports/chat-input/` has been created, but no raw transcript files are present yet. Until transcripts are added, this roadmap must not claim full recovery of "every chat so far"; it is a repo-evidence baseline with an explicit intake gap.

## Source Reconciliation

Older reports such as `reports/ALIGNMENT_COMPLETION_REPORT.md`, `reports/CHANGES_SUMMARY.md`, and `reports/PROJECT_UPDATE_SUMMARY.md` describe major TODO completion and production-ready template alignment. The newer `reports/cohesion-benchmark-audit.md` changes the standard of completeness: the app is broad and increasingly operational, but still incomplete as a cohesive ML engineer workspace.

For the next release, "complete" means the local beta feels whole from the start:

- A user keeps one active ML context across domains.
- Runs, models, plots, production checks, and registry objects can be compared.
- Workflows expose evidence, readiness, and next actions.
- The repo can be shared locally without stale docs, copied templates, or generated runtime state obscuring the source release.

## Recovered Completeness Themes

| Theme | Evidence class | Deferred or incomplete idea | Current gap | Local beta priority | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| Global context contract | Repo-backed | Carry dataset, model, study, run, inference pair, artifact, device, and environment across Panel, Store, Feature, Training, Visualization, Production, Operations, Registry, and Assistant. | Each page still has its own selectors and partial state. Panel context is powerful but slow and not yet the shared product center. | P0 | Selecting a dataset filters compatible models, studies, runs, plots, and inference pairs across at least Panel, Training, Visualization, and Production. Changing context survives navigation and reload. |

| Baseline/candidate comparison | Repo-backed | Treat comparison as a first-class workflow rather than a visual afterthought. | Training, Visualization, Panel, Registry, and Production expose related evidence, but baseline and candidate meaning is spread across run history, plots, MLflow, and production views. | P0 | A completed run can be pinned as baseline, a new run can be pinned as candidate, and both roles render consistently in run tables, plot cards, Panel widgets, and Registry details. |

| Visualization decision layer | Repo-backed | Make plots answer ML engineering questions, with intent, source object, artifact metadata, and inspector details. | `/plot`, `/visualization`, Panel widgets, Dash embeds, and plot artifacts overlap without one authoritative plot intent model. | P0 | Plot cards show intent, source object, generated time, artifact path, and related dataset/model/run. Users can inspect a plot, pin it as baseline or candidate, and keep graceful offline fallback behavior. |

| Registry lineage and readiness | Repo-backed | Move Registry from CRUD-first browsing to the product spine for lineage, dependencies, readiness, artifacts, and next actions. | Dependency routes and object aliases exist, but relationship navigation is not yet the primary experience. | P0 | Each registry object exposes upstream/downstream links, readiness badges, key artifacts, linked runs, and next valid actions. E2E covers dependency navigation. |

| Operations as runtime evidence | Repo-backed | Make `/operations` the shared ledger for queued work, terminal output, logs, retries, artifacts, MLflow links, and downstream changes. | Runtime workers, Operations, Production activity, MLflow, and assistant jobs exist as adjacent surfaces instead of one evidence layer. | P1 | Execute, training, study, assistant, ONNX, Store, and production jobs appear in Operations with status, source context, payload inspection, logs, artifacts, retry where supported, and latency visibility. |

| Assistant as reviewer | Repo-backed | Keep assistant output reviewable and evidence-backed until a real model bundle, eval dataset, graders, and trace dashboards are stable. | Assistant management is broad, but missing model/runtime readiness still makes generation quality unreliable. | P1 | Assistant UI separates provider health, model bundle readiness, eval readiness, and draft quality. Missing bundles show explicit unavailable state. Assistant artifacts link to Registry and Operations. |

| Store and Feature readiness | Repo-backed | Reframe Store and Feature as feature definitions, materialization, validation, freshness, and source lineage. | Store providers and Feature workspace are useful but not yet a coherent feature-store model with online/offline semantics. | P1 | Store providers, feature views, source datasets, materialization records, freshness, and readiness are represented distinctly. Training preflight consumes feature readiness. |

| Production control plane | Repo-backed | Make production workflows explicit about service lifecycle, rollback, thresholds, audit trail, and retraining pressure. | Production has monitoring, simulation, history, and MLflow tabs, but control-plane semantics are still partial. | P1 | Active inference pair anchors the page. Health thresholds, drift/retraining pressure, simulation comparison, lifecycle events, rollback readiness, and audit history are visible together. |

| ONNX and Netron promotion evidence | Repo-backed | Fold ONNX validation and Netron inspection into model promotion readiness. | ONNX/Netron exists as a side route, not a promotion gate tied to Registry and Production readiness. | P2 | Registry model details show ONNX readiness, validation result, Netron link/status, export artifact, and why a model is or is not deployable. |

| Dashboard extensions lifecycle | Repo-backed | Keep extension loading safe while adding discovery, compatibility, trust, versioning, and failure isolation. | Tests cover loading safety, but it is not yet a mature extension ecosystem. | P2 | Extension cards show metadata, compatibility, trust state, enabled/disabled state, load errors, and recovery guidance without breaking the main app. |

| Distribution-clean local beta | Repo-backed | Separate source from generated runtime artifacts, stale docs, local MLflow DB state, screenshots, logs, imported assistant models, and copied templates. | Current worktree includes source changes plus runtime artifacts and generated MLflow state. README still conflicts with active dashboard/Store/Operations reality. Legacy copied templates remain visible. | P0 | A clean checkout can run the documented CPU stack. README route/service tables match compose and active routes. Runtime outputs are ignored, fixture-scoped, or documented. Copied legacy templates are removed or archived intentionally. |

| E2E confidence gate | Repo-backed | Make first-load flows reliable enough for release confidence. | The cohesion audit recorded route/timeouts around Registry, Upload, and Panel save/context behavior. | P0 | E2E route smoke and core workflow tests pass repeatedly for Panel, Upload, Registry, Training, Visualization, Production, and Operations under documented local beta setup. |

| External chat recovery | Pending transcript intake | Recover ideas that only exist in prior project chats. | No transcript files are present in `reports/chat-input/` yet. | P0 for audit completeness | Once transcripts are added, each chat-backed item gets a source filename, recovered idea, disposition, priority, acceptance criteria, and revisit trigger. |

## Source References

No transcript-backed findings are listed yet because `reports/chat-input/` has no raw transcript files.

- Global context, comparison, Registry, Operations, distribution, and E2E gaps: `reports/cohesion-benchmark-audit.md`.
- Hand-designed ML engineer cockpit, plot intent, comparison, and view-level redesign direction: `docs/ml-engineer-hand-design-blueprint.md`.
- Local-first product posture, live workflows, stale service notes, and future plans: `README.md`.
- Manual local beta confidence criteria: `docs/manual-signoff-checklist.md` and `docs/manual-signoff-report-template.md`.
- Historical completion claims and deferred implementation notes: `reports/ALIGNMENT_COMPLETION_REPORT.md`, `reports/CHANGES_SUMMARY.md`, and `reports/PROJECT_UPDATE_SUMMARY.md`.
- Assistant runtime and model-bundle constraints: `api/assistant_server/README.md` and `api/assistant_server/pytorch/README.md`.
- Template/test evidence from targeted searches: `api/templates`, `api/static`, and `api/tests`.

## Next Release Sequence

1. Stabilize the release baseline.
   - Update README route/service documentation for dashboard, Store, Operations, assistant-server, CPU worker, and GPU profile reality.
   - Remove or archive copied legacy templates after confirming they are not referenced.
   - Separate runtime/generated artifacts from source release state.
   - Fix or document the E2E timeout causes for Registry, Upload, Panel save, and Panel context.

2. Add the shared context contract.
   - Define one typed active-context payload for dataset, model, study, run, inference pair, artifact, device, and environment.
   - Persist locally and server-side where practical.
   - Expose compact context in the shell and reuse it in Panel, Training, Visualization, Production, Registry, and Operations.

3. Make comparison visible everywhere it matters.
   - Add baseline and candidate slots to the active context.
   - Render metric deltas, parameter diffs, artifact diffs, and readiness decisions.
   - Connect comparison state to plot artifacts and Registry object details.

4. Promote Visualization and Registry into decision surfaces.
   - Add plot intent metadata, grouped plot sections, selected-plot inspector, and pinning.
   - Make Registry detail pages relationship-first with readiness badges and next actions.

5. Consolidate operational evidence.
   - Make Operations the common view for queued jobs, terminal/log output, MLflow links, artifacts, retries, and downstream registry changes.
   - Keep assistant actions reviewable and linked to Operations and Registry until eval quality is visible.

## Continual Completeness Loop

Use this loop for every release planning pass:

1. Add new chat exports, pasted planning notes, or design notes to `reports/chat-input/`.
2. Re-run targeted searches for `TODO`, `FIXME`, `future`, `deferred`, `placeholder`, `coming soon`, `not implemented`, `gap`, and `roadmap` across docs, reports, templates, static assets, and tests.
3. Update this roadmap with new evidence, changed priority, and resolved items.
4. For each idea, choose one disposition: promoted to next release, deferred with rationale, closed as obsolete, or blocked pending evidence.
5. For every deferred item, record the revisit trigger, such as transcript evidence, user feedback, failed signoff, benchmark mismatch, or dependency readiness.
6. Before release, reconcile this roadmap against the manual signoff checklist and the latest cohesion audit.

## Current Blockers And Limits

- Transcript-backed recovery is blocked until raw chat files are placed in `reports/chat-input/`.
- Current git status is mixed with runtime artifacts and source changes, so distribution cleanliness remains a release blocker.
- Older completion reports should be treated as historical milestones, not as final readiness proof.
- GPU worker wiring should not be marketed as acceleration until CUDA-backed runtime proof exists.
- Assistant generation should not be presented as trustworthy autonomous building until model bundle readiness and eval quality are visible.
