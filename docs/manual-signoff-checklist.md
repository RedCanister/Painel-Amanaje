# Painel Amanaje Manual Full-Coverage Signoff Checklist

This checklist is the manual signoff pass for the **entire currently routed app surface**:

- `/`
- `/upload`
- `/create`
- `/feature`
- `/training`
- `/optimization`
- `/editor`
- `/production`
- `/registry`
- supporting JSON/API surfaces under `api/main_app.py`

Run it twice:

1. `Fresh stack pass`: clean state, no manually created datasets/models/studies/inference pairs.
2. `Stateful recovery pass`: after you have already created datasets, models, studies, inference pairs, and completed at least one training/study/production cycle.

## Before You Start

1. Generate or refresh the manual fixtures:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_manual_test_fixtures.py
```

2. Optionally prepare the full stack and fixture package together:

```powershell
.\scripts\prepare_manual_signoff.ps1
```

3. Open `api/tests/fixtures/manual/manifest.json` after generation. That file is the source of truth for the fixture paths you will use.

4. Use this baseline set during manual testing:

- Existing dataset fixtures:
  - `api/tests/fixtures/datasets/narrow_sample.csv`
  - `api/tests/fixtures/datasets/narrow_sample.tsv`
  - `api/tests/fixtures/datasets/narrow_sample.jsonl`
  - `api/tests/fixtures/datasets/wide_columns.csv`
  - `api/tests/fixtures/datasets/github_vs_git_schema_mismatch.csv`
- Generated manual dataset fixtures:
  - `api/tests/fixtures/manual/datasets/narrow_sample.xlsx`
  - `api/tests/fixtures/manual/datasets/narrow_sample.parquet`
- Generated manual model fixtures:
  - `api/tests/fixtures/manual/models/valid_sklearn.joblib`
  - `api/tests/fixtures/manual/models/valid_importable.pkl`
  - `api/tests/fixtures/manual/models/broken_missing_module.pkl`
  - `api/tests/fixtures/manual/models/valid_torchscript.pt`
  - `api/tests/fixtures/manual/models/invalid_state_dict.pt`
  - `api/tests/fixtures/manual/models/identity.onnx`

## Pass / Fail Rule

Mark the run as **pass** only if:

- every routed page renders
- every major workflow can be exercised
- known failure classes return structured user-facing errors
- JSON-heavy displays remain usable
- no page exceeds viewport width
- no browser console/network errors appear unexpectedly

Mark the run as **fail** if you see:

- raw traceback output
- silent crash or stuck spinner
- unstructured `failed to fetch`
- broken run ledger state
- unexplained dependency crash
- registry foreign-key delete crash
- MLflow page collapse
- horizontal overflow

For every failure, capture:

- URL
- active tab or panel
- exact user action
- browser console error
- failing network request
- response body
- whether refresh reproduces it

## Checklist

### A. Stack, Shell, and Global Layout

- [ ] Start the full stack and confirm `api`, `postgres`, `mlflow`, `airflow-webserver`, and `pgadmin` are reachable before opening the UI.
- [ ] Open `/` and verify every navigation link works and no page returns 500.
- [ ] Confirm no generic `failed to fetch` message appears during simple navigation.
- [ ] Test layout at full width, docked-left half screen, docked-right half screen, tablet-like width, and narrow mobile-like width.
- [ ] Confirm no horizontal overflow at any tested width.
- [ ] Confirm every page respects viewport width and panels resize instead of spilling outside the window.
- [ ] On every page with large JSON, confirm JSON is scrollable, collapsible, or both.
- [ ] Verify the JSON usability rule specifically on Upload, Create, Feature, Training, Production, Registry, and Editor.

### B. Upload

- [ ] Open `/upload` and expand/collapse the supported file types panel.
- [ ] Confirm dataset/model capability claims are readable and match actual behavior.
- [ ] Upload `narrow_sample.csv`.
- [ ] Verify registry entry creation, parser report, column explorer, dataset history entry, and data analysis output.
- [ ] Upload `narrow_sample.tsv` and confirm delimiter detection is correct.
- [ ] Upload `narrow_sample.jsonl` and confirm row/column counts are correct.
- [ ] Upload `narrow_sample.xlsx` and verify registration plus analysis.
- [ ] Upload `narrow_sample.parquet` and verify registration plus analysis.
- [ ] Upload `wide_columns.csv` and verify wide-column parsing, explorer usability, and stable layout.
- [ ] Upload one unsupported dataset file type and confirm a clear structured error instead of a crash.
- [ ] Upload `valid_sklearn.joblib` and verify the artifact manifest shows runnable prediction/simulation capability.
- [ ] Upload `valid_importable.pkl` and verify guarded pickle support is explained clearly.
- [ ] Upload `broken_missing_module.pkl` and confirm registration works only as guarded support.
- [ ] After registering `broken_missing_module.pkl`, use later runtime actions to confirm dependency/runtime failure is structured and readable.
- [ ] Upload `valid_torchscript.pt` and verify it is recognized as directly runnable.
- [ ] Upload `invalid_state_dict.pt` and verify it is flagged as guarded/not directly runnable.
- [ ] Upload `identity.onnx` and verify ONNX metadata inspection works.

### C. Create

- [ ] Open `/create` and expand/collapse the supported file types panel.
- [ ] Confirm the support content and behavior matches `/upload`.
- [ ] Load the dataset template.
- [ ] Execute it, extract metadata, and create the dataset object.
- [ ] Verify the new dataset appears in history and registry.
- [ ] Load the sklearn/joblib model template.
- [ ] Execute it, extract metadata, and create the model object.
- [ ] Verify the new model appears in history and registry.
- [ ] Save a script **without executing it first** and verify save succeeds.
- [ ] Reload the saved script and confirm content and metadata survive the round trip.
- [ ] Export the document JSON and verify it contains script, metadata, and extracted variables together.
- [ ] Intentionally omit `csv_text` or `joblib_bytes` and confirm the UI shows a structured readable error rather than failing silently.

### D. Editor

- [ ] Open `/editor`.
- [ ] Repeat execute, extract variables, save, load, and export flows independently from `/create`.
- [ ] Confirm the legacy editor still works even if `/create` already has saved scripts.
- [ ] Verify large JSON and metadata displays are still usable and bounded.

### E. Feature Workspace

- [ ] Open `/feature` and load a dataset.
- [ ] Verify dataset summary, metric cards, transform builder, column explorer, preview, and plots render on the same page.
- [ ] Use the visual builder to rename a column and confirm preview plus explorer update correctly.
- [ ] Cast a column type and confirm preview reflects the cast without silent corruption.
- [ ] Fill missing values and confirm preview and summary update accordingly.
- [ ] Drop a column and confirm it disappears from preview and explorer.
- [ ] Add a filter and verify the preview row set changes correctly.
- [ ] Add a sort and verify row order changes correctly.
- [ ] Add a row limit and verify row count changes correctly.
- [ ] Switch to Advanced JSON mode, edit operations manually, and verify visual state and preview stay in sync.
- [ ] Materialize a transformed feature view.
- [ ] Verify a new dataset is created, CSV output exists, Postgres materialization status is shown, and the new dataset appears in registry/history.

### F. Training, Optimization, and ONNX

- [ ] Open `/training` and verify the global context bar persists while switching tabs.
- [ ] Select an inference pair and confirm model/dataset hydrate automatically.
- [ ] Run a training job from model + dataset directly.
- [ ] Verify immediate acceptance, `run_id` creation, live ledger updates, and completion without leaving the page.
- [ ] Run a training job from an inference pair and verify `inferenceId` remains attached through the ledger.
- [ ] Run a study optimization from the Study tab without preselecting it elsewhere and verify the selected study is the one that runs.
- [ ] Confirm progress stages are visible and meaningful: queued, resolving context, validating schema, preparing dataset, training or optimizing, evaluating, persisting artifacts, syncing inference, logging MLflow, completed or failed.
- [ ] Confirm Tracking and ONNX shows key metric cards near the top and snapshots remain collapsible.
- [ ] Run ONNX prepare on a compatible model and confirm success output is readable.
- [ ] Run ONNX validate on a compatible model and confirm success output is readable.
- [ ] Run ONNX validate on an incompatible or missing ONNX artifact and confirm the failure is structured and readable.
- [ ] Reproduce the GitHub-vs-Git mismatch using `github_vs_git_schema_mismatch.csv` with a model/study expecting `public_repos`, `followers`, `account_age_days`, `actor_login`, `repo_language`, and `expected_engagement_score`.
- [ ] Confirm the app returns a structured preflight error with `missing_columns`, `available_columns`, and recommendation details instead of `failed to fetch`.

### G. Production, Monitoring, Simulation, and MLflow

- [ ] Open `/production`.
- [ ] Create or select at least two inference pairs and add both to the watch flow.
- [ ] Confirm the watchlist keeps separate state per inference pair.
- [ ] Start production from an inference pair and verify deployment snapshot, runtime state, monitoring cards, and history populate correctly.
- [ ] Stop production and confirm the state changes are visible immediately.
- [ ] Restart production and confirm state remains isolated to the selected watch context.
- [ ] Refresh monitoring repeatedly and confirm no other watch context is corrupted.
- [ ] Trigger retraining from production and verify activity is recorded without breaking the active watch context.
- [ ] Run simulation on a supported model and confirm summary, plots, and the full-width predicted steps table all appear on the production page.
- [ ] Inspect the simulation table and confirm long input payloads remain readable through scroll or collapsible JSON presentation.
- [ ] Run production simulation with a missing dependency case and verify the error names the missing module and loader in structured form.
- [ ] Run production simulation with an unsupported artifact and confirm you get a structured `400`-style user-facing error instead of a raw `500`.
- [ ] Open the Production MLflow tab and verify experiment summary cards appear first.
- [ ] Verify a scrollable run list appears below the summary cards.
- [ ] Expand at least one MLflow run and inspect params, metrics, tags, and artifacts.
- [ ] Paginate or scroll through MLflow runs and confirm `has_more` behavior is consistent and the page remains responsive.

### H. Registry

- [ ] Open `/registry`.
- [ ] Manually create at least one entry of each major type in current use: DatasetModel, LearningModel, InferenceModel, StudyModel, and CodeModel.
- [ ] Verify list view works.
- [ ] Verify table view works.
- [ ] Verify search/detail view works.
- [ ] Verify edit modal works for each major type.
- [ ] Verify download works for each major type where applicable.
- [ ] Verify object analysis works for dataset, model, inference, and study records.
- [ ] Try deleting a model or dataset that is still referenced by a study or inference pair.
- [ ] Confirm deletion is blocked with a dependency report rather than a foreign-key crash.
- [ ] Edit JSON-heavy fields such as parameters, metrics, history, inference params, best params, code, and variables.
- [ ] Confirm edited JSON saves correctly and still displays in a usable structured format.

### I. Direct JSON/API Surface Checks

- [ ] Open `/upload/support` directly and verify the response is clean JSON.
- [ ] Open `/features` directly and verify the response is clean JSON.
- [ ] Open `/features/extract` for a valid dataset and verify the response is clean JSON.
- [ ] Open `/runs/list` and verify the response is clean JSON.
- [ ] Open `/runs/get/{run_id}` for at least one training and one study run and verify the response is clean JSON.
- [ ] Open `/registry/dependencies/{registry_type}/{item_id}` and verify the response is clean JSON.
- [ ] Open `/mlflow/experiments` and verify the response is clean JSON.
- [ ] Open `/mlflow/experiments/{id}` and verify the response is clean JSON.
- [ ] Open `/mlflow/health` and verify the response is clean JSON.
- [ ] Open `/analysis/data` and verify the response is clean JSON.
- [ ] Open `/analysis/model` and verify the response is clean JSON.
- [ ] Open `/analysis/object` and verify the response is clean JSON.

### J. Stateful Recovery, Restarts, and Stress-Like Manual Checks

- [ ] During an active training or study run, restart the browser and verify the run ledger/history still shows the in-flight or completed run after reopening.
- [ ] Restart the API process or container after runs already exist and verify MLflow pages still load and degrade gracefully if artifact paths are stale.
- [ ] If possible, interrupt a run mid-flight once and verify the app surfaces failure state cleanly.
- [ ] Re-run a study or training flow with logically equivalent params like `0` and `0.0` and verify MLflow does not break on immutable param conflicts.
- [ ] Repeatedly click preview, refresh, monitor, or similar actions several times in a row.
- [ ] Confirm the app does not deadlock, duplicate jobs uncontrollably, or overwrite another active context.

### K. Final Sweep

- [ ] Revisit all main routes and confirm there are no browser console errors.
- [ ] Confirm there are no uncaught network failures.
- [ ] Confirm there is no generic `failed to fetch`.
- [ ] Confirm there are no unbounded JSON dumps left in the UI.
- [ ] Confirm there is no horizontal overflow anywhere.
- [ ] Mark the overall pass as `PASS` or `FAIL` in the report template.

## Scope Boundaries

This signoff covers the **currently active routed product surface** and related services wired into the main stack. It does **not** treat commented-out services such as the Dash dashboard or Feast samples as required signoff scope.
