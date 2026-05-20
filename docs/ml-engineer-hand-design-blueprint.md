# Painel Amanaje ML Engineer Hand Design Blueprint

This blueprint is the starting point for moving Painel Amanaje from generated feature coverage toward a hand-designed ML engineering workspace. The goal is not to make the app prettier in isolation. The goal is to make every view answer a working ML engineer's next decision.

## Product Posture

Painel Amanaje should feel like a local-first ML operations cockpit:

- dense enough for repeated technical use
- calm enough to inspect runs, datasets, and production behavior without visual noise
- explicit about object relationships: dataset, learning model, study, training run, inference pair, deployment, and plot artifact
- optimized for comparing states, not just creating records

The current app already has broad coverage across data, training, registry, production, visualization, assistant, ONNX, and settings. The hand-design work should now consolidate duplicated selectors, strengthen context persistence, and make plots part of the decision flow.

## Core User

Primary user: a machine learning engineer working locally or inside a controlled internal environment.

They need to:

- choose a dataset and understand whether it is trainable
- choose or create a model specification
- run training or a study
- compare run outputs and hyperparameters
- inspect whether a model is good enough to promote
- monitor inference behavior after deployment
- understand drift, degradation, and retraining pressure
- trace everything back to registry objects and artifacts

Painel Amanaje should reward this user's attention by keeping context stable and reducing repeated setup.

## Design Principles

1. Persistent context first

   The active context should be visible across the app: dataset, model, study, run, inference pair, framework, and device. A user should not have to re-select the same objects on every tab.

2. Plots must answer a question

   Every plot should have a declared intent such as "Is the model learning?", "Which run won?", "Is production drifting?", or "Which feature is driving risk?"

3. Comparison beats presentation

   ML engineers rarely need a single isolated card. They need current versus previous, train versus validation, baseline versus candidate, expected versus observed, and before versus after.

4. Dense, readable, and predictable

   The interface should avoid marketing composition. It should behave like a technical workbench with strong hierarchy, compact controls, and repeatable placement.

5. Evidence trail over magic

   Assistant features should draft, explain, and validate, but promotion actions should remain inspectable and reversible.

## First Target Experience: ML Engineer Cockpit

Create one hand-designed cockpit surface that becomes the product's center of gravity. It can either replace the current home/panel route or become a new route first, then migrate into the shell.

### Cockpit Layout

Top context rail:

- Active Dataset
- Active Model
- Active Study
- Active Run
- Active Inference Pair
- Runtime Device
- Environment health

Main workspace:

- left column: object and run navigator
- center column: primary decision view
- right column: evidence inspector

Bottom rail:

- latest activity
- queued operations
- warnings
- recoverable errors

### Cockpit Modes

1. Build

   Dataset readiness, schema fit, feature quality, model configuration, parameter source, and training launch.

2. Train

   Live run timeline, loss curves, metric cards, parameter diffs, accelerator status, and artifact output.

3. Compare

   Candidate run table, metric deltas, hyperparameter parallel coordinates, confusion or residual analysis, and promotion readiness.

4. Operate

   Inference pair status, production metrics, drift pressure, simulation, retraining recommendation, and monitoring history.

5. Trace

   Registry dependencies, artifacts, MLflow links, lineage, exported files, and assistant review trail.

## Plot Design System

Painel Amanaje should define plot types by engineering question, not by chart widget.

### Training Plots

Learning curve:

- x: epoch or step
- y: train loss and validation loss
- required annotation: best epoch, early stopping point, latest epoch
- decision answered: is the model learning or overfitting?

Metric progression:

- x: epoch, step, or trial
- y: selected metric
- overlays: baseline, previous best, current candidate
- decision answered: is this run improving enough to continue?

Hyperparameter importance:

- chart: ranked bars or SHAP-like contribution list from study results
- required interaction: click parameter to filter trials
- decision answered: which knobs matter?

Trial landscape:

- chart: scatter or parallel coordinates
- dimensions: objective, learning rate, batch size, architecture fields, runtime
- decision answered: where should the next study search?

### Evaluation Plots

Prediction versus actual:

- chart: scatter with residual coloring
- required overlay: ideal diagonal
- decision answered: where does the model miss?

Residual distribution:

- chart: histogram plus summary bands
- required metrics: mean error, p95 absolute error, skew
- decision answered: is the model biased or unstable?

Confusion matrix:

- chart: matrix with row/column totals
- required interaction: click cell to inspect examples
- decision answered: which class errors matter?

Feature drift:

- chart: per-feature drift bars with severity tiers
- required sorting: worst first
- decision answered: which inputs are unsafe?

### Production Plots

Inference health strip:

- x: time
- y: request rate, error rate, latency p95
- required annotation: deployment changes and retraining events
- decision answered: is production healthy?

Prediction distribution shift:

- chart: overlay or small multiples
- compares: training baseline, previous window, current window
- decision answered: are outputs changing?

Retraining pressure:

- chart: compact stacked signal
- dimensions: drift, metric degradation, data volume, age since training
- decision answered: should we retrain now?

Simulation sensitivity:

- chart: feature override controls plus output response curve
- required interaction: change feature value and compare baseline versus scenario
- decision answered: how does the model react to controlled changes?

## Dynamic View Behavior

Global context synchronization:

- selecting a dataset filters compatible models, studies, runs, plots, and inference pairs
- selecting a run hydrates framework, parameters, metrics, artifacts, and plot deck
- selecting an inference pair hydrates production status, dataset, model, monitoring plots, and simulation controls

Progressive disclosure:

- show the primary decision view first
- move raw JSON, registry payloads, and debug data behind inspectors
- keep advanced controls visible only after the object context is known

Stateful comparison:

- allow pinning a baseline run
- allow marking a candidate run
- all comparison plots use baseline and candidate consistently

Empty states:

- every empty state should explain the next concrete action
- empty states should offer direct actions such as "select dataset", "run training", "load artifacts", or "open registry object"

Error states:

- errors should identify the failing object and operation
- where possible, offer recovery actions: retry, inspect payload, open logs, or reset context

## View-Level Redesign Notes

### Training

Current strength: the Training Orchestrator already has a useful pipeline summary and step timeline.

Hand-design direction:

- convert the pipeline summary into a persistent right inspector
- add a live plot deck below the timeline
- replace scattered parameter controls with grouped parameter families
- add "baseline run" and "candidate run" slots
- add one-click transition from completed run to Compare mode

### Visualization

Current strength: broad source support and generator preview already exist.

Hand-design direction:

- rename gallery sections by decision intent instead of source type
- add plot intent metadata to every plot spec
- group plots into Training, Evaluation, Production, Data Quality, and Registry Trace
- add a plot inspector with source object, generated time, artifact path, and raw spec
- add "pin to cockpit" and "compare with baseline" actions

### Production

Current strength: production has context, monitoring, simulation, history, MLflow, and activity tabs.

Hand-design direction:

- make the active inference pair the anchor object
- merge duplicate model/dataset selectors into a single context rail
- make monitoring status a top-level health strip
- put simulation beside current production baseline for immediate comparison
- show retraining pressure as a composed signal, not just a yes/no flag

### Registry

Current strength: it can create, upload, inspect, edit, download, and organize multiple object types.

Hand-design direction:

- make registry detail pages relationship-first
- surface lineage: dataset -> model -> study -> run -> inference pair -> production monitor
- add object readiness badges: trainable, simulatable, deployable, inspectable
- let each object expose its most relevant plots directly

### Assistant

Current strength: assistant management already spans providers, datasets, references, examples, drafts, execution, and evals.

Hand-design direction:

- use the assistant as an inspector and reviewer, not the main designer
- every assistant draft should include confidence, required review actions, and affected objects
- assistant panels should be collapsible side inspectors in context-sensitive views

## First Implementation Slice

Build the first hand-designed slice around the Visualization workspace because it touches all major user decisions without risking core training behavior.

Scope:

1. Add plot intent metadata to rendered plot cards.
2. Create grouped plot sections: Training, Evaluation, Production, Data Quality, Registry Trace.
3. Add a right-side plot inspector for selected plots.
4. Add "pin as baseline" and "pin as candidate" actions in the UI state.
5. Preserve existing endpoints and rendering fallbacks.

Success criteria:

- a user can load plots and immediately understand why each plot exists
- a user can inspect one plot without reading raw JSON first
- a user can select baseline and candidate plots for comparison
- no existing Plotly fallback or artifact browser behavior regresses

## Manual Design Checklist

For each future screen, answer these before writing code:

- What decision does this screen help the ML engineer make?
- What is the active object context?
- What should stay visible while the user changes tabs?
- What is the primary comparison?
- What plot answers the main question?
- What raw evidence must be accessible but visually secondary?
- What action moves the workflow forward?
- What should happen when there is no data?
- What should happen when the backend returns partial data?

## Suggested Near-Term Sequence

1. Hand-design Visualization as the plot decision layer.
2. Refactor Production around the inference pair and retraining pressure.
3. Refactor Training around baseline/candidate run comparison.
4. Add Registry lineage detail views.
5. Connect Assistant review summaries into the right inspector pattern.

