from __future__ import annotations

import textwrap
from typing import Protocol

from app.models.assistant_objects import AssistantDraftRequest, WorkflowDraft


class AssistantProvider(Protocol):
    name: str

    def create_draft(self, request: AssistantDraftRequest) -> WorkflowDraft:
        ...


class LocalWorkflowProvider:
    name = "local"

    def create_draft(self, request: AssistantDraftRequest) -> WorkflowDraft:
        draft_type = request.target_type or _infer_draft_type(request)
        if draft_type == "model_generation":
            return _model_generation_draft(request)
        if draft_type == "feature_operations":
            return _feature_operations_draft(request)
        if draft_type == "training_run":
            return _training_draft(request)
        if draft_type == "study":
            return _study_draft(request)
        if draft_type == "registry_object":
            return _registry_draft(request)
        return _dataset_generation_draft(request)


def get_assistant_provider(name: str | None = None) -> AssistantProvider:
    normalized = (name or "local").strip().lower()
    if normalized != "local":
        raise ValueError(f"Assistant provider '{normalized}' is not configured. Available providers: local")
    return LocalWorkflowProvider()


def _infer_draft_type(request: AssistantDraftRequest) -> str:
    context = dict(request.context or {})
    prompt = f"{request.workflow_goal or ''} {request.prompt} {context.get('operationId') or ''}".lower()
    if any(token in prompt for token in ("feature", "column", "rename", "transform", "fill missing")):
        return "feature_operations"
    if any(token in prompt for token in ("model", "joblib", "sklearn", "estimator", "random forest", "regressor")):
        return "model_generation"
    if any(token in prompt for token in ("train", "training", "fit model", "launch run")):
        return "training_run"
    if any(token in prompt for token in ("study", "optuna", "hyperparameter", "optimization")):
        return "study"
    if any(token in prompt for token in ("registry", "register", "object", "metadata")):
        return "registry_object"
    return "dataset_generation"


def _dataset_generation_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    prompt_comment = "\n".join(f"# {line.strip()}" for line in request.prompt.splitlines() if line.strip())
    code = textwrap.dedent(
        f"""
        # Assistant dataset draft
        {prompt_comment}

        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        row_count = 100
        df = pd.DataFrame({{
            "feature_a": rng.normal(10, 2, row_count).round(4),
            "feature_b": rng.integers(0, 5, row_count),
            "segment": rng.choice(["baseline", "growth", "risk"], row_count),
        }})
        df["target"] = (df["feature_a"] * 0.7 + df["feature_b"] * 1.5 + rng.normal(0, 0.5, row_count)).round(4)

        csv_text = df.to_csv(index=False)
        name = "assistant_generated_dataset"
        description = "Dataset drafted by the Amanaje assistant."
        object_type = "dataset"
        dataset_type = "dataset"
        version = 1
        connection_string = ""
        """
    ).strip()
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="dataset_generation",
        title="Generated Dataset Draft",
        summary="Creates a tabular dataset and exposes it as csv_text for the existing upload/create flow.",
        prompt=request.prompt,
        provider=request.provider,
        code=code,
        execution_profile="dataset_generation",
        materialized_outputs=["csv_text", "name", "description", "object_type", "dataset_type"],
        assumptions=[
            "The generated dataset is synthetic.",
            "The draft only prepares csv_text; it does not write files or registry rows.",
        ],
        risks=["Synthetic data may not represent the target production distribution."],
        context=request.context,
        next_actions=["Review safety report", "Approve draft", "Execute through the guarded execution path"],
    )


def _model_generation_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    prompt_comment = "\n".join(f"# {line.strip()}" for line in request.prompt.splitlines() if line.strip())
    code = textwrap.dedent(
        f"""
        # Assistant model draft
        {prompt_comment}

        import base64
        import io
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=80, random_state=42)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        joblib_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
        name = "assistant_generated_model"
        description = "Scikit-learn model artifact drafted by the Amanaje assistant."
        object_type = "learning_model"
        path = "generated/assistant_generated_model.joblib"
        version = 1
        model_type = "supervised_model"
        parameters = {{"framework": "sklearn", "estimator_class": "RandomForestRegressor", "n_estimators": 80, "random_state": 42}}
        metrics = {{"task": "regression", "status": "untrained_template"}}
        reference_data = "assistant_generated_dataset.csv"
        input_features = ["feature_a", "feature_b"]
        output_features = ["target"]
        is_trained = False
        is_tested = False
        is_deployed = False
        """
    ).strip()
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="model_generation",
        title="Generated Model Draft",
        summary="Creates a scikit-learn joblib artifact and matching LearningModel metadata for the Create workspace.",
        prompt=request.prompt,
        provider=request.provider,
        code=code,
        execution_profile="model_generation",
        materialized_outputs=["joblib_bytes", "name", "parameters", "metrics", "input_features", "output_features"],
        assumptions=[
            "The first assistant-generated model artifact is a scikit-learn joblib template.",
            "The generated artifact is intended for registry creation, not immediate production deployment.",
        ],
        risks=["The model is not trained until it is paired with a dataset and submitted to the training workflow."],
        context=request.context,
        next_actions=["Review safety report", "Run through guarded execution", "Create the model through the existing upload backend"],
    )


def _feature_operations_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    operations = list(context.get("transforms") or context.get("operations") or [])
    if not operations:
        operations = [{"type": "limit", "value": int(context.get("limitRows") or context.get("limit_rows") or 200)}]
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="feature_operations",
        title="Feature Transformation Draft",
        summary="Prepares visual feature operations for /features/preview before materialization.",
        prompt=request.prompt,
        provider=request.provider,
        execution_profile="feature_operations",
        operations=operations,
        artifacts={"preview_endpoint": "/features/preview", "materialize_endpoint": "/features/materialize"},
        assumptions=["Existing Feature Workspace operation schemas remain the source of truth."],
        risks=["Column names must match the selected dataset before preview can succeed."],
        context=context,
        next_actions=["Preview transforms", "Approve materialization", "Register the derived dataset"],
    )


def _training_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    training_request = {
        "modelId": context.get("modelId") or context.get("model_id"),
        "datasetId": context.get("datasetId") or context.get("dataset_id"),
        "inputType": "Assistant",
        "studyId": context.get("studyId") or context.get("study_id"),
        "parameters": dict(context.get("parameters") or {}),
    }
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="training_run",
        title="Training Run Draft",
        summary="Prepares a reviewed training submission using the existing async run ledger.",
        prompt=request.prompt,
        provider=request.provider,
        execution_profile="training_run",
        training_request=training_request,
        assumptions=["Training will use the selected registry model and dataset after preflight validation."],
        risks=["Training cannot start until modelId and datasetId are present and schema preflight passes."],
        context=context,
        next_actions=["Review selected model/dataset", "Approve training submission", "Track /runs/get/{run_id}"],
    )


def _study_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    study_request = {
        "studyId": context.get("studyId") or context.get("study_id"),
        "nTrials": context.get("nTrials") or context.get("n_trials") or 10,
        "plotResults": context.get("plotResults", True),
        "parameters": dict(context.get("parameters") or {}),
    }
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="study",
        title="Optimization Study Draft",
        summary="Prepares a reviewed Optuna study request for the existing study runner.",
        prompt=request.prompt,
        provider=request.provider,
        execution_profile="study",
        study_request=study_request,
        assumptions=["The existing StudyModel registry record defines the model and dataset pairing."],
        risks=["Search-space quality depends on the current StudyModel parameters."],
        context=context,
        next_actions=["Review study settings", "Approve optimization", "Track the study run ledger"],
    )


def _registry_draft(request: AssistantDraftRequest) -> WorkflowDraft:
    context = dict(request.context or {})
    registry_type = str(context.get("registryType") or context.get("modelType") or "DatasetModel")
    payload = dict(context.get("registry_payload") or {})
    if not payload:
        payload = _default_registry_payload(registry_type, context)
    return WorkflowDraft(
        session_id=request.session_id,
        draft_type="registry_object",
        title=f"{registry_type} Draft",
        summary="Prepares registry metadata for review before the user submits the manual create form.",
        prompt=request.prompt,
        provider=request.provider,
        registry_type=registry_type,
        registry_payload=payload,
        form_payload=payload,
        execution_profile="registry_object",
        materialized_outputs=["registry_payload"],
        assumptions=["The target registry route will perform final Pydantic and ORM validation."],
        risks=["References to datasets, models, or studies must exist before the object can be saved."],
        context=context,
        next_actions=["Review metadata", "Approve persistence", "Check dependency report before deletion"],
    )


def _slug(value: object, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    text = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in text.split("_") if part) or fallback


def _base_registry_payload(registry_type: str, context: dict) -> dict:
    name = _slug(context.get("name") or context.get("objectName"), f"assistant_{registry_type.lower()}")
    return {
        "name": name,
        "description": context.get("description") or f"{registry_type} drafted by the Amanaje assistant.",
        "size": float(context.get("size") or context.get("objectSize") or 0),
        "date": context.get("date"),
        "version": int(context.get("version") or context.get("objectVersion") or 1),
        "history": [{"operation": "assistant_draft"}],
    }


def _default_registry_payload(registry_type: str, context: dict) -> dict:
    payload = _base_registry_payload(registry_type, context)
    name = payload["name"]

    if registry_type == "LearningModel":
        return {
            **payload,
            "object_type": "learning_model",
            "path": context.get("path") or context.get("objectPath") or f"models/{name}.joblib",
            "model_type": context.get("model_type") or "supervised_model",
            "parameters": dict(context.get("parameters") or {"framework": "sklearn", "estimator_class": "RandomForestRegressor"}),
            "metrics": dict(context.get("metrics") or {"status": "draft"}),
            "reference_data": context.get("reference_data") or context.get("referenceData") or "",
            "input_features": list(context.get("input_features") or ["feature_a", "feature_b"]),
            "output_features": list(context.get("output_features") or ["target"]),
            "is_trained": bool(context.get("is_trained") or False),
            "is_tested": bool(context.get("is_tested") or False),
            "is_deployed": bool(context.get("is_deployed") or False),
        }

    if registry_type == "InferenceModel":
        learning_model_id = context.get("learning_model_id") or context.get("learningModelId") or context.get("model_id") or 1
        dataset_id = context.get("dataset_id") or context.get("datasetId") or 1
        return {
            **payload,
            "object_type": "inference_model",
            "path": context.get("path") or context.get("objectPath") or f"runtime_artifacts/inference/{name}.json",
            "learning_model_id": int(learning_model_id),
            "dataset_id": int(dataset_id),
            "input_features": list(context.get("input_features") or ["feature_a", "feature_b"]),
            "output_features": list(context.get("output_features") or ["target"]),
            "inference_params": dict(context.get("inference_params") or {"status": "draft", "source": "assistant"}),
        }

    if registry_type == "StudyModel":
        learning_model_id = context.get("learning_model_id") or context.get("learningModelId") or context.get("model_id") or 1
        dataset_id = context.get("dataset_id") or context.get("datasetId") or 1
        return {
            **payload,
            "object_type": "study_model",
            "path": context.get("path") or context.get("objectPath") or f"runtime_artifacts/optuna/{name}.json",
            "learning_model_id": int(learning_model_id),
            "dataset_id": int(dataset_id),
            "sampler": context.get("sampler") or "TPESampler",
            "objective": context.get("objective") or "maximize_accuracy",
            "best_trial": dict(context.get("best_trial") or {}),
            "best_params": dict(context.get("best_params") or {}),
            "study_params": dict(context.get("study_params") or {"n_trials": 20, "direction": "maximize", "framework": "sklearn"}),
        }

    if registry_type == "CodeModel":
        return {
            **payload,
            "object_type": "code_model",
            "path": context.get("path") or context.get("objectPath") or f"generated/{name}.py",
            "variables": dict(context.get("variables") or {"source": "assistant"}),
            "code": dict(context.get("code") or {"language": "python", "script": "# Assistant-drafted code document"}),
        }

    return {
        **payload,
        "object_type": "dataset",
        "path": context.get("path") or context.get("objectPath") or f"data/datasets/{name}.csv",
        "dataset_type": context.get("dataset_type") or "dataset",
        "shape": list(context.get("shape") or [100, 4]),
        "has_features": bool(context.get("has_features") if "has_features" in context else True),
        "features_list": list(context.get("features_list") or ["feature_a", "feature_b", "segment", "target"]),
        "connection_string": context.get("connection_string") or "",
    }
