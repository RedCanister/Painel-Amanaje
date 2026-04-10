"""
utils/deployment_utils.py

Model deployment and registry helpers centered on MLflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Mapping, Optional, Sequence

from .io import load_pickle, save_json, save_pickle
from .logging import get_logger
from .mlflow_utils import log_json

logger = get_logger("deployment_utils")

try:
    import mlflow
    from mlflow.exceptions import RestException
    from mlflow.tracking import MLflowClient

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - only used when MLflow is unavailable.
    mlflow = None  # type: ignore[assignment]
    RestException = Exception  # type: ignore[assignment]
    MLflowClient = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


def _require_mlflow() -> None:
    if not MLFLOW_AVAILABLE or mlflow is None or MLflowClient is None:
        raise RuntimeError("MLflow Model Registry is not available in the current environment.")


def _client() -> Any:
    _require_mlflow()
    return MLflowClient()


def _serialize_model_version(model_version: Any) -> Dict[str, Any]:
    return {
        "name": model_version.name,
        "version": str(model_version.version),
        "stage": getattr(model_version, "current_stage", None) or "None",
        "run_id": getattr(model_version, "run_id", None),
        "status": str(getattr(model_version, "status", "")),
        "source": getattr(model_version, "source", None),
    }


def _wait_for_model_version(
    registered_model_name: str,
    version: str,
    timeout_seconds: int = 60,
    poll_interval: float = 2.0,
) -> Dict[str, Any]:
    """
    Poll the registry until the requested model version becomes ready.
    """

    client = _client()
    deadline = time.time() + timeout_seconds
    last_seen = None

    while time.time() < deadline:
        model_version = client.get_model_version(registered_model_name, version)
        last_seen = model_version
        status = str(getattr(model_version, "status", ""))
        if "READY" in status or "FAILED" in status:
            break
        time.sleep(poll_interval)

    if last_seen is None:
        raise RuntimeError(
            f"Could not fetch model version '{registered_model_name}' version '{version}'."
        )

    return _serialize_model_version(last_seen)


def save_model_local(model: Any, path: str) -> str:
    """
    Save a model locally via pickle and return the saved path.
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(model, file_path)
    logger.info("Saved model locally to '%s'.", file_path)
    return str(file_path)


def save_movel_local(model: Any, path: str) -> str:
    """
    Backward-compatible alias for the previous misspelled helper name.
    """

    return save_model_local(model, path)


def load_model_local(path: str) -> Any:
    """
    Load a locally saved pickle model.
    """

    file_path = Path(path)
    model = load_pickle(file_path)
    logger.info("Loaded model from '%s'.", file_path)
    return model


def register_model_from_uri(
    model_uri: str,
    registered_model_name: str,
    await_registration_for: int = 60,
    poll_interval: float = 2.0,
) -> Dict[str, Any]:
    """
    Register a model artifact URI in the MLflow Model Registry.
    """

    _require_mlflow()
    logger.info("Registering model URI '%s' as '%s'.", model_uri, registered_model_name)

    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    except RestException:
        logger.exception("MLflow registry rejected model registration.")
        raise

    return _wait_for_model_version(
        registered_model_name=registered_model_name,
        version=str(model_version.version),
        timeout_seconds=await_registration_for,
        poll_interval=poll_interval,
    )


def get_latest_versions(
    registered_model_name: str,
    stages: Optional[Sequence[str]] = None,
) -> list[Dict[str, Any]]:
    """
    Return the latest model version per stage for a registered model.
    """

    client = _client()
    all_versions = list(client.search_model_versions(f"name='{registered_model_name}'"))
    stage_filter = {stage.lower() for stage in stages or []}

    latest_by_stage: Dict[str, Any] = {}
    for model_version in all_versions:
        stage = getattr(model_version, "current_stage", None) or "None"
        if stage_filter and stage.lower() not in stage_filter:
            continue

        existing = latest_by_stage.get(stage)
        if existing is None or int(model_version.version) > int(existing.version):
            latest_by_stage[stage] = model_version

    result = [
        _serialize_model_version(model_version)
        for model_version in sorted(
            latest_by_stage.values(),
            key=lambda item: int(item.version),
            reverse=True,
        )
    ]
    logger.info("Latest registered versions for '%s': %s", registered_model_name, result)
    return result


def transition_model_stage(
    registered_model_name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = True,
    comment: Optional[str] = None,
) -> None:
    """
    Transition a model version to a target stage.
    """

    client = _client()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )
    if comment:
        client.set_model_version_tag(
            name=registered_model_name,
            version=str(version),
            key="promotion_comment",
            value=comment,
        )

    logger.info(
        "Transitioned model '%s' version %s to stage '%s'.",
        registered_model_name,
        version,
        stage,
    )


def promote_to_production(
    registered_model_name: str,
    version: str,
    comment: Optional[str] = None,
) -> None:
    """
    Promote a model version to the Production stage.
    """

    transition_model_stage(
        registered_model_name=registered_model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
        comment=comment,
    )


def rollback_to_version(
    registered_model_name: str,
    version: str,
    comment: Optional[str] = None,
) -> None:
    """
    Roll back production traffic to a previous model version.
    """

    promote_to_production(registered_model_name, version, comment=comment)
    logger.info("Rolled back model '%s' to version %s.", registered_model_name, version)


def get_model_uri_for_run(run_id: str, artifact_path: str = "model") -> str:
    """
    Build a ``runs:/`` URI for a model artifact logged under a run.
    """

    uri = f"runs:/{run_id}/{artifact_path}"
    logger.info("Constructed model URI '%s'.", uri)
    return uri


def get_model_uri_for_stage(registered_model_name: str, stage: str = "Production") -> str:
    """
    Build a ``models:/`` URI for a registered model stage.
    """

    uri = f"models:/{registered_model_name}/{stage}"
    logger.info("Constructed stage model URI '%s'.", uri)
    return uri


FASTAPI_TEMPLATE = dedent(
    """
    from __future__ import annotations

    from typing import Any

    import pandas as pd
    import mlflow.pyfunc
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()
    MODEL_URI = "{model_uri}"
    model = mlflow.pyfunc.load_model(MODEL_URI)


    class PredictRequest(BaseModel):
        inputs: list[Any]


    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        dataframe = pd.DataFrame(request.inputs)
        predictions = model.predict(dataframe)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        return {{"predictions": predictions}}
    """
).strip() + "\n"


def create_fastapi_serving_file(output_path: str, model_uri: str) -> str:
    """
    Create a minimal FastAPI serving file for an MLflow pyfunc model.
    """

    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(FASTAPI_TEMPLATE.format(model_uri=model_uri), encoding="utf-8")
    logger.info("Created FastAPI serving file at '%s'.", file_path)
    return str(file_path)


def deploy_model_from_run(
    run_id: str,
    artifact_path: str,
    registered_model_name: str,
    promote_to_stage: Optional[str] = "Staging",
    comment: Optional[str] = None,
    await_registration_for: int = 60,
) -> Dict[str, Any]:
    """
    Register a model from a run artifact and optionally promote it to a stage.
    """

    model_uri = get_model_uri_for_run(run_id, artifact_path)
    registration = register_model_from_uri(
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
    )
    version = str(registration["version"])

    if promote_to_stage:
        transition_model_stage(
            registered_model_name=registered_model_name,
            version=version,
            stage=promote_to_stage,
            archive_existing_versions=False,
            comment=comment,
        )

    summary = {
        "registered_model": registered_model_name,
        "version": version,
        "stage": promote_to_stage,
        "model_uri": model_uri,
        "run_id": run_id,
    }
    log_json(summary, filename="deployment_summary.json", artifact_path="deployment")
    logger.info("Deployment summary: %s", summary)
    return summary


def list_registered_models() -> list[Dict[str, Any]]:
    """
    List registered models and their latest versions.
    """

    client = _client()
    result = []
    for model in client.search_registered_models():
        result.append(
            {
                "name": model.name,
                "latest_versions": [
                    _serialize_model_version(version) for version in getattr(model, "latest_versions", [])
                ],
            }
        )

    logger.info("Registered models: %s", [item["name"] for item in result])
    return result


def save_deployment_summary(summary: Mapping[str, Any], output_path: str) -> None:
    """
    Persist a deployment summary to disk.
    """

    save_json(dict(summary), output_path)
