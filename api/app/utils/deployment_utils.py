"""
utils/deployment_utils.py

Model deployment & registry helpers oriented to MLflow + FastAPI

Responsibilites:
- Save / load models (MLflow pyfunc-compatible)
- Register models into MLflow Model Registry
- Promote / transition model stages (None -> staging -> production)
- Rollback to previous model version
- Provide a lightweight FastAPI serving skeleton that loads an MLflow model
and exposes a /predict endpoint (pyfunc)
- Helper to produce a reproducible model URI for artifacts

Notes:
- Assumes MLflow tracking URI & registry are configured via environment or utils/mlflow_utils.
- For production, combine with a model serving platform (Sagemaker, KFServing, BentoML, Docker+Uvicorn).
"""

import os
import time
import tempfile
import json
from typing import Any, Dict, Optional

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MLflowClient

from app.utils.logging import get_logger
from app.utils.io import save_json, save_pickle, load_pickle, ensure_dir
from app.utils.mlflow_utils import log_artifact, log_json

logger = get_logger("deployment_utils")
mlflow_client = MLflowClient()

# Save / Load Helpers

def save_movel_local(pyfunc_model, path: str) -> str:
    """
    Saves a pyfunc-compatible model object to a local path using MLflow.
    Returns the local path.
    Example: mlflow.pyfunc.log_model(...) often is a preferred route,
    but this helper writes a pickle and returns the path for `mlflow.log_artifact`.
    """
    ensure_dir(os.path.dirname(path) or ".")
    try:
        save_pickle(pyfunc_model, path)
        logger.info(f"✅ Model saved locally at {path}")
        return path
    except Exception as e:
        logger.error(f"❌ Failed to save model locally: {e}")
        raise


def load_model_local(path: str):
    """
    Loads a pickle model from local path.
    """
    try:
        model = load_pickle(path)
        logger.info(f"✅ Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from {path}: {e}")
        raise

    
# MLflow Model Registration

def register_model_from_uri(model_uri: str, registered_model_name: str) -> Dict[str, Any]:
    """
    Registers a model artifact (model_uri) under MLflow Model Registry name.
    Returns the registration info (name, version)
    model_uri examples:
    - "runs:/<run_id>/model"
    - path to local artifact if mlflow server can access it
    """
    try:
        logger.info(f"📦 Registering model URI '{model_uri}' as '{registered_model_name}'.")
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        reg_info = {"name": registered_model_name, "version": result.version}
        logger.info(f"✅ Registered model: {reg_info}")
        return reg_info
    except RestException as e:
        logger.error(f"❌ MLflow RestException during register: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        raise

def get_latest_versions(registered_model_name: str, stages: Optional[list] = None) -> list:
    """
    Returns list of latest model versions info for a given model name.
    If `stages` provided (e.g. ["Production", "Staging"]) it filters to them.
    """
    try:
        versions = mlflow_client.get_latest_versions(name=registered_model_name, stages=stages or [])
        result = [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id, "status": v.status} for v in versions]
        logger.info(f"🔎 Latest versions for '{registered_model_name}': {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Could not fetch versions for {registered_model_name}: {e}")
        raise


def transition_model_stage(
    registered_model_name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = True,
    comment: Optional[str] = None,
) -> None:
    """
    Transition a model version to a target stage (e.g., "Staging", "Production", "Archived").
    If archive_existing_versions=True, existing versions at that stage will be archived.
    """
    try:
        logger.info(f"🔁 Transitioning model {registered_model_name} v{version} -> {stage}")
        mlflow_client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )
        if comment:
            mlflow_client.set_model_version_tag(registered_model_name, version, "promotion_comment", comment)
        logger.info(f"✅ Transitioned model {registered_model_name} v{version} to {stage}")

    except Exception as e:
        logger.error(f"❌ Failed to transition model stage: {e}")
        raise

def promote_to_production(
    registered_model_name: str,
    version: str,
    comment: Optional[str] = None
) -> None:
    """
    Convenience wrapper to move version -> Production.
    """
    transition_model_stage(registered_model_name, version, "Production", archive_existing_versions=True, comment=comment)


def rollback_to_version(registered_model_name: str, version: str, comment: Optional[str] = None) -> None:
    """
    Rollback production to a specific version (move that version to Production).
    This will archive the currently promoted version automatically (archive_existing_versions=True).
    """
    promote_to_production(registered_model_name, version, comment=comment)
    logger.info(f"↩️ Rolled back {registered_model_name} to version {version}")


# Model URI Helpers

def get_model_uri_for_run(run_id: str, artifact_path: str = "model") -> str:
    """
    Build a 'runs:/' URI for the artifact produced by a run.
    Commonly used when logging model artifat under 'model' path in the run.
    """
    uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"🔗 Constructed model URI: {uri}")
    return uri

# Simple FastAPI Serving Skeleton

FASTAPI_TEMPLATE = """
    # Minimal serving app generated by utils/deploymeny_utils
    from fastapi import FastAPI
    from pydantic import BaseModel
    import mlflow.pyfunc


    app = FastAPI()
    MODEL_URI = "{model_uri}" # e.g. "models:/MyModel/Production" or "runs:/<run_id>/model

    # Load model at startup
    model = mlflow.pyfunc.load_model(MODEL_URI)

    class PredictRequest(BaseModel):
        inputs: list # list of records (dict) or numeric lists depending on model

    @app.post("/predict")
    def predict(req: PredictRequest)
        # Adapt the input format to what your pyfunc expects.
        inputs = req.inputs
        preds = model.predict(inputs)
        return {{"predictions": preds.tolist() if hasattr(preds, 'tolist') else preds}}
"""

def create_fastapi_serving_file(output_path: str, model_uri: str) -> str:
    """
    Writes a minimal FastAPI serving file that loads the Mlflow model URI.
    Returns the path to the saved serving app file.
    Note: user should adapt the input/output schema and containerize for production.
    """
    ensure_dir(os.path.dirname(output_path) or ".")
    content = FASTAPI_TEMPLATE.format(model_uri=model_uri)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info(f"🧩 FastAPI serving template written to {output_path}")
    return output_path


# High-level Deploymeny Flow Helpers

def deploy_model_from_run(
    run_id: str,
    artifact_path: str,
    registered_model_name: str,
    promote_to_stage: Optional[str] = "Staging",
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper:
    - constructs runs:/ URI for the trained model
    - registers it to the registry under registered_model_name
    - optionally promotes it to a stage (Staging/Production)
    Returns registration metadata.
    """
    model_uri = get_model_uri_for_run(run_id, artifact_path)
    reg = register_model_from_uri(model_uri, registered_model_name)
    version = str(reg["version"])

    if promote_to_stage:
        transition_model_stage(registered_model_name, version, promote_to_stage, 
        archive_existing_versions=False, comment=comment)

    summary = {
        "registered_model": registered_model_name,
        "version": version,
        "model_uri": model_uri,
    }
    logger.info(f"🚀 Deployment summary: {summary}")
    return summary


# Safety / Utility

def list_registered_models() -> list:
    """
    List registered models names and summary info.
    """
    try:
        models = mlflow.client.list_registered_models()
        out = [{"name": m.name, "latest_versions": [v.version for v in m.latest_versions]} for m in models]
        logger.info(f"📚 Registered models: {[m['name'] for m in out]}")
        return out
    except Exception as e:
        logger.error(f"❌ Failed to list registered models: {e}")
        raise
