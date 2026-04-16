"""
utils/mlflow_utils.py

Utility functions and decorators to simplify MLflow experiment tracking.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional
from urllib.parse import urlparse
from uuid import uuid4

from .logging import get_logger
from .serialization import safe_log_params, to_json

logger = get_logger("mlflow_utils")


def _resolve_workspace_dir(env_key: str, default_path: Path) -> Path:
    override = os.getenv(env_key)
    resolved = Path(override) if override else default_path
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


_WORKSPACE_TMP_DIR = _resolve_workspace_dir("MLFLOW_TMP_DIR", Path.cwd() / ".mlflow_tmp")
_LOCAL_MLFLOW_DIR = _resolve_workspace_dir("MLFLOW_LOCAL_DIR", Path.cwd() / "mlruns")
_LOCAL_MLFLOW_URI = _LOCAL_MLFLOW_DIR.resolve().as_uri()
_ARTIFACT_FALLBACK_DIR = Path.cwd() / "runtime_artifacts" / "mlflow_fallback"
_ARTIFACT_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
_INVALID_MLFLOW_KEY_PATTERN = re.compile(r"[^0-9A-Za-z_\-\. :/]")

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - only used when MLflow is unavailable.
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


def _require_mlflow() -> Any:
    if not MLFLOW_AVAILABLE or mlflow is None:
        raise RuntimeError("MLflow is not available in the current environment.")
    return mlflow


def sanitize_mlflow_key(key: Any) -> str:
    """
    Normalize metric, parameter, and tag keys to MLflow-safe names.
    """

    normalized = _INVALID_MLFLOW_KEY_PATTERN.sub("_", str(key or "")).strip()
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed"


def _fallback_artifact_root(artifact_path: Optional[str] = None) -> Path:
    run_id = active_run_id() or "no_active_run"
    destination = _ARTIFACT_FALLBACK_DIR / run_id
    if artifact_path:
        for part in Path(str(artifact_path)).parts:
            if part in {"", ".", ".."}:
                continue
            destination /= part
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _copy_artifact_to_fallback(
    source: Path,
    artifact_path: Optional[str] = None,
    *,
    destination_name: Optional[str] = None,
) -> Optional[str]:
    try:
        destination = _fallback_artifact_root(artifact_path) / (destination_name or source.name)
        shutil.copy2(source, destination)
        return str(destination)
    except Exception:
        logger.exception("Unable to create a fallback copy for MLflow artifact '%s'.", source)
        return None


def _copy_artifact_dir_to_fallback(source_dir: Path, artifact_path: Optional[str] = None) -> Optional[str]:
    try:
        destination = _fallback_artifact_root(artifact_path) / source_dir.name
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
        return str(destination)
    except Exception:
        logger.exception("Unable to create a fallback copy for MLflow artifact directory '%s'.", source_dir)
        return None


def _normalize_mlflow_uri(uri: Optional[str], *, kind: str) -> Optional[str]:
    if not uri:
        return uri

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme not in {"http", "https"} or not parsed_uri.hostname:
        return uri

    try:
        socket.gethostbyname(parsed_uri.hostname)
        return uri
    except OSError:
        logger.warning(
            "MLflow %s host '%s' is not reachable from this environment. Using local store '%s'.",
            kind,
            parsed_uri.hostname,
            _LOCAL_MLFLOW_URI,
        )
        return _LOCAL_MLFLOW_URI


def ensure_tracking_uri() -> str:
    """
    Ensure the active MLflow tracking URI is reachable and return the normalized value.
    """

    client = _require_mlflow()
    current_uri = client.get_tracking_uri()
    normalized_uri = _normalize_mlflow_uri(current_uri, kind="tracking") or _LOCAL_MLFLOW_URI
    if normalized_uri != current_uri:
        client.set_tracking_uri(normalized_uri)
    return client.get_tracking_uri()


def configure_tracking(
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
) -> None:
    """
    Configure MLflow tracking and registry URIs when values are provided.
    """

    client = _require_mlflow()
    if tracking_uri:
        client.set_tracking_uri(_normalize_mlflow_uri(tracking_uri, kind="tracking"))
    if registry_uri:
        client.set_registry_uri(_normalize_mlflow_uri(registry_uri, kind="registry"))

    ensure_tracking_uri()


def get_client() -> Any:
    """
    Return an MLflow client instance.
    """

    _require_mlflow()
    if MlflowClient is None:
        raise RuntimeError("MLflow client is not available in the current environment.")
    return MlflowClient(tracking_uri=ensure_tracking_uri())


def is_active_run() -> bool:
    """
    Return ``True`` when an MLflow run is currently active.
    """

    if not MLFLOW_AVAILABLE or mlflow is None:
        return False
    return mlflow.active_run() is not None


def active_run_id() -> Optional[str]:
    """
    Return the current active run id, if any.
    """

    if not is_active_run():
        return None
    return mlflow.active_run().info.run_id  # type: ignore[union-attr]


def list_experiments(max_results: int = 50) -> list[Dict[str, Any]]:
    """
    Return a normalized list of experiments sorted by most recent update.
    """

    client = get_client()
    experiments = client.search_experiments(max_results=max_results)
    normalized = []
    for experiment in experiments:
        normalized.append(
            {
                "id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "creation_time": getattr(experiment, "creation_time", None),
                "last_update_time": getattr(experiment, "last_update_time", None),
                "tags": dict(getattr(experiment, "tags", {}) or {}),
            }
        )

    normalized.sort(
        key=lambda experiment: (
            experiment.get("last_update_time") or 0,
            experiment.get("creation_time") or 0,
        ),
        reverse=True,
    )
    return normalized


def get_experiment_summary(
    experiment_id: str,
    max_runs: int = 20,
) -> Dict[str, Any]:
    """
    Return a normalized experiment payload including recent runs.
    """

    client = get_client()
    experiment = client.get_experiment(str(experiment_id))
    runs = client.search_runs(
        experiment_ids=[str(experiment_id)],
        max_results=max_runs,
        order_by=["attributes.start_time DESC"],
    )

    run_summaries = []
    for run in runs:
        run_summaries.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "params": dict(run.data.params),
                "metrics": {key: float(value) for key, value in run.data.metrics.items()},
                "tags": dict(run.data.tags),
            }
        )

    return {
        "id": experiment.experiment_id,
        "name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
        "creation_time": getattr(experiment, "creation_time", None),
        "last_update_time": getattr(experiment, "last_update_time", None),
        "tags": dict(getattr(experiment, "tags", {}) or {}),
        "runs": run_summaries,
    }


def start_run(
    experiment_name: Optional[str],
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> Any:
    """
    Start a new MLflow run.

    If ``experiment_name`` is provided, the experiment is created on demand and
    selected before the run starts.
    """

    client = _require_mlflow()
    ensure_tracking_uri()

    def _start_current_run() -> Any:
        if experiment_name:
            client.set_experiment(experiment_name)
        run = client.start_run(run_name=run_name, nested=nested)
        if tags:
            client.set_tags(tags)
        logger.info(
            "Started MLflow run '%s' (%s).",
            run.info.run_name or run_name or "unnamed",
            run.info.run_id,
        )
        return run

    try:
        return _start_current_run()
    except Exception:
        logger.warning(
            "MLflow tracking URI was unavailable. Falling back to local store '%s'.",
            _LOCAL_MLFLOW_URI,
            exc_info=True,
        )
        client.set_tracking_uri(_LOCAL_MLFLOW_URI)
        return _start_current_run()


def end_run(status: str = "FINISHED") -> None:
    """
    Safely end the active MLflow run.
    """

    if not MLFLOW_AVAILABLE or mlflow is None:
        return

    if not mlflow.active_run():
        return

    try:
        mlflow.end_run(status=status.upper())
        logger.info("Ended MLflow run with status '%s'.", status.upper())
    except Exception:
        logger.exception("Failed to end the active MLflow run.")
        raise


def set_tags(tags: Mapping[str, str]) -> None:
    """
    Set MLflow tags on the active run.
    """

    if not tags:
        return

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping tag logging.")
        return

    safe_tags = {sanitize_mlflow_key(key): "" if value is None else str(value) for key, value in dict(tags).items()}
    try:
        mlflow.set_tags(safe_tags)
        logger.info("Logged %d MLflow tags.", len(safe_tags))
    except Exception:
        logger.exception("Failed to log MLflow tags.")


def log_params(params: Any) -> None:
    """
    Log parameters to the active MLflow run.
    """

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping parameter logging.")
        return

    safe_params = safe_log_params(params)
    if not safe_params:
        return

    normalized_params = {sanitize_mlflow_key(key): value for key, value in safe_params.items()}
    try:
        mlflow.log_params(normalized_params)
        logger.info("Logged %d parameters to MLflow.", len(normalized_params))
    except Exception:
        logger.exception("Failed to log MLflow parameters.")


def log_metrics(metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
    """
    Log numeric metrics to the active MLflow run.
    """

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping metric logging.")
        return

    safe_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue

        normalized_key = sanitize_mlflow_key(key)

        if isinstance(value, bool):
            safe_metrics[normalized_key] = float(int(value))
            continue

        if isinstance(value, (int, float)):
            safe_metrics[normalized_key] = float(value)
            continue

        try:
            safe_metrics[normalized_key] = float(value)
        except (TypeError, ValueError):
            logger.debug("Skipping non-numeric MLflow metric '%s'.", key)

    if not safe_metrics:
        return

    try:
        mlflow.log_metrics(safe_metrics, step=step)
        logger.info("Logged %d metrics to MLflow.", len(safe_metrics))
    except Exception:
        logger.exception("Failed to log MLflow metrics.")


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a file artifact to the active MLflow run.
    """

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping artifact logging.")
        return

    artifact = Path(path)
    if not artifact.exists():
        logger.warning("Artifact path does not exist: %s", artifact)
        return

    try:
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
        logger.info("Logged artifact '%s'.", artifact)
    except Exception:
        fallback_path = _copy_artifact_to_fallback(artifact, artifact_path=artifact_path)
        logger.exception(
            "Failed to log artifact '%s' to MLflow. Fallback copy: %s",
            artifact,
            fallback_path or "unavailable",
        )


def log_artifacts(path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a directory of artifacts to the active MLflow run.
    """

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping artifact directory logging.")
        return

    artifact_dir = Path(path)
    if not artifact_dir.exists():
        logger.warning("Artifact directory does not exist: %s", artifact_dir)
        return

    try:
        mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_path)
        logger.info("Logged artifact directory '%s'.", artifact_dir)
    except Exception:
        fallback_path = _copy_artifact_dir_to_fallback(artifact_dir, artifact_path=artifact_path)
        logger.exception(
            "Failed to log artifact directory '%s' to MLflow. Fallback copy: %s",
            artifact_dir,
            fallback_path or "unavailable",
        )


def log_json(
    data: Any,
    filename: str = "data_snapshot.json",
    artifact_path: Optional[str] = None,
) -> None:
    """
    Serialize data to JSON and log it as an MLflow artifact.
    """

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping JSON logging.")
        return

    temp_file = _WORKSPACE_TMP_DIR / f"{uuid4().hex}_{Path(filename).name}"
    logged_to_mlflow = False
    try:
        temp_file.write_text(to_json(data), encoding="utf-8")
        try:
            mlflow.log_artifact(str(temp_file), artifact_path=artifact_path)
            logged_to_mlflow = True
        except Exception:
            fallback_path = _copy_artifact_to_fallback(
                temp_file,
                artifact_path=artifact_path,
                destination_name=Path(filename).name,
            )
            logger.exception(
                "Failed to log JSON artifact '%s' to MLflow. Fallback copy: %s",
                filename,
                fallback_path or "unavailable",
            )
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not remove temporary MLflow JSON artifact '%s'.", temp_file)

    if logged_to_mlflow:
        logger.info("Logged JSON artifact '%s'.", filename)


def log_text(
    text: str,
    filename: str = "notes.txt",
    artifact_path: Optional[str] = None,
) -> None:
    """
    Log plain text as an MLflow artifact.
    """

    if not is_active_run():
        logger.warning("No active MLflow run found. Skipping text logging.")
        return

    temp_file = _WORKSPACE_TMP_DIR / f"{uuid4().hex}_{Path(filename).name}"
    logged_to_mlflow = False
    try:
        temp_file.write_text(text, encoding="utf-8")
        try:
            mlflow.log_artifact(str(temp_file), artifact_path=artifact_path)
            logged_to_mlflow = True
        except Exception:
            fallback_path = _copy_artifact_to_fallback(
                temp_file,
                artifact_path=artifact_path,
                destination_name=Path(filename).name,
            )
            logger.exception(
                "Failed to log text artifact '%s' to MLflow. Fallback copy: %s",
                filename,
                fallback_path or "unavailable",
            )
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not remove temporary MLflow text artifact '%s'.", temp_file)

    if logged_to_mlflow:
        logger.info("Logged text artifact '%s'.", filename)


@contextmanager
def managed_run(
    experiment_name: Optional[str],
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> Iterator[Any]:
    """
    Context manager that owns an MLflow run lifecycle.
    """

    run = start_run(experiment_name, run_name=run_name, tags=tags, nested=nested)
    try:
        yield run
    except Exception:
        end_run("FAILED")
        raise
    else:
        end_run("FINISHED")


@contextmanager
def resume_run(run_id: Optional[str]) -> Iterator[bool]:
    """
    Resume an existing MLflow run when possible and yield whether tracking is active.
    """

    if not run_id or not MLFLOW_AVAILABLE or mlflow is None:
        yield False
        return

    ensure_tracking_uri()

    current_run = mlflow.active_run()
    if current_run is not None:
        yield current_run.info.run_id == run_id
        return

    resumed_run = None
    try:
        resumed_run = mlflow.start_run(run_id=run_id)
        logger.info("Resumed MLflow run '%s'.", run_id)
        yield True
    except Exception:
        logger.exception("Failed to resume MLflow run '%s'.", run_id)
        yield False
    finally:
        if resumed_run is not None and mlflow.active_run() is not None:
            active = mlflow.active_run()
            if active is not None and active.info.run_id == run_id:
                try:
                    mlflow.end_run(status="FINISHED")
                    logger.info("Closed resumed MLflow run '%s'.", run_id)
                except Exception:
                    logger.exception("Failed to close resumed MLflow run '%s'.", run_id)


def track_experiment(
    experiment_name: Optional[str],
    run_name: Optional[str] = None,
    params: Optional[Any] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> Callable:
    """
    Decorator that wraps a function in an MLflow run.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with managed_run(
                experiment_name=experiment_name,
                run_name=run_name or func.__name__,
                tags=tags,
                nested=nested,
            ):
                if params is not None:
                    log_params(params)
                result = func(*args, **kwargs)
                return result

        wrapper.__name__ = getattr(func, "__name__", "tracked_function")
        wrapper.__doc__ = getattr(func, "__doc__")
        wrapper.__module__ = getattr(func, "__module__")
        return wrapper

    return decorator
