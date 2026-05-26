"""
utils/training.py

Training helpers that standardize scikit-learn and PyTorch workflows across
the project.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - only used when numpy is unavailable.
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

from .io import save_json
from .logging import get_logger
from .metrics import evaluate_and_log_metrics
from .mlflow_utils import (
    active_run_id,
    end_run,
    is_active_run,
    log_json,
    log_metrics,
    log_params,
    start_run,
)

logger = get_logger("training")

REGRESSION_METRICS = {"rmse", "mse", "mae", "mape", "r2"}
CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1"}
UNSUPERVISED_TASK_TYPES = {"clustering", "clusterer", "transformer", "outlier", "unsupervised", "estimator"}


def _require_numpy() -> None:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("numpy is required for training utilities.")


@dataclass
class TrainingResult:
    """
    Normalized result object shared by training-related utilities.
    """

    model: Any
    framework: str
    metrics: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, list[float]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    artifact_paths: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_model: bool = False) -> Dict[str, Any]:
        """
        Convert the result into a JSON-serializable dictionary.
        """

        payload = {
            "framework": self.framework,
            "metrics": dict(self.metrics),
            "history": {key: list(values) for key, values in self.history.items()},
            "parameters": dict(self.parameters),
            "run_id": self.run_id,
            "artifact_paths": list(self.artifact_paths),
            "metadata": dict(self.metadata),
        }
        if include_model:
            payload["model"] = repr(self.model)
        return payload


def _maybe_start_tracking(
    experiment_name: Optional[str],
    run_name: Optional[str],
    tags: Optional[Mapping[str, str]],
    log_to_mlflow: bool,
) -> tuple[bool, Optional[str]]:
    """
    Start an MLflow run when requested and when no run is already active.
    """

    if not log_to_mlflow or not experiment_name:
        return False, active_run_id()

    if is_active_run():
        return False, active_run_id()

    run = start_run(experiment_name, run_name=run_name, tags=dict(tags or {}))
    return True, run.info.run_id


def _finalize_tracking(started_run: bool, failed: bool = False) -> None:
    """
    End an MLflow run only when it was started by the current helper.
    """

    if started_run:
        end_run("FAILED" if failed else "FINISHED")


def _infer_task_type(y: Any) -> str:
    """
    Infer whether a target array represents a regression or classification task.
    """

    _require_numpy()
    values = np.asarray(y)
    if values.dtype.kind in {"b", "O", "S", "U"}:
        return "classification"

    flat = values.reshape(-1)
    if flat.size == 0:
        return "regression"

    unique_values = np.unique(flat)
    if np.allclose(flat, flat.astype(int), equal_nan=True) and unique_values.size <= max(20, int(flat.size * 0.05) or 2):
        return "classification"
    return "regression"


def _constructor_param_names(model_or_factory: Any) -> set[str] | None:
    try:
        signature = inspect.signature(model_or_factory)
    except (TypeError, ValueError):
        return None

    names: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            names.add(name)
    return names


def _estimator_param_names(estimator_or_factory: Any) -> set[str] | None:
    if hasattr(estimator_or_factory, "get_params"):
        try:
            return set(estimator_or_factory.get_params(deep=True).keys())
        except Exception:
            pass
    if inspect.isclass(estimator_or_factory) or callable(estimator_or_factory):
        return _constructor_param_names(estimator_or_factory)
    return None


def _split_known_params(
    estimator_or_factory: Any,
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str] | None]:
    valid_names = _estimator_param_names(estimator_or_factory)
    if valid_names is None:
        return dict(params), {}, None

    accepted = {key: value for key, value in params.items() if key in valid_names}
    ignored = {key: value for key, value in params.items() if key not in valid_names}
    return accepted, ignored, sorted(valid_names)


def _attach_estimator_build_details(
    estimator: Any,
    *,
    accepted_params: Mapping[str, Any],
    ignored_params: Mapping[str, Any],
    valid_param_names: list[str] | None,
) -> None:
    try:
        setattr(
            estimator,
            "_amanaje_build_details",
            {
                "accepted_estimator_params": dict(accepted_params),
                "ignored_estimator_params": dict(ignored_params),
                "valid_estimator_params": list(valid_param_names or []),
            },
        )
    except Exception:
        pass


def _apply_estimator_params(estimator: Any, params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str] | None]:
    accepted, ignored, valid_names = _split_known_params(estimator, params)
    if ignored:
        logger.warning("Ignoring invalid estimator parameters for %s: %s", estimator, sorted(ignored))
    if accepted and hasattr(estimator, "set_params"):
        estimator.set_params(**accepted)
    return accepted, ignored, valid_names


def _estimator_capabilities(estimator: Any) -> dict[str, bool]:
    return {
        "fit": hasattr(estimator, "fit"),
        "predict": hasattr(estimator, "predict"),
        "predict_proba": hasattr(estimator, "predict_proba"),
        "fit_predict": hasattr(estimator, "fit_predict"),
        "transform": hasattr(estimator, "transform"),
        "fit_transform": hasattr(estimator, "fit_transform"),
        "score": hasattr(estimator, "score"),
        "score_samples": hasattr(estimator, "score_samples"),
        "decision_function": hasattr(estimator, "decision_function"),
    }


def _estimator_family(estimator: Any, task_type: Optional[str] = None) -> str:
    normalized_task = str(task_type or "").strip().lower()
    if normalized_task in {"classification", "classifier"}:
        return "classifier"
    if normalized_task in {"regression", "regressor"}:
        return "regressor"
    if normalized_task in UNSUPERVISED_TASK_TYPES:
        if normalized_task == "clusterer":
            return "clustering"
        return normalized_task

    estimator_type = str(getattr(estimator, "_estimator_type", "") or "").strip().lower()
    if estimator_type == "classifier":
        return "classifier"
    if estimator_type == "regressor":
        return "regressor"
    if estimator_type in {"clusterer", "cluster"}:
        return "clustering"
    if estimator_type in {"outlier_detector", "outlier"}:
        return "outlier"

    capabilities = _estimator_capabilities(estimator)
    if capabilities["transform"] or capabilities["fit_transform"]:
        return "transformer"
    if capabilities["fit_predict"]:
        return "clustering"
    if capabilities["score_samples"] or capabilities["decision_function"]:
        return "outlier"
    return "estimator"


def _is_multi_output_target(y: Any) -> bool:
    shape = getattr(y, "shape", None)
    if shape is None:
        try:
            _require_numpy()
            shape = np.asarray(y).shape
        except Exception:
            return False
    return len(shape) > 1 and int(shape[1] or 0) > 1


def _clone_for_retry(estimator: Any) -> Any:
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:
        base_params = {}
        if hasattr(estimator, "get_params"):
            try:
                base_params = estimator.get_params(deep=False)
            except Exception:
                base_params = {}
        return estimator.__class__(**base_params)


def _wrap_multi_output_estimator(estimator: Any, family: str) -> Any:
    try:
        if family == "classifier":
            from sklearn.multioutput import MultiOutputClassifier

            return MultiOutputClassifier(_clone_for_retry(estimator))
        from sklearn.multioutput import MultiOutputRegressor

        return MultiOutputRegressor(_clone_for_retry(estimator))
    except Exception as exc:
        raise RuntimeError(f"Unable to prepare multi-output sklearn estimator: {exc}") from exc


def _fit_estimator(
    estimator: Any,
    x: Any,
    y: Any,
    *,
    family: str,
    fit_kwargs: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    metadata = {
        "fit_uses_target": family in {"classifier", "regressor"},
        "multi_output": _is_multi_output_target(y),
        "multi_output_strategy": None,
    }
    kwargs = dict(fit_kwargs or {})

    if family in {"classifier", "regressor"}:
        try:
            estimator.fit(x, y, **kwargs)
            if metadata["multi_output"]:
                metadata["multi_output_strategy"] = "native"
            return estimator, metadata
        except Exception as exc:
            if not metadata["multi_output"]:
                raise
            wrapped = _wrap_multi_output_estimator(estimator, family)
            try:
                wrapped.fit(x, y, **kwargs)
            except Exception:
                raise exc
            metadata["multi_output_strategy"] = "wrapper"
            metadata["multi_output_wrapper"] = wrapped.__class__.__name__
            return wrapped, metadata

    estimator.fit(x, **kwargs)
    return estimator, metadata


def _output_summary_metrics(prefix: str, values: Any) -> dict[str, float]:
    _require_numpy()
    array = np.asarray(values)
    metrics: dict[str, float] = {}
    if array.size == 0:
        return metrics
    metrics[f"{prefix}_rows"] = float(array.shape[0]) if array.ndim else 1.0
    if array.ndim > 1:
        metrics[f"{prefix}_columns"] = float(array.shape[1])
    if np.issubdtype(array.dtype, np.number):
        flattened = array.astype(float).reshape(-1)
        metrics[f"{prefix}_mean"] = float(np.nanmean(flattened))
        metrics[f"{prefix}_std"] = float(np.nanstd(flattened))
    return metrics


def _estimator_outputs(estimator: Any, x: Any, *, family: str) -> tuple[Any, dict[str, Any]]:
    metadata: dict[str, Any] = {"output_method": None}
    if hasattr(estimator, "predict"):
        metadata["output_method"] = "predict"
        return estimator.predict(x), metadata
    if family == "transformer" and hasattr(estimator, "transform"):
        metadata["output_method"] = "transform"
        return estimator.transform(x), metadata
    if hasattr(estimator, "score_samples"):
        metadata["output_method"] = "score_samples"
        return estimator.score_samples(x), metadata
    if hasattr(estimator, "decision_function"):
        metadata["output_method"] = "decision_function"
        return estimator.decision_function(x), metadata
    if hasattr(estimator, "fit_predict"):
        metadata["output_method"] = "fit_predict"
        metadata["output_method_warning"] = "fit_predict was used because the estimator does not expose predict."
        return estimator.fit_predict(x), metadata
    if family == "transformer" and hasattr(estimator, "fit_transform"):
        metadata["output_method"] = "fit_transform"
        metadata["output_method_warning"] = "fit_transform was used because the estimator does not expose transform."
        return estimator.fit_transform(x), metadata
    return None, metadata


def _metric_compatibility(
    task_type: str,
    requested_metrics: Optional[Any],
    objective_metric: Optional[str],
) -> dict[str, Any]:
    requested = [str(item).strip().lower() for item in (requested_metrics or []) if str(item).strip()]
    normalized_task = task_type.lower()
    if normalized_task == "regression":
        compatible = REGRESSION_METRICS | {"fit_duration_sec"}
        default_objective = "rmse"
    elif normalized_task == "classification":
        compatible = CLASSIFICATION_METRICS | {"fit_duration_sec"}
        default_objective = "accuracy"
    else:
        compatible = {"fit_duration_sec", "output_rows", "output_columns", "output_mean", "output_std"}
        default_objective = "fit_duration_sec"

    ignored = [metric for metric in requested if metric not in compatible]
    accepted = [metric for metric in requested if metric in compatible]
    objective = str(objective_metric or default_objective).strip().lower()
    resolved_objective = objective if objective in compatible else default_objective
    return {
        "requested_metrics": requested,
        "accepted_metrics": accepted,
        "ignored_incompatible_metrics": ignored,
        "objective_metric": objective or None,
        "resolved_objective_metric": resolved_objective,
    }


def _build_estimator(
    model_or_factory: Any,
    params: Optional[Mapping[str, Any]],
    random_state: Optional[int],
    *,
    clone_estimator: bool = True,
) -> Any:
    """
    Instantiate or clone an estimator-like object.
    """

    estimator_params = dict(params or {})
    ignored_params: dict[str, Any] = {}
    accepted_params: dict[str, Any] = {}
    valid_param_names: list[str] | None = None

    if hasattr(model_or_factory, "fit"):
        if not clone_estimator:
            estimator = model_or_factory
            if estimator_params and hasattr(estimator, "set_params"):
                accepted_params, ignored_params, valid_param_names = _apply_estimator_params(estimator, estimator_params)
            _attach_estimator_build_details(
                estimator,
                accepted_params=accepted_params,
                ignored_params=ignored_params,
                valid_param_names=valid_param_names,
            )
            return estimator
        try:
            from sklearn.base import clone

            estimator = clone(model_or_factory)
        except Exception:
            base_params = {}
            if hasattr(model_or_factory, "get_params"):
                try:
                    base_params = model_or_factory.get_params(deep=True)
                except Exception:
                    base_params = {}
            estimator = model_or_factory.__class__(**base_params)

        if random_state is not None and hasattr(estimator, "get_params"):
            try:
                if "random_state" in estimator.get_params(deep=True) and "random_state" not in estimator_params:
                    estimator_params["random_state"] = random_state
            except Exception:
                pass

        if estimator_params and hasattr(estimator, "set_params"):
            accepted_params, ignored_params, valid_param_names = _apply_estimator_params(estimator, estimator_params)
        _attach_estimator_build_details(
            estimator,
            accepted_params=accepted_params,
            ignored_params=ignored_params,
            valid_param_names=valid_param_names,
        )
        return estimator

    if inspect.isclass(model_or_factory):
        valid_param_set = _estimator_param_names(model_or_factory)
        if random_state is not None and (valid_param_set is None or "random_state" in valid_param_set) and "random_state" not in estimator_params:
            estimator_params["random_state"] = random_state
        accepted_params, ignored_params, valid_param_names = _split_known_params(model_or_factory, estimator_params)
        if ignored_params:
            logger.warning("Ignoring invalid constructor parameters for %s: %s", model_or_factory, sorted(ignored_params))
        estimator = model_or_factory(**accepted_params)
        _attach_estimator_build_details(
            estimator,
            accepted_params=accepted_params,
            ignored_params=ignored_params,
            valid_param_names=valid_param_names,
        )
        return estimator

    if callable(model_or_factory):
        try:
            signature = inspect.signature(model_or_factory)
            valid_param_set = _estimator_param_names(model_or_factory)
            if random_state is not None and (valid_param_set is None or "random_state" in valid_param_set) and "random_state" not in estimator_params:
                estimator_params["random_state"] = random_state
            accepted_params, ignored_params, valid_param_names = _split_known_params(model_or_factory, estimator_params)
            if ignored_params:
                logger.warning("Ignoring invalid factory parameters for %s: %s", model_or_factory, sorted(ignored_params))
            call_kwargs = dict(accepted_params)
            if random_state is not None and "random_state" in signature.parameters and "random_state" not in call_kwargs:
                call_kwargs["random_state"] = random_state
            estimator = model_or_factory(**call_kwargs)
            _attach_estimator_build_details(
                estimator,
                accepted_params=accepted_params,
                ignored_params=ignored_params,
                valid_param_names=valid_param_names,
            )
            return estimator
        except (TypeError, ValueError):
            if random_state is not None:
                return model_or_factory(estimator_params, random_state)
            return model_or_factory(estimator_params)

    raise TypeError("model_or_factory must be an estimator instance, class, or factory callable.")


def train_sklearn(
    model_or_factory: Any,
    x: Any,
    y: Any,
    params: Optional[Mapping[str, Any]] = None,
    random_state: Optional[int] = None,
    *,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    eval_data: Optional[tuple[Any, Any]] = None,
    task_type: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Mapping[str, str]] = None,
    log_to_mlflow: bool = True,
    clone_estimator: bool = True,
    metrics_to_track: Optional[Any] = None,
    objective_metric: Optional[str] = None,
) -> TrainingResult:
    """
    Train a scikit-learn estimator and return a normalized ``TrainingResult``.
    """

    started_run, run_id = _maybe_start_tracking(experiment_name, run_name, tags, log_to_mlflow)
    fit_parameters = dict(params or {})
    fit_duration_sec = 0.0

    failed = False
    try:
        estimator = _build_estimator(model_or_factory, fit_parameters, random_state, clone_estimator=clone_estimator)
        family = _estimator_family(estimator, task_type)
        compatibility = _metric_compatibility(task_type or family, metrics_to_track, objective_metric)
        if log_to_mlflow and fit_parameters:
            log_params(fit_parameters)

        start_time = time.perf_counter()
        estimator, fit_metadata = _fit_estimator(
            estimator,
            x,
            y,
            family=family,
            fit_kwargs=dict(fit_kwargs or {}),
        )
        fit_duration_sec = time.perf_counter() - start_time

        metrics: Dict[str, float] = {"fit_duration_sec": fit_duration_sec}
        output_metadata: dict[str, Any] = {}
        if eval_data is not None:
            x_eval, y_eval = eval_data
            predictions, output_metadata = _estimator_outputs(estimator, x_eval, family=family)
            if predictions is not None and family in {"classifier", "regressor"}:
                normalized_task_type = "classification" if family == "classifier" else "regression"
                try:
                    metrics.update(
                        evaluate_and_log_metrics(
                            y_true=y_eval,
                            y_pred=predictions,
                            task_type=normalized_task_type,
                            prefix="eval_",
                            log_to_mlflow=log_to_mlflow,
                        )
                    )
                except Exception as exc:
                    output_metadata["metric_warning"] = str(exc)
                    logger.warning("Unable to compute sklearn evaluation metrics: %s", exc)
            elif predictions is not None:
                metrics.update(
                    {
                        f"eval_{key}": value
                        for key, value in _output_summary_metrics("output", predictions).items()
                    }
                )

        if log_to_mlflow:
            log_metrics({"fit_duration_sec": fit_duration_sec})

        build_details = dict(getattr(estimator, "_amanaje_build_details", {}) or {})
        accepted_result_params = dict(build_details.get("accepted_estimator_params") or {})
        metadata = {
            "fit_duration_sec": fit_duration_sec,
            "estimator_family": family,
            "estimator_capabilities": _estimator_capabilities(estimator),
            "fit": fit_metadata,
            "output": output_metadata,
            "metric_compatibility": compatibility,
            **build_details,
        }
        if compatibility["ignored_incompatible_metrics"]:
            metadata["warnings"] = [
                f"Ignored incompatible metrics for {task_type or family}: {', '.join(compatibility['ignored_incompatible_metrics'])}"
            ]

        result = TrainingResult(
            model=estimator,
            framework="sklearn",
            metrics=metrics,
            history={},
            parameters=accepted_result_params or fit_parameters or getattr(estimator, "get_params", lambda **_: {})(),
            run_id=run_id or active_run_id(),
            metadata=metadata,
        )

        return result
    except Exception:
        failed = True
        raise
    finally:
        _finalize_tracking(started_run, failed=failed)


def _extract_model_output(output: Any) -> Any:
    """
    Normalize PyTorch model outputs so loss functions receive the prediction tensor.
    """

    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _align_tensors(outputs: Any, targets: Any) -> tuple[Any, Any]:
    """
    Align common prediction/target shapes such as ``[N, 1]`` vs ``[N]``.
    """

    output_shape = getattr(outputs, "shape", None)
    target_shape = getattr(targets, "shape", None)
    if output_shape is None or target_shape is None:
        return outputs, targets

    if len(output_shape) == len(target_shape) + 1 and output_shape[-1] == 1:
        outputs = outputs.squeeze(-1)
    if len(target_shape) == len(output_shape) + 1 and target_shape[-1] == 1:
        targets = targets.squeeze(-1)
    return outputs, targets


def _move_to_device(value: Any, device: Any) -> Any:
    """
    Move tensors and nested tensor containers to the selected device.
    """

    if hasattr(value, "to"):
        try:
            return value.to(device)
        except Exception:
            return value

    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _build_torch_batches(x: Any, y: Any, batch_size: Optional[int]) -> Any:
    """
    Build a batch iterator from tensors, arrays, or an existing DataLoader.
    """

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if y is None and hasattr(x, "__iter__") and not hasattr(x, "shape"):
        return x

    if y is None:
        raise ValueError("y must be provided unless x is already an iterable of batches.")

    tensor_x = x if hasattr(x, "shape") else torch.as_tensor(x)
    tensor_y = y if hasattr(y, "shape") else torch.as_tensor(y)
    if not isinstance(tensor_x, torch.Tensor):
        tensor_x = torch.as_tensor(tensor_x)
    if not isinstance(tensor_y, torch.Tensor):
        tensor_y = torch.as_tensor(tensor_y)

    if batch_size is None:
        return [(tensor_x, tensor_y)]

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _evaluate_torch_loss(
    model: Any,
    criterion: Any,
    data: tuple[Any, Any] | Any,
    *,
    batch_size: Optional[int],
    device: Any,
    forward_kwargs: Mapping[str, Any],
) -> float:
    """
    Evaluate average loss on validation data.
    """

    import torch

    if isinstance(data, tuple) and len(data) == 2:
        batches = _build_torch_batches(data[0], data[1], batch_size)
    else:
        batches = data

    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_x, batch_y in batches:
            batch_x = _move_to_device(batch_x, device)
            batch_y = _move_to_device(batch_y, device)
            outputs = _extract_model_output(model(batch_x, **dict(forward_kwargs)))
            outputs, batch_y = _align_tensors(outputs, batch_y)
            loss = criterion(outputs, batch_y)
            total_loss += float(loss.detach().item())
            batch_count += 1

    model.train()
    return total_loss / max(batch_count, 1)


def train_pytorch(
    model: Any,
    x: Any,
    y: Any = None,
    criterion: Any = None,
    optimizer: Any = None,
    epochs: int = 1,
    accum_steps: int = 1,
    *,
    batch_size: Optional[int] = None,
    val_data: Optional[tuple[Any, Any] | Any] = None,
    scheduler: Any = None,
    device: Optional[str] = None,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    log_every: int = 1,
    log_to_mlflow: bool = True,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> TrainingResult:
    """
    Train a PyTorch model and return a normalized ``TrainingResult``.
    """

    if criterion is None or optimizer is None:
        raise ValueError("criterion and optimizer must be provided for PyTorch training.")

    import torch

    started_run, run_id = _maybe_start_tracking(experiment_name, run_name, tags, log_to_mlflow)
    training_params = dict(params or {})
    forward_parameters = dict(forward_kwargs or {})
    history: Dict[str, list[float]] = {"train_loss": []}

    failed = False
    try:
        device_obj = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = model.to(device_obj)
        model.train()

        if log_to_mlflow and training_params:
            log_params(training_params)

        for epoch in range(1, epochs + 1):
            train_batches = _build_torch_batches(x, y, batch_size)
            optimizer.zero_grad()
            epoch_loss = 0.0
            batch_count = 0
            pending_backward_steps = 0
            epoch_start = time.perf_counter()

            for batch_x, batch_y in train_batches:
                batch_x = _move_to_device(batch_x, device_obj)
                batch_y = _move_to_device(batch_y, device_obj)

                outputs = _extract_model_output(model(batch_x, **forward_parameters))
                outputs, batch_y = _align_tensors(outputs, batch_y)
                loss = criterion(outputs, batch_y)

                (loss / max(accum_steps, 1)).backward()
                pending_backward_steps += 1
                if pending_backward_steps >= max(accum_steps, 1):
                    optimizer.step()
                    optimizer.zero_grad()
                    pending_backward_steps = 0

                epoch_loss += float(loss.detach().item())
                batch_count += 1

            if pending_backward_steps:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = epoch_loss / max(batch_count, 1)
            history["train_loss"].append(avg_train_loss)
            epoch_metrics: Dict[str, float] = {
                "train_loss": avg_train_loss,
                "epoch_duration_sec": time.perf_counter() - epoch_start,
            }

            if val_data is not None:
                avg_val_loss = _evaluate_torch_loss(
                    model=model,
                    criterion=criterion,
                    data=val_data,
                    batch_size=batch_size,
                    device=device_obj,
                    forward_kwargs=forward_parameters,
                )
                history.setdefault("val_loss", []).append(avg_val_loss)
                epoch_metrics["val_loss"] = avg_val_loss

            if scheduler is not None:
                try:
                    scheduler.step(epoch_metrics.get("val_loss", avg_train_loss))
                except TypeError:
                    scheduler.step()

            if log_to_mlflow:
                log_metrics(epoch_metrics, step=epoch)

            if log_every > 0 and epoch % log_every == 0:
                logger.info("Epoch %d/%d metrics: %s", epoch, epochs, epoch_metrics)
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "epoch": epoch,
                            "epochs_total": epochs,
                            "train_loss": avg_train_loss,
                            "val_loss": epoch_metrics.get("val_loss"),
                            "epoch_duration_sec": epoch_metrics.get("epoch_duration_sec"),
                        }
                    )
                except Exception:
                    logger.debug("PyTorch progress callback failed at epoch %d.", epoch, exc_info=True)

        summary_metrics = {key: values[-1] for key, values in history.items() if values}
        result = TrainingResult(
            model=model,
            framework="pytorch",
            metrics=summary_metrics,
            history=history,
            parameters=training_params,
            run_id=run_id or active_run_id(),
            metadata={"device": str(device_obj), "epochs": epochs},
        )

        return result
    except Exception:
        failed = True
        raise
    finally:
        _finalize_tracking(started_run, failed=failed)


def run_training_pipeline(
    train_fn: Callable[..., TrainingResult],
    *,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    log_to_mlflow: bool = True,
    **train_kwargs: Any,
) -> TrainingResult:
    """
    Execute a training callable inside a standardized tracking workflow.
    """

    started_run, _ = _maybe_start_tracking(experiment_name, run_name, tags, log_to_mlflow)
    failed = False
    try:
        if log_to_mlflow and params:
            log_params(params)

        result = train_fn(**train_kwargs)
        if log_to_mlflow:
            log_json(result.to_dict(), filename="training_summary.json", artifact_path="training")
        return result
    except Exception:
        failed = True
        raise
    finally:
        _finalize_tracking(started_run, failed=failed)


def save_training_summary(result: TrainingResult, output_path: str) -> None:
    """
    Persist a training summary to disk.
    """

    save_json(result.to_dict(), output_path)
