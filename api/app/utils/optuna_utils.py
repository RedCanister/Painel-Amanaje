"""
utils/optuna_utils.py

Utility functions for Optuna hyperparameter optimization integrated with MLflow.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .io import ensure_dir, save_json
from .logging import get_logger
from .mlflow_utils import (
    active_run_id,
    end_run,
    is_active_run,
    log_artifact,
    log_json,
    log_metrics,
    log_params,
    set_tags,
    start_run,
)

logger = get_logger("optuna_utils")

try:
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - only used when Optuna is unavailable.
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False


def _require_optuna() -> Any:
    if not OPTUNA_AVAILABLE or optuna is None:
        raise RuntimeError("Optuna is not available in the current environment.")
    return optuna


def create_study(
    study_name: str,
    direction: str = "minimize",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Optional[Any] = None,
    pruner: Optional[Any] = None,
    study_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Create or load an Optuna study.
    """

    client = _require_optuna()
    logger.info("Creating Optuna study '%s' (direction=%s).", study_name, direction)
    return client.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
        **dict(study_kwargs or {}),
    )


def mlflow_objective_wrapper(
    objective_fn: Callable[[Any], float],
    experiment_name: str,
    run_prefix: str = "trial_",
    tags: Optional[Mapping[str, str]] = None,
    *,
    study_name: Optional[str] = None,
    objective_metric: Optional[str] = None,
    direction: Optional[str] = None,
    static_context: Optional[Mapping[str, Any]] = None,
) -> Callable[[Any], float]:
    """
    Wrap an Optuna objective so each trial is tracked in MLflow.
    """

    def wrapped(trial: Any) -> float:
        run_name = f"{run_prefix}{trial.number}"
        run_tags = {
            **dict(tags or {}),
            "run_type": "optuna_trial",
            "trial_number": str(trial.number),
        }
        start_run(
            experiment_name=experiment_name,
            run_name=run_name,
            tags=run_tags,
            nested=is_active_run(),
        )
        run_id = active_run_id()
        if hasattr(trial, "set_user_attr"):
            try:
                trial.set_user_attr("mlflow_run_id", run_id)
            except Exception:
                logger.debug("Unable to attach MLflow run id to trial %s.", trial.number, exc_info=True)
        try:
            score = float(objective_fn(trial))
            log_params(trial.params)
            log_metrics({"objective": score}, step=trial.number)
            set_tags(
                {
                    "study_name": study_name,
                    "objective_metric": objective_metric,
                    "optimization_direction": direction,
                }
            )
            trial_summary = {
                "study_name": study_name,
                "trial_number": trial.number,
                "run_name": run_name,
                "run_id": run_id,
                "value": score,
                "state": "COMPLETE",
                "params": dict(trial.params),
                "objective_metric": objective_metric,
                "direction": direction,
                "context": dict(static_context or {}),
            }
            log_json(
                trial_summary,
                filename=f"trial_{trial.number:04d}_summary.json",
                artifact_path="optuna/trials",
            )
            if hasattr(trial, "set_user_attr"):
                try:
                    trial.set_user_attr("objective_value", score)
                except Exception:
                    logger.debug("Unable to store objective value on trial %s.", trial.number, exc_info=True)
            end_run("FINISHED")
            logger.info("Completed Optuna trial %d with value %.6f.", trial.number, score)
            return score
        except Exception as exc:
            failure_payload = {
                "study_name": study_name,
                "trial_number": trial.number,
                "run_name": run_name,
                "run_id": run_id,
                "state": "FAIL",
                "params": dict(getattr(trial, "params", {}) or {}),
                "objective_metric": objective_metric,
                "direction": direction,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "context": dict(static_context or {}),
            }
            log_json(
                failure_payload,
                filename=f"trial_{trial.number:04d}_failure.json",
                artifact_path="optuna/trials",
            )
            if hasattr(trial, "set_user_attr"):
                try:
                    trial.set_user_attr("failure_reason", str(exc))
                except Exception:
                    logger.debug("Unable to store failure reason on trial %s.", trial.number, exc_info=True)
            end_run("FAILED")
            logger.exception("Optuna trial %d failed.", trial.number)
            raise

    return wrapped


def optimize_study(
    study: Any,
    objective: Callable[[Any], float],
    n_trials: int = 50,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
    callbacks: Optional[Iterable[Callable[[Any, Any], None]]] = None,
) -> Any:
    """
    Execute an Optuna study.
    """

    logger.info("Starting Optuna optimization for %d trials.", n_trials)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
        callbacks=list(callbacks or []),
    )
    logger.info("Optuna study '%s' completed.", study.study_name)
    return study


def _build_study_summary(study: Any) -> Dict[str, Any]:
    """
    Build a normalized Optuna study summary.
    """

    states = [trial.state.name for trial in study.trials]
    try:
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = dict(study.best_params)
    except Exception:
        best_trial = None
        best_value = None
        best_params = {}

    return {
        "study_name": study.study_name,
        "direction": study.direction.name if hasattr(study.direction, "name") else str(study.direction),
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "completed_trials": states.count("COMPLETE"),
        "pruned_trials": states.count("PRUNED"),
        "failed_trials": states.count("FAIL"),
        "has_best_trial": best_trial is not None,
    }


def log_study_results(
    study: Any,
    output_dir: str = "optuna_results",
    artifact_path: str = "optuna",
) -> Dict[str, Any]:
    """
    Persist and log Optuna study results.
    """

    output_directory = ensure_dir(output_dir)
    summary = _build_study_summary(study)
    trials_summary = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": dict(trial.params),
            "state": trial.state.name,
        }
        for trial in study.trials
    ]

    summary_path = Path(output_directory) / "study_summary.json"
    best_params_path = Path(output_directory) / "best_params.json"
    trials_path = Path(output_directory) / "trials_summary.json"

    save_json(summary, summary_path)
    save_json(summary["best_params"], best_params_path)
    save_json(trials_summary, trials_path)

    log_json(summary, filename="study_summary.json", artifact_path=artifact_path)
    log_json(summary["best_params"], filename="best_params.json", artifact_path=artifact_path)
    log_json(trials_summary, filename="trials_summary.json", artifact_path=artifact_path)

    if summary["best_params"]:
        log_params(summary["best_params"])
    if summary["best_value"] is not None:
        log_metrics({"best_value": float(summary["best_value"]), "n_trials": float(summary["n_trials"])})

    logger.info("Saved Optuna results to '%s'.", output_directory)
    return summary


def plot_optuna_results(
    study: Any,
    output_dir: str = "optuna_results",
    artifact_path: str = "optuna",
) -> list[str]:
    """
    Generate and log a small set of Optuna matplotlib visualizations.
    """

    _require_optuna()
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as optuna_matplotlib

    output_directory = ensure_dir(output_dir)
    plot_builders = {
        "optimization_history.png": optuna_matplotlib.plot_optimization_history,
        "param_importances.png": optuna_matplotlib.plot_param_importances,
    }
    saved_paths: list[str] = []

    for filename, builder in plot_builders.items():
        try:
            plot_object = builder(study)
            figure = plot_object.figure if hasattr(plot_object, "figure") else plot_object
            file_path = Path(output_directory) / filename
            figure.savefig(file_path, bbox_inches="tight", dpi=200)
            plt.close(figure)
            log_artifact(str(file_path), artifact_path=artifact_path)
            saved_paths.append(str(file_path))
            logger.info("Saved Optuna plot '%s'.", file_path)
        except Exception:
            logger.exception("Unable to generate Optuna plot '%s'.", filename)

    return saved_paths


def optimize_with_tracking(
    objective_fn: Callable[[Any], float],
    study_name: str,
    experiment_name: str,
    *,
    direction: str = "minimize",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Optional[Any] = None,
    pruner: Optional[Any] = None,
    n_trials: int = 50,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
    output_dir: str = "optuna_results",
    tags: Optional[Mapping[str, str]] = None,
    plot_results: bool = False,
    objective_metric: Optional[str] = None,
    search_space: Optional[Mapping[str, Any]] = None,
    study_context: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Create, run, and log an Optuna study under a parent MLflow run.
    """

    study = create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )

    parent_nested = is_active_run()
    start_run(
        experiment_name=experiment_name,
        run_name=f"{study_name}_study",
        tags={**dict(tags or {}), "run_type": "optuna_study", "study_name": study_name},
        nested=parent_nested,
    )
    try:
        if hasattr(study, "set_user_attr"):
            try:
                study.set_user_attr("mlflow_experiment_name", experiment_name)
                if objective_metric:
                    study.set_user_attr("objective_metric", objective_metric)
            except Exception:
                logger.debug("Unable to store MLflow metadata on study '%s'.", study_name, exc_info=True)
        log_params(
            {
                "study_name": study_name,
                "direction": direction,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "objective_metric": objective_metric or "",
            }
        )
        set_tags(
            {
                "study_name": study_name,
                "objective_metric": objective_metric,
                "optimization_direction": direction,
            }
        )
        log_json(
            {
                "study_name": study_name,
                "experiment_name": experiment_name,
                "direction": direction,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "objective_metric": objective_metric,
                "search_space": dict(search_space or {}),
                "context": dict(study_context or {}),
            },
            filename="study_context.json",
            artifact_path="optuna",
        )
        wrapped_objective = mlflow_objective_wrapper(
            objective_fn=objective_fn,
            experiment_name=experiment_name,
            tags=tags,
            study_name=study_name,
            objective_metric=objective_metric,
            direction=direction,
            static_context=study_context,
        )
        callbacks = []
        if progress_callback is not None:
            def _on_trial_complete(study_obj: Any, trial_obj: Any) -> None:
                completed_trials = len(getattr(study_obj, "trials", []) or [])
                try:
                    best_value = getattr(study_obj, "best_value", None)
                except Exception:
                    best_value = None
                progress_callback(
                    {
                        "trial_number": getattr(trial_obj, "number", None),
                        "trial_state": getattr(getattr(trial_obj, "state", None), "name", None),
                        "completed_trials": completed_trials,
                        "total_trials": n_trials,
                        "best_value": best_value,
                    }
                )
            callbacks.append(_on_trial_complete)
        optimize_study(
            study=study,
            objective=wrapped_objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks,
        )
        summary = log_study_results(study, output_dir=output_dir)
        plot_paths = plot_optuna_results(study, output_dir=output_dir) if plot_results else []
        end_run("FINISHED")
        return {"study": study, "summary": summary, "plot_paths": plot_paths}
    except Exception:
        end_run("FAILED")
        logger.exception("Optuna optimization failed for study '%s'.", study_name)
        raise
