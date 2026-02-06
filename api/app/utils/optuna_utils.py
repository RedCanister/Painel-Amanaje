"""
utils/optuna_utils.py

Utility functions for managing Optuna hyperparameter optimization
integrated with MLflow tracking and project logging.

Features:
- Unified study creation and management
- MLflow logging of best params, metrics and trial summaries
- Decoratores for automated optimization tracking
- Optional result visualization and export
"""

import os
import optuna
import mlflow
from typing import Any, Callable, Dict, Optional

from utils.logging import get_logger
from utils.serialization import safe_log_params
from utils.mlflow_utils import log_metrics, log_params, log_json, start_run, end_run
from utils.io import save_json

logger = get_logger("optuna_utils")

# Study Creation
def create_study(
    study_name: str,
    direction: str = "minimize",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
) -> optuna.Study:
    """
    Creates or loads an Optuna study.

    Args:
        study_name: name of the study
        direction: "minimize" or "maximize"
        storage: e.g., "sqlite:///optuna.db"
        load_if_exists: whether to reuse an existing study
        sampler: optional custom sampler (e.g. TPSampler, RandomSampler)
    """
    logger.info(f"🎯 Creating Optuna study '{study_name}' (direction={direction})")
    
    study = optuna.create_study(
        study_name = study_name,
        direction = direction,
        storage = storage,
        load_if_exists = load_if_exists,
        sampler = sampler
    )

    return study

def mlflow_objective_wrapper(
    objective_fn: Callable[[optuna.Trial], float],
    experiment_name: str,
    run_prefix: str = "trial_",
) -> Callable[[optuna.Trial], float]:
    """
    Wraps an Optuna objective function to automatically track each trial in Mlflow with
    parameters and metrics.

    Args:
        objectie_fn: the user's objective(trial) function
        experiment_name: MLflow experiment to log into
        run_prefix: prefix for run names
    """

    def wrapped(trial: optuna.Trial) -> float:
        run_name = f"{run_prefix}{trial.number}"
        run = start_run(experiment_name, run_name)
        try:
            # Execute objective and return score
            score = objective_fn(trial)

            # Log trial data to MLflow
            log_params(trial.params)
            log_metrics({"objective": score}, step=trial.number)
            end_run("FINISHED")

            logger.info(f"✅ Trial {trial.number} complete with value={score:.6f}")
            return score
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {e}")
            end_run("FAILED")
            raise e

    return wrapped


# Study Execution

def optimize_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int = 50,
    n_jobs: int = 1,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """
    Runs an Optuna optimization study.
    """
    logger.info(f"🚀 Starting optimization for {n_trials} trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )
    logger.info(f"🏁 Study Complete. Best value: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")
    return study


# Logging & Export

def log_study_results(study: optuna.Study, output_dir: str = "optuna_results") -> None:
    """
    Saves and logs Optuna study results (best params, trials summary, etc.) to MLflow.
    """
    os.makedirs(output_dir, exist_ok=True)

    best_params_path = os.path.join(output_dir, "best_params.json")
    all_trials_path = os.path.join(output_dir, "trials_summary.json")

    best_params = safe_log_params(study.best_params)
    all_trials = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    save_json(best_params, best_params_path)
    save_json(all_trials, all_trials_path)

    log_json(best_params, filename="best_params.json")
    log_json(all_trials, filename="trials_summary.json")

    logger.info(f"💾 Best params and trials saved in {output_dir}/")
    logger.info(f"Best params: {best_params}")


    # Visualization

    def plot_optuna_results(study: optuna.Study, output_dir: str = "optuna_results") -> None:
        """ 
        Generates and logs standard Optuna visualizations.
        """

        import optuna.visualization.matplotlib as ov

        os.makedirs(output_dir, exist_ok = True)

        plots = {
            "optimization_history.png": ov.plot_optimization_history(study),
            "param_importances.png": ov.plot_param_importances(study),
        }

        for filename, fig in plots.items():
            path = os.path.join(output_dir, filename)
            fig.savefig(path, bbox_inches="tight", dpi=200)
            log_metrics({"plot_logged": 1})
            log_json({"plot": filename}, filename="plot_info.json")
            logger.info(f"📊 Saved Optuna plot: {path}")