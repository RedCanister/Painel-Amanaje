import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.utils.training import TrainingResult, run_training_pipeline
from app.utils import accelerators


def test_run_training_pipeline_logs_summary_without_name_error(monkeypatch):
    logged = {}

    def fake_log_json(data, filename="data_snapshot.json", artifact_path=None):
        logged["data"] = data
        logged["filename"] = filename
        logged["artifact_path"] = artifact_path

    monkeypatch.setattr("app.utils.training.log_json", fake_log_json)

    result = run_training_pipeline(
        train_fn=lambda: TrainingResult(model=None, framework="pytorch", metrics={"eval_accuracy": 0.9}),
        log_to_mlflow=True,
    )

    assert result.framework == "pytorch"
    assert logged["filename"] == "training_summary.json"
    assert logged["artifact_path"] == "training"
    assert logged["data"]["metrics"]["eval_accuracy"] == 0.9


def test_resolve_torch_device_falls_back_to_cpu_without_torch(monkeypatch):
    monkeypatch.setattr(accelerators, "_load_torch", lambda: None)

    resolution = accelerators.resolve_torch_device("cuda")

    assert resolution["requested_device"] == "cuda"
    assert resolution["resolved_device"] == "cpu"
    assert resolution["fallback_applied"] is True


def test_resolve_torch_device_selects_cuda_when_available(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(_index):
            return "Test GPU"

        @staticmethod
        def get_device_properties(_index):
            return type("Props", (), {"total_memory": 1024})()

    fake_torch = type(
        "FakeTorch",
        (),
        {
            "__version__": "test",
            "cuda": FakeCuda,
            "version": type("Version", (), {"cuda": "test-cuda"})(),
        },
    )()
    monkeypatch.setattr(accelerators, "_load_torch", lambda: fake_torch)

    resolution = accelerators.resolve_torch_device("cuda")

    assert resolution["resolved_device"] == "cuda"
    assert resolution["devices"][0]["name"] == "Test GPU"


def test_sklearn_train_filters_control_params_for_random_forest():
    pytest.importorskip("sklearn")
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from app.utils.training import train_sklearn

    x = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([1.2, 1.9, 3.1, 3.8])
    estimator = RandomForestRegressor(n_estimators=3, random_state=42, warm_start=True)

    result = train_sklearn(
        estimator,
        x,
        y,
        params={"n_estimators": 4, "metrics_to_track": ["mae", "accuracy"], "warm_start_increment": 10},
        random_state=42,
        eval_data=(x, y),
        task_type="regression",
        log_to_mlflow=False,
        clone_estimator=False,
        metrics_to_track=["mae", "accuracy"],
        objective_metric="accuracy",
    )

    assert result.model.n_estimators == 4
    assert "metrics_to_track" in result.metadata["ignored_estimator_params"]
    assert "warm_start_increment" in result.metadata["ignored_estimator_params"]
    assert result.metadata["metric_compatibility"]["ignored_incompatible_metrics"] == ["accuracy"]
    assert result.metadata["metric_compatibility"]["resolved_objective_metric"] == "rmse"


def test_sklearn_dynamic_estimator_resolution_and_required_params():
    pytest.importorskip("sklearn")
    from app.utils import main_utils

    estimator = main_utils._select_sklearn_estimator("RandomForestRegressor", "regression")
    qualified = main_utils._select_sklearn_estimator("sklearn.ensemble.RandomForestRegressor", "regression")

    class RequiredEstimator:
        def __init__(self, required, optional=1):
            self.required = required
            self.optional = optional

        def fit(self, x, y=None):
            return self

    partitioned = main_utils._partition_sklearn_parameters(
        {
            "required": "value",
            "optional": 2,
            "metrics_to_track": ["loss"],
            "unknown_knob": 7,
        },
        RequiredEstimator,
    )

    assert estimator is qualified
    assert partitioned["estimator_params"] == {"required": "value", "optional": 2}
    assert partitioned["ignored_params"] == {"unknown_knob": 7}
    with pytest.raises(ValueError):
        main_utils._select_sklearn_estimator("NotARealEstimator", "regression")


def test_prepare_dataset_supports_multi_output_and_removes_target_leakage(tmp_path):
    pytest.importorskip("sklearn")
    import pandas as pd

    from app.utils import main_utils

    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [5, 4, 3, 2, 1],
            "target_a": [1.0, 1.5, 2.0, 2.5, 3.0],
            "target_b": [2.0, 2.5, 3.0, 3.5, 4.0],
        }
    )
    dataset_path = tmp_path / "multi_output.csv"
    dataframe.to_csv(dataset_path, index=False)
    dataset_record = SimpleNamespace(id=1, name="multi-output", path=str(dataset_path), dataset_type="tabular")
    model_record = SimpleNamespace(id=2, name="rf", input_features=None, output_features=None, metrics={})

    prepared = main_utils._prepare_dataset_for_training(
        dataset_record,
        model_record,
        {
            "framework": "sklearn",
            "estimator_class": "RandomForestRegressor",
            "input_features": ["feature_a", "feature_b", "target_a"],
            "output_features": ["target_a", "target_b"],
            "test_size": 0.4,
        },
    )

    assert prepared.output_features == ["target_a", "target_b"]
    assert prepared.output_feature == "target_a"
    assert "target_a" not in prepared.input_features
    assert "target_b" not in prepared.input_features
    assert list(prepared.y_train.columns) == ["target_a", "target_b"]


def test_sklearn_capability_training_for_cluster_transformer_and_outlier():
    pytest.importorskip("sklearn")
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    from app.utils.training import train_sklearn

    x = pd.DataFrame({"x1": [0.0, 0.1, 8.0, 8.2, 0.2, 8.1], "x2": [0.0, 0.2, 8.0, 7.9, 0.1, 8.3]})

    cluster_result = train_sklearn(
        KMeans,
        x,
        None,
        params={"n_clusters": 2, "n_init": 1, "random_state": 42},
        eval_data=(x, None),
        task_type="clustering",
        log_to_mlflow=False,
    )
    transformer_result = train_sklearn(
        StandardScaler,
        x,
        None,
        eval_data=(x, None),
        task_type="transformer",
        log_to_mlflow=False,
    )
    outlier_result = train_sklearn(
        IsolationForest,
        x,
        None,
        params={"random_state": 42},
        eval_data=(x, None),
        task_type="outlier",
        log_to_mlflow=False,
    )

    assert cluster_result.metadata["estimator_family"] == "clustering"
    assert cluster_result.metadata["output"]["output_method"] == "predict"
    assert transformer_result.metadata["estimator_family"] == "transformer"
    assert transformer_result.metrics["eval_output_columns"] == 2.0
    assert outlier_result.metadata["estimator_family"] == "outlier"
    assert outlier_result.metadata["output"]["output_method"] in {"predict", "score_samples", "decision_function"}


def test_sklearn_pipeline_metadata_records_ignored_unknown_params(monkeypatch, tmp_path):
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd

    from app.utils import main_utils

    monkeypatch.setattr(main_utils, "run_training_pipeline", lambda train_fn, **_kwargs: train_fn())
    monkeypatch.setattr(main_utils, "_log_training_tracking_context", lambda *args, **kwargs: None)

    dataframe = pd.DataFrame({"x": np.arange(8, dtype=float), "y": np.arange(8, dtype=float) * 2.0})
    dataset_path = tmp_path / "rf.csv"
    dataframe.to_csv(dataset_path, index=False)
    dataset_record = SimpleNamespace(id=1, name="rf-data", path=str(dataset_path), dataset_type="tabular")
    model_record = SimpleNamespace(id=2, name="rf-model", input_features=None, output_features=None, metrics={}, parameters={})
    parameters = {
        "framework": "sklearn",
        "estimator_class": "RandomForestRegressor",
        "n_estimators": 4,
        "metrics_to_track": ["mae", "accuracy"],
        "objective_metric": "accuracy",
        "unknown_knob": "ignored",
        "test_size": 0.25,
    }
    prepared = main_utils._prepare_dataset_for_training(dataset_record, model_record, parameters)

    result, predictions = main_utils._train_with_sklearn(prepared, model_record, parameters)

    assert predictions.shape[0] == prepared.x_test.shape[0]
    assert result.metadata["sklearn"]["accepted_estimator_params"]["n_estimators"] == 4
    assert result.metadata["sklearn"]["ignored_unknown_params"] == {"unknown_knob": "ignored"}
    assert result.metadata["metric_compatibility"]["ignored_incompatible_metrics"] == ["accuracy"]
