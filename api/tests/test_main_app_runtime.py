import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd

import main_app


def test_main_app_imports_and_registers_expected_routes():
    route_paths = {route.path for route in main_app.app.routes}

    assert {
        "/",
        "/upload",
        "/upload/support",
        "/create",
        "/feature",
        "/training",
        "/production",
        "/registry",
        "/production/status",
        "/production/simulate",
        "/runs/list",
        "/runs/get/{run_id}",
        "/analysis/data",
        "/analysis/model",
        "/datasetmodel/list",
        "/learningmodel/list",
        "/inferencemodel/list",
        "/studymodel/list",
    }.issubset(route_paths)
    assert main_app._build_bar_plot("x", [("a", 1)])["series"] == [{"label": "a", "value": 1.0}]
    assert callable(main_app._find_inference_record)


def test_upload_support_exposes_capability_matrices():
    response = asyncio.run(main_app.upload_support())
    payload = response.body.decode("utf-8")

    assert response.status_code == 200
    assert '"datasets"' in payload
    assert '".csv"' in payload
    assert '"models"' in payload
    assert '".joblib"' in payload


def test_production_simulate_handles_mixed_feature_rows(monkeypatch):
    model_record = SimpleNamespace(
        id=10,
        name="model-a",
        path="runtime_artifacts/models/model.pkl",
        parameters={"framework": "sklearn"},
        metrics={"eval_rmse": 0.2},
        input_features=["numeric_feature", "category_feature"],
        output_features=["target"],
    )
    dataset_record = SimpleNamespace(id=20, name="dataset-a")
    inference_record = SimpleNamespace(
        id=30,
        name="model-a :: dataset-a",
        learning_model_id=10,
        dataset_id=20,
        input_features=["numeric_feature", "category_feature"],
        output_features=["target"],
        inference_params={},
        history=[],
        path="runtime_artifacts/models/model.pkl",
    )
    prepared = SimpleNamespace(
        feature_frame=pd.DataFrame(
            {
                "numeric_feature": [10.0],
                "category_feature": ["stable"],
            }
        ),
        task_type="regression",
        label_mapping={},
    )

    class FakeRequest:
        headers = {"content-type": "application/json"}

        async def json(self):
            return {
                "modelId": 10,
                "datasetId": 20,
                "inferenceId": 30,
                "steps": 2,
                "amplitude": 0.1,
                "trend": "up",
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    async def fake_upsert_inference_record(*_args, **_kwargs):
        return inference_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, SimpleNamespace()

    def fake_predict_with_result(_runtime_result, frame, _task_type):
        assert list(frame.columns) == ["numeric_feature", "category_feature"]
        return np.array([1.5, 1.8])

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)
    monkeypatch.setattr(main_app, "_upsert_inference_record", fake_upsert_inference_record)
    monkeypatch.setattr(main_app._utils, "_predict_with_result", fake_predict_with_result)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = response.body.decode("utf-8")

    assert response.status_code == 200
    assert '"status":"ok"' in payload
    assert '"category_feature":"stable"' in payload
    assert '"Feature Drift Applied"' in payload


def test_production_simulate_returns_user_facing_runtime_errors(monkeypatch):
    model_record = SimpleNamespace(id=10, name="model-a", path="runtime_artifacts/models/model.pkl")
    dataset_record = SimpleNamespace(id=20, name="dataset-a")
    inference_record = SimpleNamespace(id=30, name="model-a :: dataset-a")

    class FakeRequest:
        headers = {"content-type": "application/json"}

        async def json(self):
            return {
                "modelId": 10,
                "datasetId": 20,
                "inferenceId": 30,
                "watchContextId": "watch:test-runtime-error",
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        raise ValueError("Only TorchScript or app-generated PyTorch bundles can run directly.")

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = response.body.decode("utf-8")

    assert response.status_code == 400
    assert '"status":"error"' in payload
    assert '"watch_context_id":"watch:test-runtime-error"' in payload
    assert 'TorchScript' in payload
