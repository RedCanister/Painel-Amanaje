import asyncio
import json
import logging
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
from starlette.requests import Request

import main_app


def _payload(response):
    return json.loads(response.body.decode("utf-8"))


def _settings_test_path():
    return main_app._utils.CONFIG_SNAPSHOT_DIR / f"pytest_settings_{uuid.uuid4().hex}.json"


def _cleanup_settings_test_path(path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _settings_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/settings/config",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "client": ("testclient", 50000),
            "app": main_app.app,
            "router": main_app.app.router,
        }
    )


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
        "/settings",
        "/settings/config",
        "/production/status",
        "/production/simulate",
        "/runs/list",
        "/runs/get/{run_id}",
        "/registry/dependencies/{registry_type}/{item_id}",
        "/analysis/data",
        "/analysis/model",
        "/datasetmodel/list",
        "/learningmodel/list",
        "/inferencemodel/list",
        "/studymodel/list",
    }.issubset(route_paths)
    assert main_app._build_bar_plot("x", [("a", 1)])["series"] == [{"label": "a", "value": 1.0}]
    assert callable(main_app._find_inference_record)


def test_settings_defaults_persist_and_hot_apply(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)
    monkeypatch.setenv("APP_ENV", "pytest")

    try:
        response = asyncio.run(main_app.settings_config())
        payload = _payload(response)
        assert response.status_code == 200
        assert payload["feature_flags"] == {"debug_mode": False, "assistant_visible": True}
        assert {row["key"] for row in payload["environment_variables"]} >= {"APP_ENV", "MLFLOW_TRACKING_URI"}

        response = asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "feature_flags": {"debug_mode": True, "assistant_visible": False},
                    "env_overrides": {"APP_ENV": "staging"},
                }
            )
        )
        payload = _payload(response)
        env_by_key = {row["key"]: row for row in payload["environment_variables"]}

        assert response.status_code == 200
        assert settings_path.exists()
        assert payload["feature_flags"]["debug_mode"] is True
        assert payload["feature_flags"]["assistant_visible"] is False
        assert env_by_key["APP_ENV"]["saved_override"] == "staging"
        assert payload["restart_required"]["required"] is True
        assert "APP_ENV" in payload["restart_required"]["keys"]
        assert main_app.LOGGER.level == logging.DEBUG
    finally:
        main_app._utils.apply_runtime_settings_state(
            {"feature_flags": {"debug_mode": False, "assistant_visible": True}, "env_overrides": {}}
        )
        _cleanup_settings_test_path(settings_path)


def test_settings_reject_secret_like_environment_keys(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)

    try:
        response = asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "env_overrides": {
                        "POSTGRES_PASSWORD": "secret-value",
                    }
                }
            )
        )
        payload = _payload(response)

        assert response.status_code == 400
        assert payload["status"] == "error"
        assert payload["errors"][0]["reason"] == "secret_key_blocked"
        assert not settings_path.exists()
    finally:
        _cleanup_settings_test_path(settings_path)


def test_settings_validation_debug_includes_request_context(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)
    try:
        asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "feature_flags": {
                        "debug_mode": True,
                    }
                },
            )
        )
        response = asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "env_overrides": {
                        "POSTGRES_PASSWORD": "secret-value",
                    }
                },
            )
        )
        payload = _payload(response)

        assert response.status_code == 400
        assert payload["debug"]["error_type"] == "SettingsValidationError"
        assert payload["debug"]["request"]["path"] == "/settings/config"
        assert payload["debug"]["request"]["method"] == "POST"
        assert payload["debug"]["debug_type"]["name"] == "SettingsValidationError"
    finally:
        main_app._utils.apply_runtime_settings_state(
            {"feature_flags": {"debug_mode": False, "assistant_visible": True}, "env_overrides": {}}
        )
        _cleanup_settings_test_path(settings_path)


def test_render_page_includes_assistant_visibility_settings(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)
    try:
        asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "feature_flags": {
                        "assistant_visible": False,
                    }
                }
            )
        )
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
                "query_string": b"",
                "server": ("testserver", 80),
                "scheme": "http",
                "client": ("testclient", 50000),
                "app": main_app.app,
                "router": main_app.app.router,
            }
        )

        response = main_app._render_page("base_template.html", request)

        assert response.context["amanaje_settings"]["feature_flags"]["assistant_visible"] is False
    finally:
        _cleanup_settings_test_path(settings_path)


def test_json_error_excludes_debug_payload_by_default(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)
    try:
        response = main_app._utils._json_error("broken", status_code=500, exc=ValueError("broken"))
        payload = _payload(response)

        assert response.status_code == 500
        assert payload["status"] == "error"
        assert "debug" not in payload
    finally:
        _cleanup_settings_test_path(settings_path)


def test_json_error_includes_debug_payload_when_debug_mode_is_enabled(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)
    debug_calls = []

    def fake_debug_type(value):
        debug_calls.append(value)
        return {"type": str(type(value)), "name": getattr(value, "__class__", type(value)).__name__}

    monkeypatch.setattr(main_app._utils, "debug_type", fake_debug_type)
    try:
        asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "feature_flags": {
                        "debug_mode": True,
                    }
                }
            )
        )
        exc = ValueError("debug failure")
        response = main_app._utils._json_error("debug failure", status_code=500, exc=exc)
        payload = _payload(response)

        assert response.status_code == 500
        assert payload["debug"]["error_type"] == "ValueError"
        assert "ValueError: debug failure" in payload["debug"]["traceback"]
        assert payload["debug"]["debug_type"]["name"] == "ValueError"
        assert debug_calls == [exc]
    finally:
        main_app._utils.apply_runtime_settings_state(
            {"feature_flags": {"debug_mode": False, "assistant_visible": True}, "env_overrides": {}}
        )
        _cleanup_settings_test_path(settings_path)


def test_generic_exception_handler_includes_request_debug_when_enabled(monkeypatch):
    settings_path = _settings_test_path()
    monkeypatch.setattr(main_app._utils, "SETTINGS_STATE_PATH", settings_path)
    try:
        asyncio.run(
            main_app.settings_update_config(
                _settings_request(),
                {
                    "feature_flags": {
                        "debug_mode": True,
                    }
                }
            )
        )
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/debug-test",
                "headers": [],
                "query_string": b"sample=1",
                "server": ("testserver", 80),
                "scheme": "http",
                "client": ("testclient", 50000),
                "app": main_app.app,
                "router": main_app.app.router,
            }
        )
        response = asyncio.run(main_app._utils.generic_exception_handler(request, RuntimeError("kaput")))
        payload = _payload(response)

        assert response.status_code == 500
        assert payload["status"] == "error"
        assert payload["debug"]["error_type"] == "RuntimeError"
        assert payload["debug"]["request"]["path"] == "/debug-test"
        assert payload["debug"]["request"]["method"] == "GET"
    finally:
        main_app._utils.apply_runtime_settings_state(
            {"feature_flags": {"debug_mode": False, "assistant_visible": True}, "env_overrides": {}}
        )
        _cleanup_settings_test_path(settings_path)


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


def test_production_simulate_returns_structured_dependency_errors(monkeypatch):
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
                "watchContextId": "watch:test-dependency-error",
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        raise main_app._utils.RuntimeArtifactDependencyError(
            "Missing runtime dependency 'lightfm' required to load 'model.pkl'.",
            missing_dependencies=["lightfm"],
            artifact_loader="pickle",
        )

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = response.body.decode("utf-8")

    assert response.status_code == 400
    assert '"missing_dependencies":["lightfm"]' in payload
    assert '"artifact_loader":"pickle"' in payload
    assert '"watch_context_id":"watch:test-dependency-error"' in payload


def test_post_training_returns_preflight_validation_error(monkeypatch):
    async def fake_model_read(_db, _model_id):
        return SimpleNamespace(id=11, name="broken-model", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, _dataset_id):
        return SimpleNamespace(id=21, name="broken-dataset")

    async def fake_study_read(_db, _study_id):
        return None

    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app.StudyModel, "read", fake_study_read)
    monkeypatch.setattr(
        main_app._utils,
        "_merge_training_parameters",
        lambda *_args, **_kwargs: {"framework": "sklearn"},
    )
    monkeypatch.setattr(
        main_app._utils,
        "_build_training_preflight",
        lambda *_args, **_kwargs: {
            "ok": False,
            "missing_columns": ["expected_engagement_score"],
            "available_columns": ["followers", "public_repos"],
            "recommended_pairing": {"output_feature": "followers"},
        },
    )

    payload = main_app.TrainingRequest(datasetId=21, inputType="Manual", studyId=None, parameters={})
    response = asyncio.run(main_app.post_training(11, payload=payload, db=object()))
    body = response.body.decode("utf-8")

    assert response.status_code == 400
    assert '"missing_columns":["expected_engagement_score"]' in body
    assert '"available_columns":["followers","public_repos"]' in body
