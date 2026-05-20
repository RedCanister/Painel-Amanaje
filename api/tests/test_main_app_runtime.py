import asyncio
import importlib.util
import json
import logging
import shutil
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.utils import log_stream
from app.utils.operation_queue import QueueUnavailableError, run_editor_execution_job
from app.utils.run_ledger import create_run_entry, read_run_entry
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


def _log_test_dir():
    path = main_app._utils.CONFIG_SNAPSHOT_DIR / f"pytest_log_stream_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _cleanup_log_test_dir(path):
    shutil.rmtree(path, ignore_errors=True)


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


def _settings_logs_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/settings/logs",
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
        "/panel",
        "/panel/context",
        "/panel/dashboards",
        "/panel/dashboards/{dashboard_id}",
        "/upload",
        "/upload/support",
        "/create",
        "/feature",
        "/training",
        "/onnx",
        "/production",
        "/visualization",
        "/plot",
        "/registry",
        "/settings",
        "/settings/config",
        "/settings/logs",
        "/runtime/accelerators",
        "/runtime/workers",
        "/execute/jobs",
        "/editor/context",
        "/assistant/jobs/{operation}",
        "/production/status",
        "/production/simulation/context",
        "/production/simulate",
        "/runs/list",
        "/runs/get/{run_id}",
        "/runs/{run_id}/{action}",
        "/plots/artifacts",
        "/plots/artifacts/{plot_id}/file",
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


def test_workflow_domain_route_contracts_are_registered():
    route_paths = {route.path for route in main_app.app.routes}
    workflow_domains = {
        "panel": {"/", "/panel", "/panel/context", "/panel/dashboards", "/panel/dashboards/{dashboard_id}"},
        "upload": {"/upload", "/upload/support", "/upload/{operation_id}"},
        "create": {"/create", "/execute", "/execute/jobs", "/editor/context"},
        "feature": {"/feature", "/features", "/features/extract", "/features/preview", "/features/materialize"},
        "training": {"/training", "/training/{model_id}", "/studies/{study_id}/optimize"},
        "production": {
            "/production",
            "/production/status",
            "/production/simulation/context",
            "/production/simulate",
            "/production/start",
            "/production/stop",
        },
        "visualization": {"/visualization", "/plot", "/plots/artifacts", "/plots/artifacts/{plot_id}/file"},
        "registry": {"/registry", "/registry/dependencies/{registry_type}/{item_id}"},
        "assistant": {"/assistant", "/assistant/management/overview", "/assistant/draft", "/assistant/jobs/{operation}", "/examples/catalog"},
        "settings": {"/settings", "/settings/config", "/settings/logs", "/runtime/accelerators", "/runtime/workers"},
    }

    missing = {
        domain: sorted(paths - route_paths)
        for domain, paths in workflow_domains.items()
        if paths - route_paths
    }

    assert missing == {}


def test_shared_templates_render_global_operation_stack():
    client = TestClient(main_app.app)
    for route_path in ["/", "/feature", "/training", "/production", "/registry", "/editor"]:
        response = client.get(route_path)
        assert response.status_code == 200
        assert 'id="globalOperationStack"' in response.text
        assert 'id="globalOperationList"' in response.text


def test_panel_layout_coercion_handles_invalid_client_values():
    widgets = [{"id": "objective"}, {"id": "notes"}]

    layout = main_app._coerce_panel_layout(
        {"version": "bad", "columns": "not-a-number", "order": "bad-order"},
        widgets,
    )

    assert layout["version"] == 1
    assert layout["columns"] == 12
    assert layout["order"] == ["objective", "notes"]
    assert main_app._next_panel_version("bad") == 2


def test_queue_runtime_dependencies_are_declared_and_importable(project_root):
    api_requirements = (project_root / "api" / "requirements.txt").read_text(encoding="utf-8")
    shared_requirements = (project_root / "dependencies" / "requirements.txt").read_text(encoding="utf-8")

    assert importlib.util.find_spec("redis") is not None
    assert importlib.util.find_spec("rq") is not None
    assert "redis>=5.0,<7.0" in api_requirements
    assert "rq>=2.0,<3.0" in api_requirements
    assert "redis>=5.0,<7.0" in shared_requirements
    assert "rq>=2.0,<3.0" in shared_requirements


def test_compose_api_waits_for_redis_before_starting(project_root):
    compose_text = (project_root / "docker-compose.yaml").read_text(encoding="utf-8")
    api_service_start = compose_text.index("  api:\n")
    worker_service_start = compose_text.index("  api-worker-cpu:\n")
    api_service = compose_text[api_service_start:worker_service_start]

    assert "AMANAJE_REDIS_URL: \"redis://redis:6379/0\"" in api_service
    assert "redis:\n        condition: service_healthy" in api_service


def test_compose_worker_has_healthcheck_and_host_metadata(project_root):
    compose_text = (project_root / "docker-compose.yaml").read_text(encoding="utf-8")
    worker_service_start = compose_text.index("  api-worker-cpu:\n")
    assistant_service_start = compose_text.index("\n  assistant-server:\n", worker_service_start)
    worker_service = compose_text[worker_service_start:assistant_service_start]

    assert "image: painel-amanaje-api:local" in worker_service
    assert "AMANAJE_WORKER_HOST_ID: \"compose-cpu\"" in worker_service
    assert "python\", \"-m\", \"app.workers.healthcheck\", \"amanaje:default\"" in worker_service


def test_execute_job_endpoint_enqueues_and_records_ledger(monkeypatch, tmp_path):
    captured = {}

    def fake_enqueue(run_id, *, code):
        captured["run_id"] = run_id
        captured["code"] = code
        return {"backend": "rq", "queue": "amanaje:default", "job_id": run_id}

    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app, "enqueue_editor_execution", fake_enqueue)

    response = TestClient(main_app.app).post("/execute/jobs", json={"code": "x = 1"})
    payload = response.json()
    ledger = main_app._utils.read_run_entry(tmp_path, payload["run_id"])

    assert response.status_code == 202
    assert captured["code"] == "x = 1"
    assert ledger["run_type"] == "editor_execution"
    assert ledger["queue"]["queue"] == "amanaje:default"


def test_execute_job_endpoint_accepts_registry_context(monkeypatch, tmp_path):
    captured = {}
    dataset_record = SimpleNamespace(id=3, name="dataset-a", path="dataset.csv")
    model_record = SimpleNamespace(id=5, name="model-a", path="model.pkl", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 3
        return dataset_record

    async def fake_model_read(_db, model_id):
        assert model_id == 5
        return model_record

    def fake_enqueue(run_id, *, code, registry_context=None):
        captured["run_id"] = run_id
        captured["code"] = code
        captured["registry_context"] = registry_context
        return {"backend": "rq", "queue": "amanaje:default", "job_id": run_id}

    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app, "enqueue_editor_execution", fake_enqueue)

    response = TestClient(main_app.app).post(
        "/execute/jobs",
        json={"code": "df = load_dataset()", "registryContext": {"dataset_ids": [3], "model_ids": [5]}},
    )
    payload = response.json()
    ledger = main_app._utils.read_run_entry(tmp_path, payload["run_id"])

    assert response.status_code == 202
    assert captured["registry_context"]["datasets"][0]["id"] == 3
    assert captured["registry_context"]["models"][0]["id"] == 5
    assert ledger["context"]["dataset_ids"] == [3]
    assert ledger["context"]["model_ids"] == [5]


def test_run_control_routes_update_cooperative_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app, "cancel_rq_job", lambda run_id: {"backend": "rq", "job_id": run_id, "cancelled": True})
    monkeypatch.setattr(
        main_app,
        "enqueue_training_run",
        lambda run_id, *, model_id, payload_data, prefer_gpu=False: {
            "backend": "rq",
            "queue": "amanaje:test",
            "job_id": run_id,
            "model_id": model_id,
            "payload_dataset_id": payload_data["datasetId"],
            "prefer_gpu": prefer_gpu,
        },
    )
    queued = create_run_entry(tmp_path, run_type="training", status="queued")
    completed = create_run_entry(
        tmp_path,
        run_type="training",
        status="completed",
        context={"model_id": 5, "dataset_id": 7, "study_id": None, "inference_id": 9},
        parameters={"framework": "sklearn", "alpha": 0.1},
    )
    unsupported = create_run_entry(tmp_path, run_type="editor_execution", status="completed")
    client = TestClient(main_app.app)

    paused = client.post(f"/runs/{queued['run_id']}/pause")
    resumed = client.post(f"/runs/{queued['run_id']}/resume")
    cancelled = client.post(f"/runs/{queued['run_id']}/cancel")
    completed_cancel = client.post(f"/runs/{completed['run_id']}/cancel")
    rerun = client.post(f"/runs/{completed['run_id']}/rerun")
    active_rerun = client.post(f"/runs/{queued['run_id']}/rerun")
    unsupported_rerun = client.post(f"/runs/{unsupported['run_id']}/rerun")
    missing = client.post("/runs/missing_run/cancel")

    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "queued"
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert completed_cancel.status_code == 400
    assert rerun.status_code == 202
    assert rerun.json()["status"] == "accepted"
    assert rerun.json()["source_run_id"] == completed["run_id"]
    assert rerun.json()["ledger"]["context"]["rerun_of"] == completed["run_id"]
    assert read_run_entry(tmp_path, rerun.json()["run_id"])["context"]["model_id"] == 5
    assert active_rerun.status_code == 400
    assert unsupported_rerun.status_code == 400
    assert missing.status_code == 404


def test_execute_job_endpoint_returns_structured_queue_error(monkeypatch, tmp_path):
    def fake_enqueue(_run_id, *, code):
        assert code == "x = 1"
        raise QueueUnavailableError("Redis/RQ dependencies are not available: No module named 'redis'")

    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app, "enqueue_editor_execution", fake_enqueue)

    response = TestClient(main_app.app).post("/execute/jobs", json={"code": "x = 1"})
    payload = response.json()
    ledger = main_app._utils.read_run_entry(tmp_path, payload["run_id"])

    assert response.status_code == 503
    assert payload["status"] == "error"
    assert payload["error_code"] == "queue_backend_unavailable"
    assert payload["domain"] == "queue_runtime"
    assert payload["queue_backend"] == "rq"
    assert payload["errors"][0]["reason"] == "redis_rq_unavailable"
    assert ledger["status"] == "failed"
    assert ledger["error_code"] == "queue_backend_unavailable"


def test_assistant_job_endpoint_enqueues_and_records_ledger(monkeypatch, tmp_path):
    captured = {}

    def fake_enqueue(run_id, *, operation, payload_data):
        captured["run_id"] = run_id
        captured["operation"] = operation
        captured["payload_data"] = payload_data
        return {"backend": "rq", "queue": "amanaje:default", "job_id": run_id}

    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app, "enqueue_assistant_operation", fake_enqueue)

    response = TestClient(main_app.app).post("/assistant/jobs/draft", json={"prompt": "Draft a dataset."})
    payload = response.json()
    ledger = main_app._utils.read_run_entry(tmp_path, payload["run_id"])

    assert response.status_code == 202
    assert payload["operation"] == "draft"
    assert captured["operation"] == "draft"
    assert ledger["run_type"] == "assistant_operation"
    assert ledger["context"]["operation"] == "draft"
    assert ledger["queue"]["queue"] == "amanaje:default"


def test_editor_execution_worker_records_result(monkeypatch, tmp_path):
    monkeypatch.setattr(main_app._utils, "RUN_LEDGER_DIR", tmp_path)
    entry = create_run_entry(tmp_path, run_type="editor_execution", status="queued")

    run_editor_execution_job(entry["run_id"], "x = 42")
    ledger = read_run_entry(tmp_path, entry["run_id"])

    assert ledger["status"] == "completed"
    assert ledger["result"]["status"] == "success"
    assert "x" in ledger["result"]["variables"]


def test_runtime_accelerators_endpoint_is_stable_when_torch_missing(monkeypatch):
    monkeypatch.setattr(
        main_app,
        "get_torch_accelerator_status",
        lambda: {"torch_available": False, "cuda_available": False, "device_count": 0, "devices": []},
    )

    response = TestClient(main_app.app).get("/runtime/accelerators")
    payload = response.json()

    assert response.status_code == 200
    assert payload["torch_available"] is False
    assert payload["cuda_available"] is False


def test_runtime_workers_endpoint_reports_queue_status(monkeypatch):
    monkeypatch.setattr(
        main_app,
        "get_worker_runtime_status",
        lambda: {"status": "ok", "summary": {"active_workers": 1, "queued_jobs": 0}, "queues": []},
    )

    response = TestClient(main_app.app).get("/runtime/workers")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["summary"]["active_workers"] == 1


def test_training_gpu_queue_preference_only_for_explicit_cuda():
    assert main_app._prefers_gpu_queue({"framework": "pytorch", "device": "cuda"}) is True
    assert main_app._prefers_gpu_queue({"framework": "pytorch", "device": "auto"}) is False
    assert main_app._prefers_gpu_queue({"framework": "sklearn", "device": "cuda"}) is False


def test_log_snapshot_reads_known_logs_and_redacts_secrets():
    test_dir = _log_test_dir()
    try:
        log_dir = test_dir / "logs"
        log_dir.mkdir()
        activity_path = test_dir / "activity_log.jsonl"
        compose_file = test_dir / "docker-compose.yaml"
        compose_file.write_text("services: {}\n", encoding="utf-8")
        (log_dir / "app.log").write_text(
            "2026-05-06 10:00:00 | painel_amanaje | INFO | Boot token=super-token postgresql://airflow:db-pass@postgres/db\n",
            encoding="utf-8",
        )
        (log_dir / "main_app.log").write_text("", encoding="utf-8")
        activity_path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-05-06T10:01:00",
                    "event_type": "settings.test",
                    "message": "Saved password=activity-pass",
                    "details": {"api_key": "activity-key", "safe": "visible"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        payload = log_stream.collect_log_snapshot(
            log_dir=log_dir,
            activity_log_path=activity_path,
            compose_file=compose_file,
            source="all",
            tail=20,
            include_docker=False,
        )
        serialized = json.dumps(payload)

        assert payload["status"] == "ok"
        assert {entry["source"] for entry in payload["entries"]} >= {"app", "activity"}
        assert "super-token" not in serialized
        assert "db-pass" not in serialized
        assert "activity-pass" not in serialized
        assert "activity-key" not in serialized
        assert "[REDACTED]" in serialized
        assert "visible" in serialized
    finally:
        _cleanup_log_test_dir(test_dir)


def test_log_snapshot_reports_docker_unavailable_without_failing(monkeypatch):
    test_dir = _log_test_dir()
    try:
        log_dir = test_dir / "logs"
        log_dir.mkdir()
        activity_path = test_dir / "activity_log.jsonl"
        compose_file = test_dir / "docker-compose.yaml"
        compose_file.write_text("services: {}\n", encoding="utf-8")
        monkeypatch.setattr(log_stream.shutil, "which", lambda _name: None)

        payload = log_stream.collect_log_snapshot(
            log_dir=log_dir,
            activity_log_path=activity_path,
            compose_file=compose_file,
            source="container:api",
            tail=20,
            include_docker=True,
        )

        assert payload["status"] == "ok"
        assert payload["entries"] == []
        assert any("Docker CLI is unavailable" in warning for warning in payload["warnings"])
    finally:
        _cleanup_log_test_dir(test_dir)


def test_settings_logs_endpoint_returns_known_sources(monkeypatch):
    test_dir = _log_test_dir()
    try:
        log_dir = test_dir / "logs"
        log_dir.mkdir()
        activity_path = test_dir / "activity_log.jsonl"
        activity_path.write_text(
            json.dumps({"timestamp": "2026-05-06T10:02:00", "event_type": "test", "message": "hello"}) + "\n",
            encoding="utf-8",
        )
        (test_dir / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
        monkeypatch.setattr(main_app._utils, "LOG_DIR", log_dir)
        monkeypatch.setattr(main_app._utils, "ACTIVITY_LOG_PATH", activity_path)
        monkeypatch.setattr(main_app._utils, "REPO_ROOT", test_dir)

        response = asyncio.run(
            main_app.settings_logs(
                _settings_logs_request(),
                source="activity",
                tail=5,
                include_docker=False,
            )
        )
        payload = _payload(response)

        assert response.status_code == 200
        assert payload["source"] == "activity"
        assert payload["entries"][0]["message"] == "hello"
        assert {"all", "app", "main_app", "activity", "container:api"}.issubset(
            {source["id"] for source in payload["sources"]}
        )
    finally:
        _cleanup_log_test_dir(test_dir)


def test_settings_logs_endpoint_rejects_unknown_sources():
    response = asyncio.run(
        main_app.settings_logs(
            _settings_logs_request(),
            source="../logs/app.log",
            tail=5,
            include_docker=False,
        )
    )
    payload = _payload(response)

    assert response.status_code == 400
    assert payload["status"] == "error"
    assert payload["errors"][0]["reason"] == "unknown_log_source"


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


def test_feature_extraction_route_profiles_registered_dataset(monkeypatch):
    dataset = SimpleNamespace(id=12, name="feature-source", history=[{"operation": "upload"}])
    dataframe = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [10.0, 11.0, 12.0]})
    captured: dict[str, object] = {}

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 12
        return dataset

    async def fake_update_entry(_db, orm, item_id, payload):
        captured["updated"] = {"orm": orm, "item_id": item_id, "payload": payload}
        return SimpleNamespace(**payload)

    def fake_save_json(payload, path):
        captured["artifact"] = {"payload": payload, "path": path}

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())
    monkeypatch.setattr(main_app, "save_json", fake_save_json)
    monkeypatch.setattr(main_app, "update_entry", fake_update_entry)

    response = asyncio.run(main_app.extract_feature_candidates(12, db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["dataset_id"] == 12
    assert payload["dataset_name"] == "feature-source"
    assert payload["summary"]["feature_candidates"]
    assert payload["summary"]["target_candidate"] == "target"
    assert len(payload["preview_rows"]) == 3
    assert payload["column_explorer"]
    assert payload["metric_cards"]
    assert captured["artifact"]["path"].name == "dataset_12_features.json"
    assert captured["updated"]["item_id"] == 12


def test_generate_preserves_legacy_script_shape_and_returns_reviewed_draft():
    response = TestClient(main_app.app).post(
        "/generate",
        json={"prompt": "Create a small synthetic dataset for regression.", "operationId": "datasets"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert "script" in payload
    assert "csv_text" in payload["script"]
    assert payload["assistant_plan"]["target_type"] == "dataset_generation"
    assert payload["draft"]["draft_type"] == "dataset_generation"
    assert payload["review"]["status"] in {"approved", "needs_revision"}


def test_mlflow_redirect_uses_browser_public_url_for_container_tracking_uri(monkeypatch):
    monkeypatch.setitem(main_app._utils.RUNTIME_CONFIG["mlflow"], "tracking_uri", "http://mlflow:5000")

    response = asyncio.run(main_app.mlflow_redirect())

    assert response.status_code in {307, 308}
    assert response.headers["location"] == "http://localhost:5000"


def test_analysis_dataset_response_includes_object_metadata_and_artifacts(monkeypatch):
    dataset = SimpleNamespace(
        id=5,
        name="analysis-dataset",
        path="runtime_artifacts/training/analysis-dataset.csv",
        history=[{"operation": "upload"}],
    )
    dataframe = pd.DataFrame({"feature": [1.0, 2.0], "target": [0.1, 0.2]})

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 5
        return dataset

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())
    monkeypatch.setattr(main_app, "_list_plot_artifacts", lambda **_kwargs: [{"id": "plot-1", "kind": "legacy_image"}])
    monkeypatch.setattr(main_app, "save_json", lambda *_args, **_kwargs: None)

    response = asyncio.run(main_app.analysis_dataset(5, db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["registry_type"] == "DatasetModel"
    assert payload["item_id"] == 5
    assert payload["artifacts"] == [{"id": "plot-1", "kind": "legacy_image"}]
    assert payload["history"] == [{"operation": "upload"}]


def test_analysis_model_response_includes_object_metadata_and_artifacts(monkeypatch):
    model = SimpleNamespace(
        id=8,
        name="analysis-model",
        model_type="supervised_model",
        parameters={"framework": "sklearn"},
        metrics={"eval_rmse": 0.4},
        history=[{"operation": "training"}],
        path=None,
    )

    async def fake_model_read(_db, model_id):
        assert model_id == 8
        return model

    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app, "_list_plot_artifacts", lambda **_kwargs: [{"id": "plot-2", "kind": "legacy_image"}])

    response = asyncio.run(main_app.analysis_model(8, db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["registry_type"] == "LearningModel"
    assert payload["item_id"] == 8
    assert payload["artifacts"] == [{"id": "plot-2", "kind": "legacy_image"}]
    assert payload["history"] == [{"operation": "training"}]


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
    payload = json.loads(response.body.decode("utf-8"))
    payload_text = json.dumps(payload, separators=(",", ":"))

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["output_features"] == ["target"]
    assert payload["output_feature"] == "target"
    assert payload["prediction_summary"]["label"] == "target"
    assert payload["simulation"]["series"][0]["output_feature"] == "target"
    assert '"category_feature":"stable"' in payload_text
    assert '"Feature Drift Applied"' in payload_text


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


def test_production_simulation_context_returns_feature_metadata(monkeypatch):
    model_record = SimpleNamespace(
        id=10,
        name="model-a",
        path="runtime_artifacts/models/model.pkl",
        parameters={"framework": "sklearn"},
        metrics={"eval_rmse": 0.2},
        input_features=["numeric_feature", "category_feature"],
        output_features=["target"],
    )
    dataset_record = SimpleNamespace(id=20, name="dataset-a", path="runtime_artifacts/datasets/dataset.csv")
    inference_record = SimpleNamespace(
        id=30,
        name="model-a :: dataset-a",
        learning_model_id=10,
        dataset_id=20,
        input_features=["numeric_feature", "category_feature"],
        output_features=["target"],
        inference_params={
            "simulation_defaults": {"steps": 6, "amplitude": 0.2},
            "latest_simulation": {"series": [{"step": 1, "prediction": 1.0}]},
        },
        history=[],
        path="runtime_artifacts/models/model.pkl",
    )
    prepared = SimpleNamespace(
        input_features=["numeric_feature", "category_feature"],
        output_feature="target",
        feature_frame=pd.DataFrame({"numeric_feature": [1.0, 3.0], "category_feature": ["a", "b"]}),
        task_type="regression",
        label_mapping={},
    )
    runtime_result = SimpleNamespace(
        framework="sklearn",
        parameters={"artifact_manifest": {"runtime_capabilities": {"predict": True}}},
    )

    class FakeRequest:
        query_params = {"modelId": "10", "datasetId": "20", "inferenceId": "30"}

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, runtime_result

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)

    response = asyncio.run(main_app.production_simulation_context(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["context"]["output_features"] == ["target"]
    assert payload["context"]["runtime_ready"] is True
    assert payload["context"]["simulation_defaults"]["steps"] == 6
    assert payload["context"]["feature_stats"][0]["name"] == "numeric_feature"
    assert payload["context"]["latest_simulation"]["series"][0]["prediction"] == 1.0
    assert payload["context"]["simulation_profile"]["family"] == "regression"
    assert payload["context"]["simulation_profile"]["default_mode"] == "tabular_what_if"
    assert payload["context"]["simulation_profile"]["controls_schema"]["sections"]


def test_production_simulate_supports_named_scenarios_and_sensitivity(monkeypatch):
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
        input_features=["numeric_feature", "category_feature"],
        output_feature="target",
        feature_frame=pd.DataFrame({"numeric_feature": [10.0], "category_feature": ["stable"]}),
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
                "baselineSource": "last",
                "scenarios": [
                    {"name": "Category Boost", "overrides": {"category_feature": "boost"}, "steps": 2, "amplitude": 0},
                    {"name": "Numeric Lift", "overrides": {"numeric_feature": 20}, "steps": 2, "amplitude": 0},
                ],
                "sensitivity": {"enabled": True, "features": ["numeric_feature"], "amplitude": 0.1},
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    async def fake_upsert_inference_record(*_args, **_kwargs):
        return inference_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, SimpleNamespace(framework="sklearn", parameters={}, model=SimpleNamespace())

    def fake_predict_with_result(_runtime_result, frame, _task_type):
        return np.asarray(
            [
                float(row["numeric_feature"]) + (5.0 if row["category_feature"] == "boost" else 0.0)
                for _, row in frame.iterrows()
            ]
        )

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)
    monkeypatch.setattr(main_app, "_upsert_inference_record", fake_upsert_inference_record)
    monkeypatch.setattr(main_app._utils, "_predict_with_result", fake_predict_with_result)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))
    payload_text = json.dumps(payload, separators=(",", ":"))

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert len(payload["simulation"]["scenarios"]) == 2
    assert payload["simulation"]["scenarios"][0]["name"] == "Category Boost"
    assert payload["simulation"]["scenarios"][0]["prediction_summary"]["final_prediction"] == 15.0
    assert payload["simulation"]["plots"]
    assert payload["simulation"]["scenarios"][0]["plots"]
    assert any(plot["title"] == "Category Boost Prediction Path" for plot in payload["simulation"]["scenarios"][0]["plots"])
    assert payload["simulation"]["sensitivity"]["enabled"] is True
    assert payload["simulation"]["sensitivity"]["features"][0]["feature"] == "numeric_feature"
    assert '"category_feature":"boost"' in payload_text


def test_production_simulate_supports_classification_probabilities(monkeypatch):
    model_record = SimpleNamespace(
        id=10,
        name="classifier-a",
        path="runtime_artifacts/models/model.pkl",
        parameters={"framework": "sklearn"},
        metrics={"accuracy": 0.9},
        input_features=["numeric_feature"],
        output_features=["target"],
    )
    dataset_record = SimpleNamespace(id=20, name="dataset-a")
    inference_record = SimpleNamespace(
        id=30,
        name="classifier-a :: dataset-a",
        learning_model_id=10,
        dataset_id=20,
        input_features=["numeric_feature"],
        output_features=["target"],
        inference_params={},
        history=[],
        path="runtime_artifacts/models/model.pkl",
    )
    prepared = SimpleNamespace(
        input_features=["numeric_feature"],
        output_feature="target",
        feature_frame=pd.DataFrame({"numeric_feature": [10.0]}),
        task_type="classification",
        label_mapping={"no": 0, "yes": 1},
    )

    class FakeClassifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            assert len(frame) == 2
            return np.asarray([[0.7, 0.3], [0.2, 0.8]])

    class FakeRequest:
        headers = {"content-type": "application/json"}

        async def json(self):
            return {
                "modelId": 10,
                "datasetId": 20,
                "inferenceId": 30,
                "mode": "classification_threshold",
                "steps": 2,
                "familyPayload": {"threshold": 0.4, "topK": 2},
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    async def fake_upsert_inference_record(*_args, **_kwargs):
        return inference_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, SimpleNamespace(framework="sklearn", parameters={}, model=FakeClassifier())

    def fake_predict_with_result(_runtime_result, _frame, _task_type):
        return np.asarray([0, 1])

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)
    monkeypatch.setattr(main_app, "_upsert_inference_record", fake_upsert_inference_record)
    monkeypatch.setattr(main_app._utils, "_predict_with_result", fake_predict_with_result)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["simulation"]["family"] == "classification"
    assert payload["simulation"]["mode"] == "classification_threshold"
    assert payload["simulation"]["classification"]["threshold"] == 0.4
    assert payload["simulation"]["classification"]["top_classes"][0]["class"] == "yes"
    assert payload["simulation"]["prediction_summary"]["final_probabilities"]["yes"] == 0.8
    assert any("Class Probabilities" in plot["title"] for plot in payload["simulation"]["scenarios"][0]["plots"])


def test_production_simulation_context_respects_saved_family_profile(monkeypatch):
    model_record = SimpleNamespace(
        id=10,
        name="model-a",
        path="runtime_artifacts/models/model.pkl",
        parameters={"framework": "sklearn"},
        metrics={"eval_rmse": 0.2},
        input_features=["numeric_feature"],
        output_features=["target"],
    )
    dataset_record = SimpleNamespace(id=20, name="dataset-a", path="runtime_artifacts/datasets/dataset.csv")
    inference_record = SimpleNamespace(
        id=30,
        name="model-a :: dataset-a",
        learning_model_id=10,
        dataset_id=20,
        input_features=["numeric_feature"],
        output_features=["target"],
        inference_params={"simulation_profile": {"family": "recommendation", "default_mode": "recommendation_top_n"}},
        history=[],
        path="runtime_artifacts/models/model.pkl",
    )
    prepared = SimpleNamespace(
        input_features=["numeric_feature"],
        output_feature="target",
        feature_frame=pd.DataFrame({"numeric_feature": [1.0, 3.0]}),
        task_type="regression",
        label_mapping={},
    )
    runtime_result = SimpleNamespace(framework="sklearn", parameters={}, model=SimpleNamespace())

    class FakeRequest:
        query_params = {"modelId": "10", "datasetId": "20", "inferenceId": "30"}

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, runtime_result

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)

    response = asyncio.run(main_app.production_simulation_context(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["context"]["simulation_profile"]["family"] == "recommendation"
    assert payload["context"]["simulation_profile"]["default_mode"] == "recommendation_top_n"
    assert payload["context"]["simulation_profile"]["disabled_reasons"]


def test_production_simulate_returns_readiness_for_non_runnable_family(monkeypatch):
    model_record = SimpleNamespace(
        id=10,
        name="model-a",
        path="runtime_artifacts/models/model.pkl",
        parameters={"framework": "sklearn"},
        input_features=["numeric_feature"],
        output_features=["target"],
    )
    dataset_record = SimpleNamespace(id=20, name="dataset-a")
    inference_record = SimpleNamespace(
        id=30,
        name="model-a :: dataset-a",
        learning_model_id=10,
        dataset_id=20,
        input_features=["numeric_feature"],
        output_features=["target"],
        inference_params={"simulation_profile": {"family": "recommendation", "default_mode": "recommendation_top_n"}},
        history=[],
        path="runtime_artifacts/models/model.pkl",
    )
    prepared = SimpleNamespace(
        input_features=["numeric_feature"],
        output_feature="target",
        feature_frame=pd.DataFrame({"numeric_feature": [10.0]}),
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
                "mode": "recommendation_top_n",
                "familyPayload": {"userId": "u-1", "topN": 5},
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    async def fake_upsert_inference_record(*_args, **_kwargs):
        return inference_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, SimpleNamespace(framework="sklearn", parameters={}, model=SimpleNamespace())

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)
    monkeypatch.setattr(main_app, "_upsert_inference_record", fake_upsert_inference_record)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["simulation"]["family"] == "recommendation"
    assert payload["simulation"]["status"] == "not_ready"
    assert payload["simulation"]["readiness"]["ready"] is False
    assert "recommend" in payload["simulation"]["readiness"]["disabled_reasons"][0].lower()


def test_production_simulate_rejects_unknown_scenario_features(monkeypatch):
    model_record = SimpleNamespace(id=10, name="model-a", path="runtime_artifacts/models/model.pkl", output_features=["target"])
    dataset_record = SimpleNamespace(id=20, name="dataset-a")
    inference_record = SimpleNamespace(id=30, name="model-a :: dataset-a", output_features=["target"])
    prepared = SimpleNamespace(
        input_features=["numeric_feature"],
        output_feature="target",
        feature_frame=pd.DataFrame({"numeric_feature": [10.0]}),
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
                "scenario": {"ghost_feature": 1.0},
            }

    async def fake_resolve_runtime_context(*_args, **_kwargs):
        return inference_record, model_record, dataset_record

    def fake_prepare_runtime_execution(*_args, **_kwargs):
        return prepared, SimpleNamespace(framework="sklearn", parameters={}, model=SimpleNamespace())

    monkeypatch.setattr(main_app, "_resolve_runtime_context", fake_resolve_runtime_context)
    monkeypatch.setattr(main_app, "_prepare_runtime_execution", fake_prepare_runtime_execution)

    response = asyncio.run(main_app.production_simulate(FakeRequest(), db=object()))
    payload = response.body.decode("utf-8")

    assert response.status_code == 400
    assert "Unknown simulation feature" in payload
    assert "ghost_feature" in payload


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
