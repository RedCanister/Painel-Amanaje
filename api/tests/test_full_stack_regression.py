from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main_app
from app.utils import main_utils
from app.utils.operation_queue import QueueUnavailableError
from tests.support import dataset_fixture


pytestmark = pytest.mark.integration


@pytest.fixture()
def page_client():
    return TestClient(main_app.app)


@pytest.mark.parametrize(
    ("route_path", "expected_fragment"),
    [
        ("/upload", "Data Upload"),
        ("/create", "Data Creation"),
        ("/feature", "Feature Workspace"),
        ("/training", "Global Training Context"),
        ("/onnx", "ONNX Framework Workspace"),
        ("/production", "Global Production Watch"),
        ("/visualization", "Visualization"),
        ("/plot", "Visualization"),
        ("/registry", "Registry"),
    ],
)
def test_primary_pages_render_without_server_errors(page_client, route_path, expected_fragment):
    response = page_client.get(route_path)

    assert response.status_code == 200
    assert expected_fragment in response.text


def test_github_vs_git_fixture_reproduces_schema_mismatch_preflight(monkeypatch):
    dataset_path = dataset_fixture("github_vs_git_schema_mismatch.csv")
    dataset_record = SimpleNamespace(id=21, name="github-vs-git", path=str(dataset_path))
    model_record = SimpleNamespace(
        id=15,
        name="github_predictor",
        input_features=[
            "public_repos",
            "followers",
            "account_age_days",
            "actor_login",
            "repo_language",
        ],
        output_features=["expected_engagement_score"],
    )

    monkeypatch.setattr(main_utils, "_load_dataset_frame_from_record", lambda _record: pd.read_csv(dataset_path))

    preflight = main_utils._build_training_preflight(dataset_record, model_record, {})

    assert preflight["ok"] is False
    assert preflight["missing_columns"] == [
        "public_repos",
        "followers",
        "account_age_days",
        "actor_login",
        "repo_language",
        "expected_engagement_score",
    ]
    assert "repo_name" in preflight["available_columns"]


def test_feature_preview_accepts_visual_transform_payloads(monkeypatch):
    dataset = SimpleNamespace(id=13, name="wide-dataset")
    dataframe = pd.DataFrame(
        {
            "raw_amount": [1.0, None, 3.0],
            "category": ["a", "b", "a"],
            "target": [0, 1, 0],
        }
    )

    class FakeRequest:
        async def json(self):
            return {
                "datasetId": 13,
                "limitRows": 2,
                "transforms": [
                    {"type": "rename", "column": "raw_amount", "value": "normalized_amount"},
                    {"type": "fill", "column": "normalized_amount", "value": 0},
                    {"type": "sort", "column": "target", "ascending": False},
                ],
            }

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 13
        return dataset

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())

    response = asyncio.run(main_app.preview_feature_workspace(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["dataset_name"] == "wide-dataset"
    assert payload["operations"]["rename_columns"]["raw_amount"] == "normalized_amount"
    assert payload["preview_rows"][0]["target"] == 1
    assert any(column["name"] == "normalized_amount" for column in payload["column_explorer"])


def test_feature_preview_accepts_explicit_limit_transform(monkeypatch):
    dataset = SimpleNamespace(id=14, name="limit-dataset")
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    class FakeRequest:
        async def json(self):
            return {
                "datasetId": 14,
                "previewRows": 10,
                "transforms": [{"type": "limit", "value": 1}],
            }

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 14
        return dataset

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())

    response = asyncio.run(main_app.preview_feature_workspace(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["preview"]["shape_after"] == [1, 2]
    assert payload["preview_rows"] == [{"feature": 1, "target": 0}]


@pytest.mark.parametrize(
    ("sql_query", "expected_columns"),
    [
        ("SELECT feature, target FROM source_dataset", ["feature", "target"]),
        (
            "WITH transformed AS (SELECT feature * 2 AS feature_x2, target FROM source_dataset) "
            "SELECT feature_x2, target FROM transformed",
            ["feature_x2", "target"],
        ),
    ],
)
def test_feature_sql_preview_accepts_select_and_cte_queries(monkeypatch, sql_query, expected_columns):
    dataset = SimpleNamespace(id=15, name="sql-dataset")
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    class FakeRequest:
        async def json(self):
            return {
                "mode": "sql",
                "datasetId": 15,
                "sqlQuery": sql_query,
                "previewRows": 2,
            }

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 15
        return dataset

    async def fake_execute_sql(_db, frame, query, *, preview_rows=None):
        statement = main_utils.validate_feature_sql_query(query)
        assert statement == sql_query
        assert preview_rows == 2
        assert frame.equals(dataframe)
        if "feature_x2" in query:
            result = pd.DataFrame({"feature_x2": [2, 4], "target": [0, 1]})
        else:
            result = frame[expected_columns].head(2).copy()
        return result, {
            "shape_before": [int(frame.shape[0]), int(frame.shape[1])],
            "shape_after": [int(result.shape[0]), int(result.shape[1])],
            "steps": [{"operation": "sql_query"}],
            "preview_rows": result.to_dict(orient="records"),
            "column_explorer": [],
        }

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())
    monkeypatch.setattr(main_app._utils, "execute_feature_sql_query", fake_execute_sql)

    response = asyncio.run(main_app.preview_feature_workspace(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["operations"]["mode"] == "sql"
    assert payload["operations"]["source_relation"] == "source_dataset"
    assert payload["operations"]["sql_hash"]
    assert list(payload["preview_rows"][0].keys()) == expected_columns


@pytest.mark.parametrize(
    "sql_query",
    [
        "SELECT * FROM source_dataset; SELECT * FROM source_dataset",
        "DROP TABLE source_dataset",
        "UPDATE source_dataset SET feature = 0",
    ],
)
def test_feature_sql_preview_rejects_unsafe_queries(monkeypatch, sql_query):
    dataset = SimpleNamespace(id=16, name="unsafe-sql-dataset")
    dataframe = pd.DataFrame({"feature": [1], "target": [0]})

    class FakeRequest:
        async def json(self):
            return {"mode": "sql", "datasetId": 16, "sqlQuery": sql_query}

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 16
        return dataset

    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())

    response = asyncio.run(main_app.preview_feature_workspace(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert payload["status"] == "error"


def test_feature_sql_materialization_records_dataset_metadata(monkeypatch, tmp_path):
    source_dataset = SimpleNamespace(id=17, name="sql-source")
    source_frame = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    transformed = pd.DataFrame({"feature_x2": [2, 4, 6], "target": [0, 1, 0]})
    captured: dict[str, object] = {}

    class FakeRequest:
        async def json(self):
            return {
                "mode": "sql",
                "datasetId": 17,
                "name": "sql_feature_view",
                "sqlQuery": "SELECT feature * 2 AS feature_x2, target FROM source_dataset",
            }

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 17
        return source_dataset

    async def fake_execute_sql(_db, frame, query, *, preview_rows=None):
        assert preview_rows is None
        assert frame.equals(source_frame)
        return transformed.copy(), {
            "shape_before": [int(frame.shape[0]), int(frame.shape[1])],
            "shape_after": [int(transformed.shape[0]), int(transformed.shape[1])],
            "steps": [{"operation": "sql_query"}],
            "preview_rows": transformed.head(20).to_dict(orient="records"),
            "column_explorer": [],
        }

    async def fake_create_entry(_db, orm, payload):
        captured["payload"] = payload
        return SimpleNamespace(id=99, **payload)

    monkeypatch.setattr(main_app, "FEATURE_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: source_frame.copy())
    monkeypatch.setattr(main_app._utils, "execute_feature_sql_query", fake_execute_sql)
    monkeypatch.setattr(main_app, "create_entry", fake_create_entry)
    monkeypatch.setattr(main_app, "save_json", lambda _payload, _path: None)

    response = asyncio.run(main_app.materialize_feature_workspace(FakeRequest(), db=object()))
    payload = json.loads(response.body.decode("utf-8"))
    created = captured["payload"]
    history = created["history"][0]

    assert response.status_code == 200
    assert payload["dataset"]["name"] == "sql_feature_view"
    assert payload["postgres"] == {"table_name": "feature_17_sqlfeatureview", "materialized": False}
    assert created["shape"] == [3, 2]
    assert created["features_list"] == ["feature_x2", "target"]
    assert created["connection_string"] == "table:feature_17_sqlfeatureview"
    assert history["mode"] == "sql"
    assert history["sql_hash"]
    assert history["source_relation"] == "source_dataset"


def test_feature_visual_materialization_ignores_preview_limit_without_explicit_limit(monkeypatch, tmp_path):
    source_dataset = SimpleNamespace(id=18, name="visual-source")
    source_frame = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    captured: dict[str, object] = {}

    class FakeRequest:
        async def json(self):
            return {
                "datasetId": 18,
                "name": "visual_feature_view",
                "limitRows": 1,
                "transforms": [{"type": "rename", "column": "feature", "value": "renamed_feature"}],
            }

    async def fake_dataset_read(_db, dataset_id):
        assert dataset_id == 18
        return source_dataset

    async def fake_create_entry(_db, orm, payload):
        captured["payload"] = payload
        return SimpleNamespace(id=100, **payload)

    monkeypatch.setattr(main_app, "FEATURE_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_load_dataset_frame_from_record", lambda _record: source_frame.copy())
    monkeypatch.setattr(main_app, "create_entry", fake_create_entry)
    monkeypatch.setattr(main_app, "save_json", lambda _payload, _path: None)

    response = asyncio.run(main_app.materialize_feature_workspace(FakeRequest(), db=object()))
    created = captured["payload"]

    assert response.status_code == 200
    assert created["shape"] == [3, 2]
    assert created["features_list"] == ["renamed_feature", "target"]
    assert created["history"][0]["mode"] == "structured"


def test_feature_template_exposes_editable_sql_mode_without_implicit_limit_push():
    template = (main_app.PROJECT_ROOT / "templates" / "base_feature.html").read_text(encoding="utf-8")

    assert "Feature Workspace" in template
    assert "Preview Row Limit" in template
    assert "source_dataset" in template
    assert "mode: 'sql'" in template
    assert "sqlQuery:" in template
    assert 'id="featureSqlQuery" class="sql-preview" spellcheck="false" readonly' not in template
    assert "transforms.push({ type: 'limit'" not in template


def test_registry_dependencies_route_returns_dependency_report(monkeypatch):
    monkeypatch.setattr(
        main_app,
        "_get_registry_binding",
        lambda registry_type: {"label": f"{registry_type}-label", "orm": object()},
    )

    async def fake_get_entry_dependencies(_db, _orm_model, item_id):
        assert item_id == 16
        return {
            "exists": True,
            "can_delete": False,
            "dependencies": [{"label": "Linked studies", "count": 2, "field": "learning_model_id"}],
            "message": "This registry item is still referenced by other records.",
        }

    monkeypatch.setattr(main_app, "get_entry_dependencies", fake_get_entry_dependencies)

    response = asyncio.run(main_app.registry_dependencies("learningmodel", 16, db=object()))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["item_id"] == 16
    assert payload["dependency_report"]["can_delete"] is False
    assert payload["dependency_report"]["dependencies"][0]["label"] == "Linked studies"


def test_mlflow_experiment_route_exposes_scroll_metadata(monkeypatch):
    monkeypatch.setattr(main_app._utils, "_build_mlflow_health_snapshot", lambda: {"status": "ok", "warnings": []})
    monkeypatch.setattr(
        main_app,
        "get_experiment_summary",
        lambda exp_id, max_runs, offset: {
            "experiment_id": exp_id,
            "run_summary": {"has_more": True, "count": max_runs, "offset": offset},
            "runs": [{"run_id": "abc123"}],
        },
    )

    response = asyncio.run(main_app.mlflow_get_experiment("7", limit=25, offset=10))
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert payload["limit"] == 25
    assert payload["offset"] == 10
    assert payload["has_more"] is True
    assert payload["experiment"]["runs"][0]["run_id"] == "abc123"


def test_training_submission_preserves_inference_context_in_queued_ledger(monkeypatch):
    captured = {}

    async def fake_model_read(_db, _model_id):
        return SimpleNamespace(id=11, name="model-a", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, _dataset_id):
        return SimpleNamespace(id=21, name="dataset-a")

    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_merge_training_parameters", lambda *_args, **_kwargs: {"framework": "sklearn"})
    monkeypatch.setattr(main_app._utils, "_build_training_preflight", lambda *_args, **_kwargs: {"ok": True})

    def fake_create_run_entry(_run_dir, *, run_type, context, parameters, status):
        captured.update({"run_type": run_type, "context": context, "parameters": parameters, "status": status})
        return {"run_id": "training-test-run", "context": context, "parameters": parameters, "status": status}

    def fake_update_run_entry(_run_dir, run_id, **kwargs):
        captured["updated_run_id"] = run_id
        captured["update"] = kwargs
        return {"run_id": run_id, **kwargs}

    def fake_enqueue_training_run(run_id, *, model_id, payload_data, prefer_gpu=False):
        captured["enqueue"] = {
            "run_id": run_id,
            "model_id": model_id,
            "payload_data": payload_data,
            "prefer_gpu": prefer_gpu,
        }
        return {"backend": "rq", "queue": "amanaje:default", "job_id": run_id}

    monkeypatch.setattr(main_app._utils, "create_run_entry", fake_create_run_entry)
    monkeypatch.setattr(main_app._utils, "update_run_entry", fake_update_run_entry)
    monkeypatch.setattr(main_app, "enqueue_training_run", fake_enqueue_training_run)

    payload = main_app.TrainingRequest.model_validate(
        {"datasetId": 21, "inputType": "Manual", "studyId": None, "parameters": {}, "inferenceId": 31}
    )

    response = asyncio.run(main_app.post_training(11, payload=payload, db=object()))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 202
    assert captured["run_type"] == "training"
    assert captured["context"]["inference_id"] == 31
    assert captured["enqueue"]["run_id"] == "training-test-run"
    assert body["ledger"]["context"]["inference_id"] == 31


def test_training_submission_returns_structured_queue_unavailable_error(monkeypatch, tmp_path):
    async def fake_model_read(_db, _model_id):
        return SimpleNamespace(id=11, name="model-a", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, _dataset_id):
        return SimpleNamespace(id=21, name="dataset-a")

    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_merge_training_parameters", lambda *_args, **_kwargs: {"framework": "sklearn"})
    monkeypatch.setattr(main_app._utils, "_build_training_preflight", lambda *_args, **_kwargs: {"ok": True})

    def fake_enqueue_training_run(_run_id, *, model_id, payload_data, prefer_gpu=False):
        raise QueueUnavailableError("Redis/RQ dependencies are not available: No module named 'redis'")

    monkeypatch.setattr(main_app, "enqueue_training_run", fake_enqueue_training_run)

    payload = main_app.TrainingRequest.model_validate(
        {"datasetId": 21, "inputType": "Manual", "studyId": None, "parameters": {}}
    )
    response = asyncio.run(main_app.post_training(11, payload=payload, db=object()))
    body = json.loads(response.body.decode("utf-8"))
    ledger = main_app._utils.read_run_entry(tmp_path, body["run_id"])

    assert response.status_code == 503
    assert body["error_code"] == "queue_backend_unavailable"
    assert body["domain"] == "queue_runtime"
    assert body["errors"][0]["reason"] == "redis_rq_unavailable"
    assert ledger["status"] == "failed"


def test_study_submission_preserves_inference_context_in_queued_ledger(monkeypatch):
    captured = {}
    study_record = SimpleNamespace(id=15, learning_model_id=11, dataset_id=21)

    async def fake_study_read(_db, _study_id):
        return study_record

    async def fake_model_read(_db, _model_id):
        return SimpleNamespace(id=11, name="model-a", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, _dataset_id):
        return SimpleNamespace(id=21, name="dataset-a")

    monkeypatch.setattr(main_app.StudyModel, "read", fake_study_read)
    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_build_training_preflight", lambda *_args, **_kwargs: {"ok": True})

    def fake_create_run_entry(_run_dir, *, run_type, context, parameters, status):
        captured.update({"run_type": run_type, "context": context, "parameters": parameters, "status": status})
        return {"run_id": "study-test-run", "context": context, "parameters": parameters, "status": status}

    def fake_update_run_entry(_run_dir, run_id, **kwargs):
        captured["updated_run_id"] = run_id
        captured["update"] = kwargs
        return {"run_id": run_id, **kwargs}

    def fake_enqueue_study_run(run_id, *, study_id, payload_data, prefer_gpu=False):
        captured["enqueue"] = {
            "run_id": run_id,
            "study_id": study_id,
            "payload_data": payload_data,
            "prefer_gpu": prefer_gpu,
        }
        return {"backend": "rq", "queue": "amanaje:default", "job_id": run_id}

    monkeypatch.setattr(main_app._utils, "create_run_entry", fake_create_run_entry)
    monkeypatch.setattr(main_app._utils, "update_run_entry", fake_update_run_entry)
    monkeypatch.setattr(main_app, "enqueue_study_run", fake_enqueue_study_run)

    payload = main_app.StudyOptimizationRequest.model_validate({"nTrials": 3, "inferenceId": 31})
    response = asyncio.run(main_app.post_optimize_study(15, payload=payload, db=object()))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 202
    assert captured["run_type"] == "study"
    assert captured["context"]["inference_id"] == 31
    assert captured["enqueue"]["run_id"] == "study-test-run"
    assert body["ledger"]["context"]["inference_id"] == 31


def test_study_submission_returns_structured_queue_unavailable_error(monkeypatch, tmp_path):
    study_record = SimpleNamespace(id=15, learning_model_id=11, dataset_id=21)

    async def fake_study_read(_db, _study_id):
        return study_record

    async def fake_model_read(_db, _model_id):
        return SimpleNamespace(id=11, name="model-a", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, _dataset_id):
        return SimpleNamespace(id=21, name="dataset-a")

    monkeypatch.setattr(main_app, "RUN_LEDGER_DIR", tmp_path)
    monkeypatch.setattr(main_app.StudyModel, "read", fake_study_read)
    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
    monkeypatch.setattr(main_app._utils, "_build_training_preflight", lambda *_args, **_kwargs: {"ok": True})

    def fake_enqueue_study_run(_run_id, *, study_id, payload_data, prefer_gpu=False):
        raise QueueUnavailableError("Redis/RQ dependencies are not available: No module named 'redis'")

    monkeypatch.setattr(main_app, "enqueue_study_run", fake_enqueue_study_run)

    payload = main_app.StudyOptimizationRequest.model_validate({"nTrials": 3})
    response = asyncio.run(main_app.post_optimize_study(15, payload=payload, db=object()))
    body = json.loads(response.body.decode("utf-8"))
    ledger = main_app._utils.read_run_entry(tmp_path, body["run_id"])

    assert response.status_code == 503
    assert body["error_code"] == "queue_backend_unavailable"
    assert body["domain"] == "queue_runtime"
    assert body["errors"][0]["reason"] == "redis_rq_unavailable"
    assert ledger["status"] == "failed"


@pytest.mark.recovery
def test_study_submission_returns_preflight_validation_error(monkeypatch):
    study_record = SimpleNamespace(id=15, learning_model_id=11, dataset_id=21)

    async def fake_study_read(_db, _study_id):
        return study_record

    async def fake_model_read(_db, _model_id):
        return SimpleNamespace(id=11, name="broken-model", parameters={"framework": "sklearn"})

    async def fake_dataset_read(_db, _dataset_id):
        return SimpleNamespace(id=21, name="broken-dataset")

    monkeypatch.setattr(main_app.StudyModel, "read", fake_study_read)
    monkeypatch.setattr(main_app.LearningModel, "read", fake_model_read)
    monkeypatch.setattr(main_app.DatasetModel, "read", fake_dataset_read)
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

    response = asyncio.run(
        main_app.post_optimize_study(
            15,
            payload=main_app.StudyOptimizationRequest.model_validate({"nTrials": 5}),
            db=object(),
        )
    )
    payload = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert payload["missing_columns"] == ["expected_engagement_score"]
    assert payload["available_columns"] == ["followers", "public_repos"]
