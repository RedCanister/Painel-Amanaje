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
        ("/production", "Global Production Watch"),
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

    def fake_create_task(coro):
        coro.close()
        return SimpleNamespace(done=lambda: False)

    monkeypatch.setattr(main_app._utils, "create_run_entry", fake_create_run_entry)
    monkeypatch.setattr(main_app.asyncio, "create_task", fake_create_task)

    payload = main_app.TrainingRequest.model_validate(
        {"datasetId": 21, "inputType": "Manual", "studyId": None, "parameters": {}, "inferenceId": 31}
    )

    response = asyncio.run(main_app.post_training(11, payload=payload, db=object()))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 202
    assert captured["run_type"] == "training"
    assert captured["context"]["inference_id"] == 31
    assert body["ledger"]["context"]["inference_id"] == 31


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

    def fake_create_task(coro):
        coro.close()
        return SimpleNamespace(done=lambda: False)

    monkeypatch.setattr(main_app._utils, "create_run_entry", fake_create_run_entry)
    monkeypatch.setattr(main_app.asyncio, "create_task", fake_create_task)

    payload = main_app.StudyOptimizationRequest.model_validate({"nTrials": 3, "inferenceId": 31})
    response = asyncio.run(main_app.post_optimize_study(15, payload=payload, db=object()))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 202
    assert captured["run_type"] == "study"
    assert captured["context"]["inference_id"] == 31
    assert body["ledger"]["context"]["inference_id"] == 31


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
