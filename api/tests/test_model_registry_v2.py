from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.v2.model_registry import ModelRegistry, include_registered_routers, register_default_models
from app.models.v2.model_schemas import CodeRead, DatasetRead, LearningRead, ObjectRead, StudyRead


class FakeORM:
    __table__ = SimpleNamespace(columns=[])

    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


def _install_app(monkeypatch):
    register_default_models()
    app = FastAPI()
    include_registered_routers(app)

    async def fake_create_entry(db, orm_model, payload):
        _ = (db, orm_model)
        return FakeORM(id=1, **payload)

    async def fake_get_entry(db, orm_model, item_id):
        _ = (db, orm_model)

        if str(item_id) == "404":
            return None

        if orm_model.__name__ == "StudyORM":
            return FakeORM(
                id=1,
                name="study",
                description="desc",
                object_type="study_model",
                size=1.0,
                path="/tmp/study",
                date="2026-01-01T00:00:00",
                version=1,
                history=[],
                learning_model_id=2,
                dataset_id=3,
                sampler="tpe",
                objective="maximize",
                best_trial=None,
                best_params=None,
                study_params=None,
                learning_model=SimpleNamespace(id=2, name="model-a"),
                dataset=SimpleNamespace(id=3, name="dataset-a"),
            )

        base_payload = {
            "id": 1,
            "name": "sample",
            "description": "desc",
            "object_type": "dataset" if orm_model.__name__ == "DatasetORM" else "learning_model",
            "size": 1.0,
            "path": "/tmp/sample",
            "date": "2026-01-01T00:00:00",
            "version": 1,
            "history": [],
        }

        if orm_model.__name__ == "ObjectORM":
            return FakeORM(**base_payload)
        if orm_model.__name__ == "DatasetORM":
            return FakeORM(**base_payload, dataset_type="csv", shape=[1, 2], has_features=True, features_list=["a"], connection_string=None)
        if orm_model.__name__ == "LearningORM":
            return FakeORM(
                **base_payload,
                model_type="supervised",
                parameters={},
                metrics={},
                reference_data=None,
                input_features=["x1"],
                output_features=["y"],
                is_trained=False,
                is_tested=False,
                is_deployed=False,
            )
        if orm_model.__name__ == "CodeORM":
            return FakeORM(**base_payload, object_type="code_model", code={"script": "print(1)"}, variables={"x": 1})
        raise AssertionError(f"Unexpected ORM model {orm_model.__name__}")

    async def fake_get_all_entries(db, orm_model):
        entry = await fake_get_entry(db, orm_model, 1)
        return [entry] if entry else []

    async def fake_update_entry(db, orm_model, item_id, payload):
        existing = await fake_get_entry(db, orm_model, item_id)
        if existing is None:
            return None
        for key, value in payload.items():
            setattr(existing, key, value)
        return existing

    async def fake_delete_entry(db, orm_model, item_id):
        _ = (db, orm_model)
        return str(item_id) != "404"

    async def fake_get_db():
        yield object()

    monkeypatch.setattr("app.models.v2.model_registry.create_entry", fake_create_entry)
    monkeypatch.setattr("app.models.v2.model_registry.get_entry", fake_get_entry)
    monkeypatch.setattr("app.models.v2.model_registry.get_all_entries", fake_get_all_entries)
    monkeypatch.setattr("app.models.v2.model_registry.update_entry", fake_update_entry)
    monkeypatch.setattr("app.models.v2.model_registry.delete_entry", fake_delete_entry)
    monkeypatch.setattr("app.models.v2.model_registry.get_db", fake_get_db)

    return TestClient(app)


def test_default_models_are_registered():
    register_default_models()

    registered = set(ModelRegistry._registry)
    assert {ObjectRead, DatasetRead, LearningRead, CodeRead, StudyRead}.issubset(registered)


def test_dataset_crud_routes(monkeypatch):
    client = _install_app(monkeypatch)

    payload = {
        "name": "dataset-1",
        "description": "desc",
        "object_type": "dataset",
        "size": 1.5,
        "path": "/tmp/data.csv",
        "date": "2026-01-01T00:00:00",
        "version": 1,
        "history": [],
        "dataset_type": "csv",
        "shape": [10, 3],
        "has_features": True,
        "features_list": ["a", "b"],
        "connection_string": None,
    }

    assert client.post("/datasets-v2/create", json=payload).status_code == 201
    assert client.get("/datasets-v2/get/1").status_code == 200
    assert client.get("/datasets-v2/list").status_code == 200
    assert client.put("/datasets-v2/update/1", json={"description": "updated"}).status_code == 200
    assert client.delete("/datasets-v2/delete/1").status_code == 200


def test_learning_crud_routes(monkeypatch):
    client = _install_app(monkeypatch)

    payload = {
        "name": "model-1",
        "description": "desc",
        "object_type": "learning_model",
        "size": 2.5,
        "path": "/tmp/model.onnx",
        "date": "2026-01-01T00:00:00",
        "version": 1,
        "history": [],
        "model_type": "supervised",
        "parameters": {"alpha": 0.1},
        "metrics": {"rmse": 1.2},
        "reference_data": None,
        "input_features": ["x1", "x2"],
        "output_features": ["y"],
        "is_trained": False,
        "is_tested": False,
        "is_deployed": False,
    }

    assert client.post("/learning-models-v2/create", json=payload).status_code == 201
    assert client.get("/learning-models-v2/get/1").status_code == 200
    assert client.get("/learning-models-v2/list").status_code == 200
    assert client.put("/learning-models-v2/update/1", json={"is_trained": True}).status_code == 200
    assert client.delete("/learning-models-v2/delete/1").status_code == 200


def test_code_crud_routes(monkeypatch):
    client = _install_app(monkeypatch)

    payload = {
        "name": "code-1",
        "description": "desc",
        "object_type": "code_model",
        "size": 0.5,
        "path": "/tmp/code.py",
        "date": "2026-01-01T00:00:00",
        "version": 1,
        "history": [],
        "code": {"script": "print(1)"},
        "variables": {"x": 1},
    }

    assert client.post("/code-models-v2/create", json=payload).status_code == 201
    assert client.get("/code-models-v2/get/1").status_code == 200
    assert client.get("/code-models-v2/list").status_code == 200
    assert client.put("/code-models-v2/update/1", json={"variables": {"x": 2}}).status_code == 200
    assert client.delete("/code-models-v2/delete/1").status_code == 200


def test_study_crud_routes(monkeypatch):
    client = _install_app(monkeypatch)

    payload = {
        "name": "study-1",
        "description": "desc",
        "object_type": "study_model",
        "size": 0.5,
        "path": "/tmp/study.json",
        "date": "2026-01-01T00:00:00",
        "version": 1,
        "history": [],
        "learning_model_id": 2,
        "dataset_id": 3,
        "sampler": "tpe",
        "objective": "maximize",
        "best_trial": None,
        "best_params": None,
        "study_params": {"n_trials": 10},
    }

    assert client.post("/study-models-v2/create", json=payload).status_code == 201
    assert client.get("/study-models-v2/get/1").status_code == 200
    assert client.get("/study-models-v2/list").status_code == 200
    assert client.put("/study-models-v2/update/1", json={"objective": "minimize"}).status_code == 200
    assert client.delete("/study-models-v2/delete/1").status_code == 200


def test_object_crud_routes(monkeypatch):
    client = _install_app(monkeypatch)

    payload = {
        "name": "object-1",
        "description": "desc",
        "object_type": "object",
        "size": 0.2,
        "path": "/tmp/object",
        "date": "2026-01-01T00:00:00",
        "version": 1,
        "history": [],
    }

    assert client.post("/objects-v2/create", json=payload).status_code == 201
    assert client.get("/objects-v2/get/1").status_code == 200
    assert client.get("/objects-v2/list").status_code == 200
    assert client.put("/objects-v2/update/1", json={"description": "updated"}).status_code == 200
    assert client.delete("/objects-v2/delete/1").status_code == 200


def test_returns_404_when_resource_is_missing(monkeypatch):
    client = _install_app(monkeypatch)

    assert client.get("/datasets-v2/get/404").status_code == 404
    assert client.put("/datasets-v2/update/404", json={"description": "missing"}).status_code == 404
    assert client.delete("/datasets-v2/delete/404").status_code == 404
