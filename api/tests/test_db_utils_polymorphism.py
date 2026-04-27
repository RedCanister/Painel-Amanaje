import sys
from types import ModuleType
from datetime import datetime
import asyncio

from sqlalchemy.orm import declarative_base

db_session_stub = ModuleType("app.database.db_session")
db_session_stub.Base = declarative_base()
db_session_stub.AsyncSessionLocal = None
db_session_stub.init_models = lambda: None


async def _fake_wait_for_database():
    return None


async def _fake_get_db():
    yield object()


db_session_stub.get_db = _fake_get_db
db_session_stub.wait_for_database = _fake_wait_for_database
sys.modules.setdefault("app.database.db_session", db_session_stub)

from app.database.db_utils import (
    JSONB_CONTAINER_COLUMNS,
    create_entry,
    get_entry,
    _resolve_polymorphic_model,
    _sync_legacy_name_payload,
    normalize_legacy_jsonb_containers,
    normalize_legacy_polymorphic_identities,
)
from app.models.model_orm import (
    CodeORM,
    DatasetORM,
    LearningORM,
    SupervisedModelORM,
    TimeSeriesDatasetORM,
)


def test_resolve_dataset_subclass_from_known_identity():
    orm_model, payload = _resolve_polymorphic_model(
        DatasetORM,
        {"dataset_type": "timeseries_dataset"},
    )

    assert orm_model is TimeSeriesDatasetORM
    assert payload["dataset_type"] == "timeseries_dataset"


def test_resolve_dataset_file_format_alias_to_base_dataset():
    orm_model, payload = _resolve_polymorphic_model(
        DatasetORM,
        {"dataset_type": "csv"},
    )

    assert orm_model is DatasetORM
    assert payload["dataset_type"] == "dataset"


def test_resolve_model_alias_to_polymorphic_subclass():
    orm_model, payload = _resolve_polymorphic_model(
        LearningORM,
        {"model_type": "supervised"},
    )

    assert orm_model is SupervisedModelORM
    assert payload["model_type"] == "supervised_model"


def test_resolve_object_type_alias_to_code_model():
    orm_model, payload = _resolve_polymorphic_model(
        CodeORM,
        {"object_type": "code"},
    )

    assert orm_model is CodeORM
    assert payload["object_type"] == "code_model"


def test_sync_legacy_name_payload_for_dataset_models():
    payload = _sync_legacy_name_payload(
        DatasetORM,
        {
            "name": "sample_dataset",
            "date": datetime(2026, 4, 3, 21, 5, 5),
        },
    )

    assert payload["legacy_name"] == "sample_dataset"


def test_normalize_legacy_polymorphic_identities_repairs_aliases():
    class FakeResult:
        def __init__(self, rowcount: int):
            self.rowcount = rowcount

    class FakeDB:
        def __init__(self):
            self.calls = []
            self.commit_count = 0

        async def execute(self, statement, params):
            self.calls.append((str(statement), params))
            rowcount = 1 if params["legacy_value"] == "code" else 0
            return FakeResult(rowcount)

        async def commit(self):
            self.commit_count += 1

    fake_db = FakeDB()

    normalized = asyncio.run(normalize_legacy_polymorphic_identities(fake_db))

    assert normalized == {"object_type": 1}
    assert fake_db.commit_count == 1
    assert any("UPDATE objects" in statement for statement, _ in fake_db.calls)


def test_normalize_legacy_jsonb_containers_repairs_stringified_json():
    class FakeResult:
        def __init__(self, rowcount: int):
            self.rowcount = rowcount

    class FakeDB:
        def __init__(self):
            self.calls = []
            self.commit_count = 0

        async def execute(self, statement, params):
            self.calls.append((str(statement), params))
            rowcount = 1 if params["leading_char"] == "[" else 0
            return FakeResult(rowcount)

        async def commit(self):
            self.commit_count += 1

    fake_db = FakeDB()

    normalized = asyncio.run(normalize_legacy_jsonb_containers(fake_db))

    assert "objects" in JSONB_CONTAINER_COLUMNS
    assert normalized["objects"] == 1
    assert normalized["learning_models"] == 2
    assert normalized["inference_models"] == 2
    assert fake_db.commit_count == 1
    assert any("::jsonb" in statement for statement, _ in fake_db.calls)


def test_get_entry_uses_numeric_string_as_id_before_name_lookup():
    class FakeScalarResult:
        def __init__(self, value):
            self.value = value

        def first(self):
            return self.value

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def scalars(self):
            return FakeScalarResult(self.value)

    class FakeDB:
        def __init__(self):
            self.calls = []

        async def execute(self, statement):
            self.calls.append(str(statement))
            return FakeResult({"id": 1, "name": "dataset-1"})

    fake_db = FakeDB()

    entry = asyncio.run(get_entry(fake_db, DatasetORM, "1"))

    assert entry == {"id": 1, "name": "dataset-1"}
    assert len(fake_db.calls) == 1
    assert "WHERE" in fake_db.calls[0]
    assert "datasets.id" in fake_db.calls[0]


def test_create_entry_normalizes_stringified_jsonb_payloads_for_code_models():
    class FakeDB:
        def __init__(self):
            self.added = None
            self.refreshed = None
            self.commit_count = 0
            self.rollback_count = 0

        def add(self, obj):
            self.added = obj

        async def commit(self):
            self.commit_count += 1

        async def refresh(self, obj):
            self.refreshed = obj

        async def rollback(self):
            self.rollback_count += 1

    payload = {
        "name": "script_2026-04-24",
        "description": "Dataset created from the editor",
        "object_type": "code_model",
        "size": 0,
        "path": "generated/galactic_dataset.csv",
        "date": "2026-04-24T22:07:15.090Z",
        "version": 1,
        "history": '[{"operation": "save_script", "when": "2026-04-24T22:07:15.090Z"}]',
        "variables": '{"dataset_name": "github_vs_git"}',
        "code": '{"script": "print(1)", "language": "python"}',
    }

    fake_db = FakeDB()
    created = asyncio.run(create_entry(fake_db, CodeORM, payload))

    assert created is fake_db.added
    assert fake_db.commit_count == 1
    assert isinstance(created.history, list)
    assert created.history[0]["operation"] == "save_script"
    assert created.variables == {"dataset_name": "github_vs_git"}
    assert created.code["language"] == "python"
    assert created.date.tzinfo is None
