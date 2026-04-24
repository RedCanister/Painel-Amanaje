import sys
from types import ModuleType
from types import SimpleNamespace

from sqlalchemy.orm import declarative_base


db_session_stub = ModuleType("app.database.db_session")
db_session_stub.Base = declarative_base()


async def _fake_get_db():
    yield object()


db_session_stub.get_db = _fake_get_db
sys.modules.setdefault("app.database.db_session", db_session_stub)

optuna_stub = ModuleType("optuna")
optuna_stub.trial = SimpleNamespace(Trial=object)
sys.modules.setdefault("optuna", optuna_stub)


from app.models.model_objects import InferenceModel, LearningModel, StudyModel
from app.models.model_orm import InferenceORM, StudyORM
from app.models.model_registry import ModelRegistry


def test_orm_to_dict_includes_inherited_object_columns_for_studies():
    study = StudyORM(
        id=45,
        name="study-45",
        description="classification search",
        object_type="study_model",
        size=0.5,
        path="/tmp/study-45.json",
        learning_model_id=6,
        dataset_id=1,
        sampler="RandomSampler",
        objective="maximize_accuracy",
        best_trial=None,
        best_params=None,
        study_params={
            "metric": "accuracy",
            "n_trials": 25,
            "direction": "maximize",
            "framework": "pytorch",
        },
    )

    payload = ModelRegistry._orm_to_dict(study)

    assert payload["id"] == 45
    assert payload["name"] == "study-45"
    assert payload["object_type"] == "study_model"
    assert payload["size"] == 0.5
    assert payload["path"] == "/tmp/study-45.json"
    assert payload["learning_model_id"] == 6
    assert payload["dataset_id"] == 1
    assert payload["study_params"]["framework"] == "pytorch"


def test_study_model_accepts_registry_payload_with_flexible_study_params():
    payload = {
        "id": 45,
        "name": "study-45",
        "description": "classification search",
        "object_type": "study_model",
        "size": 0.5,
        "path": "/tmp/study-45.json",
        "learning_model_id": 6,
        "dataset_id": 1,
        "sampler": "RandomSampler",
        "objective": "maximize_accuracy",
        "best_trial": None,
        "best_params": None,
        "study_params": {
            "metric": "accuracy",
            "n_trials": 25,
            "direction": "maximize",
            "framework": "pytorch",
        },
    }

    study = StudyModel.model_validate(payload)

    assert study.id == 45
    assert study.learning_model is None
    assert study.dataset is None
    assert study.study_params == payload["study_params"]


def test_inference_orm_to_dict_includes_runtime_fields():
    inference = InferenceORM(
        id=12,
        name="model-a :: dataset-b",
        description="linked runtime pair",
        object_type="inference_model",
        size=1.5,
        path="/tmp/model.pkl",
        learning_model_id=6,
        dataset_id=1,
        input_features=["Date", "Open"],
        output_features=["MACD_Under"],
        inference_params={
            "status": "active",
            "latest_metrics": {"accuracy": 0.82},
        },
    )

    payload = ModelRegistry._orm_to_dict(inference)

    assert payload["id"] == 12
    assert payload["object_type"] == "inference_model"
    assert payload["learning_model_id"] == 6
    assert payload["dataset_id"] == 1
    assert payload["inference_params"]["status"] == "active"


def test_inference_model_coerces_stringified_runtime_fields():
    payload = {
        "id": 12,
        "name": "model-a :: dataset-b",
        "description": "linked runtime pair",
        "object_type": "inference_model",
        "size": 1.5,
        "path": "/tmp/model.pkl",
        "date": "2026-04-04T00:00:00",
        "version": 1,
        "history": "[]",
        "learning_model_id": 6,
        "dataset_id": 1,
        "input_features": "[\"Date\", \"Open\"]",
        "output_features": "[\"MACD_Under\"]",
        "inference_params": "{\"status\": \"active\", \"latest_metrics\": {\"accuracy\": 0.82}}",
    }

    inference = InferenceModel.model_validate(payload)

    assert inference.history == []
    assert inference.input_features == ["Date", "Open"]
    assert inference.output_features == ["MACD_Under"]
    assert inference.inference_params == {"status": "active", "latest_metrics": {"accuracy": 0.82}}


def test_inference_model_is_registered_with_its_orm_pair():
    assert ModelRegistry.get_orm(InferenceModel) is InferenceORM


def test_learning_model_coerces_stringified_feature_lists():
    payload = {
        "id": 6,
        "name": "model.pkl",
        "description": "Uploaded file model.pkl",
        "object_type": "learning_model",
        "size": 0.0,
        "path": "/models/model.pkl",
        "date": "2026-04-04T00:00:00",
        "version": 1,
        "history": "[]",
        "model_type": "learning_model",
        "parameters": "{\"framework\": \"pytorch\"}",
        "metrics": "{\"eval_accuracy\": 0.67}",
        "reference_data": None,
        "input_features": "[\"Date\", \"Open\"]",
        "output_features": "[\"MACD_Under\"]",
        "is_trained": True,
        "is_tested": True,
        "is_deployed": False,
    }

    model = LearningModel.model_validate(payload)

    assert model.history == []
    assert model.parameters == {"framework": "pytorch"}
    assert model.metrics == {"eval_accuracy": 0.67}
    assert model.input_features == ["Date", "Open"]
    assert model.output_features == ["MACD_Under"]
