from pathlib import Path
from types import SimpleNamespace
import re

import pytest
from sklearn.dummy import DummyRegressor

from app.models.model_objects import InferenceModel, LearningModel, StudyModel
from app.models.model_orm import InferenceORM, StudyORM


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def _template_text(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def test_learning_and_inference_models_coerce_stringified_containers():
    learning = LearningModel.model_validate(
        {
            "id": 1,
            "name": "model-a",
            "object_type": "learning_model",
            "size": 1.0,
            "path": "/tmp/model.pkl",
            "model_type": "learning_model",
            "parameters": '{"framework": "sklearn"}',
            "metrics": '{"eval_rmse": 0.1}',
            "input_features": '["x1", "x2"]',
            "output_features": '["target"]',
            "history": "[]",
        }
    )
    inference = InferenceModel.model_validate(
        {
            "id": 2,
            "name": "model-a :: dataset-a",
            "object_type": "inference_model",
            "size": 1.0,
            "path": "/tmp/model.pkl",
            "learning_model_id": 1,
            "dataset_id": 7,
            "input_features": '["x1"]',
            "output_features": '["target"]',
            "inference_params": '{"status": "active"}',
        }
    )

    assert learning.parameters == {"framework": "sklearn"}
    assert learning.metrics == {"eval_rmse": 0.1}
    assert learning.input_features == ["x1", "x2"]
    assert inference.inference_params == {"status": "active"}


def test_study_objective_sklearn_runs_estimator_score():
    score = StudyModel.objective_sklearn(
        estimator=DummyRegressor(strategy="mean"),
        X=[[1.0], [2.0], [3.0]],
        y=[2.0, 2.0, 2.0],
    )

    assert score == pytest.approx(1.0)


def test_study_objective_torch_runs_one_training_step():
    torch = pytest.importorskip("torch")

    class FakeTrial:
        def suggest_int(self, _name, low, _high, step=1):
            return low + step - 1

        def suggest_float(self, _name, low, _high, step=None):
            return low if step is None else low + step - step

        def suggest_categorical(self, _name, choices):
            return choices[0]

    class TinyModel(torch.nn.Module):
        def __init__(self, hidden_size=1, learning_rate=0.01):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate

        def forward(self, value):
            return self.linear(value)

    loss = StudyModel.objective_torch(
        [[1.0], [2.0]],
        [[1.0], [2.0]],
        TinyModel,
        param_list={
            "hidden_size": {"type": "int", "low": 1, "high": 2},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1},
        },
        trial=FakeTrial(),
    )

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_orm_mappings_include_inference_and_study_relationship_columns():
    inference = InferenceORM(
        name="model :: dataset",
        object_type="inference_model",
        size=0.0,
        path="/tmp/model.pkl",
        learning_model_id=1,
        dataset_id=2,
    )
    study = StudyORM(
        name="study-a",
        object_type="study_model",
        size=0.0,
        path="/tmp/study.json",
        learning_model_id=1,
        dataset_id=2,
        sampler="TPESampler",
        objective="minimize_loss",
    )

    assert inference.inference_params is None
    assert InferenceORM.__mapper__.polymorphic_identity == "inference_model"
    assert StudyORM.__mapper__.polymorphic_identity == "study_model"
    assert {column.name for column in StudyORM.__table__.columns}.issuperset(
        {"learning_model_id", "dataset_id", "sampler", "objective"}
    )
    assert study.learning_model_id == 1


def test_target_templates_do_not_reference_missing_static_ids():
    for template_name in [
        "base_blue.html",
        "base_create.html",
        "base_green.html",
        "base_onnx.html",
        "base_purple.html",
        "base_red.html",
    ]:
        text = _template_text(template_name)
        ids = set(re.findall(r"id=[\"']([^\"']+)[\"']", text))
        refs = set(re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", text))
        missing_refs = {ref for ref in refs - ids if "${" not in ref}
        assert missing_refs == set(), f"{template_name} missing ids: {sorted(missing_refs)}"


def test_upload_template_uses_backend_analysis_routes_and_valid_selector_instantiation():
    text = _template_text("base_red.html")

    assert "/analysis/data?dataset_id=" in text
    assert "/analysis/model?model_id=" in text
    assert "new MultiObjectSelector({" in text
    assert "document.getElementById('analysisResults')" not in text
