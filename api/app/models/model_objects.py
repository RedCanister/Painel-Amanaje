import ast
import json
from pydantic import Field, field_validator, model_validator
from typing import Any, List, Optional, Dict
from datetime import datetime
import optuna as op
from .model_schemas import AsyncCRUDMixin


def _parse_container_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    trimmed = value.strip()
    if not trimmed:
        return value

    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(trimmed)
    except (ValueError, SyntaxError):
        return value


def _coerce_list_field(value: Any) -> Any:
    parsed = _parse_container_literal(value)

    if parsed == "":
        return None
    if isinstance(parsed, (tuple, set)):
        return list(parsed)

    return parsed


def _coerce_dict_field(value: Any) -> Any:
    parsed = _parse_container_literal(value)
    return None if parsed == "" else parsed

# Modelo pydantic básico para construção das colunas principais
class ObjectModel(AsyncCRUDMixin):

    id:  int
    name: str                               # Model name
    description: Optional[str] = None       # Model description
    object_type: str
    size: float                             # Size in MB
    path: str                               # File path to the model
    date: datetime = Field(default_factory=datetime.now)                     # Date of model creation or training
    version: Optional[int] = None           # Version of the model (date or numeric)
    history: Optional[List[dict]] = None    # History of model training runs

    @field_validator("history", mode="before")
    @classmethod
    def _validate_history(cls, value: Any) -> Any:
        return _coerce_list_field(value)


# Data models - Entities: Datasets, Features, Samples, Templates. Using postgres and redis for storage
# Definitions - Send to database | Receive from database | Update in database | Delete from database 
class DatasetModel(ObjectModel):   

    dataset_type: str
    shape: List[int]
    has_features: Optional[bool] = None     # Whether dataset has features
    features_list: Optional[List[str]] = None  # List of feature names
    connection_string: Optional[str] = None # For database connections  
    # Implement shape into the set

    @field_validator("shape", "features_list", mode="before")
    @classmethod
    def _validate_dataset_lists(cls, value: Any) -> Any:
        return _coerce_list_field(value)


class AssistantTrainingDatasetModel(DatasetModel):
    """DatasetModel-compatible JSONL snapshot for AssistantModel training."""

    dataset_type: str = "assistant_training_dataset"
    shape: List[int] = Field(default_factory=lambda: [0, 0])
    has_features: Optional[bool] = True
    features_list: Optional[List[str]] = Field(
        default_factory=lambda: [
            "source_type",
            "label",
            "workflow_type",
            "prompt",
            "context_pack_hash",
            "asset_summary",
            "content",
            "registry_metadata",
            "source_path",
            "source_hash",
            "draft",
            "review",
            "user_edits",
            "quality_signals",
            "assistant_success",
            "assistant_quality_label",
        ]
    )
    connection_string: Optional[str] = None


# Machine Learning Models - Entities: Learning Models, ONNX Models, Template Models. Using ./mlflow-server for model management. Or ./mlruns for run storage
# Definitions - Send to database | Receive from database | Update in database | Delete from database | Train | Study | Deploy |
class LearningModel(ObjectModel):
    
    model_type: str
    parameters: dict                        # Model parameters
    metrics: dict                           # Model performance metrics
    reference_data: Optional[str] = None    # Reference to referenced dataset name
    input_features: Optional[List[str]] = None                   # List of input feature names
    output_features: Optional[List[str]] = None                  # List of output feature names
    is_trained: Optional[bool] = False      # Whether the model is trained
    is_tested: Optional[bool] = False       # Whether the model is tested
    is_deployed: Optional[bool] = False     # Whether the model is deployed

    @field_validator("parameters", "metrics", mode="before")
    @classmethod
    def _validate_learning_dicts(cls, value: Any) -> Any:
        return _coerce_dict_field(value)

    @field_validator("input_features", "output_features", mode="before")
    @classmethod
    def _validate_learning_lists(cls, value: Any) -> Any:
        return _coerce_list_field(value)
    

_ASSISTANT_MODEL_PARAMETER_FIELDS = (
    "provider_type",
    "base_url",
    "chat_endpoint",
    "model_name",
    "model_version",
    "api_key_env",
    "adapter_path",
    "prompt_template_version",
    "context_pack_version",
    "safety_profile_version",
    "evaluation_profile",
    "supported_draft_types",
    "behavior_profile",
    "reference_policy",
    "training_profile",
    "runtime_config",
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "enabled",
    "is_default",
)


class AssistantModel(LearningModel):
    """LearningModel-compatible registry entry for assistant LLM/SLM runtimes."""

    model_type: str = "assistant_model"
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    reference_data: Optional[str] = "runtime_artifacts/assistant_datasets/interactions.jsonl"
    input_features: Optional[List[str]] = Field(default_factory=lambda: ["prompt", "context_pack", "target_type"])
    output_features: Optional[List[str]] = Field(default_factory=lambda: ["workflow_draft"])
    provider_type: Optional[str] = "openai_compatible"
    base_url: Optional[str] = None
    chat_endpoint: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    api_key_env: Optional[str] = None
    adapter_path: Optional[str] = None
    prompt_template_version: Optional[str] = None
    context_pack_version: Optional[str] = None
    safety_profile_version: Optional[str] = None
    evaluation_profile: Optional[str] = None
    supported_draft_types: Optional[List[str]] = None
    behavior_profile: Optional[Dict[str, Any]] = None
    reference_policy: Optional[Dict[str, Any]] = None
    training_profile: Optional[Dict[str, Any]] = None
    runtime_config: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    enabled: Optional[bool] = True
    is_default: Optional[bool] = False

    @model_validator(mode="after")
    def _sync_assistant_parameters(self) -> "AssistantModel":
        parameters = dict(self.parameters or {})
        assistant_parameters = dict(parameters.get("assistant") or {})
        for field_name in _ASSISTANT_MODEL_PARAMETER_FIELDS:
            value = getattr(self, field_name, None)
            if value is None and field_name in assistant_parameters:
                setattr(self, field_name, assistant_parameters[field_name])
            elif value is not None:
                assistant_parameters[field_name] = value
        assistant_parameters.setdefault("provider_type", self.provider_type or "openai_compatible")
        parameters["assistant"] = assistant_parameters
        self.parameters = parameters
        return self

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        parameters = dict(data.get("parameters") or {})
        assistant_parameters = dict(parameters.get("assistant") or {})
        for field_name in _ASSISTANT_MODEL_PARAMETER_FIELDS:
            value = data.pop(field_name, None)
            if value is not None:
                assistant_parameters[field_name] = value
        assistant_parameters.setdefault("provider_type", "openai_compatible")
        parameters["assistant"] = assistant_parameters
        data["parameters"] = parameters
        data["model_type"] = "assistant_model"
        return data


class CodeModel(ObjectModel):
    
    code: dict
    variables: dict

    @field_validator("code", "variables", mode="before")
    @classmethod
    def _validate_code_dicts(cls, value: Any) -> Any:
        return _coerce_dict_field(value)


class InferenceModel(ObjectModel):

    learning_model_id: int
    dataset_id: int
    input_features: Optional[List[str]] = None
    output_features: Optional[List[str]] = None
    inference_params: Optional[Dict[str, Any]] = None

    @field_validator("input_features", "output_features", mode="before")
    @classmethod
    def _validate_inference_lists(cls, value: Any) -> Any:
        return _coerce_list_field(value)

    @field_validator("inference_params", mode="before")
    @classmethod
    def _validate_inference_dicts(cls, value: Any) -> Any:
        return _coerce_dict_field(value)


InfereceModel = InferenceModel


class StudyModel(InferenceModel):

    learning_model: Optional[LearningModel] = None
    dataset: Optional[DatasetModel] = None
    sampler: str
    objective: str
    best_trial: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    study_params: Optional[Dict[str, Any]] = None # n_trials, direction, metrics, n_jobs

    @field_validator("best_trial", "best_params", "study_params", mode="before")
    @classmethod
    def _validate_study_dicts(cls, value: Any) -> Any:
        return _coerce_dict_field(value)

    class Config:
        from_attributes = True

    # Objective fuctions per library
    
    # Tensors - Dataloaders
    @staticmethod
    def objective_torch(
        X,
        y,
        PyTorchModel,
        nn_criterion=None,
        nn_optimizer=None,
        param_list: Optional[Dict[str, Any]] = None,
        trial: Optional[op.trial.Trial] = None,
        epochs: int = 1,
    ):
        import torch
        import torch.nn as nn

        if trial is None:
            raise ValueError("trial is required for StudyModel.objective_torch")

        # Hyperparameter search space
        # - Parameter definition
        # - Range and parameter suggestion function

        # The incoming param_list has 3 integers behind each parameter, to be provided in the front-end
        # e.g., {'param_name': {'type': 'int', 'low': 1, 'high': 10, 'step': 1}}
        suggested_params: Dict[str, Any] = {}
        for param, values in (param_list or {}).items():
            if values["type"] == "int":
                suggested_params[param] = trial.suggest_int(param, values["low"], values["high"], step=values.get("step", 1))
            elif values["type"] == "float":
                suggested_params[param] = trial.suggest_float(param, values["low"], values["high"], step=values.get("step"))
            elif values["type"] == "categorical":
                suggested_params[param] = trial.suggest_categorical(param, values["choices"])
        
        # Model definition
        # - Criterion and optimizer
        # - Training function
        try:
            model = PyTorchModel(**suggested_params)
        except TypeError:
            model = PyTorchModel(suggested_params)

        criterion = nn_criterion() if isinstance(nn_criterion, type) else nn_criterion
        criterion = criterion or nn.MSELoss()

        learning_rate = float(suggested_params.get("learning_rate", 0.001))
        if nn_optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif isinstance(nn_optimizer, type):
            optimizer = nn_optimizer(model.parameters(), lr=learning_rate)
        else:
            optimizer = nn_optimizer

        # Loss Validation
        # - Gradient
        # - Model prediction

        # Training function
        # model, metric_a, metric_b = train_torch(model, X, y, kw1, kw1, criterion, optimizer, epochs, steps)
        def _as_tensor(value):
            if hasattr(value, "to_numpy"):
                value = value.to_numpy()
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.reshape(-1, 1)
            return tensor

        x_tensor = _as_tensor(X)
        y_tensor = _as_tensor(y)
        if hasattr(model, "train"):
            model.train()
        latest_loss = None
        for _ in range(max(1, int(epochs))):
            optimizer.zero_grad()
            predictions = model(x_tensor)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            latest_loss = loss

        # Front-end request model.train() or model.eval()
        # model.eval()

        # Front-end request with_grad() or .no_grad()
        # with torch.no_grad():
        # predictions, _ = model(val_x)
        # loss = criterion(predictions, val_y)

        # Optuna Prunning
        # if trial.should_prune():
        #   logging.warning(trial.number)
        #   raise optuna.exceptions.TrialPruned()

        # Exception catching


        return float(latest_loss.detach().cpu().item()) if latest_loss is not None else None

    # fit(X, y)
    @staticmethod
    def objective_sklearn(estimator=None, X=None, y=None, metric=None):
        if estimator is None or X is None or y is None:
            return None

        fitted_estimator = estimator.fit(X, y)
        if metric is not None:
            return metric(fitted_estimator, X, y)
        if hasattr(fitted_estimator, "score"):
            return float(fitted_estimator.score(X, y))
        return None


    @staticmethod
    def objective_tensorflow():
        return None


    @staticmethod
    def objective_xgboost():
        return None
