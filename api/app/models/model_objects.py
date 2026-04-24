import ast
import json
from pydantic import Field, field_validator
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
