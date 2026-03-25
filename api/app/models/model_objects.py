from pydantic import Field
from typing import List, Optional, Dict
from datetime import datetime
import optuna as op
from .model_schemas import AsyncCRUDMixin

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


# Data models - Entities: Datasets, Features, Samples, Templates. Using postgres and redis for storage
# Definitions - Send to database | Receive from database | Update in database | Delete from database 
class DatasetModel(ObjectModel):   

    dataset_type: str
    shape: List[int]
    has_features: Optional[bool] = None     # Whether dataset has features
    features_list: Optional[List[str]] = None  # List of feature names
    connection_string: Optional[str] = None # For database connections  
    # Implement shape into the set


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
    

class CodeModel(ObjectModel):
    
    code: dict
    variables: dict
    

class StudyModel(ObjectModel):

    learning_model_id: int
    learning_model: LearningModel
    dataset_id: int
    dataset: DatasetModel
    sampler: str
    objective: str
    best_trial: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, float]] = None
    study_params: Optional[Dict[str, float]] = None # n_trials, direction, metrics, n_jobs

    class Config:
        from_attributes = True

    # Objective fuctions per library
    
    # Tensors - Dataloaders
    def objective_torch(X, y, PyTorchModel, nn_criterion, nn_optimizer, 
            param_list: Dict[str, int], trial: op.trial.Trial
        ):
        # Hyperparameter search space
        # - Parameter definition
        # - Range and parameter suggestion function

        # The incoming param_list has 3 integers behind each parameter, to be provided in the front-end
        # e.g., {'param_name': {'type': 'int', 'low': 1, 'high': 10, 'step': 1}}
        for param, values in param_list.items():
            if values['type'] == 'int':
                param_value = trial.suggest_int(param, values['low'], values['high'], step=values.get('step', 1))
            elif values['type'] == 'float':
                param_value = trial.suggest_float(param, values['low'], values['high'], step=values.get('step', 0.1))
            elif values['type'] == 'categorical':
                param_value = trial.suggest_categorical(param, values['choices'])
            # Store or use param_value as needed
        
        # Model definition
        # - Criterion and optimizer
        # - Training function
        model = PyTorchModel(param_list, )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Loss Validation
        # - Gradient
        # - Model prediction

        # Training function
        # model, metric_a, metric_b = train_torch(model, X, y, kw1, kw1, criterion, optimizer, epochs, steps)

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


        return None

    # fit(X, y)
    def objective_sklearn():
        return None


    def objective_tensorflow():
        return None


    def objective_xgboost():
        return None