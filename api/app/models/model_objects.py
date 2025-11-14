from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from .model_schemas import AsyncCRUDMixin

# Modelo pydantic básico para construção das colunas principais
class ObjectModel(AsyncCRUDMixin):

    id:  int
    name: str                               # Model name
    description: Optional[str] = None       # Model description
    object_type: str
    size: float                             # Size in MB
    path: str                               # File path to the model
    date: datetime = Field(default_factory=datetime.now().isoformat())                               # Date of model creation or training
    version: Optional[int] = None           # Version of the model (date or numeric)
    history: Optional[List[dict]] = None    # History of model training runs



# Data models - Entities: Datasets, Features, Samples, Templates. Using postgres and redis for storage
# Definitions - Send to database | Receive from database | Update in database | Delete from database 
class DatasetModel(ObjectModel):   

    dataset_type: str
    has_features: Optional[bool] = None                      # Whether dataset has features
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
    input_features: List[str]                   # List of input feature names
    output_features: List[str]                  # List of output feature names
    is_trained: Optional[bool] = False      # Whether the model is trained
    is_tested: Optional[bool] = False       # Whether the model is tested
    is_deployed: Optional[bool] = False     # Whether the model is deployed
    