from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from .model_schemas import AsyncCRUDMixin


# Data models - Entities: Datasets, Features, Samples, Templates. Using postgres and redis for storage
# Definitions - Send to database | Receive from database | Update in database | Delete from database 
class DatasetModel(AsyncCRUDMixin):
    id: int
    name: str                               # Dataset name
    description: Optional[str] = None       # Dataset description
    dataset_type: str
    source_type: str                        # Regression, Classification, Clustering
    size: float                             # Size in MB
    path: str                               # File path or URL
    date: datetime = Field(default_factory=datetime.now())                               # Upload date in ISO format
    version: Optional[int] = None                            # Version of the feature set (date or numeric)
    history: Optional[List[dict]] = None    # Transformation history records    
    has_features: Optional[bool] = None                      # Whether dataset has features
    features_list: Optional[List[str]] = None  # List of feature names
    connection_string: Optional[str] = None # For database connections  
   
    
class StandardDataset(DatasetModel):
    has_features: bool                      # Whether dataset has features
    features_list: Optional[List[str]] = None  # List of feature names
    connection_string: Optional[str] = None # For database connections

class FeatureSet(DatasetModel):
    features: List[str]                     # List of feature paths or names

class FeatureUnit(DatasetModel):
    data_type: str                          # String, Float, Int, Boolean, etc...
    origin_dataset: str                     # Reference to DatasetUpload name

class SampleSet(DatasetModel):
    features: List[str]                     # List of feature paths or names
    origin_dataset: str                    # Reference to DatasetUpload name
    proportion: float                       # Proportion of the original dataset

class TemplateSet(DatasetModel):
    features: List[str]                     # List of feature paths or names
    source_origin: str                      # From which library or source




# Machine Learning Models - Entities: Learning Models, ONNX Models, Template Models. Using ./mlflow-server for model management. Or ./mlruns for run storage
# Definitions - Send to database | Receive from database | Update in database | Delete from database | Train | Study | Deploy |
class LearningModel(AsyncCRUDMixin):
    model_name: str                         # Model name
    size: float                             # Size in MB
    path: str                               # File path to the model
    description: Optional[str] = None       # Model description
    model_category: str                     # e.g., Regression, Classification
    date: str                               # Date of model creation or training
    version: str                            # Version of the model (date or numeric)
    
    parameters: dict                        # Model parameters
    metrics: dict                           # Model performance metrics
    reference_data: str                     # Reference to referenced dataset name
    input_data: List[str]                   # List of input feature names
    output_data: List[str]                  # List of output feature names
    is_trained: Optional[bool]              # Whether the model is trained
    is_tested: Optional[bool]               # Whether the model is tested
    is_deployed: Optional[bool]             # Whether the model is deployed
    history: Optional[List[dict]] = None    # History of model training runs

class ONNXModel(LearningModel):
    onnx_file_path: str                     # ONNX file path
    onnx_graph: dict                        # ONNX graph structure
    source_framework: str                   # e.g., PyTorch, TensorFlow

class TemplateModel(LearningModel):
    template_origin: str                    # Template file path
    source_framework: str                   # e.g., PyTorch, TensorFlow




# Form Models
class DatasetForm(BaseModel):
    name: str                               # Dataset name
    description: Optional[str] = None       # Dataset description
    source_type: str                        # Regression, Classification, Clustering
    size: float                             # Size in MB

class LearningModelForm(BaseModel):
    name: str                               # Model name                  
    description: Optional[str] = None       # Model description
    size: float                             # Size in MB
    source_code: str                        # Source code or path
    model_category: str                     # e.g., Regression, Classification

class StudyForm(BaseModel):
    study_name: str                         # Study name
    experiment_name: str                    # Experiment name
    description: Optional[str] = None       # Study description
    parameters: dict                        # Study parameters
    model_used: List[str]                   # List of model names used in the study
    data_used: List[str]                    # List of dataset names used in the study

class ParameterForm(BaseModel):
    name: str                               # Parameter set name
    parameters: dict                        # Dictionary of parameters
    description: Optional[str] = None       # Description of the parameter set
    related_model: Optional[str] = None     # Related model name




# Miscellaneous Models
class ParameterSet(BaseModel):
    name: str                               # Parameter set name
    parameters: dict                        # Dictionary of parameters
    description: Optional[str] = None       # Description of the parameter set
    related_model: Optional[str] = None     # Related model name
    date: str                               # Date of creation in ISO format

class StudySet(BaseModel):
    study_name: str                         # Study name
    experiment_name: str                    # Experiment name
    description: Optional[str] = None       # Study description
    parameters: dict                        # Study parameters
    history: Optional[List[dict]] = None    # Study history records
    model_used: List[str]                   # List of model names used in the study
    data_used: List[str]                    # List of dataset names used in the study
    date: str                               # Date of the study in ISO format
    related_dataset: List[str]              # List of related dataset names

class DashboardConfig(BaseModel):
    dashboard_name: str                     # Dashboard name
    layout: dict                            # Layout configuration
    widgets: List[dict]                     # List of widgets in the dashboard
    description: Optional[str] = None       # Dashboard description
    date: str                               # Date of creation in ISO format

class ReportModel(BaseModel):
    report_name: str                        # Report name
    content: str                            # Report content (could be markdown, HTML, etc.)
    author: str                             # Author of the report
    model_name: str                         # Related model name
    dataset_name: str                       # Related dataset name
    related_study: Optional[str] = None     # Related study name
    metrics: dict                           # Report metrics
    parameters: dict                        # Report parameters
    description: Optional[str] = None       # Report description
    date: str                               # Date of creation in ISO format

class HistoryModel(BaseModel):
    entity_name: str                        # Name of the entity (dataset, model, etc.)
    entity_type: str                        # Type of the entity
    changes: List[dict]                     # List of changes made
    dags: List[dict]                        # List of changes made
    date: str                               # Date of the change in ISO format
    changed_by: str                         # User who made the change




# Operation Models - Entities

class OperationModel(BaseModel):
    operation_name: str                     # Name of the transformation operation
    description: Optional[str] = None       # Description of the operation
    date: str                               # Date of creation in ISO format
    source_origin: str                      # From which library or source

class TransformationOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    input_features: List[str]               # List of input feature names
    output_features: List[str]              # List of output feature names

class TestingOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    test: dict                              # Test functions used in the operation

class ETLOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    input_data_sources: List[str]           # List of input data source names
    output_data_targets: List[str]          # List of output data target names
    schedule: Optional[str] = None          # Schedule for the ETL operation
    dags: Optional[List[dict]] = None       # DAGs associated with the ETL operation

class DatabaseOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    input_data_sources: List[str]           # List of input data source names
    output_data_targets: List[str]          # List of output data target names
    schedule: Optional[str] = None          # Schedule for the ETL operation
    dags: Optional[List[dict]] = None       # DAGs associated with the ETL operation

class ModelTrainingOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    input_dataset: str                      # Input dataset name
    output_model: str                       # Output model name
    training_metrics: dict                  # Metrics from the training process

class ModelEvaluationOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    input_model: str                        # Input model name
    evaluation_dataset: str                 # Evaluation dataset name
    evaluation_metrics: dict                # Metrics from the evaluation process   

class DeploymentOperation(OperationModel):
    parameters: dict                        # Parameters used in the operation
    input_model: str                        # Input model name
    deployment_target: str                  # Deployment target name
    status: str                             # Status of the deployment (e.g., successful, failed)   

