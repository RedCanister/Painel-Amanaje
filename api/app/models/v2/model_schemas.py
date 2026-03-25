from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ORMBaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ObjectBase(ORMBaseSchema):
    name: str
    description: Optional[str] = None
    object_type: str
    size: float
    path: str
    date: datetime = Field(default_factory=datetime.now)
    version: Optional[int] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


class ObjectCreate(ObjectBase):
    pass


class ObjectUpdate(ORMBaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    object_type: Optional[str] = None
    size: Optional[float] = None
    path: Optional[str] = None
    date: Optional[datetime] = None
    version: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None


class ObjectRead(ObjectBase):
    id: int


class DatasetBase(ObjectBase):
    dataset_type: str
    shape: List[int]
    has_features: Optional[bool] = None
    features_list: Optional[List[str]] = None
    connection_string: Optional[str] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(ObjectUpdate):
    dataset_type: Optional[str] = None
    shape: Optional[List[int]] = None
    has_features: Optional[bool] = None
    features_list: Optional[List[str]] = None
    connection_string: Optional[str] = None


class DatasetRead(DatasetBase):
    id: int


class LearningBase(ObjectBase):
    model_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reference_data: Optional[str] = None
    input_features: Optional[List[str]] = None
    output_features: Optional[List[str]] = None
    is_trained: bool = False
    is_tested: bool = False
    is_deployed: bool = False


class LearningCreate(LearningBase):
    pass


class LearningUpdate(ObjectUpdate):
    model_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    reference_data: Optional[str] = None
    input_features: Optional[List[str]] = None
    output_features: Optional[List[str]] = None
    is_trained: Optional[bool] = None
    is_tested: Optional[bool] = None
    is_deployed: Optional[bool] = None


class LearningRead(LearningBase):
    id: int


class CodeBase(ObjectBase):
    code: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)


class CodeCreate(CodeBase):
    pass


class CodeUpdate(ObjectUpdate):
    code: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None


class CodeRead(CodeBase):
    id: int


class StudyBase(ObjectBase):
    learning_model_id: int
    dataset_id: int
    sampler: str
    objective: str
    best_trial: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    study_params: Optional[Dict[str, Any]] = None


class StudyCreate(StudyBase):
    pass


class StudyUpdate(ObjectUpdate):
    learning_model_id: Optional[int] = None
    dataset_id: Optional[int] = None
    sampler: Optional[str] = None
    objective: Optional[str] = None
    best_trial: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    study_params: Optional[Dict[str, Any]] = None


class StudyRelationSummary(ORMBaseSchema):
    id: int
    name: str


class StudyRead(StudyBase):
    id: int
    learning_model: Optional[StudyRelationSummary] = None
    dataset: Optional[StudyRelationSummary] = None
