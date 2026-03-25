from .model_schemas import (
    CodeCreate,
    CodeRead,
    CodeUpdate,
    DatasetCreate,
    DatasetRead,
    DatasetUpdate,
    LearningCreate,
    LearningRead,
    LearningUpdate,
    ObjectCreate,
    ObjectRead,
    ObjectUpdate,
    StudyCreate,
    StudyRead,
    StudyUpdate,
)

ObjectModel = ObjectRead
DatasetModel = DatasetRead
LearningModel = LearningRead
CodeModel = CodeRead
StudyModel = StudyRead

__all__ = [
    "CodeCreate",
    "CodeModel",
    "CodeUpdate",
    "DatasetCreate",
    "DatasetModel",
    "DatasetUpdate",
    "LearningCreate",
    "LearningModel",
    "LearningUpdate",
    "ObjectCreate",
    "ObjectModel",
    "ObjectUpdate",
    "StudyCreate",
    "StudyModel",
    "StudyUpdate",
]
