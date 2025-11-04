from .model_orm import DatasetORM
from .model_pydantic import DatasetModel
from .model_registry import ModelRegistry

ModelRegistry.register_model(DatasetModel, DatasetORM)

print("DatasetModel registered:", ModelRegistry.get_orm(DatasetModel))
print("ModelRegistry", ModelRegistry._registry)