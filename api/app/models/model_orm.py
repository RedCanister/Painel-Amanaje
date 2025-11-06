from app.database.db_session import Base

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB

# Modelo ORM básico para construção das colunas principais
class ObjectORM(Base):
    __tablename__ = "objects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)               # Nome da tabela
    description = Column(String, nullable=True)        # Descrição da tabela
    object_type = Column(String)        # Tipo / Origem do objeto
    size = Column(Float)                # Tamanho em MB
    path = Column(String)               # Caminho para o conjunto de dados
    date = Column(DateTime, default=datetime.now().isoformat())               # Data de criação
    version = Column(Integer, autoincrement=True)            # Versão da tabela
    history = Column(JSONB, nullable=True)             # Histórico da tabela

    __mapper_args__ = {
        "polymorphic_identity": "object",
        "polymorphic_on": object_type,
    }


# Classe de conjuntos de dados ORM
class DatasetORM(ObjectORM):
    __tablename__ = "datasets"
    
    id = Column(Integer, ForeignKey("objects.id"), primary_key=True)

    name = Column(String, nullable=False)
    dataset_type = Column(String(50))
    has_features = Column(Boolean, nullable=True)
    features_list = Column(JSONB, nullable=True)
    connection_string = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "dataset",
        "polymorphic_on": dataset_type,
    }
    
class ImageDatasetORM(DatasetORM):

    width = Column(Integer)
    height = Column(Integer)

    __mapper_args__ = {
        "polymorphic_identity": "image_dataset"
    }
    

# Classe de modelos de aprendizado ORM
class LearningORM(ObjectORM):
    __tablename__ = "learning_models"

    id = Column(Integer, ForeignKey("objects.id"), primary_key=True)

    model_type = Column(String(50))
    parameters= Column(JSONB, nullable=False)
    metrics= Column(JSONB, nullable=True)
    reference_data = Column(String, nullable=True)
    input_features = Column(String, nullable=True)
    output_features = Column(String, nullable=True)
    is_trained = Column(Boolean, nullable=True)
    is_tested = Column(Boolean, nullable=True)
    is_deployed = Column(Boolean, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "learning_model",
        "polymorphic_on": model_type,
    }


classes = """
class FeatureSetORM(DatasetORM):
    __tablename__ = "feature_sets"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(JSONB)

class FeatureUnitORM(DatasetORM):
    __tablename__ = "feature_units"
    id = Column(Integer, primary_key=True, index=True)
    data_type = Column(String)
    origin_dataset = Column(String)

class SampleSetORM(DatasetORM):
    __tablename__ = "sample_sets"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(JSONB)
    origin_dataset = Column(String)
    proportion = Column(Float)

class TemplateSetORM(DatasetORM):
    __tablename__ = "template_sets"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(JSONB)
    source_origin = Column(String)

# Registe BaseModel to ORM pairs
# ModelRegistry.register(model_schemas.StandardDataset, StandardDatasetORM)
"""