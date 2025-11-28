from app.database.db_session import Base

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

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
    shape = Column(ARRAY(Integer))
    has_features = Column(Boolean, nullable=True)
    features_list = Column(ARRAY(String), nullable=True)
    connection_string = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "dataset",
        "polymorphic_on": dataset_type,
    }
    
class NumericalDatasetORM(DatasetORM):

    scale = Column(ARRAY(Float))
    precision = Column(Float)

    __mapper_args__ = {
        "polymorphic_identity": "numerical_dataset"
    }

class CategoricalDatasetORM(DatasetORM):

    categories = Column(Integer)

    __mapper_args__ = {
        "polymorphic_identity": "categorical_dataset"
    }

class TimeSeriesDatasetORM(DatasetORM):

    period = Column(ARRAY(DateTime))
    interval = Column(String)

    __mapper_args__ = {
        "polymorphic_identity": "timeseries_dataset"
    }

class ImageDatasetORM(DatasetORM):

    width = Column(Integer)
    height = Column(Integer)

    __mapper_args__ = {
        "polymorphic_identity": "image_dataset"
    }


class VideoAudioDatasetORM(DatasetORM):

    format = Column(String)

    __mapper_args__ = {
        "polymorphic_identity": "videoaudio_dataset"
    }

class TextDatasetORM(DatasetORM):

    embeddings = Column(String)
    language = Column(String)

    __mapper_args__ = {
        "polymorphic_identity": "text_dataset"
    }

class MixedDatasetORM(DatasetORM):

    types = Column(ARRAY(String))
    kwargs = Column(JSONB)

    __mapper_args__ = {
        "polymorphic_identity": "mixed_dataset"
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

class SupervisedModelORM(LearningORM):

    algorithm = Column(String, nullable=True)
    target_variable = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "supervised_model"
    }

class UnsupervisedModelORM(LearningORM):

    label = Column(String, nullable=True)
    clustering_method = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "unsupervised_model"
    }

class ReinforcementModelORM(LearningORM):

    environment = Column(String, nullable=True)
    policy_type = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "reinforcement_model"
    }

class DeepLearningModelORM(LearningORM):

    architecture = Column(String, nullable=True)
    framework = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "deep_learning_model"
    }

class GenerativeModelORM(LearningORM):

    model_variant = Column(String, nullable=True)
    latent_space_dim = Column(Integer, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "generative_model"
    }

class TransformativeModelORM(LearningORM):

    transformer_type = Column(String, nullable=True)
    num_layers = Column(Integer, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "transformative_model"
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