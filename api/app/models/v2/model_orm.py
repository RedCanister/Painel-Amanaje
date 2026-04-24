from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship

from app.database.db_session import Base


class ObjectORM(Base):
    __tablename__ = "objects_v2"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    object_type = Column(String(50), nullable=False)
    size = Column(Float, nullable=False)
    path = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.now, nullable=False)
    version = Column(Integer, nullable=True)
    history = Column(JSONB, nullable=False, default=list)

    __mapper_args__ = {
        "polymorphic_identity": "object",
        "polymorphic_on": object_type,
    }


class DatasetORM(ObjectORM):
    __tablename__ = "datasets_v2"

    id = Column(Integer, ForeignKey("objects_v2.id"), primary_key=True)
    dataset_type = Column(String(50), nullable=False)
    shape = Column(ARRAY(Integer), nullable=False)
    has_features = Column(Boolean, nullable=True)
    features_list = Column(ARRAY(String), nullable=True)
    connection_string = Column(String, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "dataset",
    }


class LearningORM(ObjectORM):
    __tablename__ = "learning_models_v2"

    id = Column(Integer, ForeignKey("objects_v2.id"), primary_key=True)
    model_type = Column(String(50), nullable=False)
    parameters = Column(JSONB, nullable=False, default=dict)
    metrics = Column(JSONB, nullable=False, default=dict)
    reference_data = Column(String, nullable=True)
    input_features = Column(ARRAY(String), nullable=True)
    output_features = Column(ARRAY(String), nullable=True)
    is_trained = Column(Boolean, nullable=False, default=False)
    is_tested = Column(Boolean, nullable=False, default=False)
    is_deployed = Column(Boolean, nullable=False, default=False)

    __mapper_args__ = {
        "polymorphic_identity": "learning_model",
    }


class CodeORM(ObjectORM):
    __tablename__ = "code_models_v2"

    id = Column(Integer, ForeignKey("objects_v2.id"), primary_key=True)
    variables = Column(JSONB, nullable=False, default=dict)
    code = Column(JSONB, nullable=False, default=dict)

    __mapper_args__ = {
        "polymorphic_identity": "code_model",
    }


class StudyORM(ObjectORM):
    __tablename__ = "study_models_v2"

    id = Column(Integer, ForeignKey("objects_v2.id"), primary_key=True)
    learning_model_id = Column(Integer, ForeignKey("learning_models_v2.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets_v2.id"), nullable=False)
    sampler = Column(String, nullable=False)
    objective = Column(String, nullable=False)
    best_trial = Column(JSONB, nullable=True)
    best_params = Column(JSONB, nullable=True)
    study_params = Column(JSONB, nullable=True)

    learning_model = relationship(LearningORM, foreign_keys=[learning_model_id], lazy="joined")
    dataset = relationship(DatasetORM, foreign_keys=[dataset_id], lazy="joined")

    __mapper_args__ = {
        "polymorphic_identity": "study_model",
    }
