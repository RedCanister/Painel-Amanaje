"""API Configuration Settings"""
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Painel Amanajé API"
    VERSION: str = "0.7.0"
    
    # Database Settings
    POSTGRES_USER: str = "airflow"
    POSTGRES_PASSWORD: str = "airflow"
    POSTGRES_DB: str = "airflow"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: str = "5432"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    
    # File Storage Settings
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()