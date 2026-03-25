"""
utils/validation.py

Schema and data validation utilities for ensuring data integrity
across configurations, dataset, and API inputs.

Features:
- Validate configurations (YAML/JSON)
- Validate model inputs/outputs
- Define reusable schemas for Airflow, FastAPI, and ML pipelines
- Integrate with existing utils (loggging, serialization)
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError, Field
from app.utils.logging import get_logger

logger = get_logger("validation")

# Core Validation

def validate_input(data: Any, schema: Type[BaseModel]) -> BaseModel:
    """
    Validates data against a given Pydantic schema.

    Args:
        data: dict-like input to validate
        schema: subclass of pydantic.BaseModel defining structure

    Returns:
        A validated Pydantic model instance (with type conversions)
    """

    try:
        validated = schema(**data)
        logger.info(f"✅ Data validated successfully against {schema.__name__}")
        return validated
    except ValidationError as e:
        logger.error(f"❌ Validation failed for {schema.__name__}")
        logger.error(e.json(indent=2))
        raise


# High-level helpers

def validate_config(config: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
    """
    Validates a loaded configuration dictionary and returns a clean dict.

    Example:
        cfg = load_yaml("configs/base.yaml")
        validate_cfg = validate_config(cfg["model"], ModelConfigSchema)
    """

    validated_model = validate_input(config, schema)
    return validated_model.model_dump()

def validate_dataframe_columns(df_columns: List[str], required: List[str]) -> bool:
    """
    Validates that required columns exist in a DataFrame.
    """
    
    missing = [col for col in required if col not in df_columns]
    if missing:
        logger.error(f"❌ Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("✅ All required columns present.")
    return True
