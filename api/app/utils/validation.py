"""
utils/validation.py

Schema and data validation utilities for configurations, datasets, and API
payloads.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Type

from pydantic import BaseModel, ValidationError

from .logging import get_logger

logger = get_logger("validation")


def validate_input(data: Any, schema: Type[BaseModel]) -> BaseModel:
    """
    Validate arbitrary data against a Pydantic schema.
    """

    try:
        if hasattr(schema, "model_validate"):
            validated = schema.model_validate(data)
        else:
            validated = schema(**data)
        logger.info("Validated data against schema '%s'.", schema.__name__)
        return validated
    except ValidationError:
        logger.exception("Validation failed for schema '%s'.", schema.__name__)
        raise


def validate_config(config: Mapping[str, Any], schema: Type[BaseModel]) -> dict[str, Any]:
    """
    Validate a loaded configuration mapping and return a normalized dictionary.
    """

    validated_model = validate_input(dict(config), schema)
    if hasattr(validated_model, "model_dump"):
        return validated_model.model_dump()
    return validated_model.dict()  # type: ignore[no-any-return]


def validate_dataframe_columns(df_columns: Any, required: Iterable[str]) -> bool:
    """
    Validate that all required columns exist in a DataFrame or column iterable.
    """

    available_columns = list(df_columns.columns) if hasattr(df_columns, "columns") else list(df_columns)
    required_columns = list(required)
    missing = [column for column in required_columns if column not in available_columns]

    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("All required columns are present.")
    return True


def validate_required_keys(data: Mapping[str, Any], required: Iterable[str]) -> bool:
    """
    Validate that a mapping contains every required key.
    """

    missing = [key for key in required if key not in data]
    if missing:
        logger.error("Missing required keys: %s", missing)
        raise KeyError(f"Missing required keys: {missing}")

    logger.info("All required keys are present.")
    return True
