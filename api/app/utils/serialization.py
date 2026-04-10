"""
utils/serialization.py

Utility functions for serializing Python objects, configs, and model metadata
into dictionary or JSON-compatible formats.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - only used when pydantic is unavailable.
    BaseModel = None  # type: ignore[assignment]


def deep_asdict(obj: Any) -> Any:
    """
    Recursively convert supported objects into plain Python primitives.
    """

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Enum):
        return deep_asdict(obj.value)

    if is_dataclass(obj):
        return {key: deep_asdict(value) for key, value in asdict(obj).items()}

    if BaseModel is not None and isinstance(obj, BaseModel):
        return {key: deep_asdict(value) for key, value in obj.model_dump().items()}

    if isinstance(obj, Mapping):
        return {str(key): deep_asdict(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [deep_asdict(value) for value in obj]

    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return deep_asdict(obj.tolist())
        except Exception:
            pass

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return deep_asdict(obj.item())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        return {
            key: deep_asdict(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }

    return obj


def class_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Extract instance or class attributes while filtering private and callable members.
    """

    if obj is None:
        return {}

    if isinstance(obj, Mapping):
        return {str(key): value for key, value in obj.items()}

    if BaseModel is not None and isinstance(obj, BaseModel):
        return obj.model_dump()

    if is_dataclass(obj):
        return asdict(obj)

    if inspect.isclass(obj):
        members = inspect.getmembers(obj)
        return {
            name: value
            for name, value in members
            if not name.startswith("_") and not inspect.isroutine(value) and not inspect.isdatadescriptor(value)
        }

    data: Dict[str, Any] = {}
    try:
        data.update(
            {
                name: value
                for name, value in inspect.getmembers(obj.__class__)
                if not name.startswith("_")
                and not inspect.isroutine(value)
                and not inspect.isdatadescriptor(value)
            }
        )
    except Exception:
        pass

    try:
        data.update(vars(obj))
    except TypeError:
        data["value"] = obj

    return data


def to_json(obj: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """
    Convert any supported object into a JSON string.
    """

    return json.dumps(deep_asdict(obj), indent=indent, ensure_ascii=ensure_ascii, default=str)


def from_json(json_str: str) -> Any:
    """
    Parse a JSON string into a Python object.
    """

    return json.loads(json_str)


def safe_log_params(obj: Any, max_value_length: int = 500) -> Dict[str, str]:
    """
    Convert arbitrary parameter objects into MLflow-safe string values.
    """

    params = class_to_dict(obj)
    safe_params: Dict[str, str] = {}

    for key, value in params.items():
        serialized = deep_asdict(value)
        if isinstance(serialized, (dict, list, tuple, set)):
            text = json.dumps(serialized, ensure_ascii=False, sort_keys=True, default=str)
        elif serialized is None:
            text = ""
        else:
            text = str(serialized)

        if len(text) > max_value_length:
            text = f"{text[: max_value_length - 3]}..."

        safe_params[str(key)] = text

    return safe_params


def safe_log_parames(obj: Any, max_value_length: int = 500) -> Dict[str, str]:
    """
    Backward-compatible alias for the previous misspelled helper name.
    """

    return safe_log_params(obj, max_value_length=max_value_length)
