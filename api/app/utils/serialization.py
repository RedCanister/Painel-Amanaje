"""
utils/serialization.py

Utility functions for serializing and deserializaing Python objects,
configs, and ML model metadata into dictionary or JSON-compatible formats.

These utilities support:
- Converting complex Python objects to dictionaries
- Recursive serialization of nested configs
- Safe JSON export/ import for MLflow, Airflow XComs, and FastAPI endpoints
"""

import inspect
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

def deep_asdict(obj: Any) -> Any:
    """
    Recursively converts dataclasses or nested objects into pure Python dicts.
    """

    if is_dataclass(obj):
        return {k: deep_asdict(v) for k, v in asdict(obj).items()}
    elif hasattr(obj, "__dict__"):
        return {k: deep_asdict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple, set)):
        return [deep_asdict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: deep_asdict(v) for k, v in obj.items()}
    else:
        return obj


def class_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Extracts all atributtes (instance + class) from a Python class or instance. 
    Filters out callables and private attributes.
    """

    if inspect.isclass(obj):
        members = inspect.getmembers(obj)
        data = {
            k: v for k, v in members
            if not k.startswith('__') and not inspect.isroutine(v)
        }
    else:
        data = {}
        data.update({
            k: v for k, v in inspect.getmembers(obj.__class__)
            if not k.startswith('__') and not inspect.isroutine(v)
        })

        data.update(vars(obj))

    return data


def to_json(obj: Any, indent: int = 4) -> str:
    """
    Converts any Python object into JSON string representation.
    Handles nested objects and dataclasses.
    """

    try:
        return json.dumps(deep_asdict(obj), indent=indent, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(obj), indent=indent, ensure_ascii=False)


def from_json(json_str: str) -> Any:
    """
    Converts a JSON string back into a Python dictionary.
    """
    return json.loads(json_str)


def safe_log_parames(obj: Any) -> Dict[str, Any]:
    """
    Prepares object attributes for MLflow logging.
    Converts unsupported types (lists, dict, etc.) to strings.
    """

    params = class_to_dict(obj)
    safe_params = {}
    for k, v in params.items():
        if isinstance(v, (dict, list, tuple)):
            safe_params[k] = json.dumps(v)
        else:
            safe_params[k] = str(v)
    
    return safe_params


