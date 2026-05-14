from __future__ import annotations

import inspect
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping, Optional

from pydantic import BaseModel

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except Exception:  # pragma: no cover - only used when aiofiles is unavailable.
    aiofiles = None  # type: ignore[assignment]
    AIOFILES_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - only used when pandas is unavailable.
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

from .artifact_utils import inspect_model_artifact
from .io import load_json, load_yaml
from .tabular_utils import load_tabular_from_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_UPLOAD_DIR = REPO_ROOT / "models"
DATASET_UPLOAD_DIR = REPO_ROOT / "data" / "datasets"


def _require_aiofiles() -> None:
    if not AIOFILES_AVAILABLE or aiofiles is None:
        raise RuntimeError("aiofiles is required for async file upload helpers.")


def _require_pandas() -> None:
    if not PANDAS_AVAILABLE or pd is None:
        raise RuntimeError("pandas is required for dataset payload helpers.")


def get_file_size(file_path: str | Path) -> tuple[float, float]:
    """
    Return file size in kilobytes and megabytes.
    """

    try:
        raw_size = Path(file_path).stat().st_size
    except FileNotFoundError:
        return 0.0, 0.0
    except OSError:
        return 0.0, 0.0

    kb_size = raw_size / 1024
    mb_size = kb_size / 1024
    return kb_size, mb_size


async def _seek_to_start(file: Any) -> None:
    """
    Reset an uploaded file pointer when the object exposes ``seek``.
    """

    seek_method = getattr(file, "seek", None)
    if seek_method is None:
        return

    result = seek_method(0)
    if inspect.isawaitable(result):
        await result


async def save_file_to_disk(file: Any, dest_path: str) -> float:
    """
    Save an uploaded file to disk and return its size in megabytes.
    """

    _require_aiofiles()
    destination = Path(dest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    await _seek_to_start(file)
    contents = await file.read()
    async with aiofiles.open(destination, "wb") as output_file:
        await output_file.write(contents)
    await _seek_to_start(file)

    return round(len(contents) / (1024 * 1024), 4)


def get_file_extension(file_path: str | Path) -> str:
    """
    Return a lowercase file extension.
    """

    return Path(file_path).suffix.lower()


def _load_model_metadata_from_onnx(file_path: Path) -> dict[str, Any]:
    """
    Extract lightweight metadata from an ONNX model when the dependency exists.
    """

    try:
        import onnx
    except Exception as exc:
        return {"error": f"Unable to inspect ONNX model: {exc}"}

    model = onnx.load(str(file_path))
    graph = model.graph
    return {
        "framework": "onnx",
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "node_count": len(graph.node),
        "inputs": [tensor.name for tensor in graph.input],
        "outputs": [tensor.name for tensor in graph.output],
    }


def recover_model_params(file_path: str | Path) -> dict[str, Any]:
    """
    Recover metadata and runtime capability details from a persisted model artifact.
    """

    path = Path(file_path)

    try:
        return inspect_model_artifact(path, load_runtime=False)
    except Exception as exc:
        return {"error": str(exc)}


def get_upload_dir(op_id: str) -> str:
    """
    Resolve the upload directory for the given operation identifier.
    """

    normalized = str(op_id).strip().lower().replace("_", "-")
    if normalized in {"assistant-model", "assistant-models", "assistantmodel", "assistant", "llm", "slm"}:
        return str(REPO_ROOT / "runtime_artifacts" / "assistant_models" / "uploads")
    if normalized in {"model", "models", "learning_model"}:
        return str(MODEL_UPLOAD_DIR)
    return str(DATASET_UPLOAD_DIR)


def _parse_json_field(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default

    if isinstance(value, (dict, list, bool, int, float)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def _parse_bool_field(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return default


def _parse_list_field(value: Any) -> Optional[list[str]]:
    if value is None or value == "":
        return None

    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, tuple):
        return [str(item) for item in value]

    if isinstance(value, str):
        parsed = _parse_json_field(value, value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [item.strip() for item in value.split(",") if item.strip()]

    return [str(value)]


def build_payload_model(common: Mapping[str, Any], form_data: Any) -> dict[str, Any]:
    """
    Build a normalized model payload from multipart form data.
    """

    return {
        **dict(common),
        "model_type": form_data.get("modelType") or form_data.get("model_type") or "learning_model",
        "parameters": _parse_json_field(form_data.get("parameters"), {}),
        "metrics": _parse_json_field(form_data.get("metrics"), {}),
        "reference_data": form_data.get("referenceData") or form_data.get("reference_data"),
        "input_features": _parse_list_field(form_data.get("inputFeatures") or form_data.get("input_features")),
        "output_features": _parse_list_field(form_data.get("outputFeatures") or form_data.get("output_features")),
        "is_trained": _parse_bool_field(form_data.get("isTrained") or form_data.get("is_trained"), False),
        "is_tested": _parse_bool_field(form_data.get("isTested") or form_data.get("is_tested"), False),
        "is_deployed": _parse_bool_field(form_data.get("isDeployed") or form_data.get("is_deployed"), False),
    }


async def build_payload_data(
    common: Mapping[str, Any],
    form_data: Any,
    uploaded_file: Any,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """
    Build a normalized dataset payload and return the parsed DataFrame.
    """

    _require_pandas()
    await _seek_to_start(uploaded_file)
    content = await uploaded_file.read()
    await _seek_to_start(uploaded_file)

    filename = getattr(uploaded_file, "filename", None) or str(common.get("name") or "dataset.csv")
    dataframe, parser_report = load_tabular_from_bytes(content, filename=filename)

    payload = {
        **dict(common),
        "dataset_type": form_data.get("datasetType") or form_data.get("dataset_type") or "dataset",
        "shape": list(dataframe.shape),
        "has_features": bool(len(dataframe.columns)),
        "features_list": dataframe.columns.to_list() or None,
        "connection_string": form_data.get("connectionString") or form_data.get("connection_string"),
    }

    return payload, dataframe, parser_report


def _get_object_name(obj: Any) -> str:
    try:
        return getattr(obj, "__name__", None) or obj.__class__.__name__
    except Exception:
        return "Unknown"


def _get_object_repr(obj: Any) -> str:
    try:
        return repr(obj)
    except Exception:
        return "Error getting representation"


def _get_object_dict(obj: Any) -> Any:
    try:
        return getattr(obj, "__dict__", None)
    except Exception:
        return None


def _get_pydantic_dump(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump()
        except Exception:
            return None
    return None


def _get_object_attributes(obj: Any) -> list[str]:
    try:
        return dir(obj)
    except Exception:
        return []


def _get_function_signature(obj: Any) -> Optional[str]:
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            return str(inspect.signature(obj))
        except Exception:
            return None
    return None


def _get_module_file(obj: Any) -> Optional[str]:
    if inspect.ismodule(obj):
        try:
            return str(obj.__file__)
        except Exception:
            return "Built-in or no __file__"
    return None


def debug_type(obj: Any) -> dict[str, Any]:
    """
    Print and return structured debugging information about any Python object.
    """

    payload = {
        "type": str(type(obj)),
        "name": _get_object_name(obj),
        "repr": _get_object_repr(obj),
        "dict": _get_object_dict(obj),
        "pydantic_model_dump": _get_pydantic_dump(obj),
        "attribute_count": len(_get_object_attributes(obj)),
        "signature": _get_function_signature(obj),
        "module_file": _get_module_file(obj),
    }

    print("\n" + "-" * 40)
    print("Debugging object")
    print(pformat(payload, width=100))
    print("-" * 40 + "\n")

    return payload
