from __future__ import annotations

import sys
import traceback
from io import StringIO
from types import SimpleNamespace
from typing import Any


def execute_editor_code(code: str, registry_context: dict[str, Any] | None = None) -> dict[str, Any]:
    import app.utils.main_utils as _utils

    normalized_code = str(code or "").strip()
    if not normalized_code:
        return {
            "status": "error",
            "variables": {},
            "stdout": "",
            "stderr": "",
            "error": "No code provided",
        }

    def _safe_import(module_name: str) -> Any:
        try:
            return __import__(module_name)
        except Exception:
            return None

    normalized_registry_context = registry_context if isinstance(registry_context, dict) else {}
    datasets = list(normalized_registry_context.get("datasets") or [])
    models = list(normalized_registry_context.get("models") or [])

    def _record_by_id(records: list[dict[str, Any]], identifier: Any = None) -> dict[str, Any] | None:
        if identifier in (None, ""):
            return records[0] if records else None
        return next((record for record in records if str(record.get("id")) == str(identifier) or record.get("name") == identifier), None)

    def list_datasets() -> list[dict[str, Any]]:
        return [dict(record) for record in datasets if isinstance(record, dict)]

    def list_models() -> list[dict[str, Any]]:
        return [dict(record) for record in models if isinstance(record, dict)]

    def load_dataset(identifier: Any = None) -> Any:
        record = _record_by_id(list_datasets(), identifier)
        if record is None:
            raise ValueError("Dataset is not available in the editor registry context.")
        return _utils._load_dataset_frame_from_record(SimpleNamespace(**record))

    def load_model(identifier: Any = None) -> Any:
        record = _record_by_id(list_models(), identifier)
        if record is None:
            raise ValueError("Model is not available in the editor registry context.")
        artifact = _utils.load_runtime_artifact(record.get("path"), parameters=record.get("parameters") or {})
        return artifact.get("model")

    namespace = {
        "__builtins__": __builtins__,
        "np": _safe_import("numpy"),
        "pd": _safe_import("pandas"),
        "sk": _safe_import("sklearn"),
        "torch": _safe_import("torch"),
        "ox": _safe_import("onnx"),
        "registry_context": normalized_registry_context,
        "list_datasets": list_datasets,
        "list_models": list_models,
        "load_dataset": load_dataset,
        "load_model": load_model,
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    error_message = None
    extracted_variables: dict[str, dict[str, Any]] = {}
    try:
        exec(normalized_code, namespace)
        extracted_variables = _utils._extract_variables(namespace)
    except Exception:
        error_message = traceback.format_exc()
    finally:
        stdout_output = sys.stdout.getvalue()
        stderr_output = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    editor_metadata = _utils._extract_editor_metadata(extracted_variables)
    normalized_document = _utils._normalize_code_document(
        code=normalized_code,
        variables=extracted_variables,
        metadata=editor_metadata,
    )
    defined_names = _utils._extract_defined_names(normalized_code)

    return {
        "status": "success" if error_message is None else "error",
        "variables": extracted_variables,
        "metadata": editor_metadata,
        "document": normalized_document,
        "summary": {
            "variable_count": len(extracted_variables),
            "defined_name_count": len(defined_names),
            "defined_names": defined_names,
            "has_error": error_message is not None,
        },
        "stdout": stdout_output,
        "stderr": stderr_output,
        "error": error_message,
    }
