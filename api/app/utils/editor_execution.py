from __future__ import annotations

import sys
import traceback
from io import StringIO
from typing import Any


def execute_editor_code(code: str) -> dict[str, Any]:
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

    namespace = {
        "__builtins__": __builtins__,
        "np": _safe_import("numpy"),
        "pd": _safe_import("pandas"),
        "sk": _safe_import("sklearn"),
        "torch": _safe_import("torch"),
        "ox": _safe_import("onnx"),
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
