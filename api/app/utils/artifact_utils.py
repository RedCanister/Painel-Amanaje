from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

from .io import load_json, load_yaml


try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None  # type: ignore[assignment]
    JOBLIB_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


MODEL_CAPABILITY_MATRIX: dict[str, dict[str, Any]] = {
    ".joblib": {
        "label": "scikit-learn Joblib",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": True,
        "simulate": True,
        "monitor": True,
    },
    ".pkl": {
        "label": "Python Pickle",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": "guarded",
        "simulate": "guarded",
        "monitor": "guarded",
    },
    ".pickle": {
        "label": "Python Pickle",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": "guarded",
        "simulate": "guarded",
        "monitor": "guarded",
    },
    ".pt": {
        "label": "PyTorch Artifact",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": "guarded",
        "simulate": "guarded",
        "monitor": "guarded",
    },
    ".pth": {
        "label": "PyTorch Artifact",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": "guarded",
        "simulate": "guarded",
        "monitor": "guarded",
    },
    ".onnx": {
        "label": "ONNX",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": False,
        "simulate": False,
        "monitor": False,
    },
    ".json": {
        "label": "JSON Config",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": False,
        "simulate": False,
        "monitor": False,
    },
    ".yaml": {
        "label": "YAML Config",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": False,
        "simulate": False,
        "monitor": False,
    },
    ".yml": {
        "label": "YAML Config",
        "register": True,
        "inspect": True,
        "train": False,
        "predict": False,
        "simulate": False,
        "monitor": False,
    },
}


class RuntimeArtifactDependencyError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        missing_dependencies: Optional[list[str]] = None,
        artifact_loader: Optional[str] = None,
        remediation: Optional[str] = None,
        manifest: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.missing_dependencies = list(missing_dependencies or [])
        self.artifact_loader = artifact_loader
        self.remediation = remediation or (
            "Install the missing runtime dependency or re-export the artifact into a portable format such as joblib or TorchScript."
        )
        self.manifest = dict(manifest or {})

    def to_payload(self) -> dict[str, Any]:
        return {
            "missing_dependencies": self.missing_dependencies,
            "artifact_loader": self.artifact_loader,
            "remediation": self.remediation,
            "artifact_manifest": self.manifest,
        }


def get_model_capability_matrix() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in MODEL_CAPABILITY_MATRIX.items()}


def _empty_manifest(path: Path) -> dict[str, Any]:
    extension = path.suffix.lower()
    base_capabilities = MODEL_CAPABILITY_MATRIX.get(
        extension,
        {
            "register": True,
            "inspect": False,
            "train": False,
            "predict": False,
            "simulate": False,
            "monitor": False,
        },
    )
    return {
        "artifact_format": extension or "unknown",
        "loader": None,
        "runtime_capabilities": {
            "register": bool(base_capabilities.get("register", False)),
            "inspect": bool(base_capabilities.get("inspect", False)),
            "train": bool(base_capabilities.get("train", False)),
            "predict": bool(base_capabilities.get("predict") is True),
            "simulate": bool(base_capabilities.get("simulate") is True),
            "monitor": bool(base_capabilities.get("monitor") is True),
        },
        "framework": "unknown",
        "source": str(path),
        "entrypoint": None,
        "needs_companion_spec": False,
        "warnings": [],
        "metadata": {},
        "supported_file_types": sorted(MODEL_CAPABILITY_MATRIX),
        "capability_matrix": get_model_capability_matrix(),
    }


def _safe_public_methods(obj: Any) -> list[str]:
    methods = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name, None)
        if callable(value):
            methods.append(str(name))
    return sorted(methods)[:20]


def _load_onnx_metadata(file_path: Path) -> dict[str, Any]:
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


def _infer_framework_from_loaded_object(obj: Any, fallback: str = "unknown") -> str:
    if hasattr(obj, "predict"):
        return "sklearn"
    if TORCH_AVAILABLE and torch is not None:
        if isinstance(obj, torch.nn.Module):
            return "pytorch"
        jit_module = getattr(torch.jit, "RecursiveScriptModule", None)
        if jit_module is not None and isinstance(obj, jit_module):
            return "pytorch"
    if hasattr(obj, "forward"):
        return "pytorch"
    return fallback


def _apply_predictable_runtime_capabilities(manifest: dict[str, Any], loaded_object: Any) -> dict[str, Any]:
    supports_predict = hasattr(loaded_object, "predict") or callable(getattr(loaded_object, "forward", None))
    if manifest["framework"] == "pytorch" and callable(loaded_object):
        supports_predict = True
    capabilities = {
        "predict": hasattr(loaded_object, "predict"),
        "predict_proba": hasattr(loaded_object, "predict_proba"),
        "transform": hasattr(loaded_object, "transform"),
        "fit_predict": hasattr(loaded_object, "fit_predict"),
        "score_samples": hasattr(loaded_object, "score_samples"),
        "decision_function": hasattr(loaded_object, "decision_function"),
        "forward": callable(getattr(loaded_object, "forward", None)),
    }
    manifest["runtime_capabilities"].update(
        {
            "predict": bool(supports_predict),
            "simulate": bool(supports_predict),
            "monitor": bool(supports_predict),
        }
    )
    if hasattr(loaded_object, "predict"):
        manifest["entrypoint"] = "predict"
    elif hasattr(loaded_object, "transform"):
        manifest["entrypoint"] = "transform"
    elif hasattr(loaded_object, "fit_predict"):
        manifest["entrypoint"] = "fit_predict"
    elif hasattr(loaded_object, "score_samples"):
        manifest["entrypoint"] = "score_samples"
    else:
        manifest["entrypoint"] = "forward"
    manifest["metadata"]["public_methods"] = _safe_public_methods(loaded_object)
    manifest["metadata"]["sklearn_capabilities"] = capabilities
    if hasattr(loaded_object, "get_params"):
        try:
            estimator_params = {}
            for key, value in loaded_object.get_params(deep=False).items():
                try:
                    json.dumps(value)
                    estimator_params[key] = value
                except TypeError:
                    estimator_params[key] = repr(value)
            manifest["metadata"]["estimator_params"] = estimator_params
        except Exception:
            pass
    return manifest


def inspect_model_artifact(
    file_path: str | Path,
    *,
    load_runtime: bool = True,
    parameters: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    path = Path(file_path)
    manifest = _empty_manifest(path)
    extension = path.suffix.lower()
    fallback_framework = str((parameters or {}).get("framework") or "unknown")

    try:
        if extension == ".json":
            params = load_json(path)
            manifest["loader"] = "json"
            manifest["framework"] = "config"
            manifest["metadata"] = params if isinstance(params, dict) else {"payload": params}
            return manifest

        if extension in {".yaml", ".yml"}:
            params = load_yaml(path)
            manifest["loader"] = "yaml"
            manifest["framework"] = "config"
            manifest["metadata"] = params if isinstance(params, dict) else {"payload": params}
            return manifest

        if extension == ".onnx":
            manifest["loader"] = "onnx"
            manifest["framework"] = "onnx"
            manifest["metadata"] = _load_onnx_metadata(path)
            manifest["warnings"].append(
                "ONNX artifacts can be registered and inspected, but direct runtime prediction is not enabled in this v1 path."
            )
            return manifest

        if extension == ".joblib":
            manifest["loader"] = "joblib"
            if not JOBLIB_AVAILABLE or joblib is None:
                manifest["warnings"].append("joblib is not installed in the current environment.")
                return manifest
            if not load_runtime:
                manifest["framework"] = "sklearn"
                manifest["warnings"].append("Joblib support is available for runtime loading.")
                return manifest
            loaded_object = joblib.load(path)
            manifest["framework"] = _infer_framework_from_loaded_object(loaded_object, fallback="sklearn")
            manifest["metadata"]["python_type"] = f"{type(loaded_object).__module__}.{type(loaded_object).__name__}"
            return _apply_predictable_runtime_capabilities(manifest, loaded_object)

        if extension in {".pkl", ".pickle"}:
            manifest["loader"] = "pickle"
            if not load_runtime:
                manifest["framework"] = fallback_framework
                manifest["warnings"].append(
                    "Pickle runtime support is guarded and depends on importable classes being available in the serving process."
                )
                return manifest
            with path.open("rb") as file_handle:
                loaded_object = pickle.load(file_handle)
            manifest["framework"] = _infer_framework_from_loaded_object(loaded_object, fallback=fallback_framework)
            manifest["metadata"]["python_type"] = f"{type(loaded_object).__module__}.{type(loaded_object).__name__}"
            return _apply_predictable_runtime_capabilities(manifest, loaded_object)

        if extension in {".pt", ".pth"}:
            manifest["loader"] = "torchscript"
            manifest["framework"] = "pytorch"
            if not TORCH_AVAILABLE or torch is None:
                manifest["warnings"].append("PyTorch is not installed in the current environment.")
                return manifest
            if not load_runtime:
                manifest["needs_companion_spec"] = True
                manifest["warnings"].append(
                    "Only TorchScript or app-generated PyTorch bundles are directly runnable. Plain state_dict files require a companion model definition."
                )
                return manifest
            try:
                loaded_object = torch.jit.load(str(path), map_location="cpu")
                manifest["metadata"]["python_type"] = f"{type(loaded_object).__module__}.{type(loaded_object).__name__}"
                manifest["needs_companion_spec"] = False
                return _apply_predictable_runtime_capabilities(manifest, loaded_object)
            except Exception as exc:
                manifest["loader"] = "pytorch"
                manifest["needs_companion_spec"] = True
                manifest["warnings"].append(
                    "This .pt/.pth artifact is not readable as TorchScript. It may be a state_dict or a custom bundle that needs an explicit model definition."
                )
                manifest["metadata"]["error"] = str(exc)
                return manifest

        manifest["warnings"].append(f"Unsupported file type: {extension or '[no extension]'}")
        return manifest
    except Exception as exc:
        manifest["metadata"]["error"] = str(exc)
        error_text = str(exc)
        if isinstance(exc, ModuleNotFoundError):
            manifest["metadata"]["missing_dependency"] = getattr(exc, "name", None)
            manifest["warnings"].append(
                f"Missing runtime dependency: {getattr(exc, 'name', 'unknown module')}. Install the dependency or re-export the artifact as a portable bundle."
            )
        elif "__main__" in error_text or "Can't get attribute" in error_text:
            manifest["warnings"].append(
                "This artifact references classes that are not importable in the current runtime. Re-export it as joblib, TorchScript, or an app-generated bundle."
            )
        else:
            manifest["warnings"].append(error_text)
        return manifest


def load_runtime_artifact(
    file_path: str | Path,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    path = Path(file_path)
    manifest = inspect_model_artifact(path, load_runtime=False, parameters=parameters)
    extension = path.suffix.lower()

    if extension == ".joblib":
        if not JOBLIB_AVAILABLE or joblib is None:
            raise ValueError("joblib is not available in the current environment.")
        try:
            model = joblib.load(path)
        except ModuleNotFoundError as exc:
            raise RuntimeArtifactDependencyError(
                f"Missing runtime dependency '{getattr(exc, 'name', 'unknown')}' required to load '{path.name}'.",
                missing_dependencies=[getattr(exc, "name", "unknown")],
                artifact_loader="joblib",
                manifest=manifest,
            ) from exc
    elif extension in {".pkl", ".pickle"}:
        try:
            with path.open("rb") as file_handle:
                model = pickle.load(file_handle)
        except ModuleNotFoundError as exc:
            raise RuntimeArtifactDependencyError(
                f"Missing runtime dependency '{getattr(exc, 'name', 'unknown')}' required to load '{path.name}'.",
                missing_dependencies=[getattr(exc, "name", "unknown")],
                artifact_loader="pickle",
                manifest=manifest,
            ) from exc
    elif extension in {".pt", ".pth"}:
        if not TORCH_AVAILABLE or torch is None:
            raise ValueError("PyTorch is not available in the current environment.")
        try:
            model = torch.jit.load(str(path), map_location="cpu")
        except Exception as exc:
            raise ValueError(
                "Only TorchScript or app-generated PyTorch bundles can run directly. "
                "This artifact appears to need a separate model definition."
            ) from exc
    else:
        raise ValueError(f"Unsupported runtime artifact type: {extension}")

    runtime_manifest = inspect_model_artifact(path, load_runtime=True, parameters=parameters)
    if not runtime_manifest["runtime_capabilities"]["predict"]:
        warning_message = "; ".join(runtime_manifest.get("warnings", []) or [])
        raise ValueError(
            warning_message or f"The artifact '{path.name}' is registered but not runnable in production."
        )

    return {
        "model": model,
        "framework": runtime_manifest["framework"],
        "manifest": runtime_manifest,
    }
