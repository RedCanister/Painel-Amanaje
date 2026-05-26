from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VIEWABLE_EXTENSIONS = {".onnx", ".pt", ".pth"}
PREFERRED_EXTENSION = ".onnx"
DEFAULT_NETRON_HOST = "0.0.0.0"
DEFAULT_NETRON_PORT = 8082


class NetronUnavailableError(RuntimeError):
    """Raised when the optional Netron dependency cannot start a viewer."""


@dataclass
class NetronSession:
    artifact_path: Path
    viewer_url: str
    host: str
    port: int


_ACTIVE_SESSION: NetronSession | None = None


def netron_config() -> dict[str, Any]:
    port = int(os.getenv("AMANAJE_NETRON_PORT", str(DEFAULT_NETRON_PORT)))
    public_url = os.getenv("AMANAJE_NETRON_PUBLIC_URL", f"http://localhost:{port}").rstrip("/")
    return {
        "host": os.getenv("AMANAJE_NETRON_HOST", DEFAULT_NETRON_HOST),
        "port": port,
        "public_url": public_url,
    }


def _load_netron() -> Any:
    try:
        import netron
    except Exception as exc:  # pragma: no cover - exercised by route behavior.
        raise NetronUnavailableError(f"Netron is not installed or could not be imported: {exc}") from exc
    return netron


def _safe_resolved_string(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _same_artifact(left: Path, right: Path) -> bool:
    return _safe_resolved_string(left) == _safe_resolved_string(right)


def infer_artifact_framework(extension: str, artifact_manifest: Mapping[str, Any] | None = None) -> str:
    manifest_framework = str((artifact_manifest or {}).get("framework") or "").strip().lower()
    if manifest_framework:
        return manifest_framework
    if extension == ".onnx":
        return "onnx"
    if extension in {".pt", ".pth"}:
        return "pytorch"
    return "unknown"


def build_netron_status(
    *,
    model_id: int,
    model_name: str | None = None,
    artifact_path: str | Path | None,
    artifact_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    config = netron_config()
    path = Path(artifact_path) if artifact_path else None
    extension = path.suffix.lower() if path is not None else ""
    artifact_name = path.name if path is not None else None
    exists = bool(path is not None and path.exists() and path.is_file())
    viewable = bool(exists and extension in VIEWABLE_EXTENSIONS)

    if path is None:
        warnings.append("This model does not have a registered artifact path.")
    elif not exists:
        warnings.append("The registered artifact path does not exist or is not a file.")
    elif extension not in VIEWABLE_EXTENSIONS:
        warnings.append(
            f"Netron v1 viewer supports {', '.join(sorted(VIEWABLE_EXTENSIONS))}; "
            f"the selected artifact uses {extension or 'no extension'}."
        )
    elif extension != PREFERRED_EXTENSION:
        warnings.append("ONNX artifacts are preferred; PyTorch artifacts are shown with Netron's supported/experimental parser.")

    active = bool(_ACTIVE_SESSION and path is not None and _same_artifact(_ACTIVE_SESSION.artifact_path, path))
    viewer_url = _ACTIVE_SESSION.viewer_url if active and _ACTIVE_SESSION else None
    return {
        "status": "ready" if viewable else "not_viewable",
        "model_id": model_id,
        "model_name": model_name,
        "viewable": viewable,
        "preferred_extension": PREFERRED_EXTENSION,
        "supported_extensions": sorted(VIEWABLE_EXTENSIONS),
        "artifact_name": artifact_name,
        "artifact_extension": extension or None,
        "artifact_exists": exists,
        "framework": infer_artifact_framework(extension, artifact_manifest),
        "warnings": warnings,
        "viewer_active": active,
        "viewer_url": viewer_url,
        "netron": {
            "host": config["host"],
            "port": config["port"],
            "public_url": config["public_url"],
        },
    }


def _call_netron_start(netron: Any, artifact_path: Path, *, host: str, port: int) -> Any:
    try:
        return netron.start(str(artifact_path), address=(host, port), browse=False)
    except TypeError:
        pass
    try:
        return netron.start(str(artifact_path), host=host, port=port, browse=False)
    except TypeError:
        pass
    try:
        return netron.start(str(artifact_path), port=port, browse=False)
    except TypeError:
        return netron.start(str(artifact_path), browse=False)


def start_netron_viewer(
    *,
    artifact_path: str | Path,
    status_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    global _ACTIVE_SESSION

    path = Path(artifact_path)
    status = dict(status_payload or build_netron_status(model_id=0, artifact_path=path))
    if not status.get("viewable"):
        return {**status, "status": "not_viewable"}

    config = netron_config()
    viewer_url = str(config["public_url"])
    if _ACTIVE_SESSION and _same_artifact(_ACTIVE_SESSION.artifact_path, path):
        return {**status, "status": "running", "viewer_active": True, "viewer_url": _ACTIVE_SESSION.viewer_url}

    if _ACTIVE_SESSION is not None:
        stop_netron_viewer()

    netron = _load_netron()
    _call_netron_start(netron, path, host=str(config["host"]), port=int(config["port"]))
    _ACTIVE_SESSION = NetronSession(
        artifact_path=path,
        viewer_url=viewer_url,
        host=str(config["host"]),
        port=int(config["port"]),
    )
    return {**status, "status": "running", "viewer_active": True, "viewer_url": viewer_url}


def stop_netron_viewer() -> dict[str, Any]:
    global _ACTIVE_SESSION

    previous = _ACTIVE_SESSION
    if previous is None:
        return {"status": "stopped", "viewer_active": False, "stopped": False}

    netron = _load_netron()
    try:
        netron.stop((previous.host, previous.port))
    except TypeError:
        netron.stop()
    finally:
        _ACTIVE_SESSION = None

    return {
        "status": "stopped",
        "viewer_active": False,
        "stopped": True,
        "artifact_name": previous.artifact_path.name,
    }


def reset_netron_session_for_tests() -> None:
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = None
