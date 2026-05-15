from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


EXTENSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ENTRYPOINT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")


def _split_allowlist(value: str | None) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _safe_metadata(manifest: Mapping[str, Any], extension_id: str) -> dict[str, Any]:
    return {
        "id": extension_id,
        "title": str(manifest.get("title") or extension_id.replace("_", " ").replace("-", " ").title()),
        "description": str(manifest.get("description") or ""),
        "version": str(manifest.get("version") or "0.1.0"),
        "author": str(manifest.get("author") or ""),
        "tags": [str(tag) for tag in manifest.get("tags", []) if str(tag).strip()]
        if isinstance(manifest.get("tags"), list)
        else [],
    }


@dataclass(frozen=True)
class ExtensionContext:
    extension_id: str
    manifest: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)

    def component_id(self, local_id: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", str(local_id or "component")).strip("-")
        return f"ext-{self.extension_id}-{normalized or 'component'}"


@dataclass
class LoadedExtension:
    id: str
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    context: ExtensionContext
    layout: Any = None
    module_name: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "module": self.module_name,
        }


@dataclass
class ExtensionRegistry:
    enabled: bool
    root: str
    extensions: dict[str, LoadedExtension] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def summaries(self) -> list[dict[str, Any]]:
        return [extension.summary() for extension in self.extensions.values()]

    def get(self, extension_id: str | None) -> LoadedExtension | None:
        if not extension_id:
            return None
        return self.extensions.get(str(extension_id))

    def first(self) -> LoadedExtension | None:
        return next(iter(self.extensions.values()), None)

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root": self.root,
            "extensions": self.summaries(),
            "errors": self.errors,
        }


def _load_manifest(extension_dir: Path) -> dict[str, Any]:
    manifest_path = extension_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing manifest.json.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest JSON: {exc.msg}.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    return manifest


def _resolve_entrypoint(extension_dir: Path, entrypoint: str) -> tuple[Path, str, str]:
    if not ENTRYPOINT_PATTERN.fullmatch(entrypoint):
        raise ValueError("Entrypoint must use local_module:function syntax.")
    module_name, function_name = entrypoint.split(":", 1)
    module_path = extension_dir.joinpath(*module_name.split(".")).with_suffix(".py").resolve()
    extension_root = extension_dir.resolve()
    if extension_root not in module_path.parents:
        raise ValueError("Entrypoint must resolve inside the extension directory.")
    if not module_path.is_file():
        raise ValueError(f"Entrypoint module not found: {module_name}.")
    return module_path, module_name, function_name


def _import_entrypoint_module(module_path: Path, import_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(import_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError("Unable to create module import spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_register_function(module: ModuleType, function_name: str) -> Callable[..., Any]:
    register = getattr(module, function_name, None)
    if not callable(register):
        raise ValueError(f"Entrypoint function is not callable: {function_name}.")
    return register


def _normalize_registration_result(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, Mapping):
        return result.get("layout"), dict(result.get("metadata") or {})
    return result, {}


def load_dashboard_extensions(
    root: str | Path,
    *,
    enabled: bool,
    allowlist: str | set[str] | None = None,
    app: Any = None,
    api_client: Any = None,
    config: Mapping[str, Any] | None = None,
) -> ExtensionRegistry:
    root_path = Path(root).resolve()
    registry = ExtensionRegistry(enabled=bool(enabled), root=str(root_path))
    if not enabled:
        return registry

    allowed_ids = allowlist if isinstance(allowlist, set) else _split_allowlist(allowlist)
    if not root_path.exists():
        registry.errors.append({"id": None, "reason": "extensions_dir_missing", "detail": str(root_path)})
        return registry

    for extension_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        extension_label = extension_dir.name
        try:
            manifest = _load_manifest(extension_dir)
            extension_id = str(manifest.get("id") or extension_label).strip()
            if not EXTENSION_ID_PATTERN.fullmatch(extension_id):
                raise ValueError("Extension id must be alphanumeric with optional dash or underscore.")
            if extension_id != extension_label:
                raise ValueError("Extension id must match its directory name.")
            if allowed_ids and extension_id not in allowed_ids:
                continue
            if manifest.get("enabled") is False and extension_id not in allowed_ids:
                continue

            entrypoint = str(manifest.get("entrypoint") or "extension:register")
            module_path, module_name, function_name = _resolve_entrypoint(extension_dir, entrypoint)
            import_name = f"amanaje_dashboard_extension_{extension_id}_{module_name.replace('.', '_')}"
            module = _import_entrypoint_module(module_path, import_name)
            register = _get_register_function(module, function_name)

            metadata = _safe_metadata(manifest, extension_id)
            context = ExtensionContext(extension_id=extension_id, manifest=dict(manifest), config=dict(config or {}))
            layout, registration_metadata = _normalize_registration_result(register(app, api_client, context))
            metadata.update({key: value for key, value in registration_metadata.items() if key in {"title", "description", "version", "author", "tags"}})
            registry.extensions[extension_id] = LoadedExtension(
                id=extension_id,
                metadata=metadata,
                manifest=dict(manifest),
                context=context,
                layout=layout,
                module_name=module_name,
            )
        except Exception as exc:
            registry.errors.append({"id": extension_label, "reason": exc.__class__.__name__, "detail": str(exc)})

    return registry
