from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


DASHBOARD_ROOT = Path(__file__).resolve().parents[2] / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from extension_loader import load_dashboard_extensions  # noqa: E402


def _write_extension(root: Path, extension_id: str, *, manifest_extra: dict | None = None, source: str | None = None) -> Path:
    extension_dir = root / extension_id
    extension_dir.mkdir()
    manifest = {
        "id": extension_id,
        "title": "Quality Dashboard",
        "description": "A test dashboard.",
        "version": "0.1.0",
        "author": "Tests",
        "tags": ["test"],
        "entrypoint": "extension:register",
    }
    manifest.update(manifest_extra or {})
    (extension_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (extension_dir / "extension.py").write_text(
        source
        or (
            "def register(app, api_client, context):\n"
            "    return {'layout': f\"layout:{context.component_id('output')}\", 'metadata': {'title': 'Runtime Title'}}\n"
        ),
        encoding="utf-8",
    )
    return extension_dir


def test_extension_loader_is_disabled_by_default(tmp_path):
    _write_extension(tmp_path, "quality")

    registry = load_dashboard_extensions(tmp_path, enabled=False)

    assert registry.enabled is False
    assert registry.extensions == {}
    assert registry.errors == []


def test_extension_loader_registers_trusted_local_extension(tmp_path):
    _write_extension(tmp_path, "quality")

    registry = load_dashboard_extensions(tmp_path, enabled=True, app=object(), api_client=object(), config={"mode": "test"})

    assert registry.enabled is True
    assert list(registry.extensions) == ["quality"]
    extension = registry.get("quality")
    assert extension is not None
    assert extension.layout == "layout:ext-quality-output"
    assert extension.metadata["title"] == "Runtime Title"
    assert registry.to_payload()["extensions"][0]["id"] == "quality"


def test_extension_loader_respects_allowlist_and_manifest_enabled_flag(tmp_path):
    _write_extension(tmp_path, "active")
    _write_extension(tmp_path, "sample", manifest_extra={"enabled": False})

    registry = load_dashboard_extensions(tmp_path, enabled=True)
    assert list(registry.extensions) == ["active"]

    registry = load_dashboard_extensions(tmp_path, enabled=True, allowlist="sample")
    assert list(registry.extensions) == ["sample"]


def test_extension_loader_records_invalid_manifest_errors(tmp_path):
    _write_extension(tmp_path, "bad", manifest_extra={"id": "../bad"})

    registry = load_dashboard_extensions(tmp_path, enabled=True)

    assert registry.extensions == {}
    assert registry.errors
    assert registry.errors[0]["id"] == "bad"


def test_visualization_template_exposes_custom_dashboard_tab(project_root):
    source = (project_root / "api" / "templates" / "base_plot.html").read_text(encoding="utf-8")

    assert "Custom Dashboards" in source
    assert "dashboardExtensionsEnabled" in source
    assert "Custom dashboards are disabled in Settings." in source
    assert "/extensions.json" in source
    assert "/extensions/${encodeURIComponent(extension.id)}" in source


def test_dashboard_app_exposes_extension_metadata_route(monkeypatch):
    if importlib.util.find_spec("dash") is None:
        pytest.skip("Dash is not installed in this environment.")

    monkeypatch.setenv("AMANAJE_DASH_EXTENSIONS_ENABLED", "false")
    spec = importlib.util.spec_from_file_location("dashboard_app_under_test", DASHBOARD_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    response = module.server.test_client().get("/extensions.json")

    assert response.status_code == 200
    assert response.get_json()["enabled"] is False
