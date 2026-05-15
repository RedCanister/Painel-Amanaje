from __future__ import annotations

import json
import sys
from pathlib import Path


DASHBOARD_ROOT = Path(__file__).resolve().parents[2] / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from extension_loader import load_dashboard_extensions  # noqa: E402


def _write_extension(root: Path, extension_id: str, *, manifest: dict | None = None, code: str | None = None) -> Path:
    extension_dir = root / extension_id
    extension_dir.mkdir()
    payload = {
        "id": extension_id,
        "title": "Ops Extension",
        "description": "Local dashboard extension.",
        "entrypoint": "extension:register",
        **(manifest or {}),
    }
    (extension_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (extension_dir / "extension.py").write_text(
        code
        or """
def register(app, api_client, context):
    return {
        "layout": f"loaded:{context.component_id('stage')}",
        "metadata": {"tags": ["ops", "local"]},
    }
""".strip(),
        encoding="utf-8",
    )
    return extension_dir


def test_dashboard_extensions_stay_quiet_when_disabled(tmp_path):
    registry = load_dashboard_extensions(tmp_path / "missing", enabled=False)

    assert registry.enabled is False
    assert registry.extensions == {}
    assert registry.errors == []


def test_dashboard_extensions_report_missing_enabled_directory(tmp_path):
    registry = load_dashboard_extensions(tmp_path / "missing", enabled=True)

    assert registry.enabled is True
    assert registry.extensions == {}
    assert registry.errors[0]["reason"] == "extensions_dir_missing"


def test_dashboard_extensions_load_manifest_entrypoint_and_safe_summary(tmp_path):
    _write_extension(tmp_path, "ops_panel")

    registry = load_dashboard_extensions(tmp_path, enabled=True)
    summary = registry.summaries()[0]

    assert registry.errors == []
    assert list(registry.extensions) == ["ops_panel"]
    assert summary["id"] == "ops_panel"
    assert summary["title"] == "Ops Extension"
    assert summary["tags"] == ["ops", "local"]
    assert registry.get("ops_panel").layout == "loaded:ext-ops_panel-stage"


def test_dashboard_extensions_allowlist_and_manifest_flags(tmp_path):
    _write_extension(tmp_path, "disabled_panel", manifest={"enabled": False})
    _write_extension(tmp_path, "allowed_panel", manifest={"enabled": False})
    _write_extension(tmp_path, "skipped_panel")

    registry = load_dashboard_extensions(tmp_path, enabled=True, allowlist="allowed_panel")

    assert registry.errors == []
    assert list(registry.extensions) == ["allowed_panel"]


def test_dashboard_extensions_reject_invalid_or_mismatched_manifests(tmp_path):
    _write_extension(tmp_path, "bad_id", manifest={"id": "../bad"})
    _write_extension(tmp_path, "wrong_label", manifest={"id": "different_label"})

    registry = load_dashboard_extensions(tmp_path, enabled=True)

    assert registry.extensions == {}
    assert {error["id"] for error in registry.errors} == {"bad_id", "wrong_label"}
