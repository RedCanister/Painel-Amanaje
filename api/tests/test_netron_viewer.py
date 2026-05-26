from __future__ import annotations

from types import SimpleNamespace

from app.utils import netron_viewer


def test_build_netron_status_prefers_registered_onnx_artifacts(tmp_path):
    artifact_path = tmp_path / "identity.onnx"
    artifact_path.write_bytes(b"onnx")

    payload = netron_viewer.build_netron_status(
        model_id=7,
        model_name="identity",
        artifact_path=artifact_path,
        artifact_manifest={"framework": "onnx"},
    )

    assert payload["status"] == "ready"
    assert payload["viewable"] is True
    assert payload["artifact_name"] == "identity.onnx"
    assert payload["artifact_extension"] == ".onnx"
    assert "path" not in payload


def test_build_netron_status_reports_missing_artifact_without_path_leak(tmp_path):
    payload = netron_viewer.build_netron_status(
        model_id=8,
        model_name="missing",
        artifact_path=tmp_path / "missing.onnx",
    )

    assert payload["status"] == "not_viewable"
    assert payload["viewable"] is False
    assert payload["artifact_exists"] is False
    assert payload["warnings"]
    assert "path" not in payload


def test_start_and_stop_netron_viewer_uses_lazy_netron_import(monkeypatch, tmp_path):
    netron_viewer.reset_netron_session_for_tests()
    artifact_path = tmp_path / "network.onnx"
    artifact_path.write_bytes(b"onnx")
    calls = []

    def fake_start(file_path, **kwargs):
        calls.append(("start", file_path, kwargs))
        return "http://localhost:8082"

    def fake_stop(*args, **kwargs):
        calls.append(("stop", args, kwargs))

    monkeypatch.setenv("AMANAJE_NETRON_PUBLIC_URL", "http://localhost:8082")
    monkeypatch.setattr(netron_viewer, "_load_netron", lambda: SimpleNamespace(start=fake_start, stop=fake_stop))

    status = netron_viewer.build_netron_status(model_id=9, artifact_path=artifact_path)
    opened = netron_viewer.start_netron_viewer(artifact_path=artifact_path, status_payload=status)
    stopped = netron_viewer.stop_netron_viewer()

    assert opened["status"] == "running"
    assert opened["viewer_url"] == "http://localhost:8082"
    assert stopped["status"] == "stopped"
    assert [call[0] for call in calls] == ["start", "stop"]
