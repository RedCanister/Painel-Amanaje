import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils import mlflow_utils


class _FailingArtifactMlflow:
    def active_run(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def log_artifact(self, *_args, **_kwargs):
        raise PermissionError(13, "Permission denied")


def test_sanitize_mlflow_key_replaces_invalid_characters():
    assert mlflow_utils.sanitize_mlflow_key("monitoring_drift_Williams_%R") == "monitoring_drift_Williams_R"
    assert mlflow_utils.sanitize_mlflow_key("  weird$key(name) ") == "weird_key_name"


def test_log_json_falls_back_to_local_copy_when_mlflow_artifact_upload_fails(monkeypatch):
    test_root = Path(__file__).resolve().parent / f"_tmp_mlflow_utils_{uuid.uuid4().hex}"
    workspace_tmp_dir = test_root / ".mlflow_tmp"
    fallback_dir = test_root / "mlflow_fallback"
    workspace_tmp_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)

    try:
        monkeypatch.setattr(mlflow_utils, "MLFLOW_AVAILABLE", True)
        monkeypatch.setattr(mlflow_utils, "mlflow", _FailingArtifactMlflow())
        monkeypatch.setattr(mlflow_utils, "_WORKSPACE_TMP_DIR", workspace_tmp_dir)
        monkeypatch.setattr(mlflow_utils, "_ARTIFACT_FALLBACK_DIR", fallback_dir)

        mlflow_utils.log_json({"status": "ok", "value": 7}, filename="training_summary.json", artifact_path="training")

        fallback_file = fallback_dir / "run-123" / "training" / "training_summary.json"
        assert fallback_file.exists()
        assert '"value": 7' in fallback_file.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
