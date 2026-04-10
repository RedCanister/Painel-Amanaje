import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.training import TrainingResult, run_training_pipeline


def test_run_training_pipeline_logs_summary_without_name_error(monkeypatch):
    logged = {}

    def fake_log_json(data, filename="data_snapshot.json", artifact_path=None):
        logged["data"] = data
        logged["filename"] = filename
        logged["artifact_path"] = artifact_path

    monkeypatch.setattr("app.utils.training.log_json", fake_log_json)

    result = run_training_pipeline(
        train_fn=lambda: TrainingResult(model=None, framework="pytorch", metrics={"eval_accuracy": 0.9}),
        log_to_mlflow=True,
    )

    assert result.framework == "pytorch"
    assert logged["filename"] == "training_summary.json"
    assert logged["artifact_path"] == "training"
    assert logged["data"]["metrics"]["eval_accuracy"] == 0.9
