import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.training import TrainingResult, run_training_pipeline
from app.utils import accelerators


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


def test_resolve_torch_device_falls_back_to_cpu_without_torch(monkeypatch):
    monkeypatch.setattr(accelerators, "_load_torch", lambda: None)

    resolution = accelerators.resolve_torch_device("cuda")

    assert resolution["requested_device"] == "cuda"
    assert resolution["resolved_device"] == "cpu"
    assert resolution["fallback_applied"] is True


def test_resolve_torch_device_selects_cuda_when_available(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(_index):
            return "Test GPU"

        @staticmethod
        def get_device_properties(_index):
            return type("Props", (), {"total_memory": 1024})()

    fake_torch = type(
        "FakeTorch",
        (),
        {
            "__version__": "test",
            "cuda": FakeCuda,
            "version": type("Version", (), {"cuda": "test-cuda"})(),
        },
    )()
    monkeypatch.setattr(accelerators, "_load_torch", lambda: fake_torch)

    resolution = accelerators.resolve_torch_device("cuda")

    assert resolution["resolved_device"] == "cuda"
    assert resolution["devices"][0]["name"] == "Test GPU"
