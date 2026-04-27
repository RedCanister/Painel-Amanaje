from __future__ import annotations

import pickle
from pathlib import Path
from uuid import uuid4

from app.utils.artifact_utils import inspect_model_artifact, load_runtime_artifact
from app.utils.run_ledger import create_run_entry, list_run_entries, read_run_entry, update_run_entry
from app.utils.serialization import canonicalize_scalar_for_logging
from app.utils.tabular_utils import build_dataset_analysis_summary, load_tabular_from_bytes


class PredictableModel:
    def predict(self, rows):
        return [0 for _ in range(len(rows))]


def _workspace_tmp_dir() -> Path:
    path = Path(__file__).resolve().parent / ".tmp" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_tabular_from_bytes_supports_tsv_and_reports_delimiter():
    content = b"feature_a\tfeature_b\ttarget\n1\talpha\t0\n2\tbeta\t1\n"

    dataframe, parser_report = load_tabular_from_bytes(content, filename="sample.tsv")

    assert list(dataframe.columns) == ["feature_a", "feature_b", "target"]
    assert parser_report["loader"] == "csv"
    assert parser_report["delimiter"] == "\t"
    assert parser_report["columns"] == 3


def test_build_dataset_analysis_summary_exposes_column_explorer_and_profile():
    dataframe, parser_report = load_tabular_from_bytes(
        b'feature_a,feature_b,target\n1,2026-04-23,0\n2,2026-04-24,1\n',
        filename="sample.csv",
    )

    summary = build_dataset_analysis_summary(dataframe, "sample.csv", parser_report=parser_report)

    assert summary["profile"]["recommended_target"] == "target"
    assert summary["column_explorer"]
    assert any(column["name"] == "feature_b" for column in summary["column_explorer"])
    assert "parser_report" in summary


def test_pickle_artifact_manifest_and_runtime_loader_support_predictable_models():
    tmp_path = _workspace_tmp_dir()
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(PredictableModel(), handle)

    manifest = inspect_model_artifact(artifact_path, load_runtime=True)
    loaded = load_runtime_artifact(artifact_path)

    assert manifest["framework"] == "sklearn"
    assert manifest["runtime_capabilities"]["predict"] is True
    assert loaded["framework"] == "sklearn"
    assert hasattr(loaded["model"], "predict")


def test_run_ledger_create_update_and_filter():
    tmp_path = _workspace_tmp_dir()
    entry = create_run_entry(
        tmp_path,
        run_type="training",
        context={"model_id": 7, "dataset_id": 11, "inference_id": 13},
        parameters={"framework": "sklearn"},
    )
    updated = update_run_entry(
        tmp_path,
        entry["run_id"],
        status="running",
        stage="training",
        merge={"metrics": {"eval_accuracy": 0.91}},
        event_message="Training started.",
    )

    assert updated["status"] == "running"
    assert updated["metrics"]["eval_accuracy"] == 0.91
    assert read_run_entry(tmp_path, entry["run_id"])["stage"] == "training"
    assert read_run_entry(tmp_path, entry["run_id"])["progress"]["current_stage"] == "training"
    assert read_run_entry(tmp_path, entry["run_id"])["progress"]["latest_event"] == "Training started."
    assert list_run_entries(tmp_path, model_id=7, dataset_id=11)[0]["run_id"] == entry["run_id"]
    assert list_run_entries(tmp_path, inference_id=13)[0]["run_id"] == entry["run_id"]


def test_canonicalize_scalar_for_logging_normalizes_equivalent_numeric_strings():
    assert canonicalize_scalar_for_logging(0) == "0"
    assert canonicalize_scalar_for_logging(0.0) == "0"
    assert canonicalize_scalar_for_logging("0.0") == "0"
