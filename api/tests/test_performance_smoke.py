from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd
import pytest

from app.utils import main_utils
from app.utils.tabular_utils import build_dataset_analysis_summary, load_tabular_from_bytes


pytestmark = pytest.mark.perf


def _build_wide_csv(rows: int = 500, columns: int = 80) -> bytes:
    headers = [f"feature_{index:03d}" for index in range(columns)] + ["target"]
    lines = [",".join(headers)]
    for row in range(rows):
        values = [str(row + column) for column in range(columns)] + [str(row % 2)]
        lines.append(",".join(values))
    return "\n".join(lines).encode("utf-8")


def test_wide_upload_parser_stays_within_smoke_budget():
    content = _build_wide_csv()

    started = time.perf_counter()
    dataframe, parser_report = load_tabular_from_bytes(content, filename="wide_columns.csv")
    elapsed = time.perf_counter() - started

    assert dataframe.shape == (500, 81)
    assert parser_report["columns"] == 81
    assert elapsed < 2.5


def test_feature_summary_generation_stays_within_smoke_budget():
    dataframe, parser_report = load_tabular_from_bytes(_build_wide_csv(), filename="wide_columns.csv")

    started = time.perf_counter()
    summary = build_dataset_analysis_summary(dataframe, "wide_columns.csv", parser_report=parser_report)
    elapsed = time.perf_counter() - started

    assert summary["columns"] == 81
    assert summary["column_explorer"]
    assert elapsed < 2.5


def test_training_preflight_on_wide_dataframe_stays_within_smoke_budget(monkeypatch):
    dataframe, _ = load_tabular_from_bytes(_build_wide_csv(), filename="wide_columns.csv")
    dataset_record = SimpleNamespace(id=1, name="wide-dataset")
    model_record = SimpleNamespace(id=2, name="wide-model", input_features=["feature_000", "feature_001"], output_features=["target"])

    monkeypatch.setattr(main_utils, "_load_dataset_frame_from_record", lambda _record: dataframe.copy())

    started = time.perf_counter()
    preflight = main_utils._build_training_preflight(dataset_record, model_record, {})
    elapsed = time.perf_counter() - started

    assert preflight["ok"] is True
    assert preflight["missing_columns"] == []
    assert elapsed < 1.5
