from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from app.utils.plot_registry import (
    build_bar_plot_spec,
    build_dataset_plot_specs,
    build_line_plot_spec,
    list_legacy_plot_artifacts,
    resolve_legacy_plot_path,
)


def test_plot_specs_keep_series_and_add_plotly_figure():
    bar = build_bar_plot_spec("Metrics", [("a", 1), ("bad", "x")])
    line = build_line_plot_spec("History", [(1, 0.5), (2, 0.25)])

    assert bar["kind"] == "plotly_figure"
    assert bar["series"] == [{"label": "a", "value": 1.0}]
    assert bar["figure"]["data"][0]["type"] == "bar"
    assert "marker" not in bar["figure"]["data"][0]
    assert bar["metadata"]["plot_type"] == "bar"
    assert line["figure"]["data"][0]["type"] == "scatter"
    assert "color" not in line["figure"]["data"][0]["line"]
    assert line["metadata"]["plot_type"] == "line"


def test_dataset_plot_specs_include_profile_plots_and_preview_table():
    dataframe = pd.DataFrame({"value": [1.0, 2.0], "target": [0, 1]})
    summary = {
        "profile": {"numeric_columns": ["value", "target"], "categorical_columns": [], "datetime_columns": []},
        "stats": {
            "value": {"type": "numeric", "null_count": 0, "mean": 1.5},
            "target": {"type": "numeric", "null_count": 0, "mean": 0.5},
        },
    }

    plots = build_dataset_plot_specs(summary, dataframe=dataframe)

    assert any(plot["kind"] == "table" and plot["rows"] for plot in plots)
    assert any(plot["title"] == "Numeric Mean Snapshot" and plot["figure"] for plot in plots)


def test_legacy_plot_artifact_index_and_safe_resolution():
    tmp_parent = Path(__file__).resolve().parent / "_tmp_plot_registry"
    plot_root = tmp_parent / uuid.uuid4().hex / "plots"
    try:
        artifact_dir = plot_root / "train_7_9_abc123"
        artifact_dir.mkdir(parents=True)
        artifact_path = artifact_dir / "metric_comparison.png"
        artifact_path.write_bytes(b"not-real-png")

        artifacts = list_legacy_plot_artifacts(plot_root, model_id=7, dataset_id=9)

        assert len(artifacts) == 1
        assert artifacts[0]["kind"] == "legacy_image"
        assert artifacts[0]["source"]["model_id"] == 7
        assert artifacts[0]["source"]["dataset_id"] == 9
        assert artifacts[0]["metadata"]["filename"] == "metric_comparison.png"
        assert artifacts[0]["metadata"]["relative_path"] == "train_7_9_abc123/metric_comparison.png"
        assert artifacts[0]["metadata"]["extension"] == ".png"
        assert artifacts[0]["metadata"]["size_bytes"] == len(b"not-real-png")
        assert artifacts[0]["metadata"]["modified_at"]
        assert resolve_legacy_plot_path(plot_root, artifacts[0]["id"]) == artifact_path.resolve()
        with pytest.raises(ValueError):
            resolve_legacy_plot_path(plot_root, "../metric_comparison.png")
    finally:
        shutil.rmtree(plot_root.parent, ignore_errors=True)
        try:
            tmp_parent.rmdir()
        except OSError:
            pass
