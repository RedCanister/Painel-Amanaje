from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


PLOT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _plot_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _coerce_numeric_series(series: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for label, value in series:
        if value in (None, ""):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_value):
            continue
        points.append({"label": str(label), "value": numeric_value})
    return points


def _base_spec(
    *,
    kind: str,
    title: str,
    description: Optional[str] = None,
    source: Optional[Mapping[str, Any]] = None,
    seed: Optional[str] = None,
) -> dict[str, Any]:
    plot_id = _plot_id(seed or f"{kind}:{title}:{description or ''}")
    source_payload = _json_safe(dict(source or {}))
    metadata = {
        "display_id": plot_id,
        "kind": kind,
    }
    for key in ("domain", "job_id", "model_id", "dataset_id", "inference_id", "registry_type", "registry_id"):
        value = source_payload.get(key) if isinstance(source_payload, Mapping) else None
        if value not in (None, ""):
            metadata[key] = value
    return {
        "id": plot_id,
        "kind": kind,
        "title": title,
        "description": description,
        "source": source_payload,
        "metadata": metadata,
    }


def build_bar_plot_spec(
    title: str,
    series: list[tuple[str, Any]],
    *,
    description: Optional[str] = None,
    source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    points = _coerce_numeric_series(series)
    spec = _base_spec(kind="plotly_figure", title=title, description=description, source=source)
    spec.update(
        {
            "plot_type": "bar",
            "series": points,
            "figure": {
                "data": [
                    {
                        "type": "bar",
                        "x": [point["label"] for point in points],
                        "y": [point["value"] for point in points],
                    }
                ],
                "layout": {
                    "title": {"text": title},
                    "margin": {"l": 48, "r": 20, "t": 48, "b": 80},
                    "xaxis": {"automargin": True},
                    "yaxis": {"automargin": True},
                    "template": "plotly_white",
                },
                "config": {"displaylogo": False, "responsive": True},
            },
        }
    )
    spec["metadata"]["plot_type"] = "bar"
    return spec


def build_line_plot_spec(
    title: str,
    series: list[tuple[Any, Any]],
    *,
    description: Optional[str] = None,
    source: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    points = _coerce_numeric_series(series)
    spec = _base_spec(kind="plotly_figure", title=title, description=description, source=source)
    spec.update(
        {
            "plot_type": "line",
            "series": points,
            "figure": {
                "data": [
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "x": [point["label"] for point in points],
                        "y": [point["value"] for point in points],
                        "line": {"width": 3},
                        "marker": {"size": 6},
                    }
                ],
                "layout": {
                    "title": {"text": title},
                    "margin": {"l": 48, "r": 20, "t": 48, "b": 56},
                    "xaxis": {"automargin": True},
                    "yaxis": {"automargin": True},
                    "template": "plotly_white",
                },
                "config": {"displaylogo": False, "responsive": True},
            },
        }
    )
    spec["metadata"]["plot_type"] = "line"
    return spec


def build_table_plot_spec(
    title: str,
    rows: list[Mapping[str, Any]],
    *,
    description: Optional[str] = None,
    source: Optional[Mapping[str, Any]] = None,
    max_rows: int = 20,
) -> dict[str, Any]:
    visible_rows = [_json_safe(dict(row)) for row in rows[:max_rows]]
    columns = list(visible_rows[0].keys()) if visible_rows else []
    spec = _base_spec(kind="table", title=title, description=description, source=source)
    spec.update(
        {
            "columns": columns,
            "rows": visible_rows,
            "row_count": len(rows),
            "truncated": len(rows) > len(visible_rows),
        }
    )
    spec["metadata"]["plot_type"] = "table"
    return spec


def build_dataset_plot_specs(
    summary: Mapping[str, Any],
    *,
    dataframe: Optional[pd.DataFrame] = None,
    source: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    profile = dict(summary.get("profile", {}) or {})
    stats = dict(summary.get("stats", {}) or {})

    plots.append(
        build_bar_plot_spec(
            "Column Types",
            [
                ("Numeric", len(profile.get("numeric_columns", []) or [])),
                ("Categorical", len(profile.get("categorical_columns", []) or [])),
                ("Datetime", len(profile.get("datetime_columns", []) or [])),
            ],
            description="Automatic type balance from the extracted dataset profile.",
            source=source,
        )
    )

    missing_series = sorted(
        (
            (column_name, details.get("null_count", 0))
            for column_name, details in stats.items()
            if isinstance(details, Mapping)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    plots.append(
        build_bar_plot_spec(
            "Missing Values by Column",
            missing_series,
            description="Top columns ranked by missing-value count.",
            source=source,
        )
    )

    mean_series = [
        (column_name, details.get("mean"))
        for column_name, details in stats.items()
        if isinstance(details, Mapping) and details.get("type") == "numeric"
    ][:8]
    plots.append(
        build_bar_plot_spec(
            "Numeric Mean Snapshot",
            mean_series,
            description="Quick mean comparison for the first numeric columns in the analysis.",
            source=source,
        )
    )

    if dataframe is not None and not dataframe.empty:
        preview = dataframe.head(20).replace({float("nan"): None}).to_dict(orient="records")
        plots.append(
            build_table_plot_spec(
                "Preview Rows",
                preview,
                description="First rows from the profiled table.",
                source=source,
            )
        )

    return [plot for plot in plots if plot.get("kind") == "table" or plot.get("series")]


def _safe_relative_path(plot_root: Path, file_path: Path) -> str:
    root = plot_root.resolve()
    resolved = file_path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("Plot artifact is outside the configured plot directory.")
    return resolved.relative_to(root).as_posix()


def artifact_id_for_relative_path(relative_path: str) -> str:
    return _plot_id(relative_path.replace("\\", "/"))


def _source_from_relative_path(relative_path: str) -> dict[str, Any]:
    parts = Path(relative_path).parts
    job_id = parts[0] if parts else ""
    source: dict[str, Any] = {"domain": "plot_artifact", "job_id": job_id}
    if job_id.startswith("retrain_"):
        source["domain"] = "retraining"
        pieces = job_id.split("_")
        if len(pieces) >= 2 and pieces[1].isdigit():
            source["model_id"] = int(pieces[1])
    elif job_id.startswith("train_"):
        source["domain"] = "training"
        pieces = job_id.split("_")
        if len(pieces) >= 3 and pieces[1].isdigit() and pieces[2].isdigit():
            source["model_id"] = int(pieces[1])
            source["dataset_id"] = int(pieces[2])
    elif job_id.startswith("training_"):
        source["domain"] = "training"
    elif job_id.startswith("smoke_test"):
        source["domain"] = "smoke_test"
    return source


def legacy_image_plot_spec(file_path: Path, plot_root: Path) -> dict[str, Any]:
    relative_path = _safe_relative_path(plot_root, file_path)
    plot_id = artifact_id_for_relative_path(relative_path)
    filename = file_path.name
    title = filename.replace("_", " ").replace("-", " ").rsplit(".", 1)[0].title()
    source = _source_from_relative_path(relative_path)
    file_stat = file_path.stat()
    modified_at = datetime.fromtimestamp(file_stat.st_mtime, timezone.utc).isoformat()
    spec = _base_spec(
        kind="legacy_image",
        title=title,
        description="Saved runtime plot artifact.",
        source=source,
        seed=relative_path,
    )
    spec["id"] = plot_id
    spec["artifact"] = {
        "url": f"/plots/artifacts/{plot_id}/file",
        "relative_path": relative_path,
        "filename": filename,
        "extension": file_path.suffix.lower(),
        "size_bytes": file_stat.st_size,
        "modified_at": modified_at,
    }
    spec["plot_type"] = file_path.stem
    spec["metadata"].update(
        {
            "plot_type": file_path.stem,
            "filename": filename,
            "relative_path": relative_path,
            "extension": file_path.suffix.lower(),
            "size_bytes": file_stat.st_size,
            "modified_at": modified_at,
        }
    )
    return spec


def list_legacy_plot_artifacts(
    plot_root: Path,
    *,
    job_id: Optional[str] = None,
    model_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    inference_id: Optional[int] = None,
    kind: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    if not plot_root.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    file_paths = sorted(plot_root.rglob("*")) if limit is None else plot_root.rglob("*")
    for file_path in file_paths:
        if not file_path.is_file() or file_path.suffix.lower() not in PLOT_IMAGE_SUFFIXES:
            continue
        try:
            spec = legacy_image_plot_spec(file_path, plot_root)
        except ValueError:
            continue
        source = spec.get("source", {}) or {}
        if job_id and source.get("job_id") != job_id:
            continue
        if model_id is not None and source.get("model_id") != int(model_id):
            continue
        if dataset_id is not None and source.get("dataset_id") != int(dataset_id):
            continue
        if inference_id is not None and source.get("inference_id") != int(inference_id):
            continue
        if kind and kind not in {spec.get("kind"), spec.get("plot_type")}:
            continue
        artifacts.append(spec)
        if limit is not None and len(artifacts) >= limit:
            break
    return artifacts


def legacy_plot_specs_for_job(
    plot_root: Path,
    job_id: str | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not job_id:
        return []
    return list_legacy_plot_artifacts(plot_root, job_id=str(job_id))[:limit]


def resolve_legacy_plot_path(plot_root: Path, plot_id: str) -> Path:
    if not plot_id or any(separator in plot_id for separator in ("/", "\\", "..")):
        raise ValueError("Invalid plot artifact id.")
    for file_path in plot_root.rglob("*") if plot_root.exists() else []:
        if not file_path.is_file() or file_path.suffix.lower() not in PLOT_IMAGE_SUFFIXES:
            continue
        relative_path = _safe_relative_path(plot_root, file_path)
        if artifact_id_for_relative_path(relative_path) == plot_id:
            return file_path.resolve()
    raise FileNotFoundError(f"Plot artifact not found: {plot_id}")
