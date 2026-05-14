from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Hashable
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


DATASET_CAPABILITY_MATRIX: dict[str, dict[str, Any]] = {
    ".csv": {
        "label": "CSV",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["UTF-8, UTF-8-SIG, cp1252, and latin-1 text decoding are attempted automatically."],
    },
    ".tsv": {
        "label": "TSV",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["Tab-delimited text is parsed with delimiter sniffing and low-memory safeguards."],
    },
    ".txt": {
        "label": "Delimited Text",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["Comma, tab, semicolon, and pipe delimiters are auto-detected when possible."],
    },
    ".json": {
        "label": "JSON",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["List-of-records, dict-of-lists, and single-record payloads are normalized into tables."],
    },
    ".jsonl": {
        "label": "JSON Lines",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["Each non-empty line must contain one JSON object."],
    },
    ".parquet": {
        "label": "Parquet",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["Requires a parquet engine such as pyarrow or fastparquet in the runtime environment."],
    },
    ".xlsx": {
        "label": "Excel",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["Reads the first worksheet by default. Requires an Excel engine such as openpyxl."],
    },
    ".xls": {
        "label": "Excel",
        "register": True,
        "inspect": True,
        "train": True,
        "predict": True,
        "simulate": True,
        "monitor": True,
        "notes": ["Reads the first worksheet by default. Legacy .xls support depends on the available engine."],
    },
}

_TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".jsonl"}
_DATETIME_SAMPLE_SIZE = 50
_SAMPLE_VALUE_COUNT = 5


def get_dataset_capability_matrix() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in DATASET_CAPABILITY_MATRIX.items()}


def _decode_text_bytes(content: bytes) -> tuple[str, str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode uploaded text content: {last_error}")


def _sniff_delimiter(text: str, extension: str) -> str:
    if extension == ".tsv":
        return "\t"

    sample = "\n".join(line for line in text.splitlines()[:10] if line.strip())[:4096]
    if not sample:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return str(dialect.delimiter)
    except csv.Error:
        pass

    counts = {delimiter: sample.count(delimiter) for delimiter in [",", "\t", ";", "|"]}
    return max(counts.items(), key=lambda item: item[1])[0]


def _load_json_records(text: str, extension: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if extension == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL line {line_number} is not an object.")
            records.append(payload)
        dataframe = pd.json_normalize(records)
        return dataframe, {"loader": "jsonl", "line_count": len(records)}

    payload = json.loads(text)
    if isinstance(payload, list):
        dataframe = pd.json_normalize(payload)
        return dataframe, {"loader": "json", "record_count": len(payload)}
    if isinstance(payload, Mapping):
        if any(isinstance(value, list) for value in payload.values()):
            return pd.DataFrame(payload), {"loader": "json", "record_count": None}
        return pd.json_normalize([payload]), {"loader": "json", "record_count": 1}
    raise ValueError("JSON datasets must be a list of records, a dict of columns, or a single object.")


def _load_csv_like(text: str, extension: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_content = text.strip()
    if normalized_content in {"df.to_csv(index=False)", "csv_text = df.to_csv(index=False)"}:
        raise ValueError(
            "The uploaded dataset is a Python expression, not CSV data. "
            "Run the editor code first so `csv_text` contains materialized tabular content."
        )

    delimiter = _sniff_delimiter(text, extension)
    dataframe = pd.read_csv(StringIO(text), sep=delimiter, header=0, low_memory=False)
    suspicious_single_expression = list(dataframe.columns) == ["df.to_csv(index=False)"] and dataframe.empty
    if suspicious_single_expression:
        raise ValueError(
            "The uploaded dataset still looks like editor source code instead of CSV output. "
            "Execute the editor script before creating the dataset."
        )
    return dataframe, {"loader": "csv", "delimiter": delimiter}


def load_tabular_from_bytes(
    content: bytes,
    *,
    filename: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    extension = Path(filename or "").suffix.lower()
    if extension not in DATASET_CAPABILITY_MATRIX:
        supported = ", ".join(sorted(DATASET_CAPABILITY_MATRIX))
        raise ValueError(f"Unsupported dataset file type: {extension or '[no extension]'}. Supported types: {supported}")

    warnings_list: list[str] = []
    metadata: dict[str, Any] = {
        "file_name": filename,
        "file_type": extension,
        "supported_file_types": sorted(DATASET_CAPABILITY_MATRIX),
        "capability_matrix": get_dataset_capability_matrix(),
        "warnings": warnings_list,
    }

    if extension in _TEXT_EXTENSIONS:
        text, encoding = _decode_text_bytes(content)
        metadata["encoding"] = encoding
        if extension in {".json", ".jsonl"}:
            dataframe, loader_meta = _load_json_records(text, extension)
        else:
            dataframe, loader_meta = _load_csv_like(text, extension)
        metadata.update(loader_meta)
    elif extension == ".parquet":
        try:
            dataframe = pd.read_parquet(BytesIO(content))
        except ImportError as exc:
            raise ValueError(f"Parquet support requires an installed parquet engine: {exc}") from exc
        metadata["loader"] = "parquet"
    elif extension in {".xlsx", ".xls"}:
        try:
            dataframe = pd.read_excel(BytesIO(content))
        except ImportError as exc:
            raise ValueError(f"Excel support requires an installed engine such as openpyxl: {exc}") from exc
        metadata["loader"] = "excel"
    else:
        raise ValueError(f"Unsupported dataset file type: {extension}")

    dataframe.columns = [str(column) for column in dataframe.columns]
    metadata["rows"] = int(dataframe.shape[0])
    metadata["columns"] = int(dataframe.shape[1])

    if dataframe.shape[1] > 150:
        warnings_list.append(
            f"The dataset contains {dataframe.shape[1]} columns. Use the column explorer to work with wide tables."
        )

    mixed_columns = []
    for column_name in dataframe.columns:
        column = dataframe[column_name].dropna()
        if column.empty:
            continue
        sample_types = {type(item).__name__ for item in column.head(_DATETIME_SAMPLE_SIZE).tolist()}
        if len(sample_types) > 1:
            mixed_columns.append({"name": str(column_name), "types": sorted(sample_types)})
    metadata["mixed_type_columns"] = mixed_columns
    if mixed_columns:
        warnings_list.append(f"{len(mixed_columns)} column(s) contain mixed Python value types.")

    return dataframe, metadata


def load_tabular_from_path(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    file_path = Path(path)
    return load_tabular_from_bytes(file_path.read_bytes(), filename=file_path.name)


def _safe_datetime_confidence(series: pd.Series) -> float:
    values = series.dropna().astype(str).head(_DATETIME_SAMPLE_SIZE)
    if values.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
        except TypeError:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return float(parsed.notna().mean())


def _column_family(series: pd.Series, datetime_confidence: float) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if datetime_confidence >= 0.8:
        return "datetime_candidate"
    return "categorical"


def _safe_profile_value(value: Any) -> Any:
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return [_safe_profile_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _safe_profile_value(item) for key, item in value.items()}
    if isinstance(value, Hashable):
        return value
    return str(value)


def _safe_hashable_value(value: Any) -> Any:
    normalized = _safe_profile_value(value)
    if isinstance(normalized, (list, dict)):
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    if isinstance(normalized, set):
        return json.dumps(sorted(normalized), ensure_ascii=False, default=str)
    if isinstance(normalized, Hashable):
        return normalized
    return str(normalized)


def _safe_unique_count(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        normalized = series.dropna().map(_safe_hashable_value)
        return int(normalized.nunique(dropna=True))


def _safe_mode_value(series: pd.Series) -> Any:
    try:
        mode = series.mode(dropna=True)
        return None if mode.empty else _safe_profile_value(mode.iloc[0])
    except Exception:
        normalized = series.dropna().map(_safe_hashable_value)
        mode = normalized.mode(dropna=True)
        return None if mode.empty else mode.iloc[0]


def build_column_explorer(df: pd.DataFrame) -> list[dict[str, Any]]:
    explorer: list[dict[str, Any]] = []
    total_rows = max(int(df.shape[0]), 1)

    for column_name in df.columns:
        series = df[column_name]
        null_count = int(series.isna().sum())
        unique_count = _safe_unique_count(series)
        datetime_confidence = _safe_datetime_confidence(series) if series.dtype == object else 0.0
        family = _column_family(series, datetime_confidence)
        sample_types = sorted({type(item).__name__ for item in series.dropna().head(_DATETIME_SAMPLE_SIZE).tolist()})
        high_cardinality = unique_count >= min(max(total_rows // 2, 20), total_rows)
        explorer.append(
            {
                "name": str(column_name),
                "dtype": str(series.dtype),
                "family": family,
                "null_count": null_count,
                "null_ratio": round(null_count / total_rows, 6),
                "unique_count": unique_count,
                "unique_ratio": round(unique_count / total_rows, 6),
                "mixed_type": len(sample_types) > 1,
                "sample_types": sample_types,
                "high_cardinality": bool(high_cardinality),
                "datetime_confidence": round(datetime_confidence, 4),
                "sample_values": [_safe_profile_value(item) for item in series.head(_SAMPLE_VALUE_COUNT).tolist()],
            }
        )

    explorer.sort(key=lambda item: (item["null_count"], item["unique_count"]), reverse=True)
    return explorer


def dataset_profile(df: pd.DataFrame, column_explorer: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    column_explorer = column_explorer or build_column_explorer(df)
    numeric_columns = [item["name"] for item in column_explorer if item["family"] == "numeric"]
    categorical_columns = [item["name"] for item in column_explorer if item["family"] == "categorical"]
    datetime_columns = [
        item["name"]
        for item in column_explorer
        if item["family"] in {"datetime", "datetime_candidate"}
    ]
    target = str(df.columns[-1]) if len(df.columns) else ""
    recommended_inputs = [str(column) for column in df.columns if str(column) != target]
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "recommended_target": target,
        "recommended_inputs": recommended_inputs,
    }


def build_dataset_analysis_summary(
    df: pd.DataFrame,
    df_path: str,
    *,
    parser_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    column_explorer = build_column_explorer(df)
    profile = dataset_profile(df, column_explorer=column_explorer)
    stats: dict[str, dict[str, Any]] = {}
    top_warnings: list[str] = list(parser_report.get("warnings", []) if parser_report else [])

    for column_name in df.columns:
        column = df[column_name]
        explorer_entry = next((item for item in column_explorer if item["name"] == str(column_name)), None) or {}
        summary = {
            "type": explorer_entry.get("family", "categorical"),
            "null_count": int(column.isnull().sum()),
            "unique_values": _safe_unique_count(column),
            "sample_values": explorer_entry.get("sample_values", []),
            "mixed_type": bool(explorer_entry.get("mixed_type", False)),
            "datetime_confidence": explorer_entry.get("datetime_confidence", 0.0),
        }
        if pd.api.types.is_numeric_dtype(column):
            summary.update(
                {
                    "mean": None if column.empty else float(column.mean()),
                    "std": None if column.empty else float(column.std()),
                    "min": None if column.empty else float(column.min()),
                    "max": None if column.empty else float(column.max()),
                }
            )
        else:
            summary["mode"] = _safe_mode_value(column)
        if summary["mixed_type"]:
            top_warnings.append(f"Column '{column_name}' mixes multiple observed Python value types.")
        if explorer_entry.get("high_cardinality"):
            top_warnings.append(f"Column '{column_name}' has high cardinality and may need filtering or encoding.")
        stats[str(column_name)] = summary

    unique_warnings: list[str] = []
    seen_warnings: set[str] = set()
    for warning_message in top_warnings:
        if warning_message in seen_warnings:
            continue
        seen_warnings.add(warning_message)
        unique_warnings.append(warning_message)

    type_groups = {
        "numeric": [item["name"] for item in column_explorer if item["family"] == "numeric"],
        "categorical": [item["name"] for item in column_explorer if item["family"] == "categorical"],
        "datetime": [item["name"] for item in column_explorer if item["family"] in {"datetime", "datetime_candidate"}],
    }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": int(df.isnull().sum().sum()),
        "stats": stats,
        "path": df_path,
        "profile": profile,
        "parser_report": dict(parser_report or {}),
        "column_explorer": column_explorer,
        "type_groups": type_groups,
        "top_warnings": unique_warnings[:24],
    }


def build_feature_extraction_summary(df: pd.DataFrame) -> dict[str, Any]:
    profile = dataset_profile(df)
    return {
        "feature_candidates": profile["recommended_inputs"],
        "target_candidate": profile["recommended_target"],
        "numeric_feature_count": len(profile["numeric_columns"]),
        "categorical_feature_count": len(profile["categorical_columns"]),
        "datetime_feature_count": len(profile["datetime_columns"]),
    }


def apply_feature_operations(
    dataframe: pd.DataFrame,
    operations: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    operations = dict(operations or {})
    frame = dataframe.copy()
    applied_steps: list[dict[str, Any]] = []

    rename_columns = operations.get("rename_columns") or {}
    if isinstance(rename_columns, Mapping) and rename_columns:
        frame = frame.rename(columns={str(key): str(value) for key, value in rename_columns.items()})
        applied_steps.append({"operation": "rename_columns", "count": len(rename_columns)})

    drop_columns = operations.get("drop_columns") or []
    if isinstance(drop_columns, list) and drop_columns:
        usable_drop_columns = [str(column) for column in drop_columns if str(column) in frame.columns]
        if usable_drop_columns:
            frame = frame.drop(columns=usable_drop_columns)
            applied_steps.append({"operation": "drop_columns", "columns": usable_drop_columns})

    cast_types = operations.get("cast_types") or {}
    if isinstance(cast_types, Mapping):
        for column_name, dtype_name in cast_types.items():
            column_name = str(column_name)
            if column_name not in frame.columns:
                continue
            normalized_dtype = str(dtype_name).strip().lower()
            if normalized_dtype in {"int", "integer"}:
                frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce").astype("Int64")
            elif normalized_dtype in {"float", "double", "number"}:
                frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
            elif normalized_dtype in {"datetime", "timestamp"}:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    frame[column_name] = pd.to_datetime(frame[column_name], errors="coerce", utc=True)
            elif normalized_dtype in {"string", "text"}:
                frame[column_name] = frame[column_name].astype("string")
            elif normalized_dtype in {"bool", "boolean"}:
                frame[column_name] = frame[column_name].astype("boolean")
            applied_steps.append({"operation": "cast_type", "column": column_name, "dtype": normalized_dtype})

    fill_missing = operations.get("fill_missing_values") or {}
    if isinstance(fill_missing, Mapping):
        for column_name, strategy in fill_missing.items():
            column_name = str(column_name)
            if column_name not in frame.columns:
                continue
            if isinstance(strategy, Mapping):
                mode = str(strategy.get("strategy") or "").lower()
                explicit_value = strategy.get("value")
            else:
                mode = ""
                explicit_value = strategy
            if mode == "median":
                value = frame[column_name].median(numeric_only=False)
            elif mode == "mean":
                value = frame[column_name].mean(numeric_only=False)
            elif mode == "mode":
                mode_series = frame[column_name].mode(dropna=True)
                value = None if mode_series.empty else mode_series.iloc[0]
            else:
                value = explicit_value
            frame[column_name] = frame[column_name].fillna(value)
            applied_steps.append({"operation": "fill_missing", "column": column_name})

    math_operations = operations.get("math_operations") or []
    if isinstance(math_operations, list):
        for operation in math_operations:
            if not isinstance(operation, Mapping):
                continue
            column_name = str(operation.get("column") or "").strip()
            target_column = str(operation.get("target") or column_name).strip() or column_name
            operator = str(operation.get("operator") or "").strip().lower()
            if column_name not in frame.columns or not operator:
                continue

            series = pd.to_numeric(frame[column_name], errors="coerce")
            raw_value = operation.get("value")
            numeric_value = None
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = None

            result_series = None
            if operator == "add" and numeric_value is not None:
                result_series = series + numeric_value
            elif operator == "subtract" and numeric_value is not None:
                result_series = series - numeric_value
            elif operator == "multiply" and numeric_value is not None:
                result_series = series * numeric_value
            elif operator == "divide" and numeric_value not in (None, 0):
                result_series = series / numeric_value
            elif operator == "power" and numeric_value is not None:
                result_series = series.pow(numeric_value)
            elif operator == "round":
                decimals = int(numeric_value) if numeric_value is not None else 0
                result_series = series.round(decimals)
            elif operator == "clip_min" and numeric_value is not None:
                result_series = series.clip(lower=numeric_value)
            elif operator == "clip_max" and numeric_value is not None:
                result_series = series.clip(upper=numeric_value)
            elif operator == "abs":
                result_series = series.abs()
            elif operator == "log1p":
                result_series = np.log1p(series.clip(lower=0))

            if result_series is None:
                continue

            if isinstance(result_series, pd.Series):
                result_series = result_series.replace([np.inf, -np.inf], np.nan)
            frame[target_column] = result_series
            applied_steps.append(
                {
                    "operation": "math_operation",
                    "column": column_name,
                    "target": target_column,
                    "operator": operator,
                    "value": raw_value,
                }
            )

    filters = operations.get("filters") or operations.get("filter_rows") or []
    if isinstance(filters, list):
        for condition in filters:
            if not isinstance(condition, Mapping):
                continue
            column_name = str(condition.get("column") or "")
            operator = str(condition.get("operator") or "eq").lower()
            value = condition.get("value")
            if column_name not in frame.columns:
                continue
            if operator == "eq":
                frame = frame[frame[column_name] == value]
            elif operator == "neq":
                frame = frame[frame[column_name] != value]
            elif operator == "gt":
                frame = frame[frame[column_name] > value]
            elif operator == "gte":
                frame = frame[frame[column_name] >= value]
            elif operator == "lt":
                frame = frame[frame[column_name] < value]
            elif operator == "lte":
                frame = frame[frame[column_name] <= value]
            elif operator == "contains":
                frame = frame[frame[column_name].astype(str).str.contains(str(value), case=False, na=False)]
            applied_steps.append({"operation": "filter_rows", "column": column_name, "operator": operator})

    sort_by = operations.get("sort_by") or {}
    if isinstance(sort_by, Mapping) and sort_by.get("column") in frame.columns:
        ascending = bool(sort_by.get("ascending", True))
        frame = frame.sort_values(by=str(sort_by["column"]), ascending=ascending)
        applied_steps.append({"operation": "sort_by", "column": str(sort_by["column"]), "ascending": ascending})

    limit_rows = operations.get("limit_rows")
    if limit_rows not in (None, "", 0, "0"):
        try:
            row_limit = max(1, int(limit_rows))
            frame = frame.head(row_limit)
            applied_steps.append({"operation": "limit_rows", "rows": row_limit})
        except (TypeError, ValueError):
            pass

    summary = {
        "shape_before": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "shape_after": [int(frame.shape[0]), int(frame.shape[1])],
        "steps": applied_steps,
        "preview_rows": frame.head(20).replace({np.nan: None}).to_dict(orient="records"),
        "column_explorer": build_column_explorer(frame),
    }
    return frame, summary
