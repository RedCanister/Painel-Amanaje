"""
utils/io.py

General-purpose I/O utilities for reading and writing files across multiple
formats used by the project.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - only used when pandas is unavailable.
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

PathLike = Union[str, Path]


def _require_pandas() -> None:
    if not PANDAS_AVAILABLE or pd is None:
        raise RuntimeError("pandas is required for CSV I/O helpers.")


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists and return it as a ``Path`` object.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Ensure the parent directory of a file path exists and return the file path.
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_json(
    data: Any,
    path: PathLike,
    indent: int = 4,
    ensure_ascii: bool = False,
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    """
    Save a JSON-serializable object to disk.
    """

    file_path = ensure_parent_dir(path)
    serializer = default or str
    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=indent, ensure_ascii=ensure_ascii, default=serializer)


def load_json(path: PathLike) -> Any:
    """
    Load and return the contents of a JSON file.
    """

    with Path(path).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_yaml(data: Any, path: PathLike) -> None:
    """
    Save a Python object to a YAML file.
    """

    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(data, file_handle, sort_keys=False, allow_unicode=True)


def load_yaml(path: PathLike) -> Any:
    """
    Load and return the contents of a YAML file.
    """

    with Path(path).open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def save_pickle(obj: Any, path: PathLike) -> None:
    """
    Save a Python object to a pickle file.

    Use this only with trusted data sources.
    """

    file_path = ensure_parent_dir(path)
    with file_path.open("wb") as file_handle:
        pickle.dump(obj, file_handle)


def load_pickle(path: PathLike) -> Any:
    """
    Load and return a Python object from a pickle file.
    """

    with Path(path).open("rb") as file_handle:
        return pickle.load(file_handle)


def save_csv(df: pd.DataFrame, path: PathLike, index: bool = False, sep: str = ",") -> None:
    """
    Save a ``pandas.DataFrame`` to CSV.
    """

    _require_pandas()
    file_path = ensure_parent_dir(path)
    df.to_csv(file_path, index=index, sep=sep, encoding="utf-8")


def load_csv(path: PathLike, sep: str = ",", **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Load a CSV file into a ``pandas.DataFrame``.
    """

    _require_pandas()
    return pd.read_csv(Path(path), sep=sep, encoding="utf-8", **read_csv_kwargs)


def save_txt(text: str, path: PathLike) -> None:
    """
    Save plain text to disk.
    """

    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(text)


def load_txt(path: PathLike) -> str:
    """
    Load and return a text file.
    """

    with Path(path).open("r", encoding="utf-8") as file_handle:
        return file_handle.read()


def file_exists(path: PathLike) -> bool:
    """
    Return ``True`` when the given file path exists.
    """

    return Path(path).is_file()


def get_file_size(path: PathLike) -> Optional[int]:
    """
    Return file size in bytes, or ``None`` when the file does not exist.
    """

    file_path = Path(path)
    return file_path.stat().st_size if file_path.exists() else None
