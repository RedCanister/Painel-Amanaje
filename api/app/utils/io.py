"""
utils/io.py

General-purpose I/O utilities for reading and writing files
acrpss multiple formats (YAML, JSON, CSV, Pickle, TXT).

These helpers standardize data persistence across modules like:
- Airflow task I/O (XComs, configs, intermediate files)
- MLflow model and config logging
- FastAPI configuration loading
- Feast and OPtuna experiment snapshots
"""

import os
import json
import yaml
import pickle
import pandas as pd
from typing import Any, Dict, Optional, Union


# JSON

def save_json(data: Any, path: str, indent = 4, ensure_ascii: bool = False) -> None:
    """
    Saves any Python-serializable object as a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_json(path:str) -> Any:
    """
    Loads a JSON file and returns its contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# YAML

def save_yaml(data: Dict[str, Any], path: str) -> None:
    """
    Saves a Python dictionary to a YAML file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def load_yaml(path: str) -> Dict[str, Any]:
    """
    Loads a YAML file and returns a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

# Pickle

def save_pickle(obj: Any, path: str) -> None:
    """
    Saves a Python object to a Pickle file.
    Use only for trusted objects (not secure against code injection).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> any:
    """
    Loads a Pickle file and returns the Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# CSV

def save_csv(df: pd.DataFrame, path: str, index: bool = False, sep: str = ",") -> None:
    """
    Saves Pandas DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index, sep=sep, encoding="utf-8")

def load_csv(path: str, sep: str = ",") -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.
    """
    return pd.read_csv(path, sep=sep, encoding="utf-8")


# TXT

def save_txt(text: str, path: str) -> None:
    """
    Saves a string as a plan text file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_txt(path: str) -> str:
    """
    Loads text from a plain text file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    

# Utiility helpers

def file_exists(path: str) -> bool:
    """
    Checks if a file exists.
    """
    return os.path.isfile(path)

def ensure_dir(path: str) -> None:
    """
    Ensure directory exists.
    """
    os.makedirs(path, exist_ok=True)

def get_file_size(path: str) -> Optional[int]:
    """
    Return file size is bytes if exists.
    """
    return os.path.getsize(path) if os.path.exists(path) else None