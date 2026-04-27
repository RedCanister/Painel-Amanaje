from __future__ import annotations

from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = TEST_ROOT / "fixtures"
DATASET_FIXTURE_ROOT = FIXTURE_ROOT / "datasets"
EDITOR_FIXTURE_ROOT = FIXTURE_ROOT / "editor"


def dataset_fixture(name: str) -> Path:
    path = DATASET_FIXTURE_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset fixture not found: {path}")
    return path


def editor_fixture(name: str) -> Path:
    path = EDITOR_FIXTURE_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Editor fixture not found: {path}")
    return path
