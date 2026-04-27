import os
import sys
from pathlib import Path

import pytest


API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = API_ROOT.parent
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

local_mlruns_uri = (API_ROOT / "mlruns").resolve().as_uri()
os.environ["MLFLOW_TRACKING_URI"] = local_mlruns_uri
os.environ["MLFLOW_REGISTRY_URI"] = local_mlruns_uri


@pytest.fixture(scope="session")
def api_root() -> Path:
    return API_ROOT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def full_stack_base_url() -> str:
    return os.environ.get("PAINEL_TEST_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@pytest.fixture(scope="session")
def mlflow_base_url() -> str:
    return os.environ.get("PAINEL_TEST_MLFLOW_URL", "http://127.0.0.1:5000").rstrip("/")
