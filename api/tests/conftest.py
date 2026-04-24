import os
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

local_mlruns_uri = (API_ROOT / "mlruns").resolve().as_uri()
os.environ["MLFLOW_TRACKING_URI"] = local_mlruns_uri
os.environ["MLFLOW_REGISTRY_URI"] = local_mlruns_uri
