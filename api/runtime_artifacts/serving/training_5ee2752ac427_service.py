from __future__ import annotations

from typing import Any

import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
MODEL_URI = "runs:/4abcb8aec29d43a0935ad82b5d887b9a/model"
model = mlflow.pyfunc.load_model(MODEL_URI)


class PredictRequest(BaseModel):
    inputs: list[Any]


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    dataframe = pd.DataFrame(request.inputs)
    predictions = model.predict(dataframe)
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    return {"predictions": predictions}
