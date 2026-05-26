from __future__ import annotations

from typing import Any

import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
MODEL_URI = "runs:/1da1e7a15bb14dc49852557968c32ff4/model"
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
