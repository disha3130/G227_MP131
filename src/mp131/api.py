from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "models" / "model.joblib")
    )
)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="MP131 OPD Waiting Time Prediction")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients_ahead: int = Field(ge=0)
    doctors_on_duty: int
    arrival_hour: int
    avg_consult_min: float
    lab_tests_pending: int
    distance_km: float = Field(le=70)
    department: str
    wing: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    features = pd.DataFrame([request.model_dump()])
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}
