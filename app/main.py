
from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# Import the same cleaning function used during training
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mp131.cleaning import clean_features, FEATURE_COLUMNS

# Load trained model
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="MP131 - OPD Waiting Time Prediction API",
    description="Predicts OPD waiting time in minutes.",
    version="1.0.0",
)


# Input validation
class PredictionInput(BaseModel):
    patients_ahead: int = Field(ge=0)
    doctors_on_duty: int
    arrival_hour: int = Field(ge=0, le=23)
    avg_consult_min: float
    lab_tests_pending: int
    distance_km: float = Field(ge=0, le=70)
    department: str
    wing: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestRegressor",
        "model_file": "models/model.joblib",
        "target": "wait_min",
        "unit": "minutes",
        "features": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    # Convert input into a DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Apply the same feature cleaning used during training
    cleaned_df = clean_features(input_df)

    # Generate prediction
    prediction = model.predict(cleaned_df)[0]

    return {
        "prediction": round(float(prediction), 2),
        "unit": "minutes",
    }
