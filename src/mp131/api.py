from pathlib import Path
import os


from pathlib import Path
import os
import json
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

# JSON logging configuration
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
        }

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        return json.dumps(log_data)


logger = logging.getLogger("mp131")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.propagate = False

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "models" / "model.joblib")
    )
)

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception:
    logger.exception("Model file missing or could not be loaded")
    raise

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
    start_time = time.perf_counter()

    try:
        features = pd.DataFrame([request.model_dump()])
        prediction = model.predict(features)[0]

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "prediction served",
            extra={"duration_ms": duration_ms}
        )

        return {"prediction": float(prediction)}

    except Exception:
        logger.exception("Prediction failed")
        raise
