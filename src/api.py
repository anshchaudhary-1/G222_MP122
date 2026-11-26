import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.clean_data import FEATURE_COLUMNS


# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "champion_model.joblib"


# Structured JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        })


logger = logging.getLogger("mp122_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.propagate = False


# Load the existing champion model
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="MP122 Warehouse Picking Time Prediction API",
    description="Predicts warehouse order picking time in seconds.",
    version="1.0.0"
)


# Request validation
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items_in_order: int
    aisle_distance_m: float
    picker_exp_months: float = Field(ge=0, le=150)
    shelf_height_m: float
    shift_hour: int
    order_weight_kg: float = Field(ge=0)
    zone_type: str
    shift: str


@app.get("/health")
def health():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame(
        [request.model_dump()],
        columns=FEATURE_COLUMNS
    )

    prediction = float(model.predict(input_data)[0])

    logger.info("Prediction generated successfully")

    return {
        "prediction": round(prediction, 2)
    }
