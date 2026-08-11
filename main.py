import logging

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hydroponics-api")

app = FastAPI(title="Smart Hydroponics API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "hydroponics_model.pkl"

# The order the pipeline was trained on. Keep this explicit so a future change
# to the SensorData field order can never silently reorder the model's input.
FEATURE_ORDER = [
    "pH",
    "TDS",
    "water_level",
    "DHT_temp",
    "DHT_humidity",
    "water_temp",
    "hour",
    "minute",
]

# Output order must match what the model was trained to predict.
ACTUATOR_ORDER = [
    "pH_reducer",
    "add_water",
    "nutrients_adder",
    "humidifier",
    "ex_fan",
]

# Load the saved pipeline once at startup. If it fails, don't crash the whole
# app on import -- keep the server up so /  still responds, and fail loudly
# and clearly on /predict instead.
model_pipeline = None
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except FileNotFoundError:
    logger.error("Model file '%s' not found. /predict will return 503 until it's added.", MODEL_PATH)
except Exception:
    logger.exception("Failed to load model from %s. /predict will return 503.", MODEL_PATH)


class SensorData(BaseModel):
    pH: float = Field(..., ge=0, le=14, description="Reservoir pH")
    TDS: float = Field(..., ge=0, description="Total dissolved solids (ppm)")
    water_level: float = Field(..., ge=0, le=100, description="Reservoir water level (%)")
    DHT_temp: float = Field(..., description="Ambient air temperature (°C)")
    DHT_humidity: float = Field(..., ge=0, le=100, description="Ambient air humidity (%)")
    water_temp: float = Field(..., description="Reservoir water temperature (°C)")
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)


@app.get("/")
def welcome():
    return {
        "message": "Welcome!",
        "model_loaded": model_pipeline is not None,
    }


@app.get("/health")
def health():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok"}


@app.post("/predict")
def predict_actuators(data: SensorData):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. Make sure '{MODEL_PATH}' exists next to the server.",
        )

    # Convert incoming JSON to a single-row DataFrame in the exact column
    # order the pipeline expects.
    df = pd.DataFrame([data.model_dump()])[FEATURE_ORDER]

    try:
        raw_pred = model_pipeline.predict(df)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # raw_pred is typically shape (1, 5) for a multi-output classifier, or
    # occasionally a flat (5,) array depending on how the pipeline was built.
    # Normalize both cases down to a flat sequence of 5 values.
    pred = np.asarray(raw_pred)
    if pred.ndim == 2:
        pred = pred[0]

    if pred.shape[0] != len(ACTUATOR_ORDER):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model returned {pred.shape[0]} outputs, "
                f"expected {len(ACTUATOR_ORDER)} ({', '.join(ACTUATOR_ORDER)})."
            ),
        )

    # int() to strip numpy scalar types before they hit JSON serialization.
    return {
        name: "ON" if int(value) == 1 else "OFF"
        for name, value in zip(ACTUATOR_ORDER, pred)
    }