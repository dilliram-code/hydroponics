from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Hydroponics API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorData(BaseModel):
    pH: float
    TDS: float
    water_level: float
    DHT_temp: float
    DHT_humidity: float
    water_temp: float
    hour: int
@app.get("/")
def welcome():
  return {"message": "Welcome!"}

@app.post("/predict")
def predict_actuators(data: SensorData):
    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # pred = model.predict(df)[0] 
    pred = [1, 0, 1, 0, 1] # Mock prediction: [pH_reducer, add_water, nutrients, humidifier, ex_fan]
    
    return {
        "pH_reducer": "ON" if pred[0] == 1 else "OFF",
        "add_water": "ON" if pred[1] == 1 else "OFF",
        "nutrients_adder": "ON" if pred[2] == 1 else "OFF",
        "humidifier": "ON" if pred[3] == 1 else "OFF",
        "ex_fan": "ON" if pred[4] == 1 else "OFF"
    }

