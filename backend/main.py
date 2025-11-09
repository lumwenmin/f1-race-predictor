from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS from frontend (adjust domains as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_pipeline = joblib.load('models/model_pipeline.pkl')

# Define request body schema
class RaceInput(BaseModel):
    driverId: int
    circuitId: int
    grid: int

@app.get("/")
async def root():
    return {"message": "F1 Race Prediction API"}

@app.get("/race-data")
async def get_race_data():
    drivers = [
        {"driverId": 831, "name": "Max Verstappen"},
        {"driverId": 1, "name": "Lewis Hamilton"},
        {"driverId": 844, "name": "Charles Leclerc"},
        {"driverId": 846, "name": "Lando Norris"},
        {"driverId": 857, "name": "Oscar Piastri"},
        {"driverId": 847, "name": "George Russell"}
    ]
    
    circuits = [
        {"circuitId": 32, "name": "Mexico City Grand Prix"},
        {"circuitId": 18, "name": "São Paulo Grand Prix"},
        {"circuitId": 44, "name": "Las Vegas Grand Prix"},
        {"circuitId": 78, "name": "Qatar Grand Prix"},
        {"circuitId": 24, "name": "Abu Dhabi Grand Prix"}
    ]
    
    return {
        "drivers": drivers,
        "circuits": circuits
    }

@app.post("/predict")
async def predict(data: RaceInput):
    # Prepare input array as DataFrame for consistent column names
    import pandas as pd
    input_df = pd.DataFrame([{
        'driverId': data.driverId,
        'circuitId': data.circuitId,
        'grid': data.grid
    }])

    try:
        prediction = model_pipeline.predict(input_df)[0]
        proba = model_pipeline.predict_proba(input_df)[0][prediction]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return {
        "prediction": int(prediction),
        "confidence": proba,
        "message": "Podium" if prediction == 1 else "Outside Podium",
        "driverId": data.driverId,
        "circuitId": data.circuitId
    }