
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model
bundle = joblib.load("heart_disease_model.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]

app = FastAPI(title="Heart Disease Prediction API")

class PatientData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    if len(data.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features"
        }

    patient = pd.DataFrame(
        [data.features],
        columns=feature_names
    )

    prediction = int(model.predict(patient)[0])

    return {
        "prediction": prediction,
        "result": (
            "Heart disease detected by model"
            if prediction == 1
            else "No heart disease detected by model"
        ),
        "note": "Educational prediction only; not a medical diagnosis."
    }
