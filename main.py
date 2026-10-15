
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Project folder ka path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "churn_model.joblib"

# API create karo
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer may churn.",
    version="1.0.0",
)

# Trained ML model load karo
model = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        customer_df = pd.DataFrame([customer.model_dump()])

        prediction = int(model.predict(customer_df)[0])
        probabilities = model.predict_proba(customer_df)[0]

        return {
            "churn_prediction": prediction,
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probabilities[1]), 4)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc
