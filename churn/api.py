"""FastAPI service. Run: uvicorn churn.api:app --reload   (docs at /docs)"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .predict import ModelNotFoundError, load_metadata, load_model, predict_one

YN = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", "tenure": 5,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 480.0}}}
    gender: Literal["Male", "Female"]
    SeniorCitizen: YN
    Partner: YN
    Dependents: YN
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YN
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YN
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(default=None, ge=0, description="Leave null for brand-new customers")


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: Literal["Low", "Medium", "High"]


app = FastAPI(title="Customer Churn Prediction API", version=__version__,
              description="Predicts churn probability. Decision support only: keep humans in the loop.")


@app.get("/health")
def health() -> dict:
    try:
        load_model()
        return {"status": "ok", "model": load_metadata().get("model_name")}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    try:
        return Prediction(**predict_one(customer.model_dump()))
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
