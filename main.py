import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

model = joblib.load("model/churn_pipeline.joblib")
threshold = json.load(open("model/threshold.json"))["threshold"]


class Customer(BaseModel):
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
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": "Yes" if proba >= threshold else "No",
        "threshold_used": threshold,
    }