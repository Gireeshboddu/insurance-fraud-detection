from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os


model = joblib.load("../models/fraud_model.pkl")


class ClaimData(BaseModel):
    months_as_customer: int
    age: int
    policy_number: int
    policy_deductable: int
    policy_annual_premium: float
    umbrella_limit: float
    insured_zip: int
    capital_gains: float  # mapped from 'capital-gains'
    capital_loss: float   # mapped from 'capital-loss'
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    bodily_injuries: int
    witnesses: int
    total_claim_amount: float
    injury_claim: float
    property_claim: float
    vehicle_claim: float
    auto_year: int
    claim_ratio: float

# Initialize app
app = FastAPI()

@app.post("/predict")
def predict(data: ClaimData):
    input_data = np.array([[  # input feature order
        data.months_as_customer,
        data.age,
        data.policy_number,
        data.policy_deductable,
        data.policy_annual_premium,
        data.umbrella_limit,
        data.insured_zip,
        data.capital_gains,
        data.capital_loss,
        data.incident_hour_of_the_day,
        data.number_of_vehicles_involved,
        data.bodily_injuries,
        data.witnesses,
        data.total_claim_amount,
        data.injury_claim,
        data.property_claim,
        data.vehicle_claim,
        data.auto_year,
        data.claim_ratio
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # as percentage

    # 🔍 Add risk label based on fraud probability
    if probability <= 30:
        risk_label = "✅ SAFE"
    elif probability <= 60:
        risk_label = "⚠️ MANUAL REVIEW"
    else:
        risk_label = "❌ LIKELY FRAUD"

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(probability, 2),
        "risk_label": risk_label
    }
