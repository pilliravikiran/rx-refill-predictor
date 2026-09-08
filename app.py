"""
STEP 6 - Turn the model into a web service.

Right now the model only works if you run a Python script. A pharmacy system
cannot do that. It needs to send a patient over HTTP and get a score back.

FastAPI does that. It will feel very familiar - it is minimal API in
ASP.NET Core, with Python syntax.

Note the file name: app.py, not 06_app.py. Python cannot import a file whose
name starts with a number, and uvicorn needs to import this one.

Run it:   uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs
"""

import joblib
from fastapi import FastAPI


# --- PIECE 1: load the model once, and say hello --------------------------

app = FastAPI(title="Refill Risk API", version="1.0")

# Load at start up, NOT inside the endpoint.
# Loading takes time. Doing it per request would make every call slow.
bundle  = joblib.load("models/refill_model.pkl")
model   = bundle["model"]
columns = bundle["columns"]        # the exact column order the model expects


@app.get("/health")
def health():
    """Every service needs one of these. It proves the app is alive."""
    return {"status": "ok", "features": len(columns)}


# --- PIECE 2: the real endpoint - send a patient, get a risk score --------

import pandas as pd
from pydantic import BaseModel, Field


class Patient(BaseModel):
    """
    The shape of the incoming JSON.

    This is your C# request DTO. FastAPI reads these types and does three
    jobs from them: it validates the request, it fills in the Swagger page,
    and it gives you a real typed object instead of a loose dictionary.

    Field(ge=..., le=...) means "must be between". Send pct_late = 5 and the
    caller gets a clear 422 error before your code ever runs.
    """
    patient_age: int   = Field(ge=0, le=120)
    drug_name:   str
    is_branded:  bool
    supply_days: int
    fill_number: int
    prev_gap:    float
    was_late:    int   = Field(ge=0, le=1)
    avg_gap:     float
    pct_late:    float = Field(ge=0, le=1)


@app.post("/predict")
def predict(p: Patient):
    # One incoming patient -> a one row table, the shape the model wants.
    row = pd.DataFrame([p.model_dump()])

    # Same two steps as training. They MUST match, or the model reads
    # the wrong number out of the wrong column and quietly gives nonsense.
    row = pd.get_dummies(row, columns=["drug_name"])
    row = row.reindex(columns=columns, fill_value=0).astype(float)

    prob = float(model.predict_proba(row)[0, 1])

    # Do not return a bare number. A pharmacy cannot act on 0.178.
    # Turn it into a decision they can actually follow.
    if   prob >= 0.70: risk, action = "low",    "no action"
    elif prob >= 0.40: risk, action = "medium", "send reminder"
    else:              risk, action = "high",   "pharmacist call"

    return {
        "refill_probability": round(prob, 3),
        "risk": risk,
        "action": action,
    }


# --- PIECE 3: the forecast endpoint - how much should I order? ------------

from datetime import date

forecast_bundle  = joblib.load("models/forecast_model.pkl")
forecast_model   = forecast_bundle["model"]
forecast_columns = forecast_bundle["columns"]


class DrugHistory(BaseModel):
    """
    What the caller sends: one drug, today's date, and the last 29 daily
    totals for that drug - oldest first, the LAST number being today.

    Why 29 and not 9 separate features: the caller should not have to know
    what lag_14 or roll28 mean. They send raw daily numbers, which any
    pharmacy system already has, and we work the features out here.
    """
    drug_id:     int         = Field(ge=1, le=12)
    as_of:       date
    daily_units: list[float] = Field(min_length=29, max_length=29)


@app.post("/forecast")
def forecast(h: DrugHistory):
    d = h.daily_units                      # d[-1] is today, d[0] is 28 days ago

    # Build exactly the same features file 6 trained on.
    row = {
        "units":  d[-1],                   # today
        "lag_1":  d[-2],                   # yesterday
        "lag_7":  d[-8],                   # same day last week
        "lag_14": d[-15],
        "lag_28": d[-29],
        "roll7":  sum(d[-8:-1]) / 7,       # the 7 days BEFORE today
        "roll28": sum(d[-29:-1]) / 28,     # the 28 days BEFORE today
        "dow":    h.as_of.weekday(),
        "month":  h.as_of.month,
    }
    # One hot the drug: a True in its own column, False in the other eleven.
    for i in range(1, 13):
        row[f"drug_id_{i}"] = (i == h.drug_id)

    X = pd.DataFrame([row]).reindex(columns=forecast_columns, fill_value=0).astype(float)

    units = float(forecast_model.predict(X)[0])

    return {
        "drug_id": h.drug_id,
        "as_of": str(h.as_of),
        "forecast_next_7_days": round(units),
        # A buffer, because running out is worse than holding a little extra.
        "suggested_order": round(units * 1.1),
    }
