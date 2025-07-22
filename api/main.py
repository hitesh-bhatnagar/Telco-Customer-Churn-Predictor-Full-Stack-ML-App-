from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model        = joblib.load("models/lgbm_model_tuned.pkl")
scaler       = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")
scaled_cols  = list(scaler.feature_names_in_)

class InputData(BaseModel):
    input: dict

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.input])
    df = pd.get_dummies(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    for col in scaled_cols:
        if col not in df.columns:
            df[col] = 0
    df[scaled_cols] = scaler.transform(df[scaled_cols])

    pred_class = int(model.predict(df)[0])
    pred_proba = float(model.predict_proba(df)[0][1])

    return {
        "prediction": pred_class,
        "probability": round(pred_proba, 4)
    }
