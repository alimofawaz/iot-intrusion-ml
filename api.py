from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="IoT Intrusion Detection API")

# Load models
model_stage1 = joblib.load("saved_models/model_stage1.pkl")
model_stage2 = joblib.load("saved_models/model_stage2.pkl")
le_family = joblib.load("saved_models/le_family.pkl")

family_models = joblib.load("saved_models/family_models.pkl")
family_label_encoders = joblib.load("saved_models/family_label_encoders.pkl")

feature_columns = joblib.load("saved_models/feature_columns.pkl")

class TrafficRecord(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "IoT IDS API is running"}

@app.post("/predict")
def predict(record: TrafficRecord):
    try:
        # ===== Input to DataFrame =====
        df = pd.DataFrame([record.features])

        # ===== Minimal preprocessing (IMPORTANT) =====
        df = df.reindex(columns=feature_columns, fill_value=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([float("inf"), float("-inf")], 0)
        df = df.fillna(0)

        # ===== Stage 1 =====
        pred_stage1 = model_stage1.predict(df)[0]

        if pred_stage1 == 0:
            return {
                "prediction": "BENIGN",
                "stage1": "BENIGN"
            }

        # ===== Stage 2 =====
        pred_family_enc = model_stage2.predict(df)[0]
        pred_family = le_family.inverse_transform([pred_family_enc])[0]

        # ===== Stage 3 =====
        if pred_family in family_models:
            sub_model = family_models[pred_family]
            sub_le = family_label_encoders[pred_family]

            pred_sub_enc = sub_model.predict(df)[0]
            final_label = sub_le.inverse_transform([pred_sub_enc])[0]
        else:
            final_label = pred_family  # fallback

        return {
            "prediction": final_label,
            "family": pred_family,
            "stage1": "ATTACK"
        }

    except Exception as e:
        return {"error": str(e)}