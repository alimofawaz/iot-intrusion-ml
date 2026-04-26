import pandas as pd
import numpy as np
import joblib
import os
import glob

from sklearn.metrics import accuracy_score, f1_score, classification_report

# =========================================================
# 1) LOAD SAVED MODELS
# =========================================================

models_dir = r"C:\Users\Vtouch\Desktop\ml-project\saved_modelsv2"

model_stage1 = joblib.load(os.path.join(models_dir, "model_stage1.pkl"))
model_stage2 = joblib.load(os.path.join(models_dir, "model_stage2.pkl"))
le_family = joblib.load(os.path.join(models_dir, "le_family.pkl"))
family_models = joblib.load(os.path.join(models_dir, "family_models.pkl"))
family_label_encoders = joblib.load(os.path.join(models_dir, "family_label_encoders.pkl"))
feature_cols = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))

print("Models loaded successfully.")

# =========================================================
# 2) LOAD REALWORLD DATA
# =========================================================

folder_path = r"C:\Users\Vtouch\Desktop\ml-project\realworlddata"

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))[:3]

if len(csv_files) == 0:
    raise ValueError("No CSV files found.")

print("Using file:", csv_files[0])

df = pd.read_csv(csv_files[0])

# =========================================================
# 3) CLEAN DATA
# =========================================================

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Shape after cleaning:", df.shape)

# =========================================================
# 4) PREPARE FEATURES
# =========================================================

has_labels = "Label" in df.columns

if has_labels:
    y_true = df["Label"]
    X = df.drop(columns=["Label"])
else:
    y_true = None
    X = df.copy()

missing_cols = [col for col in feature_cols if col not in X.columns]
extra_cols = [col for col in X.columns if col not in feature_cols]

print("Missing columns:", missing_cols)
print("Extra columns:", extra_cols)

if missing_cols:
    raise ValueError("Realworld data is missing required columns.")

X = X[feature_cols]

# sample سريع
sample_size = 70000

if len(X) > sample_size:
    sample_idx = X.sample(n=sample_size, random_state=42).index
    X = X.loc[sample_idx]

    if has_labels:
        y_true = y_true.loc[sample_idx]

print("Testing shape:", X.shape)

# =========================================================
# 5) FAST 3-STAGE PREDICTION
# =========================================================

preds = np.empty(len(X), dtype=object)

# Stage 1: BENIGN vs ATTACK
stage1 = model_stage1.predict(X)

benign_idx = np.where(stage1 == 0)[0]
attack_idx = np.where(stage1 == 1)[0]

preds[benign_idx] = "BENIGN"

# Stage 2 + Stage 3
if len(attack_idx) > 0:
    X_attack = X.iloc[attack_idx]

    stage2 = model_stage2.predict(X_attack)
    families = le_family.inverse_transform(stage2)

    for fam in np.unique(families):
        fam_mask = families == fam
        idx = attack_idx[fam_mask]

        X_fam = X.iloc[idx]

        if fam in family_models:
            sub_model = family_models[fam]
            sub_le = family_label_encoders[fam]

            sub_preds = sub_model.predict(X_fam)
            preds[idx] = sub_le.inverse_transform(sub_preds)
        else:
            preds[idx] = "UNKNOWN"

# =========================================================
# 6) RESULTS
# =========================================================

# =========================================================
# 6) RESULTS
# =========================================================

print("\nFirst 20 predictions:")
print(preds[:20])

if has_labels:
    print("\n===== REALWORLD SUBCLASS RESULTS =====")
    print("Subclass Accuracy:", accuracy_score(y_true, preds))
    print("Subclass Macro F1:", f1_score(y_true, preds, average="macro"))
    print("\nSubclass Classification Report:")
    print(classification_report(y_true, preds, zero_division=0))

    # =====================================================
    # FAMILY-LEVEL EVALUATION
    # =====================================================

    def map_family(label):
        if label == "BENIGN":
            return "BENIGN"
        elif label.startswith("DDOS"):
            return "DDOS"
        elif label.startswith("DOS"):
            return "DOS"
        elif label.startswith("RECON"):
            return "RECON"
        elif label.startswith("MIRAI"):
            return "MIRAI"
        elif label in [
            "SQLINJECTION", "XSS", "COMMANDINJECTION",
            "BROWSERHIJACKING", "BACKDOOR_MALWARE",
            "DICTIONARYBRUTEFORCE", "UPLOADING_ATTACK"
        ]:
            return "WEB_MISC"
        elif label in ["MITM-ARPSPOOFING", "DNS_SPOOFING", "VULNERABILITYSCAN"]:
            return "SPOOF_SCAN"
        else:
            return "OTHER"

    y_true_family = y_true.apply(map_family)
    y_pred_family = pd.Series(preds, index=y_true.index).apply(map_family)

    print("\n===== REALWORLD FAMILY-LEVEL RESULTS =====")
    print("Family Accuracy:", accuracy_score(y_true_family, y_pred_family))
    print("Family Macro F1:", f1_score(y_true_family, y_pred_family, average="macro"))
    print("\nFamily Classification Report:")
    print(classification_report(y_true_family, y_pred_family, zero_division=0))

# =========================================================
# 7) SAVE OUTPUT
# =========================================================

output_path = r"C:\Users\Vtouch\Desktop\ml-project\realworlddata\predictions.csv"

df_out = X.copy()
df_out["Predicted_Label"] = preds

if has_labels:
    df_out["True_Label"] = y_true.values

df_out.to_csv(output_path, index=False)

print("\nSaved to:", output_path)