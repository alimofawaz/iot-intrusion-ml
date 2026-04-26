import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =========================================================
# 1) Load + Clean
# =========================================================
file_path = r"C:\Users\Vtouch\Desktop\ml-project\dataset\final_dataset.csv"

df = pd.read_csv(file_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

print("Shape after removing inf/null:", df.shape)

# =========================================================
# 2) Family mapping function
# =========================================================
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
    elif label in ["SQLINJECTION", "XSS", "COMMANDINJECTION", "BROWSERHIJACKING", "BACKDOOR_MALWARE", "DICTIONARYBRUTEFORCE", "UPLOADING_ATTACK"]:
        return "WEB_MISC"
    elif label in ["MITM-ARPSPOOFING", "DNS_SPOOFING", "VULNERABILITYSCAN"]:
        return "SPOOF_SCAN"
    else:
        return "OTHER"

df["Family"] = df["Label"].apply(map_family)

print("\nFamily distribution:")
print(df["Family"].value_counts())

# =========================================================
# 3) Features
# =========================================================
X = df.drop(columns=["Label", "Family"])

# =========================================================
# 4) STAGE 1: BENIGN vs ATTACK
# =========================================================
y_stage1 = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

X_train_1, X_temp_1, y_train_1, y_temp_1 = train_test_split(
    X, y_stage1,
    test_size=0.30,
    random_state=42,
    stratify=y_stage1
)

X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(
    X_temp_1, y_temp_1,
    test_size=0.50,
    random_state=42,
    stratify=y_temp_1
)

model_stage1 = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model_stage1.fit(X_train_1, y_train_1)

val_pred_1 = model_stage1.predict(X_val_1)
test_pred_1 = model_stage1.predict(X_test_1)

print("\n==============================")
print("STAGE 1: BENIGN vs ATTACK")
print("==============================")
print("Validation Accuracy:", accuracy_score(y_val_1, val_pred_1))
print("Validation F1:", f1_score(y_val_1, val_pred_1))
print("Test Accuracy:", accuracy_score(y_test_1, test_pred_1))
print("Test F1:", f1_score(y_test_1, test_pred_1))

# =========================================================
# 5) STAGE 2: ATTACK FAMILY CLASSIFICATION
# =========================================================
df_attack = df[df["Label"] != "BENIGN"].copy()

X_attack = df_attack.drop(columns=["Label", "Family"])
y_family = df_attack["Family"]

le_family = LabelEncoder()
y_family_enc = le_family.fit_transform(y_family)

X_train_2, X_temp_2, y_train_2, y_temp_2 = train_test_split(
    X_attack, y_family_enc,
    test_size=0.30,
    random_state=42,
    stratify=y_family_enc
)

X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(
    X_temp_2, y_temp_2,
    test_size=0.50,
    random_state=42,
    stratify=y_temp_2
)

model_stage2 = XGBClassifier(
    objective="multi:softmax",
    num_class=len(le_family.classes_),
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model_stage2.fit(X_train_2, y_train_2)

val_pred_2 = model_stage2.predict(X_val_2)
test_pred_2 = model_stage2.predict(X_test_2)

print("\n==============================")
print("STAGE 2: ATTACK FAMILY")
print("==============================")
print("Families:", list(le_family.classes_))
print("Validation Accuracy:", accuracy_score(y_val_2, val_pred_2))
print("Validation Macro F1:", f1_score(y_val_2, val_pred_2, average="macro"))
print("Test Accuracy:", accuracy_score(y_test_2, test_pred_2))
print("Test Macro F1:", f1_score(y_test_2, test_pred_2, average="macro"))

# =========================================================
# 6) STAGE 3: SUBCLASS MODEL FOR EACH FAMILY
# =========================================================
family_models = {}
family_label_encoders = {}

print("\n==============================")
print("STAGE 3: SUBCLASS PER FAMILY")
print("==============================")

for family in sorted(df_attack["Family"].unique()):
    df_family = df_attack[df_attack["Family"] == family].copy()

    # إذا family فيها class وحدة بس، ما في داعي model
    if df_family["Label"].nunique() < 2:
        print(f"\nFamily {family}: skipped (only one subclass)")
        continue

    X_fam = df_family.drop(columns=["Label", "Family"])
    y_fam = df_family["Label"]

    le_sub = LabelEncoder()
    y_fam_enc = le_sub.fit_transform(y_fam)

    X_train_f, X_temp_f, y_train_f, y_temp_f = train_test_split(
        X_fam, y_fam_enc,
        test_size=0.30,
        random_state=42,
        stratify=y_fam_enc
    )

    X_val_f, X_test_f, y_val_f, y_test_f = train_test_split(
        X_temp_f, y_temp_f,
        test_size=0.50,
        random_state=42,
        stratify=y_temp_f
    )

    model_fam = XGBClassifier(
        objective="multi:softmax",
        num_class=len(le_sub.classes_),
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model_fam.fit(X_train_f, y_train_f)

    val_pred_f = model_fam.predict(X_val_f)
    test_pred_f = model_fam.predict(X_test_f)

    print(f"\nFamily: {family}")
    print("Subclasses:", list(le_sub.classes_))
    print("Validation Accuracy:", accuracy_score(y_val_f, val_pred_f))
    print("Validation Macro F1:", f1_score(y_val_f, val_pred_f, average="macro"))
    print("Test Accuracy:", accuracy_score(y_test_f, test_pred_f))
    print("Test Macro F1:", f1_score(y_test_f, test_pred_f, average="macro"))

    family_models[family] = model_fam
    family_label_encoders[family] = le_sub

# =========================================================
# 7) FULL PIPELINE PREDICTION ON TEST SET
#    stage1 -> stage2 -> stage3
# =========================================================
print("\n==============================")
print("FULL 3-STAGE PIPELINE ON GLOBAL TEST SET")
print("==============================")

# global split for full-label evaluation
y_full = df["Label"]
le_full = LabelEncoder()
y_full_enc = le_full.fit_transform(y_full)

X_train_g, X_temp_g, y_train_g, y_temp_g = train_test_split(
    X, y_full,
    test_size=0.30,
    random_state=42,
    stratify=y_full
)

X_val_g, X_test_g, y_val_g, y_test_g = train_test_split(
    X_temp_g, y_temp_g,
    test_size=0.50,
    random_state=42,
    stratify=y_temp_g
)

final_preds = []

for _, row in X_test_g.iterrows():
    row_df = pd.DataFrame([row])

    # Stage 1
    pred_stage1 = model_stage1.predict(row_df)[0]

    if pred_stage1 == 0:
        final_preds.append("BENIGN")
        continue

    # Stage 2
    pred_family_enc = model_stage2.predict(row_df)[0]
    pred_family = le_family.inverse_transform([pred_family_enc])[0]

    # Stage 3
    if pred_family in family_models:
        sub_model = family_models[pred_family]
        sub_le = family_label_encoders[pred_family]
        pred_sub_enc = sub_model.predict(row_df)[0]
        pred_label = sub_le.inverse_transform([pred_sub_enc])[0]
        final_preds.append(pred_label)
    else:
        # إذا ما في model لهالعيلة
        family_rows = df_attack[df_attack["Family"] == pred_family]
        majority_label = family_rows["Label"].value_counts().idxmax()
        final_preds.append(majority_label)

print("Pipeline Test Accuracy:", accuracy_score(y_test_g, final_preds))
print("Pipeline Test Macro F1:", f1_score(y_test_g, final_preds, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test_g, final_preds, zero_division=0))

# =========================================================
# 8) SAVE TRAINED 3-STAGE MODELS
# =========================================================

import joblib
import os

models_dir = r"C:\Users\Vtouch\Desktop\ml-project\saved_models"
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model_stage1, os.path.join(models_dir, "model_stage1.pkl"))
joblib.dump(model_stage2, os.path.join(models_dir, "model_stage2.pkl"))
joblib.dump(le_family, os.path.join(models_dir, "le_family.pkl"))
joblib.dump(family_models, os.path.join(models_dir, "family_models.pkl"))
joblib.dump(family_label_encoders, os.path.join(models_dir, "family_label_encoders.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))

print("\nModels saved successfully in:")
print(models_dir)
