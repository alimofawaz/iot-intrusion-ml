import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# =========================================================
# 1) Load + Clean
# =========================================================
file_path = r"C:\Users\Vtouch\Desktop\ml-project\dataset\final_dataset.csv"

df = pd.read_csv(file_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

print("Shape after removing inf/null:", df.shape)

# =========================================================
# 2) Family mapping
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
    elif label in ["SQLINJECTION","XSS","COMMANDINJECTION","BROWSERHIJACKING","BACKDOOR_MALWARE","DICTIONARYBRUTEFORCE","UPLOADING_ATTACK"]:
        return "WEB_MISC"
    elif label in ["MITM-ARPSPOOFING","DNS_SPOOFING","VULNERABILITYSCAN"]:
        return "SPOOF_SCAN"
    else:
        return "OTHER"

df["Family"] = df["Label"].apply(map_family)

# =========================================================
# 3) Features
# =========================================================
X = df.drop(columns=["Label", "Family"])

# =========================================================
# 4) STAGE 1
# =========================================================
y_stage1 = df["Label"].apply(lambda x: 0 if x=="BENIGN" else 1)

X_train_1, X_temp_1, y_train_1, y_temp_1 = train_test_split(
    X, y_stage1, test_size=0.30, stratify=y_stage1, random_state=42
)

X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(
    X_temp_1, y_temp_1, test_size=0.50, stratify=y_temp_1, random_state=42
)

model_stage1 = XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, random_state=42
)

model_stage1.fit(X_train_1, y_train_1)

print("\nSTAGE 1 F1:", f1_score(y_test_1, model_stage1.predict(X_test_1)))

# =========================================================
# 5) STAGE 2
# =========================================================
df_attack = df[df["Label"] != "BENIGN"]

X_attack = df_attack.drop(columns=["Label", "Family"])
y_family = df_attack["Family"]

le_family = LabelEncoder()
y_family_enc = le_family.fit_transform(y_family)

X_train_2, X_temp_2, y_train_2, y_temp_2 = train_test_split(
    X_attack, y_family_enc, test_size=0.30, stratify=y_family_enc, random_state=42
)

X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(
    X_temp_2, y_temp_2, test_size=0.50, stratify=y_temp_2, random_state=42
)

model_stage2 = XGBClassifier(
    objective="multi:softmax",
    num_class=len(le_family.classes_),
    n_estimators=150,
    max_depth=8,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model_stage2.fit(X_train_2, y_train_2)
# 🔥 Overfitting Check
train_pred_2 = model_stage2.predict(X_train_2)
test_pred_2 = model_stage2.predict(X_test_2)

print("\n===== OVERFITTING CHECK (STAGE 2) =====")
print("Train F1:", f1_score(y_train_2, train_pred_2, average="macro"))
print("Test F1:", f1_score(y_test_2, test_pred_2, average="macro"))
print("STAGE 2 Macro F1:", f1_score(y_test_2, model_stage2.predict(X_test_2), average="macro"))

# =========================================================
# 6) STAGE 3 (🔥 V2 تحسين)
# =========================================================
family_models = {}
family_label_encoders = {}

for family in sorted(df_attack["Family"].unique()):

    df_family = df_attack[df_attack["Family"] == family]

    if df_family["Label"].nunique() < 2:
        continue

    X_fam = df_family.drop(columns=["Label","Family"])
    y_fam = df_family["Label"]

    le_sub = LabelEncoder()
    y_fam_enc = le_sub.fit_transform(y_fam)

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_fam, y_fam_enc, test_size=0.30, stratify=y_fam_enc, random_state=42
    )

    # 🔥 التعديل الجديد
    if family in ["WEB_MISC", "RECON"]:
        model_fam = XGBClassifier(
            objective="multi:softmax",
            num_class=len(le_sub.classes_),
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=2.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )
    else:
        model_fam = XGBClassifier(
            objective="multi:softmax",
            num_class=len(le_sub.classes_),
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )

    model_fam.fit(X_train_f, y_train_f)

    preds = model_fam.predict(X_test_f)

    print(f"{family} F1:", f1_score(y_test_f, preds, average="macro"))

    family_models[family] = model_fam
    family_label_encoders[family] = le_sub

# =========================================================
# 7) FULL PIPELINE (⚡ FAST)
# =========================================================

X_train_g, X_temp_g, y_train_g, y_temp_g = train_test_split(
    X, df["Label"], test_size=0.30, stratify=df["Label"], random_state=42
)

X_val_g, X_test_g, y_val_g, y_test_g = train_test_split(
    X_temp_g, y_temp_g, test_size=0.50, stratify=y_temp_g, random_state=42
)

# 🔥 sample بدل dataset كامل
sample_size = 5000
X_test_g = X_test_g.sample(n=sample_size, random_state=42)
y_test_g = y_test_g.loc[X_test_g.index]

print("Pipeline sample:", X_test_g.shape)

preds = np.empty(len(X_test_g), dtype=object)

# Stage1 batch
s1 = model_stage1.predict(X_test_g)

benign = np.where(s1==0)[0]
attack = np.where(s1==1)[0]

preds[benign] = "BENIGN"

# Stage2 + Stage3
if len(attack) > 0:
    X_attack_part = X_test_g.iloc[attack]

    s2 = model_stage2.predict(X_attack_part)
    families = le_family.inverse_transform(s2)

    for fam in np.unique(families):
        idx = attack[families == fam]
        X_f = X_test_g.iloc[idx]

        if fam in family_models:
            sub_model = family_models[fam]
            sub_le = family_label_encoders[fam]

            sub_preds = sub_model.predict(X_f)
            preds[idx] = sub_le.inverse_transform(sub_preds)
        else:
            preds[idx] = "UNKNOWN"

print("\nFINAL RESULTS")
print("Accuracy:", accuracy_score(y_test_g, preds))
print("Macro F1:", f1_score(y_test_g, preds, average="macro"))

# =========================================================
# 8) SAVE
# =========================================================

models_dir = r"C:\Users\Vtouch\Desktop\ml-project\saved_modelsv2"
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model_stage1, os.path.join(models_dir, "model_stage1.pkl"))
joblib.dump(model_stage2, os.path.join(models_dir, "model_stage2.pkl"))
joblib.dump(le_family, os.path.join(models_dir, "le_family.pkl"))
joblib.dump(family_models, os.path.join(models_dir, "family_models.pkl"))
joblib.dump(family_label_encoders, os.path.join(models_dir, "family_label_encoders.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))

print("\nModels saved.")