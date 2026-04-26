import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score
)

# =========================
# CONFIG
# =========================
DATA_FILE = r"C:\Users\Vtouch\Desktop\ml-project\dataset\final_dataset.csv"
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"
RANDOM_STATE = 42

# بدّلها بين True / False للمقارنة
USE_BALANCED = True

# للـ Logistic Regression
MAX_ITER = 2000


# =========================
# HELPERS
# =========================
def load_and_clean_for_training(file_path: str, label_column: str):
    print("Loading final dataset...")
    df = pd.read_csv(file_path, low_memory=False)

    print(f"Dataset shape before numeric cleaning: {df.shape}")

    if label_column not in df.columns:
        raise ValueError(f"'{label_column}' column not found in dataset.")

    # Separate features and target
    X = df.drop(columns=[label_column]).copy()
    y = df[label_column].copy()

    # Force all features to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace inf / -inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Keep only fully valid rows
    valid_mask = X.notna().all(axis=1)
    removed_rows = int(len(X) - valid_mask.sum())

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    print(f"Rows removed بسبب inf / NaN / invalid values: {removed_rows}")
    print(f"Dataset shape after numeric cleaning: {(len(X), X.shape[1])}")

    return X, y


def build_model(use_balanced: bool, max_iter: int, random_state: int):
    if use_balanced:
        print("\nModel setting: Logistic Regression WITH class_weight='balanced'")
        return LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced"
        )
    else:
        print("\nModel setting: Logistic Regression WITHOUT class weighting")
        return LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )


def compute_benign_metrics(y_true, y_pred, benign_label: str):
    """
    نحسب:
    - BENIGN recall
    - False Positive Rate on BENIGN

    هون منحوّل المهمة مؤقتًا لـ binary:
    BENIGN vs NOT_BENIGN
    """

    y_true_binary = (y_true == benign_label).astype(int)
    y_pred_binary = (y_pred == benign_label).astype(int)

    # confusion matrix binary with labels [1, 0]
    # 1 = BENIGN, 0 = NOT_BENIGN
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])

    # cm layout:
    # [[TP, FN],
    #  [FP, TN]]
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    benign_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    benign_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return benign_recall, benign_fpr


def evaluate_split(split_name: str, y_true, y_pred, benign_label: str):
    print(f"\n=== {split_name} RESULTS ===")

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    benign_recall, benign_fpr = compute_benign_metrics(y_true, y_pred, benign_label)

    print(f"{split_name} Accuracy                : {accuracy:.4f}")
    print(f"{split_name} Macro F1                : {macro_f1:.4f}")
    print(f"{split_name} BENIGN Recall           : {benign_recall:.4f}")
    print(f"{split_name} BENIGN False Positive Rate: {benign_fpr:.4f}")

    print(f"\n{split_name} Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print(f"{split_name} Confusion Matrix shape  : {cm.shape}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "benign_recall": benign_recall,
        "benign_fpr": benign_fpr,
        "confusion_matrix_shape": cm.shape
    }


# =========================
# MAIN
# =========================
def main():
    # 1) Load + training-specific numeric cleaning
    X, y = load_and_clean_for_training(DATA_FILE, LABEL_COLUMN)

    print(f"Features shape: {X.shape}")
    print(f"Target shape  : {y.shape}")

    # 2) Split 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("\n=== SPLIT SHAPES ===")
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")

    # 3) Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaling done.")

    # 4) Build model
    model = build_model(USE_BALANCED, MAX_ITER, RANDOM_STATE)

    # 5) Train
    print("\nTraining Logistic Regression baseline...")
    model.fit(X_train_scaled, y_train)

    # 6) Validation
    y_val_pred = model.predict(X_val_scaled)
    val_results = evaluate_split("Validation", y_val, y_val_pred, BENIGN_LABEL)

    # 7) Test
    y_test_pred = model.predict(X_test_scaled)
    test_results = evaluate_split("Test", y_test, y_test_pred, BENIGN_LABEL)

    # 8) Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Use balanced weights   : {USE_BALANCED}")
    print(f"Validation Accuracy    : {val_results['accuracy']:.4f}")
    print(f"Validation Macro F1    : {val_results['macro_f1']:.4f}")
    print(f"Validation BENIGN Recall: {val_results['benign_recall']:.4f}")
    print(f"Validation BENIGN FPR  : {val_results['benign_fpr']:.4f}")

    print(f"Test Accuracy          : {test_results['accuracy']:.4f}")
    print(f"Test Macro F1          : {test_results['macro_f1']:.4f}")
    print(f"Test BENIGN Recall     : {test_results['benign_recall']:.4f}")
    print(f"Test BENIGN FPR        : {test_results['benign_fpr']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()