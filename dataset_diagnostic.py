import pandas as pd
import numpy as np

# حط اسم الملف هون
file_path = r"C:\Users\Vtouch\Desktop\ml-project\dataset\final_dataset.csv"

# =========================
# 1) Load dataset
# =========================
df = pd.read_csv(file_path)

print("\n==============================")
print("DATASET BASIC INFO")
print("==============================")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\n==============================")
print("COLUMN NAMES")
print("==============================")
print(df.columns.tolist())

print("\n==============================")
print("DATA TYPES")
print("==============================")
print(df.dtypes)

# =========================
# 2) Missing values
# =========================
print("\n==============================")
print("MISSING VALUES")
print("==============================")
missing_per_col = df.isnull().sum()
print(missing_per_col[missing_per_col > 0])

print("\nTotal missing values in dataset:", df.isnull().sum().sum())

# =========================
# 3) Infinite values
# =========================
print("\n==============================")
print("INFINITE VALUES")
print("==============================")
numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] > 0:
    inf_per_col = np.isinf(numeric_df).sum()
    inf_per_col = pd.Series(inf_per_col, index=numeric_df.columns)
    print(inf_per_col[inf_per_col > 0])
    print("\nTotal infinite values in numeric columns:", np.isinf(numeric_df).sum().sum())
else:
    print("No numeric columns found.")

# =========================
# 4) Detect possible target column
# =========================
print("\n==============================")
print("POSSIBLE TARGET COLUMN")
print("==============================")

possible_targets = ["label", "Label", "class", "Class", "target", "Target", "Attack", "attack"]

found_target = None
for col in possible_targets:
    if col in df.columns:
        found_target = col
        break

if found_target:
    print("Possible target column found:", found_target)
    print("\nClass distribution:")
    print(df[found_target].value_counts(dropna=False))
else:
    print("No obvious target column found.")
    print("You may need to choose it manually.")

# =========================
# 5) Non-numeric columns
# =========================
print("\n==============================")
print("NON-NUMERIC COLUMNS")
print("==============================")
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(non_numeric_cols)

# =========================
# 6) Duplicate rows
# =========================
print("\n==============================")
print("DUPLICATES")
print("==============================")
print("Number of duplicate rows:", df.duplicated().sum())

# =========================
# 7) Summary for ML readiness
# =========================
print("\n==============================")
print("ML READINESS SUMMARY")
print("==============================")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Total missing:", df.isnull().sum().sum())
print("Total duplicates:", df.duplicated().sum())

if numeric_df.shape[1] > 0:
    print("Total infinite:", np.isinf(numeric_df).sum().sum())
else:
    print("Total infinite: could not check, no numeric columns found")

if found_target:
    print("Target column:", found_target)
    print("Number of classes:", df[found_target].nunique())
else:
    print("Target column: not detected")