from pathlib import Path
import pandas as pd

DATASET_FOLDER = r"C:\Users\Vtouch\Desktop\ml-project\dataset"
MAX_SAMPLE_ROWS = 70000

POSSIBLE_LABEL_COLUMNS = [
    "label", "Label", "class", "Class", "attack", "Attack",
    "attack_type", "Attack_Type", "category", "Category"
]

def detect_label_column(columns):
    for c in POSSIBLE_LABEL_COLUMNS:
        if c in columns:
            return c
    return None

def try_read_csv(file_path: Path):
    try:
        df = pd.read_csv(file_path, low_memory=False, nrows=MAX_SAMPLE_ROWS)
        return df, "utf-8", None
    except Exception as e1:
        try:
            df = pd.read_csv(file_path, low_memory=False, nrows=MAX_SAMPLE_ROWS, encoding="latin1")
            return df, "latin1", None
        except Exception as e2:
            return None, None, f"{e1} | fallback failed: {e2}"

def main():
    dataset_path = Path(DATASET_FOLDER)
    files = sorted(dataset_path.glob("*.csv"))

    if not files:
        print("No CSV files found.")
        return

    print(f"CSV files found: {len(files)}")
    print("=" * 90)

    reference_columns = None
    reference_file = None

    for idx, file_path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] {file_path.name}")

        df, encoding_used, error = try_read_csv(file_path)

        if error:
            print(f"   Status          : FAILED")
            print(f"   Error           : {error}")
            continue

        print(f"   Status          : OK")
        print(f"   Encoding used   : {encoding_used}")
        print(f"   Sample rows     : {len(df)}")
        print(f"   Columns count   : {len(df.columns)}")
        print(f"   First 10 cols   : {list(df.columns[:10])}")

        missing_total = int(df.isna().sum().sum())
        print(f"   Missing values  : {missing_total}")

        try:
            duplicates = int(df.duplicated().sum())
            print(f"   Duplicates      : {duplicates}")
        except Exception:
            print("   Duplicates      : could not compute")

        label_col = detect_label_column(df.columns)
        print(f"   Label column    : {label_col}")

        if label_col is not None:
            print("   Top 10 classes:")
            print(df[label_col].astype(str).value_counts(dropna=False).head(10))

        current_columns = list(df.columns)

        if reference_columns is None:
            reference_columns = current_columns
            reference_file = file_path.name
            print(f"   Schema check    : Reference schema set from {reference_file}")
        else:
            if current_columns == reference_columns:
                print(f"   Schema check    : SAME as {reference_file}")
            else:
                print(f"   Schema check    : DIFFERENT from {reference_file}")

                missing_in_current = [c for c in reference_columns if c not in current_columns]
                extra_in_current = [c for c in current_columns if c not in reference_columns]

                print(f"   Missing cols    : {missing_in_current}")
                print(f"   Extra cols      : {extra_in_current}")

        print("-" * 90)

if __name__ == "__main__":
    main()