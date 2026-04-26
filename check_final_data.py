import pandas as pd

FILE_PATH = r"C:\Users\Vtouch\Desktop\ml-project\dataset\final_dataset.csv"
LABEL_COLUMN = "Label"


def main():
    print("Loading dataset...")
    df = pd.read_csv(FILE_PATH, low_memory=False)

    print("\n=== BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Columns count: {len(df.columns)}")

    print("\n=== COLUMN NAMES ===")
    print(list(df.columns))

    print("\n=== FIRST 10 COLUMNS ===")
    print(list(df.columns[:10]))

    print("\n=== MISSING VALUES ===")
    missing_total = int(df.isna().sum().sum())
    print(f"Total missing values: {missing_total}")

    missing_by_column = df.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]

    if len(missing_by_column) == 0:
        print("No missing values by column.")
    else:
        print("\nMissing values by column:")
        print(missing_by_column.sort_values(ascending=False))

    print("\n=== DUPLICATES ===")
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_count}")

    unique_count = len(df) - duplicate_count
    print(f"Unique rows: {unique_count}")
    print(f"Duplicate ratio: {duplicate_count / len(df):.4f}")

    print("\n=== LABEL CHECK ===")
    if LABEL_COLUMN in df.columns:
        print(f"Label column found: {LABEL_COLUMN}")
    else:
        print(f"ERROR: Label column '{LABEL_COLUMN}' not found!")
        return

    print("\n=== CLASS DISTRIBUTION (BEFORE DEDUP) ===")
    before_counts = df[LABEL_COLUMN].value_counts()
    print(before_counts)

    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())

    print("\n=== DUPLICATES BY LABEL ===")
    dup_mask = df.duplicated(keep="first")
    dup_df = df[dup_mask].copy()

    if len(dup_df) == 0:
        print("No duplicate rows found.")
    else:
        dup_label_counts = dup_df[LABEL_COLUMN].value_counts()
        print(dup_label_counts)

    print("\n=== CLASS COUNTS AFTER DEDUP ===")
    dedup_df = df.drop_duplicates().copy()
    after_counts = dedup_df[LABEL_COLUMN].value_counts()
    print(after_counts)

    print("\n=== LABEL IMPACT SUMMARY ===")
    labels = sorted(set(before_counts.index).union(set(after_counts.index)))

    summary_rows = []
    for label in labels:
        before = int(before_counts.get(label, 0))
        after = int(after_counts.get(label, 0))
        removed = before - after
        removed_ratio = removed / before if before > 0 else 0

        summary_rows.append({
            "Label": label,
            "Before": before,
            "After": after,
            "Removed": removed,
            "Removed_ratio": round(removed_ratio, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.sort_values("Removed", ascending=False).to_string(index=False))

    print("\n=== TOP REPEATED LABELS ONLY ===")
    if len(dup_df) > 0:
        print(dup_df[LABEL_COLUMN].value_counts().head(15))

    print("\n=== FINAL SHAPES ===")
    print(f"Before dedup: {df.shape}")
    print(f"After dedup : {dedup_df.shape}")

    print("\n=== DONE CHECKING FINAL DATASET ===")


if __name__ == "__main__":
    main()