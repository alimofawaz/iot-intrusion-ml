from pathlib import Path
import pandas as pd

DATASET_FOLDER = r"C:\Users\Vtouch\Desktop\ml-project\dataset"
SAMPLE_ROWS_PER_FILE = 70000
RANDOM_STATE = 42
LABEL_COLUMN = "Label"


def stratified_sample(df, label_col, n, random_state=42):
    if len(df) <= n:
        return df.copy()

    sampled_parts = []
    total_rows = len(df)

    for label_value, group in df.groupby(label_col):
        proportion = len(group) / total_rows
        n_samples = max(1, int(round(proportion * n)))

        if n_samples > len(group):
            n_samples = len(group)

        sampled_group = group.sample(n=n_samples, random_state=random_state)
        sampled_parts.append(sampled_group)

    result = pd.concat(sampled_parts, ignore_index=True)

    if len(result) > n:
        result = result.sample(n=n, random_state=random_state)
    elif len(result) < n:
        remaining = df.drop(result.index, errors="ignore")
        needed = n - len(result)
        if len(remaining) >= needed:
            extra = remaining.sample(n=needed, random_state=random_state)
            result = pd.concat([result, extra], ignore_index=True)

    return result


def process_file(file_path):
    print(f"\nProcessing: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"   Original rows      : {len(df)}")

    # cleaning
    df = df.dropna()
    print(f"   After dropna       : {len(df)}")

    df = df.drop_duplicates()
    print(f"   After duplicates   : {len(df)}")

    # sample
    df = stratified_sample(df, LABEL_COLUMN, SAMPLE_ROWS_PER_FILE, RANDOM_STATE)
    print(f"   Final sampled rows : {len(df)}")

    return df


def main():
    dataset_path = Path(DATASET_FOLDER)
    csv_files = sorted(dataset_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    all_parts = []

    for file_path in csv_files:
        try:
            cleaned_sample = process_file(file_path)
            all_parts.append(cleaned_sample)
        except Exception as e:
            print(f"Error in {file_path.name}: {e}")

    if not all_parts:
        print("No data was processed successfully.")
        return

    print("\nMerging all cleaned samples...")
    final_df = pd.concat(all_parts, ignore_index=True)

    print(f"Rows before removing final duplicates: {len(final_df)}")
    final_df = final_df.drop_duplicates()
    print(f"Rows after removing final duplicates : {len(final_df)}")

    print(f"Rows before shuffle   : {len(final_df)}")
    final_df = final_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Rows after shuffle    : {len(final_df)}")

    output_file = dataset_path / "final_dataset.csv"
    final_df.to_csv(output_file, index=False)

    print(f"\nSaved final dataset to: {output_file}")
    print(f"Final shape           : {final_df.shape}")


if __name__ == "__main__":
    main()