from pathlib import Path
import pandas as pd

DATASET_FOLDER = r"C:\Users\Vtouch\Desktop\ml-project\dataset"

def count_rows_csv(file_path):
    # minus 1 because first line is header
    return sum(1 for _ in open(file_path, "r", encoding="utf-8", errors="ignore")) - 1

def main():
    dataset_path = Path(DATASET_FOLDER)
    csv_files = sorted(dataset_path.glob("*.csv"))

    total_rows = 0

    for file_path in csv_files:
        try:
            rows = count_rows_csv(file_path)
            total_rows += rows
            print(f"{file_path.name}: {rows} rows")
        except Exception as e:
            print(f"{file_path.name}: ERROR -> {e}")

    print("-" * 50)
    print(f"Total rows in all files: {total_rows}")

if __name__ == "__main__":
    main()