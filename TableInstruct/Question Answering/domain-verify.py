import os
import csv
import random

# --- CONFIG ---
INPUT_DIR = "TQAS"
OUTPUT_FILE = "domain-verification.csv"

# optional: set a seed for reproducibility
random.seed(42)

# which files get 15 vs 14 rows
FETAQA_FILES = {"fetaqa_train_7325.csv", "fetaqa_test.csv"}
SAMPLE_15 = 15
SAMPLE_14 = 14

def sample_rows_from_csv(file_path, n_samples):
    """
    Reads a CSV, returns a header and n_samples random rows.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header, rows = reader[0], reader[1:]
        if n_samples > len(rows):
            raise ValueError(f"File {file_path} has only {len(rows)} rows, can't sample {n_samples}.")
        sampled = random.sample(rows, n_samples)
    return header, sampled

def main():
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    combined_rows = []
    header_written = False
    header_used = None

    for fname in sorted(all_files):  # sort for consistent order
        full_path = os.path.join(INPUT_DIR, fname)
        n = SAMPLE_15 if fname in FETAQA_FILES else SAMPLE_14
        header, sampled_rows = sample_rows_from_csv(full_path, n)

        # use the header from the first file
        if not header_written:
            header_used = header
            header_written = True

        combined_rows.extend(sampled_rows)

    if len(combined_rows) != 100:
        print(f"⚠️ WARNING: Expected 100 rows, got {len(combined_rows)}")

    # write combined file
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(header_used)
        writer.writerows(combined_rows)

    print(f"✅ domain-verification.csv created with {len(combined_rows)} rows at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
