import os
import csv
import random

# --- CONFIG ---
INPUT_DIR = "GTQA"
OUTPUT_FILE = "domain-verification.csv"

# optional: set a seed for reproducibility
random.seed(42)

# which files get 15 vs 14 rows
FETAQA_FILES = {"fetaqa_train_7325.csv", "fetaqa_test.csv"}
SAMPLE_15 = 15
SAMPLE_14 = 14

# explicitly skip this file from sampling altogether
SKIP_FILES = {"statistics.csv"}

def read_csv_rows(file_path):
    """Reads a CSV and returns (header, rows list)."""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header, rows = reader[0], reader[1:]
    return header, rows

def main():
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])

    combined_rows = []
    header_used = None

    # We’ll keep leftover pools to top-up if we’re short of 100 rows.
    leftover_pools = []  # list of lists (remaining rows per file)

    # First pass: sample per rule (skip statistics.csv), collect leftovers
    for fname in all_files:
        if fname in SKIP_FILES:
            print(f"⏭️ Skipping {fname} by rule.")
            continue

        full_path = os.path.join(INPUT_DIR, fname)
        header, rows = read_csv_rows(full_path)

        if header_used is None:
            header_used = header

        target_n = SAMPLE_15 if fname in FETAQA_FILES else SAMPLE_14

        # Shuffle deterministically using random.sample to avoid in-place mutation
        shuffled = random.sample(rows, k=len(rows))

        if len(shuffled) >= target_n:
            sampled = shuffled[:target_n]
            leftovers = shuffled[target_n:]
        else:
            # Take what we can; note deficit and try to top-up later from other files
            sampled = shuffled
            leftovers = []  # nothing extra here
            deficit = target_n - len(shuffled)
            print(f"⚠️ {fname} has only {len(shuffled)} rows; short by {deficit} (will try to top-up from others).")

        combined_rows.extend(sampled)
        if leftovers:
            leftover_pools.append(leftovers)

    # Top-up to exactly 100 rows if we’re short and there are leftovers available
    TARGET_TOTAL = 100
    current_total = len(combined_rows)

    if current_total < TARGET_TOTAL:
        needed = TARGET_TOTAL - current_total
        print(f"ℹ️ Topping up with {needed} extra rows from remaining pools.")
        # Flatten leftovers lazily while preserving earlier file order bias
        for pool in leftover_pools:
            if needed == 0:
                break
            take = min(len(pool), needed)
            if take > 0:
                combined_rows.extend(pool[:take])
                needed -= take
        current_total = len(combined_rows)

    # If we somehow exceeded 100 (e.g., configuration changed), trim to 100 for exact output size
    if len(combined_rows) > TARGET_TOTAL:
        print(f"ℹ️ Collected {len(combined_rows)} rows; trimming down to {TARGET_TOTAL}.")
        combined_rows = combined_rows[:TARGET_TOTAL]

    # Final checks & write
    if header_used is None:
        raise RuntimeError("No usable CSV header found. Are there CSVs in the input directory?")

    if len(combined_rows) != TARGET_TOTAL:
        print(f"⚠️ WARNING: Expected {TARGET_TOTAL} rows, got {len(combined_rows)} (not enough leftovers to top-up).")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(header_used)
        writer.writerows(combined_rows)

    print(f"✅ domain-verification.csv created with {len(combined_rows)} rows at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
