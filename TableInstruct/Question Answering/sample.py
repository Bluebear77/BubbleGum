import os
import pandas as pd
import random

# Set paths
qas_dir = "./QAS"
demo_dir = "./Demo"

# Ensure output directory exists
os.makedirs(demo_dir, exist_ok=True)

# List all CSV files in QAS directory
csv_files = [f for f in os.listdir(qas_dir) if f.endswith(".csv")]

for filename in csv_files:
    file_path = os.path.join(qas_dir, filename)
    
    # Read CSV (with header)
    df = pd.read_csv(file_path)
    
    # Skip if file has fewer than 50 rows
    if len(df) < 50:
        print(f"Skipping {filename}: less than 50 rows.")
        continue

    # Sample 50 rows randomly
    sampled_df = df.sample(n=50, random_state=42)

    # Construct output path
    output_filename = filename.replace(".csv", "-50.csv")
    output_path = os.path.join(demo_dir, output_filename)

    # Write to output CSV (including header)
    sampled_df.to_csv(output_path, index=False)

    print(f"Sampled 50 rows from {filename} -> {output_filename}")
