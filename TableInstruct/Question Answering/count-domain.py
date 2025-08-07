import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
INPUT_FOLDER = './question_domain_type'
OUTPUT_COUNT_FILE = 'domain_label_counts_pivot.csv'
OUTPUT_PLOT_FILE = 'domain_label_distribution.png'

summary_rows = []
dataset_totals = {}

# Step 1: Process input files
for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith('.csv'):
        continue

    dataset_name = os.path.splitext(filename)[0]
    file_path = os.path.join(INPUT_FOLDER, filename)

    try:
        df = pd.read_csv(file_path)

        if 'Top 1 Topic' not in df.columns:
            print(f"⚠️ Skipping {filename} — missing 'Top 1 Topic' column.")
            continue

        value_counts = df['Top 1 Topic'].value_counts()
        total = value_counts.sum()
        dataset_totals[dataset_name] = total

        for label, count in value_counts.items():
            summary_rows.append({
                'Dataset': dataset_name,
                'Domain Label': label,
                'Count': count
            })

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# Step 2: Create DataFrame
summary_df = pd.DataFrame(summary_rows)

# Step 3: Pivot for wide-format count table
count_pivot = summary_df.pivot(index='Dataset', columns='Domain Label', values='Count').fillna(0).astype(int)

# Step 4: Add Total row with frequencies
count_total_row = count_pivot.sum()
total_count = count_total_row.sum()
frequency_row = (count_total_row / total_count).round(4)
count_total_row.name = 'Total'

# Convert count to float just for the Total row (to allow frequency decimals)
count_pivot = pd.concat([count_pivot, count_total_row.to_frame().T.astype(float)])

# Replace Total row with frequencies in parentheses
for col in count_pivot.columns:
    count = int(count_total_row[col])
    freq = frequency_row[col]
    count_pivot.at['Total', col] = f"{count} ({freq})"

# Step 5: Save final CSV
count_pivot.to_csv(OUTPUT_COUNT_FILE)
print(f"✅ Saved: {OUTPUT_COUNT_FILE}")

# Step 6: Plot bar chart for frequencies from Total row
# Extract numeric frequencies
plot_data = frequency_row.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
bars = plt.bar(plot_data.index, plot_data.values)

# Apply distinct colors
for bar in bars:
    bar.set_color(plt.cm.tab20c(bars.index(bar) / len(bars)))

plt.title("Domain Label Distribution (Total Frequency)")
plt.xlabel("Domain Label")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
print(f"📊 Bar chart saved to: {OUTPUT_PLOT_FILE}")
plt.close()

