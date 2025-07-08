import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

# Paths
data_dir = "./"  # Folder containing the CSV files
output_dir = "./charts"  # Folder to save the plots
os.makedirs(output_dir, exist_ok=True)

# Find CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Store summary stats
average_scores = {}
median_scores = {}

# Color cycle for distinct plots
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# Parameters for plotting
downsample_rate = 10
smoothing_window = 20

# Process each CSV
for file in csv_files:
    filepath = os.path.join(data_dir, file)
    df = pd.read_csv(filepath)

    if 'similarity_score' in df.columns and 'question' in df.columns:
        scores = df['similarity_score'].reset_index(drop=True)
        average_scores[file] = scores.mean()
        median_scores[file] = scores.median()

        color = next(color_cycle)
        safe_filename = file.replace('.csv', '').replace(' ', '_')

        # --- Plot 1: Downsampled raw scores ---
        sampled_scores = scores[::downsample_rate]
        sampled_indices = sampled_scores.index

        plt.figure(figsize=(10, 4))
        plt.plot(sampled_indices, sampled_scores, marker='', linestyle='-', color=color, alpha=0.7, linewidth=1.0, label="Raw (Downsampled)")
        plt.title(f"Similarity Scores - {file}")
        plt.xlabel("Question Index")
        plt.ylabel("Similarity Score")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        raw_plot_path = os.path.join(output_dir, f"{safe_filename}_line_plot_raw.png")
        plt.savefig(raw_plot_path)
        plt.close()

        # --- Plot 2: Smoothed version ---
        smoothed_scores = scores.rolling(smoothing_window).mean()

        plt.figure(figsize=(10, 4))
        plt.plot(smoothed_scores, color=color, alpha=0.9, linewidth=2, label=f"Smoothed (window={smoothing_window})")
        plt.title(f"Smoothed Similarity Scores - {file}")
        plt.xlabel("Question Index")
        plt.ylabel("Similarity Score")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        smoothed_plot_path = os.path.join(output_dir, f"{safe_filename}_line_plot_smoothed.png")
        plt.savefig(smoothed_plot_path)
        plt.close()
    else:
        print(f"Skipping {file}: Missing required columns.")

# --- Bar chart: Average similarity scores ---
plt.figure(figsize=(10, 6))
plt.bar(average_scores.keys(), average_scores.values(), color=plt.cm.tab10.colors[:len(average_scores)])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Similarity Score")
plt.title("Average Similarity Score per File")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_similarity_scores.png"))
plt.close()

# --- Bar chart: Median similarity scores ---
plt.figure(figsize=(10, 6))
plt.bar(median_scores.keys(), median_scores.values(), color=plt.cm.tab10.colors[:len(median_scores)])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Median Similarity Score")
plt.title("Median Similarity Score per File")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "median_similarity_scores.png"))
plt.close()

print(f"All plots saved in: {output_dir}")
