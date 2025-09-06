import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- I/O paths --------------------
input_dir = "."
output_dir = "charts_output"
os.makedirs(output_dir, exist_ok=True)

# Collect CSVs from relatedness_outputs, excluding the global _lowest100.csv
csv_files = [
    f for f in glob.glob(os.path.join(input_dir, "*.csv"))
    if not os.path.basename(f).endswith("_lowest100.csv")
]

summary_stats = []
line_data = {}
all_scores = []  # accumulate all similarity scores for global stats

required_cols = {"similarity_score", "num_pairs", "qas", "table"}

# -------------------- Read & process --------------------
for file in tqdm(csv_files, desc="Processing CSV files"):
    base = os.path.splitext(os.path.basename(file))[0]
    dataset_name = base[:-8] if base.endswith(".grouped") else base

    try:
        df = pd.read_csv(file, quotechar='"', on_bad_lines='skip')
    except Exception as e:
        print(f"⚠️ Failed to read {file}: {e}")
        continue

    if not required_cols.issubset(df.columns):
        print(f"⚠️ Missing required columns in {file}. Found: {list(df.columns)}")
        continue

    scores = df["similarity_score"].dropna()
    if scores.empty:
        print(f"⚠️ No valid similarity scores in {file}")
        continue

    # Save for overall stats
    all_scores.append(scores)

    summary_stats.append({
        "dataset": dataset_name,
        "mean": scores.mean(),
        "median": scores.median(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max(),
        "count": len(scores)
    })

    line_data[dataset_name] = scores.reset_index(drop=True)

if not summary_stats:
    print("⚠️ No valid CSVs found to plot. Check input files in 'relatedness_outputs/'.")
    raise SystemExit(0)

# -------------------- Build summary table --------------------
summary_df = pd.DataFrame(summary_stats).sort_values("dataset")

# Add overall row (across all datasets)
if all_scores:
    combined = pd.concat(all_scores, ignore_index=True)
    overall_row = pd.DataFrame([{
        "dataset": "ALL_FILES",
        "mean": combined.mean(),
        "median": combined.median(),
        "std": combined.std(),
        "min": combined.min(),
        "max": combined.max(),
        "count": len(combined)
    }])
    summary_df = pd.concat([summary_df, overall_row], ignore_index=True)

# Save table as CSV
summary_csv_path = os.path.join(output_dir, "similarity_summary_table.csv")
summary_df.to_csv(summary_csv_path, index=False)

# -------------------- Colors --------------------
color_map = plt.cm.get_cmap('tab20', max(1, len(summary_df) - 1))  # exclude ALL_FILES from color set
dataset_colors = {name: color_map(i % 20) for i, name in enumerate(summary_df["dataset"]) if name != "ALL_FILES"}

# -------------------- Bar Chart: Mean + Std --------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(
    [d for d in summary_df["dataset"] if d != "ALL_FILES"],
    [m for d, m in zip(summary_df["dataset"], summary_df["mean"]) if d != "ALL_FILES"],
    yerr=[s for d, s in zip(summary_df["dataset"], summary_df["std"]) if d != "ALL_FILES"],
    capsize=5,
    color=[dataset_colors[name] for name in summary_df["dataset"] if name != "ALL_FILES"]
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Similarity Score")
plt.title("Average Similarity Score per Dataset")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "avg_similarity_bar_chart.png"))
plt.close()

# -------------------- Combined Line Chart --------------------
plt.figure(figsize=(14, 6))
for dataset, scores in line_data.items():
    plt.plot(
        scores,
        label=dataset,
        linewidth=1.5,
        alpha=0.8,
        color=dataset_colors[dataset]
    )
plt.legend(loc="upper right", fontsize="small")
plt.xlabel("Group Index (table)")
plt.ylabel("Similarity Score")
plt.title("Similarity Scores per Group (All Datasets)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "similarity_line_chart_combined.png"))
plt.close()

# -------------------- Individual Line Charts --------------------
for dataset, scores in line_data.items():
    plt.figure(figsize=(10, 4))
    plt.plot(
        scores,
        linewidth=1.8,
        alpha=0.9,
        color=dataset_colors[dataset]
    )
    plt.title(f"Similarity Scores - {dataset}")
    plt.xlabel("Group Index (table)")
    plt.ylabel("Similarity Score")
    plt.tight_layout()
    filename = f"{dataset}_line_chart.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# -------------------- Box Plot --------------------
plt.figure(figsize=(12, 6))
plt.boxplot(
    [scores for scores in line_data.values()],
    labels=list(line_data.keys()),
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Similarity Score")
plt.title("Similarity Score Distribution per Dataset (Box Plot)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "similarity_box_plot.png"))
plt.close()

# -------------------- Table as PNG --------------------
def render_mpl_table(data, output_path, col_width=2.5, row_height=0.6, font_size=12,
                     header_color='#40466e', row_colors=('#f1f1f2', 'w'), edge_color='w',
                     bbox=(0, 0, 1, 1)):
    fig, ax = plt.subplots(figsize=(col_width * max(1, len(data.columns)),
                                    row_height * (len(data) + 1)))
    ax.axis('off')

    mpl_table = ax.table(
        cellText=data.values,
        bbox=bbox,
        colLabels=list(data.columns),
        cellLoc='center',
        loc='center'
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for (row, col), cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[(row - 1) % len(row_colors)])

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

table_img_path = os.path.join(output_dir, "similarity_summary_table.png")
render_mpl_table(summary_df.round(4), output_path=table_img_path)

print(f"✅ Charts and summary saved to: {output_dir}")
