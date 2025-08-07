import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle

# Create an output directory
output_dir = "charts_output"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Read all CSV files in current directory
csv_files = [f for f in glob.glob("*.csv") if f.endswith(".csv")]

# Hold stats and line data
summary_stats = []
line_data = {}

# Step 2: Process files
for file in tqdm(csv_files, desc="Processing CSV files"):
    dataset_name = os.path.splitext(file)[0]

    try:
        df = pd.read_csv(file, quotechar='"', on_bad_lines='skip')
    except Exception as e:
        print(f"⚠️ Failed to read {file}: {e}")
        continue

    if "similarity_score" not in df.columns:
        print(f"⚠️ 'similarity_score' column not found in {file}")
        continue

    scores = df["similarity_score"].dropna()

    if scores.empty:
        print(f"⚠️ No valid similarity scores in {file}")
        continue

    # Add full stats
    summary_stats.append({
        "dataset": dataset_name,
        "mean": scores.mean(),
        "median": scores.median(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max()
    })

    line_data[dataset_name] = scores.reset_index(drop=True)

# Convert to DataFrame
summary_df = pd.DataFrame(summary_stats)

# Save table as CSV
summary_csv_path = os.path.join(output_dir, "similarity_summary_table.csv")
summary_df.to_csv(summary_csv_path, index=False)

# Assign distinct colors using tab20
color_map = plt.cm.get_cmap('tab20', len(summary_df))
dataset_colors = {name: color_map(i) for i, name in enumerate(summary_df["dataset"])}

# Step 3: Bar Chart - Mean + Std
plt.figure(figsize=(12, 6))
bars = plt.bar(
    summary_df["dataset"],
    summary_df["mean"],
    yerr=summary_df["std"],
    capsize=5,
    color=[dataset_colors[name] for name in summary_df["dataset"]]
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Similarity Score")
plt.title("Average Similarity Score per Dataset")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "avg_similarity_bar_chart.png"))

# Step 4: Combined Line Chart
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
plt.xlabel("Question Index")
plt.ylabel("Similarity Score")
plt.title("Similarity Scores per Question (All Datasets)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "similarity_line_chart_combined.png"))

# Step 5: Individual Line Charts
for dataset, scores in line_data.items():
    plt.figure(figsize=(10, 4))
    plt.plot(
        scores,
        linewidth=1.8,
        alpha=0.9,
        color=dataset_colors[dataset]
    )
    plt.title(f"Similarity Scores - {dataset}")
    plt.xlabel("Question Index")
    plt.ylabel("Similarity Score")
    plt.tight_layout()
    filename = f"{dataset}_line_chart.png"
    plt.savefig(os.path.join(output_dir, filename))

# Step 6: Box Plot
plt.figure(figsize=(12, 6))
plt.boxplot(
    [scores for scores in line_data.values()],
    labels=line_data.keys(),
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Similarity Score")
plt.title("Similarity Score Distribution per Dataset (Box Plot)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "similarity_box_plot.png"))

# Step 7: Plot Table as PNG Image
def render_mpl_table(data, output_path, col_width=2.5, row_height=0.6, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0):
    """
    Render a pandas DataFrame to a matplotlib table and save it.
    """
    fig, ax = plt.subplots(figsize=(col_width * len(data.columns), row_height * (len(data) + 1)))
    ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', loc='center')

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig.tight_layout()
    plt.savefig(output_path)

# Save table as PNG
table_img_path = os.path.join(output_dir, "similarity_summary_table.png")
render_mpl_table(summary_df.round(4), output_path=table_img_path)

# ✅ All charts and tables saved to output folder.
