import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# Configuration
wiki_folder = 'topic prediction wiki'
bart_folder = 'topic prediction bart'
output_folder = 'topic prediction comparison results'
file_name = 'tabfact-p8-300.csv'

os.makedirs(output_folder, exist_ok=True)

wiki_path = os.path.join(wiki_folder, file_name)
bart_path = os.path.join(bart_folder, file_name)

wiki_df = pd.read_csv(wiki_path)
bart_df = pd.read_csv(bart_path)

report = []

# 1. Top 1 Topic Agreement Rate
top1_agreement = (wiki_df['Top 1 Topic'] == bart_df['Top 1 Topic']).mean()
report.append(f"Top 1 Topic Agreement Rate: {top1_agreement:.4f}")

# 2. Top-N Topic Overlap
combined_df = pd.DataFrame({
    'Top 1 Topic_wiki': wiki_df['Top 1 Topic'],
    'Top 2 Topic_wiki': wiki_df['Top 2 Topic'],
    'Top 3 Topic_wiki': wiki_df['Top 3 Topic'],
    'Top 1 Topic_bart': bart_df['Top 1 Topic'],
    'Top 2 Topic_bart': bart_df['Top 2 Topic'],
    'Top 3 Topic_bart': bart_df['Top 3 Topic'],
})

def check_overlap(row):
    wiki_topics = {row['Top 1 Topic_wiki'], row['Top 2 Topic_wiki'], row['Top 3 Topic_wiki']}
    bart_topics = {row['Top 1 Topic_bart'], row['Top 2 Topic_bart'], row['Top 3 Topic_bart']}
    return len(wiki_topics & bart_topics) > 0

overlap_rate = combined_df.apply(check_overlap, axis=1).mean()
report.append(f"Top 3 Topic Overlap Rate: {overlap_rate:.4f}")

# 3. Score Correlation
pearson_corr, _ = pearsonr(wiki_df['Top 1 Score'], bart_df['Top 1 Score'])
spearman_corr, _ = spearmanr(wiki_df['Top 1 Score'], bart_df['Top 1 Score'])
report.append(f"Pearson Correlation of Top 1 Scores: {pearson_corr:.4f}")
report.append(f"Spearman Correlation of Top 1 Scores: {spearman_corr:.4f}")

# 4. Topic Distribution Comparison (Top 1, Top 2, Top 3)
def save_topic_counts(level, output_folder):
    wiki_counts = wiki_df[f'Top {level} Topic'].value_counts().reset_index()
    wiki_counts.columns = [f'Wiki Top {level} Topic', f'Wiki Top {level} Topic Count']

    bart_counts = bart_df[f'Top {level} Topic'].value_counts().reset_index()
    bart_counts.columns = [f'Bart Top {level} Topic', f'Bart Top {level} Topic Count']

    wiki_path = os.path.join(output_folder, f'wiki_top{level}_topic_count.csv')
    bart_path = os.path.join(output_folder, f'bart_top{level}_topic_count.csv')

    wiki_counts.to_csv(wiki_path, index=False)
    bart_counts.to_csv(bart_path, index=False)

    return wiki_counts, bart_counts, wiki_path, bart_path

wiki_top1, bart_top1, path1, path2 = save_topic_counts(1, output_folder)
wiki_top2, bart_top2, _, _ = save_topic_counts(2, output_folder)
wiki_top3, bart_top3, _, _ = save_topic_counts(3, output_folder)

report.append(f"Wiki Top 1 Topic count saved to: {path1}")
report.append(f"Bart Top 1 Topic count saved to: {path2}")

# 5. Disagreement Analysis
disagreements = combined_df[combined_df['Top 1 Topic_wiki'] != combined_df['Top 1 Topic_bart']].copy()
disagreements['Original text'] = wiki_df.loc[disagreements.index, 'Original text']

for n in [1, 2, 3]:
    score_wiki = wiki_df.loc[disagreements.index, f'Top {n} Score'].reset_index(drop=True)
    score_bart = bart_df.loc[disagreements.index, f'Top {n} Score'].reset_index(drop=True)
    disagreements[f'Top {n} Score Difference (Wiki-Bart)'] = score_wiki - score_bart

disagreements_path = os.path.join(output_folder, 'disagreements.csv')
disagreements.to_csv(disagreements_path, index=False)
report.append(f"Disagreements saved to: {disagreements_path}")

# 6. Ranking Similarity (Kendall’s Tau)
def kendall_per_row(wiki_row, bart_row):
    wiki_ranks = pd.Series({
        wiki_row['Top 1 Topic']: 1,
        wiki_row['Top 2 Topic']: 2,
        wiki_row['Top 3 Topic']: 3
    })
    bart_ranks = pd.Series({
        bart_row['Top 1 Topic']: 1,
        bart_row['Top 2 Topic']: 2,
        bart_row['Top 3 Topic']: 3
    })
    combined_topics = list(set(wiki_ranks.index) | set(bart_ranks.index))
    wiki_order = [wiki_ranks.get(topic, 4) for topic in combined_topics]
    bart_order = [bart_ranks.get(topic, 4) for topic in combined_topics]
    return kendalltau(wiki_order, bart_order).correlation

kendall_scores = [
    kendall_per_row(wiki_df.iloc[i], bart_df.iloc[i])
    for i in range(len(wiki_df))
]
average_kendall = pd.Series(kendall_scores).mean()
report.append(f"Average Kendall’s Tau: {average_kendall:.4f}")

# Save report
report_path = os.path.join(output_folder, 'comparison_report.txt')
with open(report_path, 'w') as f:
    for line in report:
        f.write(line + '\n')

print(f"Comparison complete. Report saved to: {report_path}")

# 7. Plotting and Markdown Generation

def plot_topic_comparison(wiki_counts, bart_counts, level, output_folder):
    plt.figure(figsize=(12, 6))
    plt.bar(wiki_counts.iloc[:, 0], wiki_counts.iloc[:, 1], alpha=0.6, label='Wiki', color='blue')
    plt.bar(bart_counts.iloc[:, 0], bart_counts.iloc[:, 1], alpha=0.6, label='Bart', color='orange')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title(f"Top {level} Topic Count Comparison: Wiki vs Bart")
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f'top{level}_topic_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

plot_paths = [
    plot_topic_comparison(wiki_top1, bart_top1, 1, output_folder),
    plot_topic_comparison(wiki_top2, bart_top2, 2, output_folder),
    plot_topic_comparison(wiki_top3, bart_top3, 3, output_folder),
]

# Markdown report generation
markdown_path = os.path.join(output_folder, 'comparison_report.md')
with open(markdown_path, 'w') as md_file:
    md_file.write("# Topic Prediction Comparison Report\n\n")
    for line in report:
        md_file.write(f"- {line}\n")
    for i, plot in enumerate(plot_paths, 1):
        md_file.write(f"\n## Top {i} Topic Count Comparison\n\n")
        md_file.write(f"![Top {i} Topic Count Comparison]({os.path.basename(plot)})\n")

print(f"Markdown report saved to: {markdown_path}")
