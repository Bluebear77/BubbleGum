import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr, kendalltau

# Configuration
wiki_folder = 'topic prediction wiki'
bart_folder = 'topic prediction bart'
output_folder = 'topic prediction comparison results'
file_name = 'tabfact-p8-300.csv'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load CSV files
wiki_path = os.path.join(wiki_folder, file_name)
bart_path = os.path.join(bart_folder, file_name)

wiki_df = pd.read_csv(wiki_path)
bart_df = pd.read_csv(bart_path)

# Initialize report content
report = []

# 1. Top 1 Topic Agreement Rate
top1_agreement = (wiki_df['Top 1 Topic'] == bart_df['Top 1 Topic']).mean()
report.append(f"Top 1 Topic Agreement Rate: {top1_agreement:.4f}")

# 2. Top-N Topic Overlap (Top 3 from each file)
def check_overlap(row):
    wiki_topics = {row['Top 1 Topic'], row['Top 2 Topic'], row['Top 3 Topic']}
    bart_topics = {row['Top 1 Topic_bart'], row['Top 2 Topic_bart'], row['Top 3 Topic_bart']}
    return len(wiki_topics & bart_topics) > 0

combined_df = pd.DataFrame({
    'Top 1 Topic': wiki_df['Top 1 Topic'],
    'Top 2 Topic': wiki_df['Top 2 Topic'],
    'Top 3 Topic': wiki_df['Top 3 Topic'],
    'Top 1 Topic_bart': bart_df['Top 1 Topic'],
    'Top 2 Topic_bart': bart_df['Top 2 Topic'],
    'Top 3 Topic_bart': bart_df['Top 3 Topic'],
})

overlap_rate = combined_df.apply(check_overlap, axis=1).mean()
report.append(f"Top 3 Topic Overlap Rate: {overlap_rate:.4f}")

# 3. Score Correlation
pearson_corr, _ = pearsonr(wiki_df['Top 1 Score'], bart_df['Top 1 Score'])
spearman_corr, _ = spearmanr(wiki_df['Top 1 Score'], bart_df['Top 1 Score'])
report.append(f"Pearson Correlation of Top 1 Scores: {pearson_corr:.4f}")
report.append(f"Spearman Correlation of Top 1 Scores: {spearman_corr:.4f}")

# 4. Topic Distribution Comparison
wiki_top1_counts = wiki_df['Top 1 Topic'].value_counts().rename('Wiki Count')
bart_top1_counts = bart_df['Top 1 Topic'].value_counts().rename('Bart Count')
topic_distribution = pd.concat([wiki_top1_counts, bart_top1_counts], axis=1).fillna(0).astype(int)
topic_distribution_path = os.path.join(output_folder, 'topic_distribution.csv')
topic_distribution.to_csv(topic_distribution_path)
report.append(f"Topic distribution saved to: {topic_distribution_path}")

# 5. Disagreement Analysis
disagreements = combined_df[combined_df['Top 1 Topic'] != combined_df['Top 1 Topic_bart']]
disagreements_path = os.path.join(output_folder, 'disagreements.csv')
disagreements.to_csv(disagreements_path, index=False)
report.append(f"Disagreements saved to: {disagreements_path}")

# 6. Ranking Similarity (Kendall’s Tau)
# Create lists of rank orders for each row and calculate Kendall’s Tau
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
