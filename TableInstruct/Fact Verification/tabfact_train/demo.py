import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
df = pd.read_csv('demo.csv')

# Ensure column names are stripped of spaces
df.columns = [col.strip() for col in df.columns]

# Create output directory
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# PIE CHART: Distribution of predicted_label
label_counts = df['predicted_label'].value_counts()
label_percentages = (label_counts / label_counts.sum()) * 100

# Create pie chart
plt.figure(figsize=(8, 6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Label Distribution')
plt.axis('equal')
plt.savefig(os.path.join(output_dir, 'label_distribution_pie_chart.png'))
plt.close()

# Save pie chart summary as table
summary_df = pd.DataFrame({
    'Label': label_counts.index,
    'Count': label_counts.values,
    'Percentage': label_percentages.round(2)
})
summary_df.to_csv(os.path.join(output_dir, 'label_summary.csv'), index=False)

# BAR CHART: Average score per label
avg_scores = df.groupby('predicted_label')['score'].mean()
plt.figure(figsize=(8, 6))
avg_scores.plot(kind='bar', color='mediumseagreen')
plt.title('Average Score by Label')
plt.xlabel('Label')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'average_score_bar_chart.png'))
plt.close()
