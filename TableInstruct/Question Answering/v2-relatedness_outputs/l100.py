import pandas as pd

# Load the CSV file
df = pd.read_csv("_lowest100.csv")

# Total number of rows
total_rows = len(df)

# Count occurrences of each source
source_counts = df['source'].value_counts()

# Print the results
print(f"Total rows: {total_rows}\n")
print(f"{'Source':<30}{'Count':<10}{'Percentage':<10}")
print("-" * 50)

for source, count in source_counts.items():
    percentage = (count / total_rows) * 100
    print(f"{source:<30}{count:<10}{percentage:.2f}%")
