"""
Question Type Analysis Script
-----------------------------
Processes all .csv files in ./QAS/, analyzes the first column for question types,
saves labeled CSV files in ./question_type/, generates a PNG and HTML sunburst chart,
and produces a hierarchical statistics CSV file sorted by count (descending).

Automatically installs:
- System libraries via apt-get (for Plotly image export)
- Python packages: pandas, plotly, kaleido, tqdm
"""

import subprocess
import sys
import os

def run_apt_dependencies():
    try:
        subprocess.run(
            "sudo apt update && sudo apt-get install -y "
            "libnss3 libatk-bridge2.0-0t64 libcups2t64 libxcomposite1 libxdamage1 "
            "libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2t64",
            check=True,
            shell=True
        )
    except Exception as e:
        print("⚠️  Warning: Could not automatically install system dependencies. Run manually if needed.")
        print(e)

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

run_apt_dependencies()

for pkg in ['pandas', 'plotly', 'kaleido', 'tqdm']:
    install_and_import(pkg)

import pandas as pd
import re
from collections import Counter, defaultdict
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
from urllib.parse import quote
import platform

input_folder = './QAS/'
output_folder = './question_cqw_type/'

os.makedirs(output_folder, exist_ok=True)

QUESTION_WORDS = ['what', 'who', 'how', 'which', 'when', 'where', 'does', 'did', 'is', 'are']
PREPOSITIONS = ['in', 'by']

def extract_question_type(question):
    tokens = re.findall(r'\w+', question.lower())
    cqw_index = None
    for idx, token in enumerate(tokens[:3]):
        if token in QUESTION_WORDS:
            cqw_index = idx
            break
    if cqw_index is None:
        for idx in reversed(range(len(tokens))):
            if tokens[idx] in QUESTION_WORDS:
                cqw_index = idx
                break
    if cqw_index is None:
        return "other"
    start_index = cqw_index
    if cqw_index > 0 and tokens[cqw_index - 1] in PREPOSITIONS:
        start_index -= 1
    end_index = min(cqw_index + 3, len(tokens))
    return ' '.join(tokens[start_index:end_index])

all_types_counter = Counter()
file_list = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for file_name in tqdm(file_list, desc="Processing files"):
    df = pd.read_csv(os.path.join(input_folder, file_name))
    if df.shape[1] == 0:
        continue
    questions = df.iloc[:, 0].astype(str)
    types = questions.apply(extract_question_type)
    all_types_counter.update(types)
    result_df = pd.DataFrame({'Type': types, 'Question': questions})
    result_df.to_csv(os.path.join(output_folder, file_name), index=False)

sunburst_data = []
for type_phrase, count in all_types_counter.items():
    parts = type_phrase.split()
    sunburst_data.append({
        'CQW': parts[0],
        'Next Word 1': parts[1] if len(parts) > 1 else '',
        'Next Word 2': parts[2] if len(parts) > 2 else '',
        'Count': count
    })

sunburst_df = pd.DataFrame(sunburst_data)

fig = px.sunburst(
    sunburst_df,
    path=['CQW', 'Next Word 1', 'Next Word 2'],
    values='Count',
    title='Distribution of Question Types'
)

sunburst_image_path = os.path.join(output_folder, 'question_type_sunburst.png')
sunburst_html_path = os.path.join(output_folder, 'question_type_sunburst.html')

pio.write_image(fig, sunburst_image_path, format='png', scale=2)
fig.write_html(sunburst_html_path)

parent_child_counter = defaultdict(int)

for _, row in sunburst_df.iterrows():
    count = row['Count']
    cqw = row['CQW']
    next1 = row['Next Word 1']
    next2 = row['Next Word 2']

    parent_child_counter[cqw] += count
    if next1:
        parent_child_counter[f"{cqw}/{next1}"] += count
    if next2:
        parent_child_counter[f"{cqw}/{next1}/{next2}"] += count

statistics_rows = []
for key, count in parent_child_counter.items():
    parts = key.split('/')
    label = parts[-1]
    parent = '/'.join(parts[:-1]) if len(parts) > 1 else ''
    statistics_rows.append({
        'label': label,
        'count': count,
        'parent': parent,
        'id': key
    })

statistics_df = pd.DataFrame(statistics_rows)

# Sort by count in descending order
statistics_df = statistics_df.sort_values(by='count', ascending=False)

statistics_csv_path = os.path.join(output_folder, 'question_type_sunburst_statistics.csv')
statistics_df.to_csv(statistics_csv_path, index=False)

def local_file_url(path):
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        return f"file:///{quote(path.replace(os.sep, '/'))}"
    else:
        return f"file://{quote(path)}"

print("✅ Processing complete. Outputs saved in ./question_type/")
print(f"📊 View the sunburst chart here: {local_file_url(sunburst_html_path)}")
