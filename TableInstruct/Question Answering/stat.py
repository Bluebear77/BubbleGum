import os
import csv
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
INPUT_FOLDER = "question_reason_type"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "stats")
CHART_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# Define categories
answer_types = ["Adjective", "Artwork", "Common noun", "Date", "Event", "Group", "Location", "Number", "Person", "Unknown"]
question_types = ["Composition", "Intersection", "Simple"]
all_skills = [
    "Aggregation", "All", "Average", "Comparative", "Counting", "Difference", "Filtering entity",
    "Filtering numeric", "Filtering time", "Negation", "No skill", "Ordinal", "Superlative", "Unique"
]

# -----------------------------
# Data Collection
# -----------------------------
answer_type_dist = defaultdict(lambda: Counter())
question_type_dist = defaultdict(lambda: Counter())
skills_dist = defaultdict(lambda: Counter())
overall_answer_type = Counter()
overall_question_type = Counter()
overall_skills = Counter()
skills_per_question_count = Counter()
example_answer_per_type = {}

for filename in tqdm(os.listdir(INPUT_FOLDER), desc="Reading CSVs"):
    if not filename.endswith("-type.csv"):
        continue

    dataset_name = filename.replace("-type.csv", "")
    filepath = os.path.join(INPUT_FOLDER, filename)

    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            atype = row['answer_type']
            qtype = row['question_type']
            skills = row['skills'].split(";") if row['skills'].strip() else ["No skill"]

            # Count answer type
            answer_type_dist[dataset_name][atype] += 1
            overall_answer_type[atype] += 1
            if atype not in example_answer_per_type and row["answer"].strip():
                example_answer_per_type[atype] = row["answer"].strip()

            # Count question type
            question_type_dist[dataset_name][qtype] += 1
            overall_question_type[qtype] += 1

            # Count skills
            for skill in skills:
                skills_dist[dataset_name][skill] += 1
                overall_skills[skill] += 1

            # Count skills per question
            skills_per_question_count[len(skills)] += 1

# -----------------------------
# Output 1: Answer Type Distribution Per Dataset
# -----------------------------
rows = []
for dataset, counts in answer_type_dist.items():
    row = [dataset] + [counts.get(t, 0) for t in answer_types]
    rows.append(row)
rows.append(["TOTAL"] + [overall_answer_type.get(t, 0) for t in answer_types])

df1 = pd.DataFrame(rows, columns=["Dataset"] + answer_types)
df1.to_csv(os.path.join(OUTPUT_FOLDER, "1_answer_type_distribution_per_dataset.csv"), index=False)

# -----------------------------
# Output 2: Answer Type Overview
# -----------------------------
total_answers = sum(overall_answer_type.values())
rows = []
for atype in answer_types:
    count = overall_answer_type.get(atype, 0)
    percent = round(100 * count / total_answers, 1)
    example = example_answer_per_type.get(atype, "")
    rows.append([atype, percent, example])

df2 = pd.DataFrame(rows, columns=["Answer Type", "%", "Example(s)"])
df2.to_csv(os.path.join(OUTPUT_FOLDER, "2_answer_type_distribution_overview.csv"), index=False)

# -----------------------------
# Output 3: Question Type + Skills Overview
# -----------------------------
rows = []
total_questions = sum(overall_question_type.values())
for qtype in question_types:
    count = overall_question_type.get(qtype, 0)
    freq = round(100 * count / total_questions, 1)
    rows.append(["Question Type", qtype, count, freq])

for skill in all_skills:
    count = overall_skills.get(skill, 0)
    freq = round(100 * count / total_questions, 1)
    rows.append(["Skill", skill, count, freq])

skill_one = skills_per_question_count.get(1, 0)
skill_two = skills_per_question_count.get(2, 0)
skill_three = skills_per_question_count.get(3, 0)
total_skills = skill_one + skill_two + skill_three
rows.extend([
    ["Skills per question", "One skill", skill_one, round(100 * skill_one / total_skills, 1)],
    ["Skills per question", "Two skills", skill_two, round(100 * skill_two / total_skills, 1)],
    ["Skills per question", "Three skills", skill_three, round(100 * skill_three / total_skills, 1)],
])

df3 = pd.DataFrame(rows, columns=["Section", "Label", "Count", "Freq (%)"])
df3.to_csv(os.path.join(OUTPUT_FOLDER, "3_question_type_overview.csv"), index=False)

# -----------------------------
# Output 4: Question Type Per Dataset
# -----------------------------
rows = []
for dataset, counts in question_type_dist.items():
    row = [dataset] + [counts.get(t, 0) for t in question_types]
    rows.append(row)
rows.append(["TOTAL"] + [overall_question_type.get(t, 0) for t in question_types])

df4 = pd.DataFrame(rows, columns=["Dataset"] + question_types)
df4.to_csv(os.path.join(OUTPUT_FOLDER, "4_question_type_per_dataset.csv"), index=False)

# -----------------------------
# Output 5: Skills Per Dataset
# -----------------------------
rows = []
for dataset, counts in skills_dist.items():
    row = [dataset] + [counts.get(s, 0) for s in all_skills]
    rows.append(row)
rows.append(["TOTAL"] + [overall_skills.get(s, 0) for s in all_skills])

df5 = pd.DataFrame(rows, columns=["Dataset"] + all_skills)
df5.to_csv(os.path.join(OUTPUT_FOLDER, "5_skills_per_dataset.csv"), index=False)

# -----------------------------
# Bar Chart: Output 1 - Answer Type Distribution
# -----------------------------
df1_plot = df1[df1["Dataset"] != "TOTAL"].set_index("Dataset")
df1_plot.plot(kind="bar", figsize=(14, 7), width=0.8)
plt.title("Answer Type Distribution per Dataset")
plt.ylabel("Count")
plt.xlabel("Dataset")
plt.legend(title="Answer Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "1_answer_type_distribution.png"))
plt.clf()

# -----------------------------
# Bar Chart: Output 4 - Question Type per Dataset
# -----------------------------
df4_plot = df4[df4["Dataset"] != "TOTAL"].set_index("Dataset")
df4_plot.plot(kind="bar", figsize=(10, 6), width=0.8)
plt.title("Question Type Distribution per Dataset")
plt.ylabel("Count")
plt.xlabel("Dataset")
plt.legend(title="Question Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "4_question_type_per_dataset.png"))
plt.clf()

# -----------------------------
# Bar Chart: Output 5 - Skills per Dataset
# -----------------------------
df5_plot = df5[df5["Dataset"] != "TOTAL"].set_index("Dataset")
df5_plot.plot(kind="bar", figsize=(18, 8), width=0.8)
plt.title("Skills per Dataset")
plt.ylabel("Count")
plt.xlabel("Dataset")
plt.legend(title="Skill", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "5_skills_per_dataset.png"))
plt.clf()

print(f"✅ Statistics CSVs and bar charts saved to: {OUTPUT_FOLDER}")
