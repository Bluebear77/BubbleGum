import os
import csv
import time
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# -----------------------------
# Setup
# -----------------------------

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Optional quick check
doc = nlp("Albert Einstein was a physicist.")
print([(ent.text, ent.label_) for ent in doc.ents])

# Configuration
INPUT_FOLDER = 'sample100domain'      # folder containing input CSVs
OUTPUT_FOLDER = 'sample100domain'     # final output folder
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'QAdomain-verification.csv')
USER_AGENT = "TopicPredictorBot/1.0"

# Labels for classification
LABELS = [
    "Food and Beverage", "Culture", "Media and Entertainment", "Religion and Philosophy",
    "Sports", "Art", "Business and Economic", "Education", "Warfare and Conflict",
    "Political", "Society", "Science and Technology", "Environmental", "Healthcare and Medicine"
]

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# -----------------------------
# Helper Functions
# -----------------------------

def extract_entities(text: str):
    doc = nlp(text)
    # unique entities with length > 2
    return list({ent.text for ent in doc.ents if len(ent.text) > 2})

def predict_topics_with_zero_shot(text: str, labels):
    result = classifier(text, labels, multi_label=True)
    sorted_results = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)
    # return top 3 predictions
    return sorted_results[:3]

# -----------------------------
# Main Processing
# -----------------------------

all_rows = []

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)

    try:
        # Safer CSV reading
        df_input = pd.read_csv(input_path, on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        continue

    if 'qas' not in df_input.columns:
        print(f"⚠️ Skipping {filename} — no 'qas' column found.")
        continue

    # Normalize optional columns so code path is uniform
    if 'answer' not in df_input.columns:
        df_input['answer'] = pd.NA
    if 'table' not in df_input.columns:
        df_input['table'] = pd.NA

    iterator = df_input.dropna(subset=['qas']).iterrows()
    for _, row in tqdm(iterator, desc=f"Processing {filename}", unit="row"):
        qas_text = str(row['qas']).strip()
        answer_text = str(row['answer']).strip() if pd.notna(row['answer']) else ""
        table_text = str(row['table']).strip() if pd.notna(row['table']) else ""

        # Build combined text (qas + answer when available)
        combined_text = (
            f"QAS: {qas_text} Answer: {answer_text}"
            if answer_text else qas_text
        )

        # Extract entities and predict topics on combined text
        entities = extract_entities(combined_text)
        predictions = predict_topics_with_zero_shot(combined_text, LABELS)

        # Ensure we always have 3 predictions (defensive, though model should return many)
        while len(predictions) < 3:
            predictions.append(("N/A", 0.0))

        top1_topic, top1_score = predictions[0][0], round(float(predictions[0][1]), 4)
        top2_topic, top2_score = predictions[1][0], round(float(predictions[1][1]), 4)
        top3_topic, top3_score = predictions[2][0], round(float(predictions[2][1]), 4)

        all_rows.append({
            "qas": qas_text,
            "answer": answer_text,
            "table": table_text,
            "Top 1 Topic": top1_topic,
            "Top 1 Score": top1_score,
            "Top 2 Topic": top2_topic,
            "Top 2 Score": top2_score,
            "Top 3 Topic": top3_topic,
            "Top 3 Score": top3_score,
            "Entities": ", ".join(entities)
        })

        time.sleep(0.1)  # polite delay to avoid rate limits

# -----------------------------
# Save consolidated output
# -----------------------------
df_output = pd.DataFrame(all_rows)

# Enforce exact column order and presence
desired_columns = [
    "qas", "answer", "table",
    "Top 1 Topic", "Top 1 Score",
    "Top 2 Topic", "Top 2 Score",
    "Top 3 Topic", "Top 3 Score",
    "Entities"
]
# Add any missing columns as empty, then reorder
for col in desired_columns:
    if col not in df_output.columns:
        df_output[col] = ""
df_output = df_output[desired_columns]

df_output.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved: {OUTPUT_FILE}")

# -----------------------------
# Notes:
# - The previous 'question' column is fully replaced by 'qas'.
# - Combined text now uses 'qas' instead of 'question'.
# -----------------------------
