import os
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import time

# Configuration
INPUT_FOLDER = 'test'
OUTPUT_FOLDER = 'topic prediction bart'
USER_AGENT = "TopicPredictorBot/1.0"

# Labels for classification
LABELS = [
    "Food and Beverage",
    "Culture",
    "Media and Entertainment",
    "Religion and Philosophy",
    "Sports",
    "Art",
    "Business and Economic",
    "Education",
    "Warfare and Conflict",
    "Political",
    "Society",
    "Science and Technology",
    "Environmental",
    "Healthcare and Medicine"
]

# Setup
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
nlp = spacy.load("en_core_web_sm")

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --- Helper Functions ---

def extract_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if len(ent.text) > 2))

def predict_topics_with_zero_shot(text, labels):
    result = classifier(text, labels, multi_label=True)
    # Sort results by score descending
    sorted_results = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)
    return sorted_results[:3]

# --- Main Processing Loop ---

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename.replace('.txt', '.csv'))

    rows = []

    with open(input_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in tqdm(lines, desc=f"Processing {filename}", unit="line"):
        entities = extract_entities(line)

        # Use the full line as the classification input
        predictions = predict_topics_with_zero_shot(line, LABELS)

        top1_topic, top1_score = predictions[0][0], round(predictions[0][1], 4)
        top2_topic, top2_score = predictions[1][0], round(predictions[1][1], 4)
        top3_topic, top3_score = predictions[2][0], round(predictions[2][1], 4)

        rows.append({
            "Top 1 Topic": top1_topic,
            "Top 1 Score": top1_score,
            "Top 2 Topic": top2_topic,
            "Top 2 Score": top2_score,
            "Top 3 Topic": top3_topic,
            "Top 3 Score": top3_score,
            "Entities": ", ".join(entities),
            "Original text": line
        })

        time.sleep(0.1)  # optional polite delay

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
