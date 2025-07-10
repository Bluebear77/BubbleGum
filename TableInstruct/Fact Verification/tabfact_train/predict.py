import os
import requests
import spacy
import pandas as pd
from wikipediaapi import Wikipedia
from tqdm import tqdm
import time

# Configuration
INPUT_FOLDER = 'test'
OUTPUT_FOLDER = 'topic prediction'
USER_AGENT = "TopicPredictorBot/1.0"
THRESHOLD = 0.0  # Keep 0 to always get top 3 results

# Setup
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
nlp = spacy.load("en_core_web_sm")
wiki = Wikipedia(user_agent=USER_AGENT, language='en')

# --- Helper Functions ---

def extract_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if len(ent.text) > 2))

def get_wikipedia_titles(entities):
    titles = []
    for ent in entities:
        page = wiki.page(ent)
        if page.exists():
            titles.append(page.title)
    return list(set(titles))

def predict_topic_from_title(title, lang='en', threshold=THRESHOLD):
    payload = {
        "page_title": title.replace(" ", "_"),
        "lang": lang,
        "threshold": threshold
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT
    }

    url = "https://api.wikimedia.org/service/lw/inference/v1/models/outlink-topic-model:predict"
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if "prediction" in result:
            predictions = sorted(result['prediction']['results'], key=lambda x: x['score'], reverse=True)
            return predictions[:3]
    return []

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
        titles = get_wikipedia_titles(entities)
        best_title = titles[0] if titles else ""

        top1_topic, top1_score = "", ""
        top2_topic, top2_score = "", ""
        top3_topic, top3_score = "", ""

        if best_title:
            predictions = predict_topic_from_title(best_title)
            if len(predictions) > 0:
                top1_topic = predictions[0]['topic']
                top1_score = round(predictions[0]['score'], 4)
            if len(predictions) > 1:
                top2_topic = predictions[1]['topic']
                top2_score = round(predictions[1]['score'], 4)
            if len(predictions) > 2:
                top3_topic = predictions[2]['topic']
                top3_score = round(predictions[2]['score'], 4)

        rows.append({
            "Top 1 Topic": top1_topic,
            "Top 1 Score": top1_score,
            "Top 2 Topic": top2_topic,
            "Top 2 Score": top2_score,
            "Top 3 Topic": top3_topic,
            "Top 3 Score": top3_score,
            "Entities": ", ".join(entities),
            "Wikipedia Titles": ", ".join(titles),
            "Using page title": best_title,
            "Original text": line
        })

        time.sleep(0.1)  # polite API delay

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")
