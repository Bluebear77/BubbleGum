# =============================================================================
# Script Summary:
# This script processes JSON files containing questions and table segments.
# It:
#   - Cleans input segments and questions using regex.
#   - Removes punctuation to prepare text for BERT encoding.
#   - Uses SentenceTransformer (BERT-based) to compute semantic similarity
#     between each question and its corresponding table segment.
#   - Saves similarity scores, questions, and tables into CSV files.
# Requirements:
#   pip install sentence-transformers pandas tqdm
# =============================================================================

import os
import json
import re
import string
import pandas as pd
from tqdm import tqdm
import csv  # ✅ For CSV quoting control
from sentence_transformers import SentenceTransformer, util  # ✅ Activate BERT

# Directories for input and output
input_dir = '.'
output_dir = 'relatedness_outputs'
os.makedirs(output_dir, exist_ok=True)

# ✅ Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Clean input segment using regex ===
def clean_input_seg(text):
    if not isinstance(text, str):
        return ""

    # Regex patterns to remove unwanted tokens and noise
    patterns = [
        r"\[TLE\] The Wikipedia page title of this table is",
        r"The Wikipedia section title of this table is",
        r"\[TLE\] The table caption is",
        r"\[TAB\]",
        r"\[TAB\] col:",
        r"\bcol:\b",
        r"\|",
        r"\\",            # remove backslashes
        r"\[SEP\]",
        r"[TAB] col:",
        r"col:",
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.replace('"', '')  # remove quotation marks
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    return text.strip()

# === Extract and clean question based on dataset type ===
def extract_question(entry, filename):
    question_text = entry.get("question", "")
    if not isinstance(question_text, str):
        return None

    if "fetaqa" in filename.lower():
        if "[HIGHLIGHTED_END]" in question_text:
            question = question_text.split("[HIGHLIGHTED_END]", 1)[1]
        else:
            return None
    elif "hybridqa" in filename.lower():
        if "The question:" in question_text:
            question = question_text.split("The question:", 1)[1]
        else:
            return None
    else:
        question = question_text

    # Remove special characters and normalize whitespace
    question = question.replace('"', '').replace('?', '')
    question = re.sub(r'\s+', ' ', question)
    return question.strip()

# === Remove punctuation from a string ===
def remove_punctuation(text):
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

# === Main processing loop ===
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.csv")

        print(f"🔍 Processing {filename}...")

        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping {filename}: invalid JSON format")
                continue

        results = []

        for entry in tqdm(data, desc=f"→ {filename}", leave=False):
            raw_input_seg = entry.get("input_seg", "")
            question = extract_question(entry, filename)

            if not raw_input_seg or not question:
                continue

            # Clean and prep the text
            clean_seg = clean_input_seg(raw_input_seg)
            clean_question = question

            seg_for_bert = remove_punctuation(clean_seg)
            question_for_bert = remove_punctuation(clean_question)

            # ✅ BERT processes the text directly in memory, right before writing to the CSV.
            # ✅ Compute semantic similarity using BERT
            embeddings = model.encode([seg_for_bert, question_for_bert], convert_to_tensor=True)
            similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()

            results.append({
                "similarity_score": similarity_score,
                "question": clean_question,
                "table": clean_seg
            })

        # ✅ Save results to CSV with minimal quoting
        df = pd.DataFrame(results)
        # df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"✅ Saved: {output_path}")
