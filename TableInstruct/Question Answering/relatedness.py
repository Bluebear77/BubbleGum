# pip install sentence-transformers pandas


import os
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Directory containing your JSON files
input_dir = '.'  # current directory
output_dir = 'relatedness_outputs'

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Process each JSON file
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.csv")

        print(f"🔍 Processing {filename}...")

        # Load JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping {filename}: invalid JSON format")
                continue

        results = []

        for entry in tqdm(data, desc=f"→ {filename}", leave=False):
            input_seg = entry.get("input_seg", "")
            question = entry.get("question", "")

            if not input_seg or not question:
                continue

            # Compute similarity
            embeddings = model.encode([input_seg, question], convert_to_tensor=True)
            similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()

            results.append({
                "similarity_score": round(similarity_score, 4),
                "question": question
            })

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_path}")
