import os
import json
import csv
from transformers import pipeline
from tqdm import tqdm

# Disable TensorFlow backend in Hugging Face Transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Define the 25 TabLib categories
TABLIB_CATEGORIES = [
    "Software and Technology", "Science and Research", "Financial and Economic",
    "Retail and E-commerce", "Census and Demographics", "Healthcare and Medicine",
    "Sports and Recreation", "Education", "Transportation", "Art and Design",
    "Industrial and Manufacturing", "Internet and Web Services", "Weather and Climate",
    "Media and Entertainment", "Real Estate and Construction", "Energy and Utilities",
    "Environmental", "Political", "Travel and Hospitality", "Food and Beverage",
    "Agriculture and Food", "Legal", "Telecommunications", "Religion and Philosophy",
    "Marketing and Advertising"
]

# Load the zero-shot classification pipeline using a PyTorch backend
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load your JSON input
input_path = "tabfact_train_92283_part8.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Optional: limit data size for debugging (remove or adjust as needed)
# data = data[:10]

# Prepare CSV output
output_csv_path = "classified_results.csv"
with open(output_csv_path, mode="w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ["question", "predicted_label", "score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process each question with a tqdm progress bar
    for item in tqdm(data, desc="Classifying questions"):
        question = item.get("question", "")
        if not question.strip():
            continue  # Skip empty questions

        try:
            prediction = classifier(question, TABLIB_CATEGORIES, multi_label=False)
            writer.writerow({
                "question": question,
                "predicted_label": prediction["labels"][0],
                "score": prediction["scores"][0]
            })
        except Exception as e:
            print(f"Error processing question: {question[:60]}... => {e}")
            continue

print(f"\nAll predictions saved to '{output_csv_path}'")
