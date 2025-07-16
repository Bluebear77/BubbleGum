# This script extracts questions and answers from multiple JSON files with varying formats.
# It outputs CSV files with two columns: "question" and "answer".
# Extraction rules depend on the file:
# - For FETAQA files: extract text after [HIGHLIGHTED_END] as the question, and use the "output" field as the answer.
# - For standard files (HiTab, WikiSQL, WikiTQ): use the "question" and "output" fields directly.
# - For HybridQA: extract text after "The question:" in the "question" field, and use the "output" field as the answer.
# The resulting CSV files are saved in the "question" directory with input-filename.csv naming.


import os
import json
import csv

def extract_fetaqa(item):
    """Extract question and answer for FETAQA format."""
    question_text = item.get("question", "")
    answer = item.get("output", "").strip()
    if "[HIGHLIGHTED_END]" in question_text:
        question = question_text.split("[HIGHLIGHTED_END]", 1)[1].strip()
        return question, answer
    return None, None

def extract_standard(item):
    """Extract question and answer for standard format."""
    question = item.get("question", "").strip()
    answer = item.get("output", "").strip()
    return question, answer

def extract_hybridqa(item):
    """Extract question and answer for HybridQA format."""
    question_text = item.get("question", "")
    answer = item.get("output", "").strip()
    if "The question:" in question_text:
        question = question_text.split("The question:", 1)[1].strip()
        return question, answer
    return None, None

def process_files(file_configs, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name, extraction_function in file_configs:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        extracted_data = []
        for item in data:
            question, answer = extraction_function(item)
            if question and answer:
                extracted_data.append([question, answer])

        base_name = os.path.basename(file_name).replace('.json', '.csv')
        output_path = os.path.join(output_folder, base_name)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["question", "answer"])
            writer.writerows(extracted_data)

        print(f"Extracted questions and answers saved to {output_path}")

if __name__ == "__main__":
    file_configs = [
        ("fetaqa_train_7325.json", extract_fetaqa),
        ("fetaqa_test.json", extract_fetaqa),
        ("hitab_test.json", extract_standard),
        ("wikisql_test.json", extract_standard),
        ("wikitq_test.json", extract_standard),
        ("hitab_train_7417.json", extract_standard),
        ("hybridqa_eval.json", extract_hybridqa),
    ]

    output_dir = "QAS"
    process_files(file_configs, output_dir)
