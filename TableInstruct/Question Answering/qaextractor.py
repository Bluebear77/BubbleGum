# ============================================================================================
# Script: JSON QA Extractor to CSV
# --------------------------------------------------------------------------------------------
# This script processes multiple JSON files with various question-answer (QA) formats and
# extracts standardized question and answer pairs. It outputs CSV files with two columns:
# "question" and "answer", saved in a specified output directory.
#
# Key Features:
# - Supports FETAQA, HybridQA, and standard QA formats (HiTab, WikiSQL, WikiTQ).
# - Removes all double quotation marks from questions and answers.
# - Writes CSV files using csv.QUOTE_NONE to prevent extra quoting in output.
# ============================================================================================

import os
import json
import csv

def extract_fetaqa(item):
    """
    Extract question and answer from FETAQA format.
    Question is extracted from the text after [HIGHLIGHTED_END].
    Quotation marks are removed from both fields.
    """
    question_text = item.get("question", "")
    answer = item.get("output", "").strip().replace('"', '')  # Remove quotes from answer
    if "[HIGHLIGHTED_END]" in question_text:
        question = question_text.split("[HIGHLIGHTED_END]", 1)[1].strip().replace('"', '')  # Remove quotes from question
        return question, answer
    return None, None

def extract_standard(item):
    """
    Extract question and answer from standard formats (HiTab, WikiSQL, WikiTQ).
    Fields are directly taken from 'question' and 'output'.
    Quotation marks are removed from both fields.
    """
    question = item.get("question", "").strip().replace('"', '')
    answer = item.get("output", "").strip().replace('"', '')
    return question, answer

def extract_hybridqa(item):
    """
    Extract question and answer from HybridQA format.
    Question is extracted from the text after "The question:".
    Quotation marks are removed from both fields.
    """
    question_text = item.get("question", "")
    answer = item.get("output", "").strip().replace('"', '')
    if "The question:" in question_text:
        question = question_text.split("The question:", 1)[1].strip().replace('"', '')
        return question, answer
    return None, None

def process_files(file_configs, output_folder):
    """
    Processes each JSON file with the corresponding extraction function.
    Extracted question-answer pairs are saved to CSV files with the same base filename.

    Parameters:
    - file_configs: list of tuples (file_path, extraction_function)
    - output_folder: target directory for CSV output
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist

    for file_name, extraction_function in file_configs:
        # Load JSON data from file
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        extracted_data = []
        for item in data:
            question, answer = extraction_function(item)
            if question and answer:
                extracted_data.append([question, answer])

        # Define output CSV file path
        base_name = os.path.basename(file_name).replace('.json', '.csv')
        output_path = os.path.join(output_folder, base_name)

        # Write to CSV using no quotes and custom escape character
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerow(["question", "answer"])  # Write header
            writer.writerows(extracted_data)         # Write data rows

        print(f"Extracted questions and answers saved to {output_path}")

if __name__ == "__main__":
    # Define list of files and their corresponding extraction functions
    file_configs = [
        ("fetaqa_train_7325.json", extract_fetaqa),
        ("fetaqa_test.json", extract_fetaqa),
        ("hitab_test.json", extract_standard),
        ("wikisql_test.json", extract_standard),
        ("wikitq_test.json", extract_standard),
        ("hitab_train_7417.json", extract_standard),
        ("hybridqa_eval.json", extract_hybridqa),
    ]

    # Output directory for resulting CSV files
    output_dir = "QAS"

    # Run the extraction and conversion process
    process_files(file_configs, output_dir)
