import os
import json

def extract_fetaqa_question(item):
    """Extract question part after [HIGHLIGHTED_END] for FETAQA format."""
    question_text = item.get("question", "")
    if "[HIGHLIGHTED_END]" in question_text:
        return question_text.split("[HIGHLIGHTED_END]", 1)[1].strip()
    return None

def extract_standard_question(item):
    """Extract question directly."""
    return item.get("question", "").strip()

def extract_hybridqa_question(item):
    """Extract text after 'The question:' in HybridQA format."""
    question_text = item.get("question", "")
    if "The question:" in question_text:
        return question_text.split("The question:", 1)[1].strip()
    return None

def process_files(file_configs, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name, extraction_method in file_configs:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        extracted_questions = []
        for item in data:
            question = extraction_method(item)
            if question:
                extracted_questions.append(question)

        base_name = os.path.basename(file_name).replace('.json', '.txt')
        output_path = os.path.join(output_folder, base_name)

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for question in extracted_questions:
                out_f.write(question + "\n")

        print(f"Extracted questions saved to {output_path}")

if __name__ == "__main__":
    file_configs = [
        ("fetaqa_train_7325.json", extract_fetaqa_question),
        ("fetaqa_test.json", extract_fetaqa_question),
        ("hitab_test.json", extract_standard_question),
        ("wikisql_test.json", extract_standard_question),
        ("wikitq_test.json", extract_standard_question),
        ("hitab_train_7417.json", extract_standard_question),
        ("hybridqa_eval.json", extract_hybridqa_question),
    ]

    output_dir = "question"
    process_files(file_configs, output_dir)
