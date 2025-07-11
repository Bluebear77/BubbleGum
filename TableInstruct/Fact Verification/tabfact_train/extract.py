import os
import json
import re
from collections import defaultdict

def clean_input_seg(text):
    # Extract content after "The table caption is about"
    marker = "The table caption is about"
    if marker in text:
        text = text.split(marker, 1)[1]

    # Remove [TAB], [SEP], and |
    text = re.sub(r'\[TAB\]|\[SEP\]|\|', '', text)
    return text.strip()

def extract_table_key(text):
    # Extract everything after [TLE] and before the first [TAB]
    match = re.search(r'\[TLE\](.*?)\[TAB\]', text)
    return match.group(1).strip() if match else "UNKNOWN_TABLE"

def clean_question(text):
    match = re.search(r'<(.*?)>', text)
    return match.group(1).strip() if match else text.strip()

def process_json_file(filepath, output_dir):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to parse {filepath}: {e}")
            return

    table_grouped_data = defaultdict(list)

    for item in data:
        raw_input = item.get("input_seg", "")
        cleaned_input = clean_input_seg(raw_input)
        cleaned_question = clean_question(item.get("question", ""))
        table_grouped_data[cleaned_input].append(cleaned_question)

    # Write output for this file
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{base_filename}.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        for table_text, questions in table_grouped_data.items():
            line = table_text + " " + " ".join(questions)
            f.write(line + "\n")

def main():
    input_dir = '.'
    output_dir = 'grouped_output'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            process_json_file(os.path.join(input_dir, filename), output_dir)

if __name__ == "__main__":
    main()
