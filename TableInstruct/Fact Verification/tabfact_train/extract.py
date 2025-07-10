# This script processes all JSON files in the current directory, extracting and cleaning the "input_seg" and "question" fields.
# It removes formatting tags like [TAB], [SEP], and '|' characters, extracts the relevant portion of the input segment
# (after "The table caption is about"), and keeps only the text inside angle brackets in the question.
# The cleaned results are saved to individual .txt files (one per JSON file) in the "output" directory, with each line
# containing the cleaned input segment followed by the cleaned question.

import os
import json
import re

def clean_input_seg(text):
    # Extract after "The table caption is about"
    marker = "The table caption is about"
    if marker in text:
        text = text.split(marker, 1)[1]

    # Remove [TAB], [SEP], and |
    text = re.sub(r'\[TAB\]|\[SEP\]|\|', '', text)
    return text.strip()

def clean_question(text):
    # Extract text inside < >
    match = re.search(r'<(.*?)>', text)
    return match.group(1).strip() if match else text.strip()

def process_json_file(filepath, output_dir):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to parse {filepath}: {e}")
            return

    output_lines = []
    for item in data:
        input_seg = clean_input_seg(item.get("input_seg", ""))
        question = clean_question(item.get("question", ""))
        output_lines.append(f"{input_seg} {question}")

    # Write to output file
    basename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{basename}.txt")
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write("\n".join(output_lines))

def main():
    input_dir = '.'
    output_dir = 'extracted output'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            process_json_file(os.path.join(input_dir, filename), output_dir)

if __name__ == "__main__":
    main()
