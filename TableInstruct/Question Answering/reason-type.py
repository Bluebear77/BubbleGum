"""
===================================================================================
Question Classification and Skill Inference Script for Table QA Benchmarking
===================================================================================

Purpose:
--------
This script categorizes table-based QA examples into structured reasoning types
(Simple, Intersection, Composition), identifies reasoning skills involved, and
classifies the type of the final answer (e.g., Date, Person, Number).

It is tailored for use with benchmark datasets like HybridQA, HiTab, WikiTQ, etc.

Structure:
----------
1. Reasoning Skill Patterns
   - Patterns derived from the TabFact, TANQ, and QTSumm datasets.
   - Detects skills like Aggregation, Superlative, Filtering, Average, etc.
   - Matched using regular expressions over question text.

2. Answer Type Detection
   - Uses keyword-based patterns (e.g., "km", "year", "city") to infer answer types.
   - Based on the taxonomy from HybridQA: Number, Date, Person, Location, etc.
   - Prioritizes types in order to avoid misclassification (e.g., Date > Number).

3. Question Type Classification (Core logic in `classify_question_and_skills`)
   - Based on taxonomy introduced in the TableInstruct paper (arXiv:2405.07765):
     - Simple: Single-hop queries with no nested or combined constraints.
     - Intersection: Entities must satisfy **multiple conditions** (e.g., “both X and Y”).
     - Composition: Requires **chaining** through intermediate entities (e.g., “film that X wrote”).
   - Prioritization rule: Composition > Intersection > Simple.
   - Multi-level detection:
     - Composition: Detected via nested clause cues (e.g., "that", "whose", etc.).
     - Intersection: Detected via "both", "and", co-occurring filters, or semantically combined filters (e.g., "discovered by" + "type of").
     - Fallback rules apply if patterns are not matched explicitly.

4. Main Processing Loop
   - Reads QA files from the `QASd/` directory in CSV format.
   - Outputs enriched files in `question_domain_type/`, appending new columns:
     [question_type, skills, answer_type]
   - Expects headers "question" and "answer" to be present in each input file.

-----------------------
Example Classification:
-----------------------
Input:   "Who won the World Cup in 2018?"
Output:  question_type = "Simple", skills = ["Filtering time"], answer_type = "Group"

Input:   "Who directed the movie that Quentin Tarantino wrote?"
Output:  question_type = "Composition", skills = ["Filtering entity"], answer_type = "Person"

Input:   "Which films were both written and directed by Spielberg?"
Output:  question_type = "Intersection", skills = ["Filtering entity"], answer_type = "Other proper noun"
"""


import os
import csv
import re
from collections import Counter
from tqdm import tqdm

# ---------------------------------------------------------------------
# Step 1: Define reasoning skill patterns (from TabFact, TANQ, QTSumm)
# ---------------------------------------------------------------------

skill_patterns = {
    # TabFact reasoning skills
    "Aggregation": [r"total", r"sum of", r"overall"],
    "Negation": [r" did not ", r" does not ", r" not ", r" never ", r"has never ", r"no ",r" what other ",],
    "Superlative": [r"most ", r"least ", r"highest", r"lowest", r"best", r"worst", r"largest", r"smallest", r"longest", r"shortest", r"earliest", r"latest"],
    "Comparative": [r" earlier than ", r" later than ", r" before ", r" after ", r" more than ", r" greater than ", r" less than ", r" fewer than ", r" higher than ", r" lower than ", r" older than ", r" younger than "],
    "Ordinal": [r" first ", r" second ", r" third ", r" 1st ", r" 2nd ", r" 3rd ", r" last ", r" next "],
    "Unique": [r"different", r"unique", r"only one", r"no two"],
    "All": [r"all of", r"every ", r"none of"],

    # TANQ reasoning skills
    "Filtering numeric": [r" greater than ", r" less than ", r" more than ", r" fewer than ", r">", r"<", r"at least", r"at most"],
    "Filtering time": [r" before ", r" after ", r" earlier than ", r" later than ", r"since ", r"until ", r"date", r" year of ", r" in what year", r" year was", r" on which date", r" what date", r" on what date", r"date of", r"date did", r"when was"],
    "Filtering entity": [r" whose ", r" with ", r"where .* is "],

    # QTSumm reasoning skills
    "Average": [r" sum of ", r" average of ", r"avg of", r" total of ", r"average ", r"mean "],
    "Difference": [r" how many years", r" how long", r" lived", r" years old", r"difference in years", r" difference between", r" how many more", r" how many less", r" how much more", r" difference in"]
}

# All known skills for fallback check
all_skills = set(skill_patterns)

# ---------------------------------------------------------------------
# Step 2: Define answer type patterns (based on HybridQA taxonomy)
# ---------------------------------------------------------------------

answer_type_patterns = [
    ("Date", [
        r"\b\d{1,2} (january|february|march|april|may|june|july|august|september|october|november|december) \d{4}\b",
        r"\b(\d{4}|\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December)|\d{1,2}/\d{1,2}/\d{2,4})\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2}\b"
    ]),
    ("Person", r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"),
    ("Location", r"\b(road|avenue|street|city|state|country|river|mountain|park|lake|building|airport|station|square|valley|island|region|area)\b"),
    ("Group", r"\b(team|band|group|organization|committee|council|corporation|company)\b"),
    ("Event", r"\b(war|battle|conference|summit|meeting|ceremony|game|match|tournament)\b"),
    ("Artwork", r"\b(song|album|book|movie|film|novel|painting|sculpture|opera)\b"),
    ("Adjective", r"\b(best|worst|largest|smallest|most|least|first|second|last|only)\b"),
    ("Other proper noun", r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b"),
    ("Common noun", r"\b(musician|writer|teacher|city|dog|car|food|instrument)\b"),
    ("Number", r"[-+]?[.,\d]*[\d]+(?:[.,\d]*)?(?:\s*(?:m|km|ft|kg|lbs|years|minutes|seconds|people|times|percent|%)\b)?")
]

# ---------------------------------------------------------------------
# Step 3: Reasoning classification logic
# ---------------------------------------------------------------------

def infer_answer_type(answer):
    """Infer answer type based on regex patterns (with prioritization)."""
    for atype, pattern in answer_type_patterns:
        if isinstance(pattern, list):
            for p in pattern:
                if re.search(p, str(answer), flags=re.IGNORECASE):
                    return atype
        else:
            if re.search(pattern, str(answer), flags=re.IGNORECASE):
                return atype
    return "Unknown"

def classify_question_and_skills(question, answer):
    """
    Classify question into one of three structural types:
    - Simple: neither Intersection nor Composition
    - Intersection: asks for entities that satisfy multiple conditions simultaneously
    - Composition: requires chaining through intermediate entities (e.g., via 'that', 'whose', etc.)

    Then extract up to 3 reasoning skills and answer type.

    Based on structural question type definitions from TableInstruct (arXiv:2405.07765).
    """

    q_lower = question.lower()
    ans_lower = str(answer).lower()
    qtype = "Simple"
    skills = Counter()

    # Prioritize Composition > Intersection > Simple.

    # --- Refined Composition detection (multi-hop, nested logic) ---
    # Detect questions that involve entity chaining, e.g., "Who directed the film that X wrote?"
    # Use presence of nested clause indicators like "that", "whose", etc.
    # But exclude shallow patterns like "with population" that often indicate filtering instead.
    if (
        re.search(r"\b(who|what|which|how|when|how many)\b", q_lower) and
        re.search(r"\b(that|whose|where|who has|who have|that has|that have)\b", q_lower)
        and len(re.findall(r"\b\w+\b", q_lower)) > 10  # Exclude short questions
    ):
        qtype = "Composition"

    # --- Special Composition logic for "with" ---
    # Only treat "with" as Composition if it is distant from WH-word (non-local constraint)
    elif (
        re.search(r"\b(who|what|which|how|when|how many)\b", q_lower) and
        " with " in q_lower
    ):
        wh_pos = q_lower.find(re.search(r"\b(who|what|which|how|when|how many)\b", q_lower).group())
        with_pos = q_lower.find(" with ")
        if with_pos - wh_pos > 20:  # Only trigger if "with" is far from WH-word
            qtype = "Composition"

    # --- Intersection detection (multiple constraints) ---
    # E.g., "films written and directed by X", "who won in 1951 and 1957"
    elif (
        re.search(r"\bboth\b.*\band\b", q_lower) or
        re.search(r"\b(and)\b", q_lower) and
        len(re.findall(r"\b(in|with|for|of|from|by|at|on|when|where|which)\b", q_lower)) >= 2
    ):
        qtype = "Intersection"

    # --- Additional semantic-based Intersection heuristic ---
    # E.g., "what type of structure was discovered by William"
    elif re.search(r"(discovered by|written by|founded by|built by|composed by)", q_lower) and \
         re.search(r"(type|kind|category|group|genre|form|class|species)", q_lower):
        qtype = "Intersection"

    # --- Domain-based Intersection heuristics ---
    # Look for multiple filter clues like years + sports/event terms
    intersection_filters = [
        r"\b\d{4}\b",  # year
        r"serie a|serie b|league|division|club|team",  # sports domain
        r"match|game|season|round"  # event structure
    ]

    # Only activate if nothing else has set qtype to Intersection yet
    if qtype == "Simple":
        filter_hits = sum(1 for f in intersection_filters if re.search(f, q_lower))
        if filter_hits >= 2 and (
            re.search(r"\band\b", q_lower) or
            len(re.findall(r"\b\d{4}\b", q_lower)) >= 2
        ):
            qtype = "Intersection"

    # --- Pattern-based skill detection ---
    for skill, patterns in skill_patterns.items():
        for pat in patterns:
            if re.search(pat, q_lower):
                skills[skill] += 1
                break

    # --- Additional skill cue ---
    if q_lower.strip().startswith("how many") or " how many " in q_lower or " number of " in q_lower:
        skills["Counting"] += 1

    # --- Power recovery from answer type ---
    answer_type = infer_answer_type(answer)
    if not skills:
        if answer_type == "Number":
            skills["Counting"] += 1
        elif answer_type == "Date":
            skills["Filtering time"] += 1
        else:
            skills["No skill"] += 1

    return qtype, list([k for k, _ in skills.most_common(3)]), answer_type



# ---------------------------------------------------------------------
# Step 4: Main loop to process files in input directory
# ---------------------------------------------------------------------

def process_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "QAS")
    output_dir = os.path.join(base_dir, "question_reason_type")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for filename in tqdm(input_files, desc="Processing files", unit="file"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".csv", "-type.csv"))

        with open(input_path, newline='', encoding='utf-8') as infile, \
             open(output_path, "w", newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader)
            if "question" not in header or "answer" not in header:
                print(f"Skipping {filename} - missing required columns.")
                continue

            q_idx = header.index("question")
            a_idx = header.index("answer")

            # Write header
            writer.writerow(["question", "answer", "question_type", "skills", "answer_type"])

            # Process each row
            for row in tqdm(reader, desc=filename, leave=False):
                question = row[q_idx]
                answer = row[a_idx]
                qtype, skill_list, atype = classify_question_and_skills(question, answer)
                writer.writerow([question, answer, qtype, ";".join(skill_list), atype])

# ---------------------------------------------------------------------
# Step 5: Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    process_files()
