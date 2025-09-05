# ============================================================================================
# Script: JSON → CSV (Q/A/Table + dataset-specific extras)
# --------------------------------------------------------------------------------------------
# Changes in this version:
# - Table column: keep raw from input_seg (starting at [TAB]), only remove commas.
# - All columns: commas removed, to prevent accidental extra CSV columns.
# ============================================================================================

import os
import re
import csv
import json

INPUT_FILES = [
    ("fetaqa_train_7325.json", "fetaqa"),
    ("fetaqa_test.json", "fetaqa"),
    ("hitab_test.json", "hitab"),
    ("wikisql_test.json", "standard"),
    ("wikitq_test.json", "standard"),
    ("hitab_train_7417.json", "hitab"),
    ("hybridqa_eval.json", "hybridqa"),
]

OUTPUT_DIR = "TQAS"

# ------------------------------- helpers -----------------------------------------

def _strip_commas(s: str) -> str:
    """Remove commas and surrounding whitespace."""
    return s.replace(",", "") if isinstance(s, str) else s

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else s

def _sanitize_field(s: str) -> str:
    """Remove commas + collapse whitespace."""
    return _collapse_ws(_strip_commas(s))

# ------------------------ dataset-specific metadata parse ------------------------

def parse_fetaqa_titles(input_seg: str):
    page_title, section_title = "", ""
    if not isinstance(input_seg, str):
        return page_title, section_title

    m_page = re.search(
        r"The Wikipedia page title of this table is\s+(.*?)\.",
        input_seg, flags=re.IGNORECASE | re.DOTALL
    )
    if m_page:
        page_title = _sanitize_field(m_page.group(1))

    m_sec = re.search(
        r"The Wikipedia section title of this table is\s+(.*?)\.",
        input_seg, flags=re.IGNORECASE | re.DOTALL
    )
    if m_sec:
        section_title = _sanitize_field(m_sec.group(1))

    return page_title, section_title

def parse_hitab_caption(input_seg: str):
    caption = ""
    if not isinstance(input_seg, str):
        return caption
    m = re.search(r"The table caption is\s+(.*?)\.", input_seg, flags=re.IGNORECASE | re.DOTALL)
    if m:
        caption = _sanitize_field(m.group(1))
    return caption

# ------------------------ question/answer extractors -----------------------------

def extract_question_answer(item: dict, dataset_tag: str):
    q_raw = item.get("question", "")
    a_raw = item.get("output", "")

    if dataset_tag == "fetaqa":
        if isinstance(q_raw, str) and "[HIGHLIGHTED_END]" in q_raw:
            question = q_raw.split("[HIGHLIGHTED_END]", 1)[1].strip()
        else:
            return None, None
    elif dataset_tag == "hybridqa":
        if isinstance(q_raw, str) and "The question:" in q_raw:
            question = q_raw.split("The question:", 1)[1].strip()
        else:
            return None, None
    else:
        question = q_raw if isinstance(q_raw, str) else ""

    question = _sanitize_field(question)
    answer   = _sanitize_field(a_raw if isinstance(a_raw, str) else "")

    if not question or not answer:
        return None, None
    return question, answer

# ------------------------ table extractor ---------------------------------------

def extract_table_raw(input_seg: str) -> str:
    """
    Extract table substring starting at [TAB], do not clean,
    only remove commas.
    """
    if not isinstance(input_seg, str):
        return ""
    tab_idx = input_seg.find("[TAB]")
    if tab_idx == -1:
        return ""
    table = input_seg[tab_idx:]  # raw table text including [TAB]
    return _strip_commas(table)

# ---------------------------------- main ----------------------------------------

def process_file(filename: str, dataset_tag: str, out_dir: str):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        q, a = extract_question_answer(item, dataset_tag)
        if q is None:
            continue

        input_seg = item.get("input_seg", "")
        table_str = extract_table_raw(input_seg)

        row = [q, a, table_str]

        if dataset_tag == "fetaqa":
            page_title, section_title = parse_fetaqa_titles(input_seg)
            row.extend([page_title, section_title])
        elif dataset_tag == "hitab":
            caption = parse_hitab_caption(input_seg)
            row.append(caption)

        rows.append(row)

    # Choose header
    header = ["question", "answer", "table"]
    if dataset_tag == "fetaqa":
        header += ["wikipedia_page_title", "wikipedia_section_title"]
    elif dataset_tag == "hitab":
        header += ["table_caption"]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(filename).replace(".json", ".csv"))

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved: {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname, tag in INPUT_FILES:
        if os.path.exists(fname):
            process_file(fname, tag, OUTPUT_DIR)
        else:
            print(f"Skip (not found): {fname}")
