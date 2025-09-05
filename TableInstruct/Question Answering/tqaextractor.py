# ============================================================================================
# Script: JSON → CSV (Unified 3 columns: question, answer, table)
# --------------------------------------------------------------------------------------------
# - All outputs: columns are exactly ["question", "answer", "table"]
# - FETAQA: Prepend "Wikipedia page title" and "Wikipedia section title" at the START of the
#           table column, then append the raw [TAB]... table content.
# - HiTab:  Prepend "Table caption" at the START of the table column, then append raw [TAB]...
# - HybridQA/Standard: table column is just the raw [TAB]... content.
# - Table content is kept RAW from [TAB] onward (no cleaning). No comma stripping anywhere.
# - CSV uses QUOTE_MINIMAL with lineterminator="\n" for VS Code/Rainbow CSV friendliness.
# - Outputs go to ./TQAS with matching base filenames (.csv).
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

# ------------------------------- small helpers ----------------------------------

def _collapse_ws(s: str) -> str:
    """Collapse internal whitespace (do not touch commas)."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def _strip_double_quotes(s: str) -> str:
    """Remove double quotes inside fields to keep CSV simpler."""
    if not isinstance(s, str):
        return ""
    return s.replace('"', '')

def _sanitize_field(s: str) -> str:
    """Normalize fields: remove internal double quotes + collapse whitespace."""
    return _collapse_ws(_strip_double_quotes(s))

# ------------------------ dataset-specific metadata parse ------------------------

def parse_fetaqa_titles(input_seg: str):
    """
    Extracts wikipedia page and section titles from FETAQA input_seg preamble.
    Returns (page_title, section_title).
    """
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
    """
    Extracts HiTab table caption from input_seg preamble.
    """
    caption = ""
    if not isinstance(input_seg, str):
        return caption
    m = re.search(r"The table caption is\s+(.*?)\.", input_seg, flags=re.IGNORECASE | re.DOTALL)
    if m:
        caption = _sanitize_field(m.group(1))
    return caption

# ------------------------ question/answer extractors -----------------------------

def extract_question_answer(item: dict, dataset_tag: str):
    """
    Normalizes Q/A for different datasets (no comma stripping).
    """
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

# ------------------------ table extractor & prefixers ----------------------------

def extract_table_raw(input_seg: str) -> str:
    """
    Extract raw table substring starting at [TAB]. No cleaning or comma removal.
    """
    if not isinstance(input_seg, str):
        return ""
    tab_idx = input_seg.find("[TAB]")
    return input_seg[tab_idx:] if tab_idx != -1 else ""

def build_table_with_prefix(dataset_tag: str, input_seg: str) -> str:
    """
    Build the final table column value for each dataset:
      - FETAQA: "[Wikipedia page title:{...}; Wikipedia section title:{...}; ] " + RAW_TABLE
      - HiTab : "[Table caption:{...}; ] " + RAW_TABLE
      - Else  : RAW_TABLE
    """
    raw_table = extract_table_raw(input_seg)

    if dataset_tag == "fetaqa":
        page_title, section_title = parse_fetaqa_titles(input_seg)
        prefix_parts = []
        if page_title:
            prefix_parts.append(f"Wikipedia page title:{page_title}")
        if section_title:
            prefix_parts.append(f"Wikipedia section title:{section_title}")
        if prefix_parts:
            prefix = "[" + "; ".join(prefix_parts) + "; ] "
            return prefix + raw_table if raw_table else prefix.strip()
        return raw_table

    if dataset_tag == "hitab":
        caption = parse_hitab_caption(input_seg)
        if caption:
            prefix = f"[Table caption:{caption}; ] "
            return prefix + raw_table if raw_table else prefix.strip()
        return raw_table

    # standard & hybridqa
    return raw_table

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
        table_col = build_table_with_prefix(dataset_tag, input_seg)

        rows.append([q, a, table_col])

    # Unified header for all datasets
    header = ["question", "answer", "table"]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(filename).replace(".json", ".csv"))

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        # QUOTE_MINIMAL quotes fields only when needed (commas, quotes, newlines).
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
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
