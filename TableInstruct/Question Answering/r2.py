# =============================================================================
# Script Summary:
# - Reads *.json files in the working directory.
# - Cleans table segments and questions.
# - Extracts the answer directly from the "output" field.
# - Groups ALL (question, answer) pairs by their associated table (after cleaning).
# - For each table group, builds a cleaned plain-text "qas" block by concatenating
#   question and answer (punctuation removed, whitespace normalized).
# - Computes semantic similarity (SentenceTransformer) between each table and
#   its "qas" block — batched with progress bars.
# - Saves:
#     1) relatedness_outputs/<file>.grouped.csv      (columns: similarity_score,num_pairs,qas,table,source)
#     2) relatedness_outputs/_lowest100.csv          (100 lowest across ALL files; columns ordered: source,similarity_score,num_pairs,qas,table)
#
# Speed/robustness tweaks:
#   - TOKENIZERS_PARALLELISM=false (avoid rare hangs)
#   - model.max_seq_length=256 (faster; adjust as needed)
#   - Auto-select CUDA if available
#
# Requirements:
#   pip install sentence-transformers pandas tqdm
# =============================================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid rare tokenizer hangs

import json
import re
import string
import pandas as pd
from tqdm import tqdm
import csv
import torch
from sentence_transformers import SentenceTransformer

# --------------------------- Paths & Model -----------------------------------
input_dir = '.'
output_dir = 'v2-relatedness_outputs'
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model.max_seq_length = 256  # speed up (default 512). Lower (e.g., 192) for more speed.

# --------------------------- Cleaning Helpers --------------------------------
def clean_input_seg(text: str) -> str:
    """Clean table segment text (remove prompts/tokens, collapse spaces)."""
    if not isinstance(text, str):
        return ""
    patterns = [
        r"\[TLE\]\s*The Wikipedia page title of this table is",
        r"The Wikipedia section title of this table is",
        r"\[TLE\]\s*The table caption is",
        r"\[TAB\]",
        r"\[TAB\]\s*col:",
        r"\bcol:\b",
        r"\|",
        r"\\",             # backslashes
        r"\[SEP\]",
        r"[TAB]\s*col:",
        r"col:",
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = text.replace('"', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_line(text: str) -> str:
    """Basic whitespace cleanup."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_punctuation(text: str) -> str:
    """Remove punctuation for embedding normalization / QAS output."""
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text or "")

# --------------------------- Extraction Helpers ------------------------------
def extract_question(entry: dict, filename: str) -> str | None:
    question_text = entry.get("question", "")
    if not isinstance(question_text, str):
        return None

    fname = filename.lower()
    if "fetaqa" in fname:
        if "[HIGHLIGHTED_END]" in question_text:
            question = question_text.split("[HIGHLIGHTED_END]", 1)[1]
        else:
            return None
    elif "hybridqa" in fname:
        if "The question:" in question_text:
            question = question_text.split("The question:", 1)[1]
        else:
            return None
    else:
        question = question_text

    question = question.replace('"', '')
    question = re.sub(r'\s+', ' ', question)
    return question.strip()

def extract_answer(entry: dict) -> str:
    """
    Extract the answer directly from the 'output' field.
    Handles common shapes: string, dict with 'answer', list with first item, etc.
    """
    out = entry.get("output", "")
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        for k in ("answer", "final_answer", "prediction", "result"):
            if k in out and isinstance(out[k], str):
                return out[k].strip()
        return clean_line(json.dumps(out, ensure_ascii=False))
    if isinstance(out, list):
        if out and isinstance(out[0], str):
            return out[0].strip()
        return clean_line(json.dumps(out, ensure_ascii=False))
    return clean_line(str(out))

# --------------------------- Main --------------------------------------------
all_rows = []  # accumulate rows from ALL files (for global _lowest100.csv)

json_files = [fn for fn in os.listdir(input_dir) if fn.endswith('.json')]
if not json_files:
    print("⚠️ No .json files found in the current directory.")
else:
    for filename in json_files:
        input_path = os.path.join(input_dir, filename)
        base = os.path.splitext(filename)[0]  # NAME without extension (used as 'source')
        output_all = os.path.join(output_dir, f"{base}.grouped.csv")

        print(f"🔍 Processing {filename}...")

        # Load JSON
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Skipping {filename}: invalid JSON format")
            continue
        if not isinstance(data, list):
            print(f"⚠️ Skipping {filename}: expected a list of entries")
            continue

        # 1) Aggregate Q/A pairs by CLEANED table
        groups = {}  # key: clean_seg, value: list of cleaned "qa piece" strings (punctuation removed)
        for entry in tqdm(data, desc=f"→ reading {filename}", leave=False):
            raw_seg = entry.get("input_seg", "")
            clean_seg = clean_input_seg(raw_seg)
            if not clean_seg:
                continue

            q = extract_question(entry, filename)
            if not q:
                continue
            a = extract_answer(entry)

            # Build a per-pair cleaned "qa piece": (Q + A), punctuation removed, whitespace normalized
            q_disp = clean_line(q)
            a_disp = clean_line(a)
            qa_piece = clean_line(remove_punctuation(f"{q_disp} {a_disp}"))

            groups.setdefault(clean_seg, []).append(qa_piece)

        if not groups:
            print(f"⚠️ No valid groups found in {filename}")
            continue

        # 2) Build texts for batched encoding
        tables = []
        qas_blocks = []
        metas = []  # (clean_seg, qas_block, num_pairs)

        for clean_seg, qa_pieces in groups.items():
            # Join all QA pieces into one cleaned block (space-separated)
            qas_block = clean_line(" ".join(qa_pieces))  # punctuation already removed per piece
            tables.append(remove_punctuation(clean_seg))  # embeddings use no punctuation
            qas_blocks.append(qas_block)                  # already punctuation-free
            metas.append((clean_seg, qas_block, len(qa_pieces)))

        # 3) Batched embeddings with progress bars
        seg_emb = model.encode(
            tables,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True
        )
        qas_emb = model.encode(
            qas_blocks,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True
        )

        # Cosine similarity = dot product for normalized embeddings
        sims = (seg_emb * qas_emb).sum(dim=1).tolist()

        # 4) Collect rows (per-file & global) — WITH 'source' column
        rows = []
        for (clean_seg, qas_block, num_pairs), sim in zip(metas, sims):
            row = {
                "similarity_score": sim,
                "num_pairs": num_pairs,
                "qas": qas_block,   # punctuation-free concatenation of Q & A pairs
                "table": clean_seg,
                "source": base      # origin from NAME.json (used in NAME.grouped.csv)
            }
            rows.append(row)
            all_rows.append(row)

        # 5) Save per-file grouped CSV (include source; keep similarity first for sorting)
        df = pd.DataFrame(rows).sort_values("similarity_score", ascending=False)
        df = df[["similarity_score", "num_pairs", "qas", "table", "source"]]
        df.to_csv(output_all, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"✅ Saved: {output_all}")

# --------------------------- Global Lowest 100 -------------------------------
if all_rows:
    df_all = pd.DataFrame(all_rows)
    # Reorder columns so 'source' is the FIRST column in _lowest100.csv
    df_all = df_all[["source", "similarity_score", "num_pairs", "qas", "table"]]
    df_low_global = df_all.sort_values("similarity_score", ascending=True).head(100)
    global_path = os.path.join(output_dir, "_lowest100.csv")
    df_low_global.to_csv(global_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Saved global lowest-100: {global_path}")
else:
    print("⚠️ No rows produced; global _lowest100.csv not created.")
