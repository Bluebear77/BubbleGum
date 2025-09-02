#!/usr/bin/env python3
"""
dbpedia_query_runner.py

SUMMARY
-------
This script iterates through every CSV file in the `post-T5-small-qald9` folder,
executes each SPARQL query in the column `sparql` against the official DBpedia endpoint,
and records the results.

For each row:
- If the query executes successfully and returns one or more ?obj values → store them (joined by ";").
- If the query executes successfully but matches no ?obj → record "empty".
- If the query fails (HTTP error, timeout, parse error, etc.) → record NaN in obj_values and store the error message.

Output:
- A new folder `DB-T5-small-qald9` with exactly the same filenames as the input.
- Each output CSV will contain two additional columns: `obj_values` and `error`.
- A Markdown file `summary.md` will be created containing:
  * Summary counts/percentages.
  * A Markdown table of distinct error messages (one row per unique error).
"""

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
INPUT_DIR   = Path("post-T5-small-qald9")  # Input folder: contains repaired CSVs
OUTPUT_DIR  = Path("DB-T5-small-qald9")    # Output folder: results will be saved here
SUMMARY_FILE = Path("summary.md")          # File where summary + error table will be written
SPARQL_COL  = "sparql"                     # Column containing SPARQL queries
ENDPOINT    = "https://dbpedia.org/sparql" # DBpedia SPARQL endpoint
TIMEOUT_SEC = 25                           # Per-request timeout (seconds)
PAUSE_BETWEEN_REQUESTS = 0.05              # Small delay between queries to avoid overloading endpoint
# ------------------------------------------------

def make_session():
    """
    Create a requests.Session configured with:
      * Automatic retries (5 total) on transient HTTP errors (429/500/502/503/504).
      * Exponential backoff (0.8s, 1.6s, 3.2s, ...).
      * 'Accept: application/sparql-results+json' header to request JSON results.
    """
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Accept": "application/sparql-results+json"})
    return s

def run_sparql(session: requests.Session, query: str):
    """
    Execute a SPARQL query on DBpedia and return:
        (obj_values: list[str] | None, error: str | None)

    - obj_values: a list of ?obj bindings (deduplicated, preserving order)
    - If query succeeds but no ?obj found → return [] (empty list), error=None
    - If query fails (network/HTTP/SPARQL parse error) → return None, error="description"
    """
    if not isinstance(query, str) or not query.strip():
        return None, "Empty or invalid query string"

    try:
        params = {
            "query": query,
            "format": "application/sparql-results+json",
        }
        r = session.get(ENDPOINT, params=params, timeout=TIMEOUT_SEC)
    except requests.RequestException as e:
        return None, f"Network error: {type(e).__name__}: {e}"

    # If DBpedia returned an HTTP error code, log the message body (truncated)
    if r.status_code != 200:
        snippet = r.text.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:180] + "…"
        return None, f"HTTP {r.status_code}: {snippet}"

    # Parse JSON response
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        txt = r.text.strip().replace("\n", " ")
        if len(txt) > 180:
            txt = txt[:180] + "…"
        return None, f"Invalid JSON: {type(e).__name__}: {txt}"

    results = data.get("results", {})
    bindings = results.get("bindings", [])
    if not isinstance(bindings, list):
        return None, "Unexpected results format"

    # Collect all values of ?obj if they exist
    obj_vals = []
    for b in bindings:
        if "obj" in b and "value" in b["obj"]:
            obj_vals.append(b["obj"]["value"])

    # Deduplicate while preserving order
    seen = set()
    unique_vals = []
    for v in obj_vals:
        if v not in seen:
            seen.add(v)
            unique_vals.append(v)

    return unique_vals, None

def process_file(session: requests.Session, in_path: Path, out_path: Path, counters: Counter, error_examples: set):
    """
    Process a single CSV file:
      * Read CSV.
      * Execute each SPARQL query.
      * Record results in new columns: obj_values, error.
      * Collect one example of each unique error type in error_examples.
      * Update counters for final percentage summary.
      * Write the processed CSV to the output path.
    """
    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print(f"[SKIP] {in_path.name}: failed to read CSV ({e})")
        return False, 0

    if SPARQL_COL not in df.columns:
        print(f"[SKIP] {in_path.name}: missing '{SPARQL_COL}' column")
        return False, 0

    # Ensure result columns exist (create if missing)
    if "obj_values" not in df.columns:
        df["obj_values"] = pd.NA
    if "error" not in df.columns:
        df["error"] = ""

    iter_rows = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc=f"{in_path.name}",
        leave=False,
    )

    obj_values_col = []
    error_col = []
    processed_rows = 0

    for row in iter_rows:
        query = getattr(row, SPARQL_COL, None)
        obj_vals, err = run_sparql(session, query)

        if err is not None:
            obj_values_col.append(pd.NA)
            error_col.append(err)
            counters["error_nan"] += 1
            error_examples.add(err)  # collect unique error type
        else:
            if len(obj_vals) == 0:
                obj_values_col.append("empty")
                error_col.append("")
                counters["empty"] += 1
            else:
                obj_values_col.append(";".join(obj_vals))
                error_col.append("")
                counters["non_empty"] += 1

        processed_rows += 1
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    df["obj_values"] = obj_values_col
    df["error"] = error_col

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return True, processed_rows

def generate_summary_md(total_rows: int, counters: Counter, error_examples: set):
    """
    Create a markdown-formatted string with:
      * Summary counts and percentages.
      * A table of distinct error messages (one example per type).
    """
    lines = []
    lines.append("# DBpedia Query Summary\n")
    lines.append(f"**Total rows processed:** {total_rows}\n")

    if total_rows > 0:
        def pct(n): return (n / total_rows) * 100.0
        lines.append(f"- **Non-empty results:** {counters.get('non_empty', 0)} ({pct(counters.get('non_empty', 0)):.2f}%)")
        lines.append(f"- **Empty results:** {counters.get('empty', 0)} ({pct(counters.get('empty', 0)):.2f}%)")
        lines.append(f"- **Errors (NaN):** {counters.get('error_nan', 0)} ({pct(counters.get('error_nan', 0)):.2f}%)\n")
    else:
        lines.append("No rows processed.\n")

    if error_examples:
        lines.append("## Distinct Error Types\n")
        lines.append("| # | Error Message |\n|---|---------------|")
        for i, e in enumerate(sorted(error_examples), start=1):
            safe_err = e.replace("|", "\\|")  # escape pipe for markdown table
            lines.append(f"| {i} | {safe_err} |")

    return "\n".join(lines)

def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {INPUT_DIR.resolve()}")

    session = make_session()
    ok_files = 0
    total_rows = 0
    counters = Counter()
    error_examples = set()  # store unique error messages

    for csv_path in tqdm(csv_files, desc="Files"):
        out_path = OUTPUT_DIR / csv_path.name
        ok, rows = process_file(session, csv_path, out_path, counters, error_examples)
        if ok:
            ok_files += 1
            total_rows += rows

    print(f"Done. Wrote {ok_files} file(s) to {OUTPUT_DIR.resolve()}")

    summary_text = generate_summary_md(total_rows, counters, error_examples)
    print(summary_text)  # still print to console for convenience

    # Save to summary.md
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Summary saved to {SUMMARY_FILE.resolve()}")

if __name__ == "__main__":
    main()
