#!/usr/bin/env python3
"""
post-process.py

SUMMARY:
---------
This script preprocesses SPARQL queries stored in CSV files to make them syntactically valid
for DBpedia's SPARQL endpoint. It does NOT add PREFIX declarations (keeps queries prefix-free),
but performs the following key repairs:

1. Converts square brackets [ ... ] into curly braces { ... } (valid SPARQL syntax).
2. Fixes illegal CURIE local parts (e.g., dbr:Kot:Rock_Standard → dbr:Kot_Rock_Standard).
3. Repairs malformed triple patterns:
   - Adds missing objects if a triple has only subject + predicate.
   - Splits stacked predicates like:  S P1 P2 O  →  S P1 ?v . ?v P2 O.
   - Fixes quadruple token case like: S P1 P2 S  →  two triples with fresh variable.
4. Normalizes whitespace and keywords (SELECT, ASK, ORDER BY, LIMIT).
5. Ensures braces are balanced so queries parse correctly.
6. Fixes ORDER BY and LIMIT spacing so Virtuoso accepts them.

Output: A new folder "post-T5-small-qald9" with the same CSV filenames,
but with all SPARQL queries repaired.
"""

import re
from pathlib import Path
import pandas as pd

# --- CONFIGURATION ---
INPUT_DIR  = Path("T5-small-qald9")       # Input folder containing original CSVs
OUTPUT_DIR = Path("post-T5-small-qald9") # Output folder for processed CSVs
SPARQL_COL = "sparql"                    # Column containing SPARQL queries
# ---------------------

# Regex to convert [...] → {...} in SPARQL queries (multiline safe)
BRACKET_BLOCK = re.compile(r"\[(.*?)\]", flags=re.DOTALL)

# Regex to find CURIEs (prefix:localpart) so we can sanitize illegal ':' inside local part
CURIE_RX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*):([^\s\.\,\;\)\}\{]+)")

def split_clauses(where_body: str):
    """
    Split a WHERE body into top-level clauses on '.' but ignore '.' inside braces/parens.
    This avoids breaking up subqueries, FILTERs, etc.
    """
    out, buf, depth = [], [], 0
    for ch in where_body:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        if ch == '.' and depth == 0:
            clause = ''.join(buf).strip()
            if clause:
                out.append(clause)
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        out.append(tail)
    return out

def sanitize_curie_localparts(text: str) -> str:
    """
    Fix CURIEs where local parts contain illegal characters like ':' or spaces.
    Example: dbr:Kot:Rock_Standard → dbr:Kot_Rock_Standard
    """
    def _fix(m):
        prefix = m.group(1)
        local  = m.group(2)
        fixed = local.replace(":", "_").replace(" ", "_")
        return f"{prefix}:{fixed}"
    return CURIE_RX.sub(_fix, text)

def extract_where_body(query: str):
    """
    Extract the WHERE clause body content between { ... }.
    Returns (head, body, tail) where:
    - head: everything before the body content
    - body: inside of {...}
    - tail: everything after }
    """
    m = re.search(r"(?is)where\s*\{(.*)\}\s*", query)
    if not m:
        m = re.search(r"(?is)\{(.*)\}\s*", query)
    if not m:
        return None, None, None
    start = m.start(1)
    end   = m.end(1)
    return query[:start], m.group(1), query[end:]

def smart_repair_where(where_body: str) -> str:
    """
    Apply structural repairs to the WHERE body:
      * Add fresh variables when objects are missing.
      * Split stacked predicates or quadruple-token clauses into two triples.
    """
    clauses = split_clauses(where_body)
    vcounter = 0

    def fresh_var():
        """Generate a fresh variable name like ?v1, ?v2 ..."""
        nonlocal vcounter
        vcounter += 1
        return f"?v{vcounter}"

    repaired = []
    for cl in clauses:
        tokens = re.findall(r"[^\s]+", cl.strip())
        if not tokens:
            continue

        # Skip FILTER, OPTIONAL, etc. — leave them untouched
        if re.search(r"(?i)^(filter|optional|values|bind|minus|service|graph)\b", tokens[0]):
            repaired.append(cl.strip())
            continue

        # Strip stray trailing punctuation like ';' or ','
        while tokens and tokens[-1] in {';', ','}:
            tokens = tokens[:-1]

        if len(tokens) == 2:
            # Triple missing object: add fresh variable
            s, p = tokens
            repaired.append(f"{s} {p} {fresh_var()}")
            continue

        if len(tokens) == 4:
            # Case with stacked predicates: S P1 P2 O
            s, p1, p2, o = tokens
            mid = fresh_var()
            repaired.append(f"{s} {p1} {mid}")
            repaired.append(f"{mid} {p2} {o}")
            continue

        # Otherwise, keep triple as-is
        repaired.append(' '.join(tokens))

    # Join all repaired clauses with " . " as SPARQL expects
    return ' . '.join(x.strip() for x in repaired if x.strip())

def normalize_query_text(q: str) -> str:
    """
    Normalize whitespace and SPARQL keywords for consistency.
    Also fixes ORDER BY and LIMIT spacing.
    """
    q = re.sub(r"\s+", " ", q.strip())                     # collapse whitespace
    q = re.sub(r"(?i)\bselect\b", "SELECT", q)
    q = re.sub(r"(?i)\bask\b", "ASK", q)
    q = re.sub(r"(?i)\border\s+by\b", "ORDER BY", q)
    q = re.sub(r"(?i)\blimit\b", "LIMIT", q)
    q = re.sub(r"\s*\{\s*", " { ", q)
    q = re.sub(r"\s*\}\s*", " } ", q)
    # Fix ORDER BY DESC and LIMIT spacing to match Virtuoso expectations
    q = re.sub(r"ORDER BY\s*DESC\s*\(", "ORDER BY DESC(", q)
    q = re.sub(r"\)\s*limit", ") LIMIT", q, flags=re.IGNORECASE)
    return q.strip()

def fix_query(q: str) -> str:
    """
    Full pipeline to repair a single SPARQL query.
    """
    if not isinstance(q, str):
        return q

    # Step 1: Convert [...] to {...}
    q = BRACKET_BLOCK.sub(r"{\1}", q)

    # Step 2: Normalize 'where' to uppercase to help our regex
    q = re.sub(r"(?i)\bwhere\b", "WHERE", q)

    # Step 3: Sanitize CURIE local parts with illegal characters
    q = sanitize_curie_localparts(q)

    # Step 4: Extract and repair WHERE body if present
    head, body, tail = extract_where_body(q)
    if body is not None:
        repaired_body = smart_repair_where(body)
        q = f"{head}{repaired_body}{tail}"

    # Step 5: Normalize whitespace, keywords, ORDER/LIMIT syntax
    q = normalize_query_text(q)

    # Step 6: Ensure balanced braces (add closing ones if missing)
    opens = q.count("{")
    closes = q.count("}")
    if opens > closes:
        q = q + " }" * (opens - closes)

    return q

def main():
    # Check input folder exists
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {INPUT_DIR.resolve()}")

    total_files = 0
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] {csv_path.name}: failed to read CSV ({e})")
            continue

        if SPARQL_COL not in df.columns:
            print(f"[SKIP] {csv_path.name}: no '{SPARQL_COL}' column")
            continue

        # Apply repair pipeline to SPARQL column
        before = df[SPARQL_COL].astype(str).copy()
        df[SPARQL_COL] = df[SPARQL_COL].apply(fix_query)
        changed = (before != df[SPARQL_COL].astype(str)).sum()
        print(f"[OK]   {csv_path.name}: updated {changed} row(s)")

        # Save repaired CSV to output folder
        out_path = OUTPUT_DIR / csv_path.name
        df.to_csv(out_path, index=False, encoding="utf-8")
        total_files += 1

    print(f"Done. Wrote {total_files} file(s) to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
