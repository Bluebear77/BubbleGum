#!/usr/bin/env python3
"""
post-process.py

SUMMARY
-------
Repairs SPARQL queries to avoid Virtuoso SP030 parse errors:
- Convert [...] → {...}
- Sanitize CURIEs (replace ':' in local part with '_')
- Map undefined prefixes (p:, wdt:) → dbo:, (wd:) → dbr:
- Remove bad characters like '&', stray '>' not in FILTER
- Fix unterminated or unmatched quotes
- Drop orphan dots, empty triple patterns, and stray numbers/words
- Add missing object variables and split stacked predicates
- Normalize ORDER BY / LIMIT syntax
- Ensure balanced braces
"""

import re
from pathlib import Path
import pandas as pd

INPUT_DIR  = Path("T5-small-qald9")
OUTPUT_DIR = Path("post-T5-small-qald9")
SPARQL_COL = "sparql"

BRACKET_BLOCK = re.compile(r"\[(.*?)\]", flags=re.DOTALL)
CURIE_RX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*):([^\s\.\,\;\)\}\{]+)")

def split_clauses(where_body: str):
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
    def _fix(m):
        prefix = m.group(1)
        local  = m.group(2)
        fixed = local.replace(":", "_").replace(" ", "_")
        return f"{prefix}:{fixed}"
    return CURIE_RX.sub(_fix, text)

def extract_where_body(query: str):
    m = re.search(r"(?is)where\s*\{(.*)\}\s*", query)
    if not m:
        m = re.search(r"(?is)\{(.*)\}\s*", query)
    if not m:
        return None, None, None
    return query[:m.start(1)], m.group(1), query[m.end(1):]

def clean_bad_tokens(text: str) -> str:
    # Remove illegal characters
    text = text.replace("&", " ")
    # Remove lone > not part of FILTER
    text = re.sub(r"(?<![<>=!])>+", " ", text)
    # Remove unmatched single quotes
    text = re.sub(r"'([^']*)\n", r"\1 ", text)
    text = text.replace("'", "")
    return text

def map_prefixes(text: str) -> str:
    text = re.sub(r"\bwdt:", "dbo:", text)
    text = re.sub(r"\bp:", "dbo:", text)
    text = re.sub(r"\bwd:", "dbr:", text)
    return text

def smart_repair_where(where_body: str) -> str:
    clauses = split_clauses(where_body)
    vcounter = 0
    def fresh_var():
        nonlocal vcounter
        vcounter += 1
        return f"?v{vcounter}"

    repaired = []
    for cl in clauses:
        cl = cl.strip()
        if not cl or cl == ".":
            continue
        tokens = re.findall(r"[^\s]+", cl)
        # Drop lone tokens that are just numbers or words
        if len(tokens) == 1 and not tokens[0].startswith("?"):
            continue
        if len(tokens) == 2:
            repaired.append(f"{tokens[0]} {tokens[1]} {fresh_var()}")
            continue
        if len(tokens) == 4:
            mid = fresh_var()
            repaired.append(f"{tokens[0]} {tokens[1]} {mid}")
            repaired.append(f"{mid} {tokens[2]} {tokens[3]}")
            continue
        repaired.append(' '.join(tokens))
    # Remove duplicate dots and clean trailing dots
    out = ' . '.join(x for x in repaired if x)
    out = re.sub(r"\.\s*\.", ".", out)
    return out.strip()

def normalize_query_text(q: str) -> str:
    q = re.sub(r"\s+", " ", q.strip())
    q = re.sub(r"(?i)\bselect\b", "SELECT", q)
    q = re.sub(r"(?i)\bask\b", "ASK", q)
    q = re.sub(r"(?i)\border\s+by\b", "ORDER BY", q)
    q = re.sub(r"(?i)\blimit\b", "LIMIT", q)
    q = re.sub(r"\s*\{\s*", " { ", q)
    q = re.sub(r"\s*\}\s*", " } ", q)
    q = re.sub(r"ORDER BY\s*DESC\s*\(", "ORDER BY DESC(", q)
    q = re.sub(r"\)\s*limit", ") LIMIT", q, flags=re.IGNORECASE)
    return q.strip()

def fix_query(q: str) -> str:
    if not isinstance(q, str):
        return q
    q = BRACKET_BLOCK.sub(r"{\1}", q)
    q = clean_bad_tokens(q)
    q = map_prefixes(q)
    q = sanitize_curie_localparts(q)
    head, body, tail = extract_where_body(q)
    if body is not None:
        body = smart_repair_where(body)
        q = f"{head}{body}{tail}"
    q = normalize_query_text(q)
    opens, closes = q.count("{"), q.count("}")
    if opens > closes:
        q += " }" * (opens - closes)
    return q

def main():
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
        before = df[SPARQL_COL].astype(str).copy()
        df[SPARQL_COL] = df[SPARQL_COL].apply(fix_query)
        changed = (before != df[SPARQL_COL].astype(str)).sum()
        print(f"[OK] {csv_path.name}: updated {changed} row(s)")
        out_path = OUTPUT_DIR / csv_path.name
        df.to_csv(out_path, index=False, encoding="utf-8")
        total_files += 1
    print(f"Done. Wrote {total_files} file(s) to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
