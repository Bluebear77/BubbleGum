#!/usr/bin/env python3
"""
post-process.py

SUMMARY
-------
Repairs model-generated SPARQL so DBpedia (Virtuoso) accepts it by fixing frequent syntax issues:

1) Convert any square-bracket WHERE blocks to curly braces and normalize stray '[' / ']'.
2) Sanitize CURIE localparts (replace inner ':' or spaces with '_').
3) Map undefined prefixes seen in data to DBpedia-style ones (no PREFIX lines added):
      wdt:, p:, ps:  -> dbo:   (properties)
      wd:            -> dbr:   (entities)
4) Clean illegal tokens & quoting:
   - remove stray '&' or '>' (outside operators),
   - close/remove unmatched single quotes,
   - quote clock-like literals like 12:00,
   - reduce contains(a,b,...) to contains(a,b),
   - normalize 'AS alias' -> 'AS ?alias'.
5) WHERE-body structural fixes:
   - remove orphan dots / empty clauses,
   - drop clauses whose subject looks invalid (not ?var / CURIE / <IRI>),
   - add missing object (?vN) when only S P,
   - split stacked predicates S P1 P2 O -> S P1 ?v . ?v P2 O,
   - collapse duplicated adjacent variables (?x ?x -> ?x).
6) Normalize ORDER BY / LIMIT spacing & case.
7) Re-balance braces '{' '}' and parentheses '(' ')' (fix both under/over-count).
8) Leave semantics as-is; only syntactic, structural, and token-level repairs.

Output CSVs land in 'post-T5-small-qald9' with the same filenames.
"""

import re
from pathlib import Path
import pandas as pd

# --- CONFIG ---
INPUT_DIR  = Path("T5-small-qald9")
OUTPUT_DIR = Path("post-T5-small-qald9")
SPARQL_COL = "sparql"
# -------------

# Convert [...] => {...} (greedy across lines)
BRACKET_BLOCK = re.compile(r"\[(.*?)\]", flags=re.DOTALL)

# CURIE detection: prefix:localpart  (we later sanitize localparts)
CURIE_RX = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*):([^\s\.\,\;\)\}\{]+)")

# Simple helpers
WS = r"[ \t\r\n]+"
NONWS = r"[^\s]"

def replace_square_brackets_globally(q: str) -> str:
    """
    Primary conversion of bracket blocks, then a safety net:
    aggressively replace stray '[' with '{' and stray ']' with '}'.
    """
    q = BRACKET_BLOCK.sub(r"{\1}", q)
    # If 'WHERE [' appears, make sure we normalize it (addresses error #1)
    q = re.sub(r"(?i)\bWHERE\s*\[", "WHERE {", q)
    # Safety: replace any remaining '[' or ']' with braces
    q = q.replace("[", "{").replace("]", "}")
    return q

def sanitize_curie_localparts(text: str) -> str:
    """
    Replace illegal ':' or spaces inside the local part of CURIEs, e.g.:
      dbr:Kot:Rock_Standard -> dbr:Kot_Rock_Standard
    """
    def _fix(m):
        prefix = m.group(1)
        local  = m.group(2)
        fixed = local.replace(":", "_").replace(" ", "_")
        return f"{prefix}:{fixed}"
    return CURIE_RX.sub(_fix, text)

def map_unknown_prefixes(text: str) -> str:
    """
    Map Wikidata-like prefixes to DBpedia-ish ones (without adding PREFIX lines):
      wdt:, p:, ps: -> dbo:   | wd: -> dbr:
    (This makes tokens syntactically valid in DBpedia, though semantics may differ.)
    """
    text = re.sub(r"\bwdt:", "dbo:", text)
    text = re.sub(r"\bps:",  "dbo:", text)
    text = re.sub(r"\bp:",   "dbo:", text)
    text = re.sub(r"\bwd:",  "dbr:", text)
    return text

def clean_bad_chars_and_quotes(q: str) -> str:
    """
    Remove/normalize illegal characters and broken quotes:
      - Remove stray '&' (error #1 category)
      - Remove '>' when not part of <=, >=, !=, or IRI '<...>'
      - Drop/flatten unmatched single quotes (2,3,6)
      - Quote time-like tokens '12:00' -> "12:00"
      - Fix contains() with too many args -> keep first two (3)
      - Normalize 'AS alias' -> 'AS ?alias' (19)
    """
    # Remove ampersands outright
    q = q.replace("&", " ")

    # Remove lone '>' that isn't part of a comparison and not inside an IRI
    # Keep patterns like '<http://...>' intact
    q = re.sub(r"(?<![<=>!])>(?!\s)", " ", q)

    # Flatten unmatched single quotes: remove them entirely
    # (Virtuoso errors often show unfinished single-quoted strings)
    if q.count("'") % 2 != 0:
        q = q.replace("'", "")
    else:
        # Even number of quotes but may wrap IRIs wrongly; aggressively strip all single quotes
        q = q.replace("'", "")

    # Quote clock-like tokens HH:MM (6)
    q = re.sub(r'(?<!")\b(\d{1,2}:\d{2})\b(?!")', r'"\1"', q)

    # contains() with too many args -> keep only first two args (3)
    def fix_contains(m):
        inner = m.group(1)
        # Split top-level commas naively (works for simple cases we see)
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 2:
            return f"CONTAINS({parts[0]}, {parts[1]})"
        return f"CONTAINS({inner})"
    q = re.sub(r"(?i)CONTAINS\s*\(\s*(.*?)\s*\)", fix_contains, q)

    # Normalize 'AS alias' -> 'AS ?alias' (19)
    q = re.sub(r"(?i)\bAS\s+([A-Za-z_]\w*)", r"AS ?\1", q)

    return q

def extract_where_body(query: str):
    """
    Extract first {...} body after WHERE (case-insensitive),
    else the first {...} block in the query. Returns (head, body, tail).
    """
    m = re.search(r"(?is)\bWHERE\s*\{(.*)\}\s*", query)
    if not m:
        m = re.search(r"(?is)\{(.*)\}\s*", query)
    if not m:
        return None, None, None
    return query[:m.start(1)], m.group(1), query[m.end(1):]

def split_clauses(where_body: str):
    """
    Split WHERE body into top-level clauses on '.' while respecting nested {...} () [].
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

def looks_like_subject(tok: str) -> bool:
    """
    Very light validity check for a subject token:
      - variable (?x), CURIE (prefix:local), or IRI <...>
    Reject plain words like 'Position', 'Wynn', '_Martin' (8–12).
    """
    if tok.startswith("?"):
        return True
    if tok.startswith("<") and tok.endswith(">"):
        return True
    if ":" in tok and not tok.startswith(":"):
        return True
    return False

def collapse_adjacent_duplicate_vars(tokens):
    """
    Replace sequences like '?x ?x' with a single '?x' (7).
    """
    out = []
    for t in tokens:
        if out and t.startswith("?") and out[-1] == t:
            continue
        out.append(t)
    return out

def smart_repair_where(where_body: str) -> str:
    """
    Structural repair of WHERE content:
      - Remove empty/invalid clauses,
      - Add missing object,
      - Split stacked predicates,
      - Drop stray numbers in predicate slots,
      - Collapse duplicated variables.
    """
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

        # Tokenize (keep parentheses/brackets attached to tokens)
        tokens = re.findall(r"[^\s]+", cl)
        tokens = collapse_adjacent_duplicate_vars(tokens)

        # Skip FILTER/OPTIONAL/etc. clauses as-is
        if tokens and re.match(r"(?i)^(filter|optional|values|bind|minus|service|graph|union)\b", tokens[0]):
            repaired.append(cl)
            continue

        # Strip trailing punctuation like ';' or ','
        while tokens and tokens[-1] in {";", ","}:
            tokens.pop()

        # Drop clauses with invalid-looking subject (8–12)
        if tokens and not looks_like_subject(tokens[0]):
            continue

        # If exactly two tokens: S P  -> add missing object (5/20 side-effect fix)
        if len(tokens) == 2:
            s, p = tokens
            repaired.append(f"{s} {p} {fresh_var()}")
            continue

        # If four tokens: S P1 P2 O  -> split stacked predicate (also catches 17 category)
        if len(tokens) == 4:
            s, p1, p2, o = tokens

            # If token[1] (predicate) is numeric or plain word → invalid; drop clause
            if re.fullmatch(r"\d+(\.\d+)?", p1) or (":" not in p1 and not p1.startswith("?")):
                continue

            mid = fresh_var()
            repaired.append(f"{s} {p1} {mid}")
            repaired.append(f"{mid} {p2} {o}")
            continue

        # If 3+ tokens but token[1] (predicate) is numeric → drop numeric and try to recover (5)
        if len(tokens) >= 3 and re.fullmatch(r"\d+(\.\d+)?", tokens[1]):
            tokens.pop(1)

        # If after cleanups we still have >=3, keep the first 3 as a triple; extras will be joined by spaces,
        # which is acceptable for simple object lists like "O , O2" but we avoid generating commas here.
        if len(tokens) >= 3:
            s, p, o = tokens[0], tokens[1], tokens[2]
            repaired.append(f"{s} {p} {o}")
            # If there are leftover tokens (rare junk), ignore them.
            continue

        # Otherwise, if still unusable, skip the clause.
        # (Avoid generating invalid triples.)
        continue

    # Join clauses with ' . ' and squeeze repeated dots
    out = ' . '.join(x for x in repaired if x)
    out = re.sub(r"\s*\.\s*\.\s*", " . ", out)
    return out.strip()

def normalize_query_text(q: str) -> str:
    """
    Normalize whitespace & keywords; fix ORDER BY/LIMIT spacing.
    """
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

def rebalance_braces_and_parens(q: str) -> str:
    """
    Ensure counts of '{' and '}' match, and '(' and ')' match.
    - If there are extra '}', drop from the end.
    - If there are extra ')', drop from the end.
    - If there are missing closers, append them at the end.
    """
    # Braces
    opens = q.count("{")
    closes = q.count("}")
    if closes > opens:
        # Remove extra closing braces from the end
        to_remove = closes - opens
        while to_remove and q.endswith("}"):
            q = q[:-1].rstrip()
            to_remove -= 1
    elif opens > closes:
        q += " }" * (opens - closes)

    # Parentheses
    lpar = q.count("(")
    rpar = q.count(")")
    if rpar > lpar:
        to_remove = rpar - lpar
        # remove trailing ')' first
        while to_remove and q.endswith(")"):
            q = q[:-1].rstrip()
            to_remove -= 1
    elif lpar > rpar:
        q += ")" * (lpar - rpar)

    return q

def fix_query(q: str) -> str:
    """
    Full repair pipeline for a single SPARQL query string.
    """
    if not isinstance(q, str):
        return q

    # 1) Normalize square brackets into braces (errors #1–2)
    q = replace_square_brackets_globally(q)

    # 2) Map unknown prefixes (includes new ps:)
    q = map_unknown_prefixes(q)

    # 3) Sanitize CURIE localparts (inner colons/spaces)
    q = sanitize_curie_localparts(q)

    # 4) Clean bad chars, quotes, contains(), AS alias, time literals
    q = clean_bad_chars_and_quotes(q)

    # 5) Normalize 'WHERE' to help extraction regexes
    q = re.sub(r"(?i)\bwhere\b", "WHERE", q)

    # 6) Extract WHERE body and structurally repair it
    head, body, tail = extract_where_body(q)
    if body is not None:
        body_fixed = smart_repair_where(body)
        q = f"{head}{body_fixed}{tail}"

    # 7) Normalize keywords/spacing
    q = normalize_query_text(q)

    # 8) Rebalance braces and parentheses (errors #2,20)
    q = rebalance_braces_and_parens(q)

    return q

def main():
    # Ensure folders exist
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
        print(f"[OK]   {csv_path.name}: updated {changed} row(s)")

        out_path = OUTPUT_DIR / csv_path.name
        df.to_csv(out_path, index=False, encoding="utf-8")
        total_files += 1

    print(f"Done. Wrote {total_files} file(s) to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
