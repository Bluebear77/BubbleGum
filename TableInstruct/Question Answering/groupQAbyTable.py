# ============================================================================================
# Script: Group Q/A pairs by table (from ./TQAS/*.csv) → ./GTQA/*.gtqa.csv + statistics.csv
# Changes per request:
#   1) Column "qa_pairs_json" → "qas" (plain text: "Q1 A1 Q2 A2 ...")
#   2) Column order: num_pairs, table, qas
# ============================================================================================

import os
import csv
import json
from collections import defaultdict
from statistics import mean, median, pstdev

INPUT_DIR = "TQAS"
OUTPUT_DIR = "GTQA"
STATS_FILENAME = "statistics.csv"

def _plain(s: str) -> str:
    """Collapse all whitespace/newlines to single spaces."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.split())

def read_input_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = ["question", "answer", "table"]
        if reader.fieldnames is None or [h.strip() for h in reader.fieldnames] != expected:
            raise ValueError(f"{os.path.basename(path)} must have header {expected}, got {reader.fieldnames}")
        for row in reader:
            yield {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "table": row.get("table", ""),
            }

def group_by_table(rows_iter):
    groups = defaultdict(list)  # table -> list of {"question","answer"}
    for r in rows_iter:
        groups[r["table"]].append({"question": r["question"], "answer": r["answer"]})
    return groups

def make_qas_plain_text(qa_list):
    # "Q A Q A ..." — each question followed by its answer; collapse whitespace
    parts = []
    for qa in qa_list:
        q = _plain(qa.get("question", ""))
        a = _plain(qa.get("answer", ""))
        if q or a:
            # keep a single space between Q and A, and between pairs
            parts.append((q + " " + a).strip())
    return " ".join(parts).strip()

def write_grouped_csv(out_path, groups):
    # Column order: num_pairs, table, qas
    header = ["num_pairs", "table", "qas"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(header)
        for table_text, qa_list in groups.items():
            qas_text = make_qas_plain_text(qa_list)
            w.writerow([len(qa_list), table_text, qas_text])

def collect_stats_for_file(groups):
    counts = [len(v) for v in groups.values()]
    if not counts:
        return {"num_tables": 0, "total_qas": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
    return {
        "num_tables": len(counts),
        "total_qas": sum(counts),
        "mean": float(mean(counts)),
        "median": float(median(counts)),
        "stdev": float(pstdev(counts)),
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(f"Input directory '{INPUT_DIR}' not found.")

    input_files = [fn for fn in os.listdir(INPUT_DIR) if fn.lower().endswith(".csv")]
    if not input_files:
        raise SystemExit(f"No CSV files found in '{INPUT_DIR}'.")

    all_counts = []
    per_file_stats_rows = []

    for fn in sorted(input_files):
        in_path = os.path.join(INPUT_DIR, fn)
        base = os.path.splitext(fn)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}.gtqa.csv")

        rows_iter = read_input_csv(in_path)
        groups = group_by_table(rows_iter)

        write_grouped_csv(out_path, groups)

        stats = collect_stats_for_file(groups)
        per_file_stats_rows.append({"filename": fn, **stats})
        all_counts.extend([len(v) for v in groups.values()])

        print(f"Processed: {fn} → {os.path.basename(out_path)} "
              f"(tables: {stats['num_tables']}, QAs: {stats['total_qas']})")

    # Overall row
    if all_counts:
        overall = {
            "filename": "ALL",
            "num_tables": len(all_counts),
            "total_qas": sum(all_counts),
            "mean": float(mean(all_counts)),
            "median": float(median(all_counts)),
            "stdev": float(pstdev(all_counts)),
        }
    else:
        overall = {"filename": "ALL", "num_tables": 0, "total_qas": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}

    # Write statistics.csv (unchanged schema)
    stats_path = os.path.join(OUTPUT_DIR, STATS_FILENAME)
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["filename", "num_tables", "total_qas", "mean", "median", "stdev"])
        for row in per_file_stats_rows:
            w.writerow([
                row["filename"],
                row["num_tables"],
                row["total_qas"],
                f"{row['mean']:.6f}",
                f"{row['median']:.6f}",
                f"{row['stdev']:.6f}",
            ])
        w.writerow([
            overall["filename"],
            overall["num_tables"],
            overall["total_qas"],
            f"{overall['mean']:.6f}",
            f"{overall['median']:.6f}",
            f"{overall['stdev']:.6f}",
        ])

    print(f"Saved statistics: {stats_path}")

if __name__ == "__main__":
    main()
