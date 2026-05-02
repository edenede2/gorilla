"""
Build the AI-Agent Function-Calling dataset (CSV) for the ML course assignment.

Source: Berkeley Function Calling Leaderboard v4 (BFCL) JSONL files in
`berkeley-function-call-leaderboard/bfcl_eval/data/`.

Output: agent_tasks.csv  (one row per BFCL test case)

Run:
    python build_dataset.py
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = (
    ROOT.parent
    / "berkeley-function-call-leaderboard"
    / "bfcl_eval"
    / "data"
)
OUT_CSV = ROOT / "agent_tasks.csv"

# (filename, short category label, is_live, label_is_relevant)
SOURCES = [
    ("BFCL_v4_simple_python.json",       "simple",                 0, 1),
    ("BFCL_v4_multiple.json",            "multiple",               0, 1),
    ("BFCL_v4_parallel.json",            "parallel",               0, 1),
    ("BFCL_v4_parallel_multiple.json",   "parallel_multiple",      0, 1),
    ("BFCL_v4_irrelevance.json",         "irrelevance",            0, 0),
    ("BFCL_v4_live_simple.json",         "live_simple",            1, 1),
    ("BFCL_v4_live_multiple.json",       "live_multiple",          1, 1),
    ("BFCL_v4_live_parallel.json",       "live_parallel",          1, 1),
    ("BFCL_v4_live_parallel_multiple.json","live_parallel_multiple",1, 1),
    ("BFCL_v4_live_irrelevance.json",    "live_irrelevance",       1, 0),
    ("BFCL_v4_live_relevance.json",      "live_relevance",         1, 1),
]

NUMERIC_TYPES = {"integer", "int", "float", "number", "double", "long"}
STRING_TYPES = {"string", "str"}
BOOLEAN_TYPES = {"boolean", "bool"}
ARRAY_TYPES = {"array", "list", "tuple"}
OBJECT_TYPES = {"dict", "object"}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_user_query(record: dict) -> str:
    """Return the *first* user message in the (nested) `question` field."""
    q = record.get("question", [])
    if not q:
        return ""
    msgs = q[0] if isinstance(q[0], list) else q
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return (m.get("content") or "").strip()
    # Fallback: first message regardless of role
    if msgs and isinstance(msgs[0], dict):
        return (msgs[0].get("content") or "").strip()
    return ""


def normalise_type(t) -> str:
    if isinstance(t, list):
        # Pick the first non-null type
        for x in t:
            if x and str(x).lower() != "null":
                return str(x).lower()
        return "any"
    if t is None:
        return "any"
    return str(t).lower()


def summarise_functions(functions):
    """Return a dict of features describing the available tools/functions."""
    n_funcs = len(functions)
    names = [f.get("name", "") for f in functions]

    total_params = 0
    total_required = 0
    n_string = n_numeric = n_bool = n_array = n_object = 0
    n_enum = 0
    has_default = 0
    desc_lengths = []
    tool_desc_total = 0

    for f in functions:
        tool_desc_total += len(f.get("description", "") or "")
        params = (f.get("parameters") or {}).get("properties", {}) or {}
        required = (f.get("parameters") or {}).get("required", []) or []
        total_params += len(params)
        total_required += len(required)
        for _, spec in params.items():
            if not isinstance(spec, dict):
                continue
            t = normalise_type(spec.get("type"))
            if t in STRING_TYPES:
                n_string += 1
            elif t in NUMERIC_TYPES:
                n_numeric += 1
            elif t in BOOLEAN_TYPES:
                n_bool += 1
            elif t in ARRAY_TYPES:
                n_array += 1
            elif t in OBJECT_TYPES:
                n_object += 1
            if "enum" in spec:
                n_enum += 1
            if "default" in spec:
                has_default = 1
            d = spec.get("description") or ""
            desc_lengths.append(len(d))

    if desc_lengths:
        avg_desc_len = round(sum(desc_lengths) / len(desc_lengths), 2)
    else:
        # Intentionally leave as missing -> NaN in CSV (imputation exercise)
        avg_desc_len = ""

    return {
        "num_available_tools": n_funcs,
        "tool_names": "|".join(names),
        "total_params": total_params,
        "total_required_params": total_required,
        "num_string_params": n_string,
        "num_numeric_params": n_numeric,
        "num_boolean_params": n_bool,
        "num_array_params": n_array,
        "num_object_params": n_object,
        "num_enum_params": n_enum,
        "has_default_values": has_default,
        "avg_param_description_length": avg_desc_len,
        "tool_description_total_length": tool_desc_total,
    }


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")
_UPPER_WORD_RE = re.compile(r"\b[A-Z][A-Z]+\b")


def summarise_query(text: str) -> dict:
    return {
        "query": text.replace("\r", " ").replace("\n", " ").strip(),
        "query_char_length": len(text),
        "query_word_count": len(_WORD_RE.findall(text)),
        "query_question_marks": text.count("?"),
        "query_digit_count": sum(c.isdigit() for c in text),
        "query_uppercase_words": len(_UPPER_WORD_RE.findall(text)),
    }


COLUMNS = [
    "task_id",
    "category",
    "is_live_benchmark",
    "query",
    "query_char_length",
    "query_word_count",
    "query_question_marks",
    "query_digit_count",
    "query_uppercase_words",
    "num_available_tools",
    "tool_names",
    "total_params",
    "total_required_params",
    "num_string_params",
    "num_numeric_params",
    "num_boolean_params",
    "num_array_params",
    "num_object_params",
    "num_enum_params",
    "has_default_values",
    "avg_param_description_length",
    "tool_description_total_length",
    "label_is_relevant",
]


def main() -> None:
    all_rows = []
    for fname, cat, is_live, label in SOURCES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        for rec in load_jsonl(path):
            row = {
                "task_id": rec.get("id", ""),
                "category": cat,
                "is_live_benchmark": is_live,
                "label_is_relevant": label,
            }
            row.update(summarise_query(extract_user_query(rec)))
            row.update(summarise_functions(rec.get("function", []) or []))
            all_rows.append(row)

    # Stable sort: category then id
    all_rows.sort(key=lambda r: (r["category"], r["task_id"]))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Wrote {len(all_rows):,} rows -> {OUT_CSV}")
    # Quick label balance
    pos = sum(1 for r in all_rows if r["label_is_relevant"] == 1)
    print(f"  relevant=1: {pos:,}   relevant=0: {len(all_rows)-pos:,}")


if __name__ == "__main__":
    main()
