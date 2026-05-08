"""Build a companion CSV from the Gorilla APIBench data.

Source: /Users/edeneldar/gorilla/data/apibench/{huggingface,tensorflow,torchhub}_{train,eval}.json
Each line is a JSON object with keys: code, api_call, provider, api_data.
The `code` field embeds an `###Instruction: ...` block followed by `###Output:` with
<<<domain>>>, <<<api_call>>>, <<<api_provider>>>, <<<explanation>>>, <<<code>>>.

Output: /Users/edeneldar/gorilla/ml_assignment_2026/api_recommendations.csv
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

DATA_DIR = Path("/Users/edeneldar/gorilla/data/apibench")
OUT_PATH = Path("/Users/edeneldar/gorilla/ml_assignment_2026/api_recommendations.csv")

SOURCES = [
    ("huggingface_train.json", "huggingface", "train"),
    ("huggingface_eval.json",  "huggingface", "eval"),
    ("tensorflow_train.json",  "tensorflow",  "train"),
    ("tensorflow_eval.json",   "tensorflow",  "eval"),
    ("torchhub_train.json",    "torchhub",    "train"),
    ("torchhub_eval.json",     "torchhub",    "eval"),
]

INSTRUCTION_RE = re.compile(r"###\s*Instruction:\s*(.*?)(?:###\s*Output:|<<<)", re.DOTALL)
KEYWORD_RE = re.compile(r"\b(code|function|api|library|model|pretrained)\b", re.IGNORECASE)


def parse_instruction(code: str) -> str:
    if not isinstance(code, str):
        return ""
    m = INSTRUCTION_RE.search(code)
    if m:
        return m.group(1).strip()
    # Fallback: text before "###Output"
    if "###Output" in code:
        return code.split("###Output", 1)[0].replace("###Instruction:", "").strip()
    return code.strip()


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalise_arguments(api_data: dict) -> int:
    args = safe_get(api_data, "api_arguments")
    if isinstance(args, dict):
        return len(args)
    if isinstance(args, list):
        return len(args)
    if isinstance(args, str):
        return 0 if args.strip().upper() in {"", "N/A", "NONE"} else 1
    return 0


def normalise_env(api_data: dict) -> int:
    env = safe_get(api_data, "python_environment_requirements")
    if isinstance(env, list):
        return len(env)
    if isinstance(env, str):
        return 0 if env.strip().upper() in {"", "N/A", "NONE"} else 1
    return 0


def has_example_code(api_data: dict) -> int:
    ex = safe_get(api_data, "example_code")
    if isinstance(ex, str) and ex.strip() and ex.strip().upper() != "N/A":
        return 1
    return 0


def description_length(api_data: dict) -> int:
    desc = safe_get(api_data, "description")
    return len(desc) if isinstance(desc, str) else 0


def perf_accuracy(api_data: dict) -> str:
    return _str(safe_get(api_data, "performance", "accuracy"))


def perf_dataset(api_data: dict) -> str:
    return _str(safe_get(api_data, "performance", "dataset"))


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return ", ".join(_str(x) for x in v if x not in (None, ""))
    return str(v).strip()


def build_row(obj: dict, source: str, split: str) -> dict:
    instruction = parse_instruction(obj.get("code", ""))
    api_data = obj.get("api_data") or {}
    words = instruction.split()
    return {
        "source": source,
        "split": split,
        "instruction": instruction,
        "instruction_length": len(instruction),
        "instruction_word_count": len(words),
        "instruction_question_count": instruction.count("?"),
        "instruction_keyword_hits": len(KEYWORD_RE.findall(instruction)),
        "provider": _str(obj.get("provider")),
        "domain": _str(safe_get(api_data, "domain")),
        "framework": _str(safe_get(api_data, "framework")),
        "functionality": _str(safe_get(api_data, "functionality")),
        "api_name": _str(safe_get(api_data, "api_name")),
        "api_call": _str(obj.get("api_call")),
        "num_api_arguments": normalise_arguments(api_data),
        "num_env_requirements": normalise_env(api_data),
        "has_example_code": has_example_code(api_data),
        "description_length": description_length(api_data),
        "performance_dataset": perf_dataset(api_data),
        "performance_accuracy": perf_accuracy(api_data),
    }


def main() -> None:
    rows: list[dict] = []
    for fname, source, split in SOURCES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"skip missing {path}")
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(build_row(obj, source, split))
        print(f"{fname:30s} -> {len(rows):>6d} rows so far")

    if not rows:
        raise SystemExit("no rows produced")

    fieldnames = list(rows[0].keys())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT_PATH}  ({len(rows)} rows, {len(fieldnames)} cols)")


if __name__ == "__main__":
    main()
