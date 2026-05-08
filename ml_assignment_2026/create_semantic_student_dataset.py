"""Create a richer, less redundant student CSV for ML Assignment 2.

The first student CSV exposed many low-level schema counts
(`num_string_params`, `num_boolean_params`, ...). Those columns are valid, but
they are repetitive and not very interesting for modeling discussions.

This script keeps a small set of transparent base columns and adds lightweight
aspect/semantic-alignment features inspired by aspect analysis:

* What aspects appear in the user query?
* What aspects appear in the available tool names?
* Do query aspects and tool aspects overlap?
* Does the query ask for multiple intents?
* Are the available tools action-oriented or risky?

The features are deterministic and lexicon-based, so students do not need
external NLP models or network access.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "agent_tool_tasks.csv"
STUDENT_OUT = ROOT.parent / "ml_assignment_2026_student" / "agent_tool_tasks.csv"
SOLUTION_OUT = ROOT.parent / "ml_assignment_2026_solution" / "agent_tool_tasks.csv"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']*")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "for",
    "from", "give", "have", "help", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "please", "the", "them", "this", "to", "using", "want", "with",
    "you", "your",
}

ACTION_WORDS = {
    "add", "analyze", "book", "buy", "calculate", "cancel", "check", "classify",
    "compare", "convert", "create", "delete", "download", "extract", "fetch",
    "find", "generate", "get", "list", "lookup", "make", "open", "play",
    "predict", "read", "recommend", "remove", "retrieve", "run", "schedule",
    "search", "send", "set", "show", "solve", "summarize", "translate",
    "update", "upload", "write",
}

RISKY_ACTION_WORDS = {
    "book", "buy", "cancel", "create", "delete", "download", "email", "message",
    "open", "pay", "purchase", "remove", "reserve", "run", "schedule", "send",
    "transfer", "update", "upload",
}

MULTI_INTENT_WORDS = {
    "also", "and", "another", "finally", "first", "lastly", "next", "second",
    "then", "third",
}

CODE_WORDS = {
    "api", "code", "command", "database", "debug", "endpoint", "function",
    "json", "python", "query", "sql", "script", "token", "url",
}

TEMPORAL_WORDS = {
    "after", "appointment", "calendar", "date", "day", "deadline", "hour",
    "later", "meeting", "minute", "month", "schedule", "today", "tomorrow",
    "tonight", "week", "year", "yesterday",
}

SENSITIVE_WORDS = {
    "account", "address", "bank", "card", "credit", "email", "key", "password",
    "payment", "phone", "private", "secret", "ssn", "token", "transfer",
}

ASPECT_KEYWORDS: dict[str, set[str]] = {
    "math": {
        "area", "average", "calculate", "compute", "derivative", "equation",
        "factorial", "geometry", "integral", "math", "matrix", "probability",
        "solve", "sum",
    },
    "travel_weather": {
        "airport", "book", "bus", "flight", "forecast", "hotel", "rain",
        "reservation", "restaurant", "ticket", "travel", "trip", "weather",
    },
    "scheduling": {
        "appointment", "calendar", "deadline", "event", "meeting", "remind",
        "schedule",
    },
    "communication": {
        "chat", "email", "gmail", "message", "notify", "send", "slack", "sms",
    },
    "finance": {
        "account", "bank", "bitcoin", "budget", "card", "currency", "finance",
        "loan", "money", "payment", "portfolio", "price", "stock", "transfer",
    },
    "data_code": {
        "api", "code", "database", "endpoint", "fetch", "file", "json",
        "python", "query", "retrieve", "script", "search", "sql", "url",
    },
    "media": {
        "audio", "image", "movie", "music", "photo", "picture", "play",
        "song", "speech", "video", "voice",
    },
    "language": {
        "classify", "entity", "intent", "language", "sentiment", "summarize",
        "text", "translate", "word",
    },
    "shopping": {
        "buy", "cart", "order", "price", "product", "purchase", "shop",
    },
    "security": {
        "access", "domain", "key", "password", "security", "token", "virus",
    },
}


def normalize_text(text: object) -> str:
    s = str(text)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_./|:;,-]+", " ", s)
    return s


def tokenize(text: object) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(normalize_text(text))]


def aspect_counts(tokens: list[str]) -> Counter:
    token_set = set(tokens)
    counts: Counter = Counter()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        counts[aspect] = sum(1 for kw in keywords if kw in token_set)
    return Counter({k: v for k, v in counts.items() if v > 0})


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(-sum((v / total) * math.log2(v / total) for v in counts.values()))


def pipe_keys(counts: Counter) -> str:
    if not counts:
        return "none"
    return "|".join(sorted(counts))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def specificity_score(query: str, tokens: list[str]) -> float:
    numbers = len(NUMBER_RE.findall(query))
    quoted = query.count("'") // 2 + query.count('"') // 2
    long_tokens = sum(1 for t in tokens if len(t) >= 10)
    return float(numbers + quoted + 0.5 * long_tokens)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tool_names"] = out["tool_names"].fillna("")

    query_tokens = out["query"].map(tokenize)
    tool_tokens = out["tool_names"].map(tokenize)

    query_aspect_counts = query_tokens.map(aspect_counts)
    tool_aspect_counts = tool_tokens.map(aspect_counts)

    query_aspects = query_aspect_counts.map(lambda c: set(c.keys()))
    tool_aspects = tool_aspect_counts.map(lambda c: set(c.keys()))

    out["query_aspects"] = query_aspect_counts.map(pipe_keys)
    out["tool_aspects"] = tool_aspect_counts.map(pipe_keys)
    out["query_aspect_count"] = query_aspects.map(len)
    out["tool_aspect_count"] = tool_aspects.map(len)
    out["aspect_overlap_count"] = [
        len(q & t) for q, t in zip(query_aspects, tool_aspects, strict=True)
    ]
    out["aspect_coverage_ratio"] = [
        safe_ratio(len(q & t), len(q)) for q, t in zip(query_aspects, tool_aspects, strict=True)
    ]
    out["aspect_mismatch_count"] = [
        max(len(q - t), 0) for q, t in zip(query_aspects, tool_aspects, strict=True)
    ]

    query_content = query_tokens.map(lambda ts: {t for t in ts if t not in STOPWORDS})
    tool_content = tool_tokens.map(lambda ts: {t for t in ts if t not in STOPWORDS})
    out["query_tool_token_jaccard"] = [
        jaccard(q, t) for q, t in zip(query_content, tool_content, strict=True)
    ]
    out["query_unique_token_ratio"] = [
        safe_ratio(len(set(ts)), len(ts)) for ts in query_tokens
    ]

    query_actions = query_tokens.map(lambda ts: {t for t in ts if t in ACTION_WORDS})
    tool_actions = tool_tokens.map(lambda ts: {t for t in ts if t in ACTION_WORDS})
    out["query_tool_action_overlap"] = [
        len(q & t) for q, t in zip(query_actions, tool_actions, strict=True)
    ]
    out["tool_action_verb_count"] = tool_tokens.map(
        lambda ts: sum(1 for t in ts if t in ACTION_WORDS)
    )
    out["risky_tool_action_count"] = tool_tokens.map(
        lambda ts: sum(1 for t in ts if t in RISKY_ACTION_WORDS)
    )

    out["query_multi_intent_score"] = query_tokens.map(
        lambda ts: sum(1 for t in ts if t in MULTI_INTENT_WORDS)
    )
    out["query_specificity_score"] = [
        specificity_score(q, ts) for q, ts in zip(out["query"], query_tokens, strict=True)
    ]
    out["query_code_signal"] = query_tokens.map(
        lambda ts: int(any(t in CODE_WORDS for t in ts))
    )
    out["query_temporal_signal"] = query_tokens.map(
        lambda ts: int(any(t in TEMPORAL_WORDS for t in ts))
    )
    out["query_sensitive_data_signal"] = query_tokens.map(
        lambda ts: int(any(t in SENSITIVE_WORDS for t in ts))
    )

    type_cols = [
        "num_string_params",
        "num_numeric_params",
        "num_boolean_params",
        "num_array_params",
        "num_object_params",
    ]
    out["param_type_diversity"] = (out[type_cols] > 0).sum(axis=1)
    out["structured_param_ratio"] = [
        safe_ratio(a + o, t)
        for a, o, t in zip(
            out["num_array_params"],
            out["num_object_params"],
            out["total_params"],
            strict=True,
        )
    ]
    required_ratio = [
        safe_ratio(r, t) for r, t in zip(out["total_required_params"], out["total_params"], strict=True)
    ]
    enum_ratio = [
        safe_ratio(e, t) for e, t in zip(out["num_enum_params"], out["total_params"], strict=True)
    ]
    no_defaults = 1 - out["has_default_values"].astype(int)
    out["schema_rigidity_score"] = (
        0.55 * np.array(required_ratio) + 0.25 * np.array(enum_ratio) + 0.20 * no_defaults
    ).round(4)

    keep = [
        "task_uid",
        "is_live_benchmark",
        "task_complexity",
        "task_domain",
        "query",
        "tool_names",
        "query_word_count",
        "num_available_tools",
        "total_params",
        "total_required_params",
        "avg_param_description_length",
        "tool_description_total_length",
        "query_aspects",
        "tool_aspects",
        "query_aspect_count",
        "tool_aspect_count",
        "aspect_overlap_count",
        "aspect_coverage_ratio",
        "aspect_mismatch_count",
        "query_tool_token_jaccard",
        "query_tool_action_overlap",
        "tool_action_verb_count",
        "risky_tool_action_count",
        "query_multi_intent_score",
        "query_specificity_score",
        "query_unique_token_ratio",
        "query_code_signal",
        "query_temporal_signal",
        "query_sensitive_data_signal",
        "param_type_diversity",
        "structured_param_ratio",
        "schema_rigidity_score",
        "can_answer",
    ]
    return out[keep]


def main() -> None:
    df = pd.read_csv(SOURCE)
    out = build_features(df)
    STUDENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STUDENT_OUT, index=False)
    if SOLUTION_OUT.parent.exists():
        out.to_csv(SOLUTION_OUT, index=False)

    print(f"Wrote {STUDENT_OUT} {out.shape}")
    if SOLUTION_OUT.parent.exists():
        print(f"Wrote {SOLUTION_OUT} {out.shape}")
    print(out.columns.tolist())


if __name__ == "__main__":
    main()
