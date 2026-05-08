"""Add a derived join-key column `task_type` to both CSVs.

`task_type` is a broad ML / task-domain category inferred from the natural-language
text of each row using the *same* keyword classifier on both datasets. This makes
the two otherwise-disjoint CSVs joinable on a meaningful semantic key:

    agent_tasks.csv     <-- task_type -->     api_recommendations.csv

Outputs (in-place updates):
    /Users/edeneldar/gorilla/ml_assignment_2026/agent_tasks.csv
    /Users/edeneldar/gorilla/ml_assignment_2026/api_recommendations.csv

Plus a small cross-tabulation CSV showing the relationship:
    /Users/edeneldar/gorilla/ml_assignment_2026/task_type_cross_tab.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/edeneldar/gorilla/ml_assignment_2026")

# Order matters: first matching category wins (most specific first).
TASK_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("multimodal", [
        "multimodal", "vision-language", "visual question", "image caption",
        "image-to-text", "text-to-image", "captioning", "vqa",
    ]),
    ("vision", [
        "image", "photo", "picture", "video", "object detection", "segmentation",
        "classify image", "image classification", "computer vision", "ocr",
        "pixel", "yolo", "resnet", "depth estimation", "pose", "face", "vision",
    ]),
    ("audio", [
        "audio", "speech", "voice", "transcribe", "transcription", "music",
        "speaker", "asr", "tts", "speech-to-text", "text-to-speech", "sound",
    ]),
    ("nlp", [
        "translate", "translation", "summari", "sentiment", "question answer",
        "qa ", "text", "language model", "tokeniz", "embedding", "named entity",
        "ner", "chatbot", "dialogue", "paraphras", "language",
    ]),
    ("rl", [
        "reinforcement", "policy", "reward", "agent in", "atari", "ppo", "dqn",
        "gym", "ml-agents",
    ]),
    ("tabular", [
        "csv", "dataframe", "tabular", "regression on", "predict price",
        "scikit", "joblib", "boosting", "random forest", "xgboost",
    ]),
    ("scheduling", [
        "calendar", "schedule", "meeting", "appointment", "remind", "event on",
    ]),
    ("communication", [
        "email", "send mail", "gmail", "message", "slack", "sms", "notify",
    ]),
    ("data_query", [
        "database", "sql", "query the", "fetch from", "lookup", "search for",
        "list all", "get all", "retrieve",
    ]),
    ("math", [
        "calculate", "compute", "integral", "derivative", "matrix", "equation",
        "average of", "mean of", "sum of", "factorial",
    ]),
    ("finance", [
        "stock", "price of", "exchange rate", "currency", "invest", "portfolio",
        "crypto", "bitcoin",
    ]),
    ("travel", [
        "flight", "hotel", "book a", "reservation", "weather", "forecast", "trip",
    ]),
]

DEFAULT = "other"


def classify(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return DEFAULT
    t = text.lower()
    for label, keywords in TASK_TYPE_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return label
    return DEFAULT


def main() -> None:
    agent_path = ROOT / "agent_tasks.csv"
    api_path = ROOT / "api_recommendations.csv"

    print(f"loading {agent_path.name}")
    agent = pd.read_csv(agent_path)
    # Combine the user query and the tool names so we get signal even when the
    # query is terse but the tool names are descriptive (e.g. "send_email").
    agent_text = (
        agent["query"].fillna("").astype(str) + " " +
        agent["tool_names"].fillna("").astype(str).str.replace("|", " ", regex=False)
    )
    agent["task_type"] = agent_text.map(classify)

    print(f"loading {api_path.name}")
    api = pd.read_csv(api_path)
    api_text = (
        api["instruction"].fillna("").astype(str) + " " +
        api["domain"].fillna("").astype(str) + " " +
        api["functionality"].fillna("").astype(str)
    )
    api["task_type"] = api_text.map(classify)

    agent.to_csv(agent_path, index=False)
    api.to_csv(api_path, index=False)

    # Distributions
    print("\nagent_tasks task_type distribution:")
    print(agent["task_type"].value_counts())
    print("\napi_recommendations task_type distribution:")
    print(api["task_type"].value_counts())

    # Cross-tab using task_type as the join key. We do an *aggregated* join
    # rather than a row-level join because the relationship is many-to-many.
    agent_summary = (
        agent.groupby("task_type")
             .agg(agent_n=("task_id", "count"),
                  agent_relevance_rate=("label_is_relevant", "mean"),
                  agent_avg_query_words=("query_word_count", "mean"),
                  agent_avg_tools=("num_available_tools", "mean"))
             .round(3)
    )
    api_summary = (
        api.groupby("task_type")
           .agg(api_n=("instruction", "count"),
                api_avg_instruction_words=("instruction_word_count", "mean"),
                api_avg_arguments=("num_api_arguments", "mean"),
                api_top_provider=("provider", lambda s: s.value_counts().index[0] if len(s) else ""))
           .round(3)
    )
    cross = agent_summary.join(api_summary, how="outer").fillna(0)
    cross_path = ROOT / "task_type_cross_tab.csv"
    cross.to_csv(cross_path)
    print(f"\nwrote cross-tab {cross_path.name}:")
    print(cross.to_string())


if __name__ == "__main__":
    main()
