"""Prepare leakage-reduced CSVs for the ML assignment.

The raw source files are useful for reproducibility, but they contain fields
that make the supervised task too easy:

* ``task_id`` values expose source/category names such as ``irrelevance_*``.
* ``category`` directly determines the target label.
* Rows are grouped by source category, so row order itself is mildly revealing.

This script keeps the original files untouched and writes curated assignment
files with opaque identifiers, shuffled rows, clearer column names, and a shared
``task_domain`` key for the cross-dataset section.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def minmax(s: pd.Series) -> pd.Series:
    low = float(s.min())
    high = float(s.max())
    if high == low:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - low) / (high - low)


def prepare_agent_tasks() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "agent_tasks.csv")

    curated = raw.copy()
    curated["tool_names"] = curated["tool_names"].fillna("")
    curated["avg_param_description_length"] = pd.to_numeric(
        curated["avg_param_description_length"], errors="coerce"
    )

    # A non-label convenience column for EDA. It summarizes observable task
    # complexity without using category or can_answer.
    complexity_score = (
        0.35 * minmax(curated["query_word_count"])
        + 0.35 * minmax(curated["num_available_tools"])
        + 0.30 * minmax(curated["total_params"])
    )
    curated["task_complexity"] = pd.qcut(
        complexity_score.rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)

    curated = curated.rename(
        columns={
            "label_is_relevant": "can_answer",
            "task_type": "task_domain",
        }
    )

    drop_cols = ["task_id", "category"]
    curated = curated.drop(columns=[c for c in drop_cols if c in curated.columns])

    # Shuffle before assigning IDs so neither row order nor ID reveals source
    # category. The target remains stratifiable by the students.
    curated = curated.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    curated.insert(0, "task_uid", [f"task_{i:05d}" for i in range(1, len(curated) + 1)])

    ordered = [
        "task_uid",
        "is_live_benchmark",
        "task_complexity",
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
        "task_domain",
        "can_answer",
    ]
    return curated[ordered]


def prepare_api_catalog() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "api_recommendations.csv")
    curated = raw.rename(columns={"task_type": "task_domain"}).copy()
    curated = curated.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    curated.insert(0, "api_uid", [f"api_{i:05d}" for i in range(1, len(curated) + 1)])
    return curated


def write_domain_summary(agent: pd.DataFrame, api: pd.DataFrame) -> pd.DataFrame:
    agent_summary = (
        agent.groupby("task_domain")
        .agg(
            agent_tasks=("task_uid", "count"),
            can_answer_rate=("can_answer", "mean"),
            avg_query_words=("query_word_count", "mean"),
            avg_available_tools=("num_available_tools", "mean"),
        )
        .round(3)
    )
    api_summary = (
        api.groupby("task_domain")
        .agg(
            api_rows=("api_uid", "count"),
            avg_instruction_words=("instruction_word_count", "mean"),
            avg_api_arguments=("num_api_arguments", "mean"),
            top_provider=(
                "provider",
                lambda s: s.value_counts().index[0] if len(s.value_counts()) else "",
            ),
        )
        .round(3)
    )
    summary = agent_summary.join(api_summary, how="outer").fillna(
        {
            "agent_tasks": 0,
            "can_answer_rate": 0,
            "avg_query_words": 0,
            "avg_available_tools": 0,
            "api_rows": 0,
            "avg_instruction_words": 0,
            "avg_api_arguments": 0,
            "top_provider": "none",
        }
    )
    summary["agent_tasks"] = summary["agent_tasks"].astype(int)
    summary["api_rows"] = summary["api_rows"].astype(int)
    return summary.reset_index()


def main() -> None:
    agent = prepare_agent_tasks()
    api = prepare_api_catalog()
    summary = write_domain_summary(agent, api)

    agent.to_csv(ROOT / "agent_tool_tasks.csv", index=False)
    api.to_csv(ROOT / "api_catalog.csv", index=False)
    summary.to_csv(ROOT / "domain_coverage_summary.csv", index=False)

    print("Wrote agent_tool_tasks.csv", agent.shape)
    print("Wrote api_catalog.csv", api.shape)
    print("Wrote domain_coverage_summary.csv", summary.shape)
    print("\nTarget distribution:")
    print(agent["can_answer"].value_counts(normalize=True).round(3).to_string())
    print("\nAgent columns:")
    print(", ".join(agent.columns))


if __name__ == "__main__":
    main()
