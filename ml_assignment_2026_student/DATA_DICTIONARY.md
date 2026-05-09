# Data Dictionary

This file explains the columns in `agent_tool_tasks.csv`.

The dataset has **3,491 rows** and **33 columns**. Each row is one agent task: a user query, the tools available to the agent, and the target label.

The dataset was extracted from the Berkeley Function Calling Leaderboard JSON files:

<https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data>

You do not need the original JSON files for this assignment.

---

## Important Terms

| Term | Meaning |
|---|---|
| Query | The user request. |
| Tool | A function/API the agent could call. |
| Aspect | A broad topic detected by keyword rules, such as `math`, `finance`, `media`, or `travel_weather`. |
| Query/tool match | A measure of whether the query and the available tools seem to talk about similar topics. |
| Target | The value we want to predict. Here the target is `can_answer`. |

---

## Columns

| Column | Type | Description |
|---|---|---|
| `task_uid` | string | Row ID. Do not use it as a model feature. |
| `is_live_benchmark` | 0/1 | 1 if the task came from the live benchmark, 0 otherwise. |
| `task_complexity` | category | `low`, `medium`, or `high`. This is a derived rough complexity bucket. |
| `task_domain` | category | Broad derived domain, such as `math`, `travel`, `vision`, `nlp`, or `other`. |
| `query` | text | Raw user request. Use it only to create features. Do not use the raw text directly in numeric models. |
| `tool_names` | text | Pipe-separated list of available tool names. Example: `math.factorial|get_weather`. May be missing in a few rows. |
| `query_word_count` | number | Number of words in the query. |
| `num_available_tools` | number | Number of tools available to the agent. |
| `total_params` | number | Total number of parameters across all available tools. |
| `total_required_params` | number | Total number of required parameters across all available tools. |
| `avg_param_description_length` | number | Average length of parameter descriptions. Has a few missing values. |
| `tool_description_total_length` | number | Total length of all tool descriptions. |
| `query_aspects` | category/text | Broad aspects detected in the query. Multiple aspects are separated by `|`. `none` means no aspect was detected. |
| `tool_aspects` | category/text | Broad aspects detected in the tool names. Multiple aspects are separated by `|`. `none` means no aspect was detected. |
| `query_aspect_count` | number | Number of aspects detected in the query. |
| `tool_aspect_count` | number | Number of aspects detected in the tool names. |
| `aspect_overlap_count` | number | Number of aspects that appear in both `query_aspects` and `tool_aspects`. |
| `aspect_coverage_ratio` | number | Fraction of query aspects that are also found in the tool aspects. Values are between 0 and 1. |
| `aspect_mismatch_count` | number | Number of query aspects not found in the tool aspects. |
| `query_tool_token_jaccard` | number | Token-overlap score between the query and tool names. Higher means more shared words. |
| `query_tool_action_overlap` | number | Number of action words that appear in both the query and the tool names. |
| `tool_action_verb_count` | number | Number of action-like words in tool names, such as `get`, `search`, `send`, or `calculate`. |
| `risky_tool_action_count` | number | Number of tool-name action words that may involve risk, such as `send`, `book`, `buy`, `delete`, or `update`. |
| `query_multi_intent_score` | number | Count of words that suggest multiple steps, such as `also`, `then`, `first`, `next`, or `finally`. |
| `query_specificity_score` | number | Simple score based on numbers, quoted strings, and long tokens in the query. |
| `query_unique_token_ratio` | number | Number of unique query tokens divided by total query tokens. |
| `query_code_signal` | 0/1 | 1 if the query contains code/API/data words such as `python`, `sql`, `json`, `api`, or `token`. |
| `query_temporal_signal` | 0/1 | 1 if the query contains time or scheduling words such as `today`, `meeting`, `calendar`, or `deadline`. |
| `query_sensitive_data_signal` | 0/1 | 1 if the query contains words such as `account`, `password`, `token`, `payment`, or `card`. |
| `param_type_diversity` | number | Number of different parameter type families used by the available tools. |
| `structured_param_ratio` | number | Share of parameters that are arrays/lists or objects/dicts. |
| `schema_rigidity_score` | number | Rough score for how strict the tool schemas are. Higher means more required/limited parameters. |
| `can_answer` | 0/1 | Target label. 1 means at least one tool is relevant. 0 means the agent should refuse. |

---

## Notes About Feature Use

| Column type | What to do |
|---|---|
| IDs | Do not use `task_uid` as a feature. |
| Raw text | Do not feed raw `query` or raw `tool_names` directly into numeric models. Use them only to create features. |
| Categorical columns | Encode columns such as `task_domain`, `task_complexity`, `query_aspects`, and `tool_aspects` before using them in models. |
| Multi-value columns | `query_aspects`, `tool_aspects`, and `tool_names` can contain several values separated by "pipe" (`\|`). Treat them carefully if you use them. |
| Missing values | Handle missing `avg_param_description_length` and missing/empty `tool_names`. |
| Target | Do not use `can_answer` as an input feature. |

---

## Target Meaning

| `can_answer` | Meaning |
|---|---|
| 1 | At least one available tool is relevant to the user query. |
| 0 | No available tool is relevant; the agent should refuse to call a tool. |
