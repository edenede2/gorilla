# ML Assignment 2 - Data Dictionary

## Minimal Dataset Package

Students only need the two required CSV files below to complete the assignment. The summary CSV is optional because it can be regenerated from the two required files.

| File | Required? | Shape | Purpose | Notes |
|---|---:|---:|---|---|
| `agent_tool_tasks.csv` | Yes | 3,491 x 24 | Primary dataset for EDA, preprocessing, classification, and clustering. | Predict `can_answer`: whether the agent has at least one relevant tool. |
| `api_catalog.csv` | Yes | 17,003 x 21 | Secondary dataset for cross-dataset domain analysis and API provider/framework classification. | Uses the original `split` column for train/eval in Section E. |
| `domain_coverage_summary.csv` | Optional | 13 x 9 | Precomputed domain-level join summary. | Useful for checking Section E, but students can generate it themselves. |

Files that are not needed for students: `agent_tasks.csv`, `api_recommendations.csv`, `task_type_cross_tab.csv`, solution notebooks, HTML exports, and dataset-building scripts. Keep those for instructor reproducibility only.

---

## `agent_tool_tasks.csv`

One row represents one agent task: a user query plus the tools available to the agent.

| Column | Type | Role | Use Directly As Model Input? | Description |
|---|---|---|---|---|
| `task_uid` | string | Identifier | No | Opaque row ID assigned after shuffling. It does not contain source/category information. |
| `is_live_benchmark` | integer, 0/1 | Metadata feature | Yes | 1 if the task came from the live benchmark, 0 otherwise. Useful for distribution-shift analysis. |
| `task_complexity` | categorical | EDA / optional feature | Only if encoded | Low / medium / high bucket derived from query length, number of available tools, and total parameters. |
| `query` | string | Raw text | No | Raw user request. Use only to engineer text-derived features. |
| `query_char_length` | integer | Numeric feature | Yes | Number of characters in `query`. |
| `query_word_count` | integer | Numeric feature | Yes | Number of words in `query`. |
| `query_question_marks` | integer | Numeric feature | Yes | Number of `?` characters in `query`. |
| `query_digit_count` | integer | Numeric feature | Yes | Number of digit characters in `query`. |
| `query_uppercase_words` | integer | Numeric feature | Yes | Number of fully uppercase words in `query`, such as `USA` or `API`. |
| `num_available_tools` | integer | Numeric feature | Yes | Number of tools/functions offered to the agent. |
| `tool_names` | string | Raw tool text | No | Pipe-separated list of tool names. Use only to engineer tool-derived features. May be empty/missing in a few rows. |
| `total_params` | integer | Numeric feature | Yes | Total number of parameters across all available tools. |
| `total_required_params` | integer | Numeric feature | Yes | Total number of required parameters across all available tools. |
| `num_string_params` | integer | Numeric feature | Yes | Number of tool parameters of type string. |
| `num_numeric_params` | integer | Numeric feature | Yes | Number of integer, float, or numeric parameters. |
| `num_boolean_params` | integer | Numeric feature | Yes | Number of boolean parameters. |
| `num_array_params` | integer | Numeric feature | Yes | Number of array/list parameters. |
| `num_object_params` | integer | Numeric feature | Yes | Number of object/dict parameters. |
| `num_enum_params` | integer | Numeric feature | Yes | Number of parameters constrained to a fixed enum set. |
| `has_default_values` | integer, 0/1 | Binary feature | Yes | 1 if at least one parameter has a default value, else 0. |
| `avg_param_description_length` | float | Numeric feature | Yes, after imputation | Average length of parameter descriptions. Has a few missing values when tools have no parameter descriptions. |
| `tool_description_total_length` | integer | Numeric feature | Yes | Total character length of all available tool descriptions. |
| `task_domain` | categorical | Domain feature / join key | Only if encoded | Broad keyword-derived domain such as `vision`, `nlp`, `travel`, `math`, or `other`. Used to compare with `api_catalog.csv`. |
| `can_answer` | integer, 0/1 | Target label | No | Target variable. 1 means at least one offered tool is relevant; 0 means the agent should refuse. |

Important notes:

| Issue | What Students Should Do |
|---|---|
| Target imbalance | Report class-aware metrics, especially recall and F1 for `can_answer = 0`. |
| Raw text fields | Do not feed `query` or `tool_names` directly into numeric models without transformation. |
| Missing values | Impute `avg_param_description_length`; handle empty/missing `tool_names`. |
| Categorical fields | Encode `task_complexity` and `task_domain` if using them in models. |
| Leakage prevention | `category` and original `task_id` were removed from this curated file. |

---

## `api_catalog.csv`

One row represents one API recommendation example from APIBench.

| Column | Type | Role | Use Directly As Model Input? | Description |
|---|---|---|---|---|
| `api_uid` | string | Identifier | No | Opaque row ID assigned after shuffling. |
| `source` | categorical | Metadata | No for provider/framework classifier | Source collection: `huggingface`, `tensorflow`, or `torchhub`. Useful for EDA. |
| `split` | categorical | Train/eval split | No | Original APIBench split. Use `train` rows for training and `eval` rows for testing in Section E. |
| `instruction` | string | Raw text | No | Natural-language request for an API recommendation. Use only to engineer simple text-derived features. |
| `instruction_length` | integer | Numeric feature | Yes | Number of characters in `instruction`. |
| `instruction_word_count` | integer | Numeric feature | Yes | Number of words in `instruction`. |
| `instruction_question_count` | integer | Numeric feature | Yes | Number of `?` characters in `instruction`. |
| `instruction_keyword_hits` | integer | Numeric feature | Yes | Count of simple API/model/library-related keywords. |
| `provider` | categorical | Target option / label | No if predicting provider | API provider name. Students may choose this as the target in Section E. |
| `domain` | string / categorical | Descriptive metadata | Usually no | Original APIBench domain text. Useful for interpretation. |
| `framework` | categorical | Target option / label | No if predicting framework | Framework name. Students may choose this as the target in Section E. |
| `functionality` | string / categorical | Descriptive metadata | Usually no | API functionality/category, such as text classification or image classification. |
| `api_name` | string | API identifier | No | Specific API/model name. Too specific for most simple numeric models. |
| `api_call` | string | Raw code text | No | Example API call string. Useful for inspection, not required for modeling. |
| `num_api_arguments` | integer | Numeric feature | Yes | Number of arguments in the API call metadata. |
| `num_env_requirements` | integer | Numeric feature | Yes | Number of Python/environment requirements. |
| `has_example_code` | integer, 0/1 | Binary feature | Yes | 1 if example code is available, else 0. |
| `description_length` | integer | Numeric feature | Yes | Character length of the API description. |
| `performance_dataset` | string | Optional metadata | No | Dataset name or text from API performance metadata. Often missing or inconsistent. |
| `performance_accuracy` | string | Optional metadata | No | Accuracy/performance text. Often missing, non-numeric, or inconsistently formatted. |
| `task_domain` | categorical | Domain feature / join key | Usually no for classifier; yes for aggregation | Same broad keyword-derived domain key used in `agent_tool_tasks.csv`. |

Important notes:

| Issue | What Students Should Do |
|---|---|
| Train/test split | Use `split == "train"` for training and `split == "eval"` for testing. |
| Provider/framework targets | Choose either `provider` or `framework` as the prediction target, not both at once. |
| Rare classes | Students may group rare providers/frameworks into `Other`, but must explain the rule. |
| Text fields | Use simple text-derived features only; do not rely on external embedding models. |
| Domain comparison | Use `task_domain` for aggregated comparison with the agent dataset. |

---

## `domain_coverage_summary.csv` Optional Helper

This file is a precomputed aggregated comparison between `agent_tool_tasks.csv` and `api_catalog.csv`.

| Column | Type | Description |
|---|---|---|
| `task_domain` | categorical | Shared broad domain key. |
| `agent_tasks` | integer | Number of rows in `agent_tool_tasks.csv` for this domain. |
| `can_answer_rate` | float | Mean of `can_answer` for this domain. |
| `avg_query_words` | float | Mean query word count for agent tasks in this domain. |
| `avg_available_tools` | float | Mean number of available tools for agent tasks in this domain. |
| `api_rows` | integer | Number of rows in `api_catalog.csv` for this domain. |
| `avg_instruction_words` | float | Mean instruction word count for API examples in this domain. |
| `avg_api_arguments` | float | Mean number of API arguments for this domain. |
| `top_provider` | string | Most frequent API provider in this domain, or `none` when no API rows exist. |

Students can use this file as a check, but a stronger submission should show how to recreate it with `groupby` and an aggregated join.
