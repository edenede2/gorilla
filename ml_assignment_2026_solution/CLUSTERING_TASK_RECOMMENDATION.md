# Recommended Clustering Task Revision

## Goal

Make the clustering section more satisfying by asking students to cluster **agent-tool situations** rather than blindly clustering all rows with all available features.

The clustering question should be:

> What common types of tool-use situations exist in this dataset?

This connects clustering to the assignment story: deciding whether an agent should call a tool or refuse.

---

## Recommended Student Instructions

Use clustering to discover common types of agent-tool situations.

Unlike Section C, do not try to predict `can_answer`. Instead, use unsupervised learning to identify situation profiles such as:

* clear semantic match
* weak query/tool match
* complex multi-intent request
* risky action with sensitive query
* tool-heavy but low-alignment task

Use a focused feature set from at least three of these groups:

| Feature group | Example columns |
|---|---|
| Query/tool alignment | `aspect_coverage_ratio`, `aspect_overlap_count`, `aspect_mismatch_count`, `query_tool_token_jaccard`, `query_tool_action_overlap` |
| Query complexity | `query_word_count`, `query_multi_intent_score`, `query_specificity_score`, `query_unique_token_ratio` |
| Tool complexity | `num_available_tools`, `total_params`, `total_required_params`, `schema_rigidity_score`, `param_type_diversity` |
| Risk signals | `risky_tool_action_count`, `query_sensitive_data_signal`, `query_code_signal`, `query_temporal_signal` |

Do not use:

* `can_answer`
* `task_uid`
* raw `query`
* raw `tool_names`

Apply at least two clustering algorithms, such as:

* K-Means
* Gaussian Mixture Models
* Agglomerative clustering

For each algorithm:

* Try several values of `k` or `n_components`.
* Plot WCSS or another appropriate clustering-quality curve.
* Plot silhouette score.
* Choose a final number of clusters and justify it.

For the final clustering:

* Show cluster sizes.
* Show mean/median feature values per cluster.
* Compare clusters to `can_answer` only after clustering.
* Compare clusters to `task_domain` or `task_complexity`.
* Give each cluster a short descriptive name.

Suggested profile table:

| cluster | size | answer rate | key feature pattern | descriptive name |
|---|---:|---:|---|---|

---

## Why This Is Better

The generic clustering task tends to produce broad clusters such as short/simple vs long/complex. That is valid, but not very interesting.

This revised task produces clusters that are easier to interpret in the language of agent safety and tool relevance:

* high aspect coverage and high answer rate
* sensitive/code-heavy requests with weak tool alignment
* many tools but low query-tool overlap
* long multi-intent requests
* simple clear matches

This makes the clustering section support the main assignment story instead of feeling like a disconnected unsupervised-learning exercise.

---

## Recommended Grading Emphasis

Grade the clustering section mainly on:

* sensible feature choice,
* correct exclusion of `can_answer`,
* clear tuning/evaluation plots,
* cluster profile table,
* quality of plain-English interpretation.

Do not overemphasize getting a high silhouette score. The goal is meaningful unsupervised exploration, not supervised accuracy.
