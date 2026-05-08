# Focused Clustering Expansion Results

The focused clustering expansion was run in:

`ML_Assignment_2_Solution.ipynb`

Execution status: no notebook errors.

## Setup

The expansion clustered agent-tool situations using only focused feature families:

* query/tool alignment,
* query complexity,
* tool complexity,
* risk signals.

It did not use `can_answer` as an input. The label was used only after clustering to interpret cluster profiles.

K-Means was tested for `k = 2..8`.

| k | WCSS | silhouette |
|---:|---:|---:|
| 2 | 55070.436 | 0.177 |
| 3 | 48965.241 | 0.184 |
| 4 | 42967.383 | 0.207 |
| 5 | 39059.263 | 0.218 |
| 6 | 36003.814 | 0.214 |
| 7 | 33135.493 | 0.170 |
| 8 | 31751.542 | 0.170 |

`k = 5` was selected because it had the best silhouette score and gave interpretable situation profiles.

## Cluster Profiles

| Cluster | Size | Descriptive name | Answer rate | Key pattern |
|---:|---:|---|---:|---|
| 0 | 1,928 | simple low-signal situation | 0.608 | Low aspect coverage, low risk/code/sensitive signals, mostly simple tasks. |
| 1 | 777 | clear semantic match | 0.896 | Very high aspect coverage and higher token overlap. Strongly answerable. |
| 2 | 472 | tool-heavy situation | 0.756 | Many available tools and high risky-tool-action count. All high-complexity tasks. |
| 3 | 311 | sensitive/code-heavy weak match | 0.457 | High sensitive/code signals, low aspect coverage, weak token overlap. Most refusal-prone substantial cluster. |
| 4 | 3 | sensitive/code-heavy weak match | 0.000 | Tiny outlier cluster with extreme multi-intent/sensitive/code signal. |

## Target Counts By Cluster

| Cluster | Refuse (`0`) | Can answer (`1`) |
|---:|---:|---:|
| 0 | 756 | 1,172 |
| 1 | 81 | 696 |
| 2 | 115 | 357 |
| 3 | 169 | 142 |
| 4 | 3 | 0 |

## Conclusion

This clustering task is more satisfying than generic all-feature clustering.

Why:

* The clusters have clearer plain-English meanings.
* The profiles connect directly to the agent-tool decision.
* `can_answer` rates differ meaningfully across clusters.
* The sensitive/code-heavy weak-match cluster is especially interesting because it is the most refusal-prone substantial group.
* The clear semantic-match cluster is strongly answerable, which validates the usefulness of the aspect-overlap features.

Recommended assignment change:

Replace the generic clustering section with the focused agent-tool situation clustering task described in `CLUSTERING_TASK_RECOMMENDATION.md`.
