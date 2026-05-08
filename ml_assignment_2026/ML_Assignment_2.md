# Machine Learning - Assignment 2

## Should an AI agent call a tool, or refuse?

Modern AI agents are often given a list of external tools: search APIs, calendar APIs, weather APIs, ML model APIs, finance APIs, database APIs, and so on. A useful agent must do two things well:

1. Call a tool when at least one available tool can help answer the user.
2. Refuse to call a tool when none of the available tools are relevant.

In this assignment you will build machine-learning models for this decision.

The target variable is:

**`can_answer`**

* `1`: at least one available tool is relevant to the user request.
* `0`: none of the available tools are relevant, so the agent should not call a tool.

This is a binary classification problem with an important practical meaning: a false positive means the agent calls a tool when it should refuse.

---

## Data Files

Use the two required curated CSV files below:

1. **`agent_tool_tasks.csv`** - primary dataset for Sections A-D.
2. **`api_catalog.csv`** - secondary dataset for Section E.

There is also an optional helper file, **`domain_coverage_summary.csv`**, which contains a precomputed domain-level summary for Section E. You can use it as a check, but you should be able to recreate the same table from the two required CSVs.

For the full column-level data dictionary, see **`DATA_DICTIONARY.md`**.

The original raw files are kept in the folder for transparency, but your work should use `agent_tool_tasks.csv` and `api_catalog.csv`.

### Why these curated files?

The raw benchmark data contained two direct leakage risks:

* `category` revealed the label because categories such as `irrelevance` directly map to `can_answer = 0`.
* `task_id` values exposed source/category names such as `irrelevance_123`.

The curated `agent_tool_tasks.csv` removes these fields, shuffles the rows, and assigns opaque IDs such as `task_00001`. This makes the prediction task more realistic.

---

## Dataset 1: `agent_tool_tasks.csv`

Each row is one agent decision: a user query plus the tools available to the agent.

| Column | Description |
|---|---|
| `task_uid` | Opaque row ID. Do not use as a model feature. |
| `is_live_benchmark` | 1 if the request came from the live benchmark, 0 otherwise. |
| `task_complexity` | Low / medium / high bucket derived from query length, number of tools, and number of parameters. Useful for EDA. |
| `query` | Raw user request. Use for feature engineering only. Do not feed raw text directly into numeric models unless you explicitly transform it. |
| `query_char_length` | Number of characters in the query. |
| `query_word_count` | Number of words in the query. |
| `query_question_marks` | Number of `?` characters in the query. |
| `query_digit_count` | Number of digit characters in the query. |
| `query_uppercase_words` | Number of fully uppercase words. |
| `num_available_tools` | Number of tools offered to the agent. |
| `tool_names` | Pipe-separated list of tool names. May be empty. Use for feature engineering only. |
| `total_params` | Total number of parameters across all available tools. |
| `total_required_params` | Total number of required parameters across all tools. |
| `num_string_params` | Number of string parameters. |
| `num_numeric_params` | Number of numeric parameters. |
| `num_boolean_params` | Number of boolean parameters. |
| `num_array_params` | Number of array/list parameters. |
| `num_object_params` | Number of object/dict parameters. |
| `num_enum_params` | Number of enum-constrained parameters. |
| `has_default_values` | 1 if at least one parameter has a default value. |
| `avg_param_description_length` | Average parameter-description length. Has a few missing values. |
| `tool_description_total_length` | Total length of all tool descriptions. |
| `task_domain` | Broad domain inferred by a keyword classifier, such as `vision`, `nlp`, `travel`, `math`, or `other`. |
| `can_answer` | Target label. 1 means a relevant tool exists; 0 means the agent should refuse. |

The target is moderately imbalanced: about 68% of tasks are answerable and 32% are not answerable.

---

## Dataset 2: `api_catalog.csv`

Each row is an instruction-to-API example from APIBench.

Important columns:

| Column | Description |
|---|---|
| `api_uid` | Opaque row ID. |
| `source` | Source collection: HuggingFace, TensorFlow Hub, or Torch Hub. |
| `split` | Original `train` / `eval` split. |
| `instruction` | Natural-language request for an API recommendation. |
| `instruction_word_count` | Number of words in the instruction. |
| `instruction_keyword_hits` | Count of simple API-related keywords. |
| `provider` | API provider name. |
| `framework` | Framework name. |
| `domain` | Original APIBench domain text. |
| `functionality` | Functionality/category of the API. |
| `api_name` | API/model name. |
| `num_api_arguments` | Number of API arguments. |
| `num_env_requirements` | Number of environment requirements. |
| `has_example_code` | 1 if example code exists. |
| `description_length` | Length of the API description. |
| `task_domain` | Same broad domain key used in `agent_tool_tasks.csv`. |

The two datasets do not share a row-level key. They can only be compared through aggregated `task_domain` statistics.

---

## Dataset Caveats

Good ML work includes understanding dataset limitations.

You must mention these caveats in your report:

* `can_answer` is derived from benchmark construction, not from a new manual annotation process.
* `task_domain` is a keyword-derived domain label. It is useful, but it is not a perfect semantic annotation.
* `api_catalog.csv` is mostly about ML APIs, while `agent_tool_tasks.csv` contains many everyday tools such as travel, scheduling, weather, math, communication, and finance.
* Empty or missing `tool_names` should be handled explicitly.
* Some features are correlated. This is expected, but should be considered during feature selection.

---

# Requirements

## Section A - Data Exploration and Visualization (10 pts)

Explore `agent_tool_tasks.csv`.

You must include:

* At least **5 visualizations**.
* At least **3 different plot types**.
* At least one plot comparing `can_answer = 0` vs `can_answer = 1`.
* At least one plot comparing live vs non-live tasks using `is_live_benchmark`.
* At least one plot or table about `task_domain`.
* At least one plot or table about `task_complexity`.

For every plot or table, write 2-4 sentences explaining the key observation. Avoid describing the obvious; focus on what the result teaches you about the data.

Suggested questions:

* Are answerable and non-answerable tasks different in query length?
* Do live benchmark tasks look different from non-live tasks?
* Which domains are common?
* Are some domains more likely to be answerable?
* Are tool-heavy tasks easier or harder to classify?

---

## Section B - Preprocessing and Feature Engineering (20 pts)

Prepare the data for modeling.

### B.1 Mandatory engineered features

Create all 5 features below:

1. **`required_params_ratio`**  
   `total_required_params / total_params`. If `total_params == 0`, set to 0.

2. **`avg_params_per_tool`**  
   `total_params / num_available_tools`. If `num_available_tools == 0`, set to 0.

3. **`query_avg_word_length`**  
   Average word length in the raw `query`. Compute this from tokenized words, not by dividing total character count by word count.

4. **`query_mentions_number`**  
   1 if `query_digit_count > 0`, otherwise 0.

5. **`tool_name_diversity`**  
   Number of unique tool-name prefixes. The prefix is everything before the first `.`. For example, `math.factorial` has prefix `math`; `get_weather` has prefix `get_weather`.

### B.2 Your own features

Create at least **4 additional features** of your choice.

Examples:

* Number of tool-name tokens.
* Whether the query looks like a question.
* Ratio of numeric parameters to all parameters.
* Log-transformed `total_params`.
* Average tool-description length per tool.
* Number of different parameter types used by the available tools.

For each feature, explain what it captures and why it might help.

### B.3 Cleaning and transformation

You must perform and explain:

* One imputation step, including `avg_param_description_length`.
* One transformation step, such as scaling numeric features or one-hot encoding categorical features.
* One feature exclusion or feature selection step.

Do not use these fields directly as model inputs:

* `task_uid`
* `query`
* `tool_names`
* `can_answer`

You may use `query` and `tool_names` to derive features.

You may choose whether to use `task_domain` and `task_complexity`, but if you use them, encode them properly and explain your choice.

---

## Section C - Predicting Whether The Agent Can Answer (30 pts)

Train supervised models to predict `can_answer`.

### C.1 Data split

Use:

* 80% train
* 10% validation
* 10% test

The split must be stratified by `can_answer`.

Use `random_state = 42`.

Train models only on the train set. Use the validation set for choosing hyperparameters. Report final performance once on the test set.

### C.2 Models

Train at least **3 supervised models**:

1. **k-Nearest Neighbors**
2. **One boosting model** such as AdaBoost or Gradient Boosting
3. **One additional model** covered in class, such as logistic regression, decision tree, random forest, naive Bayes, or SVM

Also include a simple baseline, such as predicting the majority class.

### C.3 Hyperparameter tuning

For each of the 3 main models:

* Tune at least **2 hyperparameters**.
* Try at least **3 values** for each tuned hyperparameter.
* Choose the best setting using the validation set.

### C.4 Evaluation

Report all of the following on the test set:

* Confusion matrix.
* Accuracy.
* Precision and recall for both classes.
* F1-score for both classes.

You must choose one main metric and justify it.

Because the important failure case is calling a tool when none is relevant, you should pay special attention to:

* Recall for `can_answer = 0`.
* F1-score for `can_answer = 0`.
* The number of false positives.

Create one comparison plot showing the test-set performance of all models.

Discuss:

* Which model worked best?
* Did the best model improve meaningfully over the baseline?
* What types of mistakes are most concerning?
* What would you change if false positives were very expensive?

---

## Section D - Clustering Agent Tasks (20 pts)

Use clustering to discover natural groups of agent tasks.

Use the same cleaned and engineered features from Section C, but do not include:

* `can_answer`
* `task_uid`
* raw `query`
* raw `tool_names`

You may include or exclude `task_domain` and `task_complexity`, but explain the choice.

### D.1 Algorithms

Apply at least **2 clustering algorithms** covered in class.

Recommended choices:

* K-Means
* Gaussian Mixture Models
* Agglomerative clustering

### D.2 Choosing the number of clusters

For each clustering algorithm:

* Try several values of `k` or `n_components`.
* Plot WCSS or another appropriate clustering-quality curve.
* Plot the silhouette score.
* Choose the final number of clusters and justify it.

If WCSS and silhouette suggest different choices, explain which one you trust more and why.

### D.3 Interpretation

For your final clustering:

* Show cluster sizes.
* Show the mean or median feature values per cluster.
* Visualize clusters in 2D using two meaningful features or a dimensionality-reduction method taught in class.
* Interpret each cluster in plain English.

After clustering, you may compare clusters to `can_answer`, `task_domain`, or `is_live_benchmark` to understand them. Do not use these comparisons as inputs unless already justified in preprocessing.

---

## Section E - Cross-Dataset Domain Analysis (10 pts)

Use `agent_tool_tasks.csv` and `api_catalog.csv`. You may use `domain_coverage_summary.csv` as a helper or validation file, but the same table can be generated from the two required CSVs.

The goal is to understand how well the API catalog covers the domains found in agent-tool tasks.

### E.1 Domain coverage

Create or load an aggregated table by `task_domain` showing:

From `agent_tool_tasks.csv`:

* Number of agent tasks.
* Mean `can_answer`.
* Mean query word count.
* Mean number of available tools.

From `api_catalog.csv`:

* Number of API rows.
* Mean instruction word count.
* Mean number of API arguments.
* Most frequent provider.

Discuss:

* Which domains appear in both datasets?
* Which domains appear mostly or only in the agent dataset?
* Which domains appear mostly or only in the API catalog?
* Does high API coverage seem related to a higher `can_answer` rate?
* Why should we be careful about drawing causal conclusions here?

### E.2 API classifier

Train one classifier to predict either `provider` or `framework` in `api_catalog.csv`.

Requirements:

* Use the original `split` column: train on `train`, test on `eval`.
* Use numeric features and simple text-derived features only.
* Use at least one model covered in class.
* Report a confusion matrix, accuracy, precision, recall, and F1-score.
* Discuss the most confused provider/framework pairs.

You may group very rare providers into an `Other` class, but you must explain your rule.

### E.3 Connection back to the agent task

Write one paragraph answering:

How could information from the API catalog help an agent decide whether to call a tool or refuse?

---

## Presentation (10 pts)

Create a short presentation of no more than **6 slides**.

Your presentation should focus on the most important findings, not on every implementation detail.

Required slides:

1. Problem and data.
2. Most important EDA insight.
3. Best classification model and test results.
4. Most important error analysis.
5. Clustering or domain-coverage insight.
6. Final recommendation or limitation.

Submit the presentation as a PDF.

---

# Bonus Options

You may earn up to **20 bonus points** total.

## Bonus 1 - Feature Importance (up to 10 pts)

Analyze which features matter most.

Use at least one method:

* Correlation with the label.
* Model-based feature importance.
* Permutation importance.
* Drop-column importance.
* Correlation between features to identify redundancy.

Report:

* 3 most useful features.
* 3 least useful features.
* Whether engineered features helped.
* Any feature that seems suspicious or hard to interpret.

## Bonus 2 - Tool-Level Analysis (up to 10 pts)

Switch the unit of analysis from tasks to tools.

Build a new dataset where each row is one unique tool name from `tool_names`.

Create at least 5 tool-level features, such as:

* Number of tasks where the tool appears.
* Average number of co-available tools.
* Fraction of tasks with `can_answer = 1`.
* Average query length for tasks containing the tool.
* Tool-name prefix.
* Average parameter counts of tasks containing the tool.

Then formulate one ML question about tools and answer it using a method covered in class.

Example questions:

* Can we cluster tools into common vs rare tools?
* Can we predict whether a tool usually appears in answerable tasks?
* Which tools are most associated with refusal cases?

---

# Grading Summary

| Section | Points |
|---|---:|
| A. EDA and visualization | 10 |
| B. Preprocessing and feature engineering | 20 |
| C. Classification | 30 |
| D. Clustering | 20 |
| E. Cross-dataset domain analysis | 10 |
| Presentation | 10 |
| **Total** | **100** |
| Bonus 1. Feature importance | +10 |
| Bonus 2. Tool-level analysis | +10 |

---

# Code and Submission Guidelines

Submit:

1. Notebook: `ML_HW2_ID1_ID2.ipynb`
2. HTML export: `ML_HW2_ID1_ID2.html`
3. Presentation PDF: `ML_HW2_ID1_ID2.pdf`

All notebook outputs must be visible in the submitted HTML.

Your code should:

* Run from top to bottom without errors.
* Use clear section headers.
* Use meaningful variable names.
* Avoid using the test set during model selection.
* Explain any library used beyond the standard course libraries.

Major penalties:

* Using `can_answer` as an input feature.
* Using raw IDs as predictive features.
* Evaluating on the test set repeatedly while tuning.
* Submitting a notebook that does not run.
* Showing plots without explaining what they mean.

Late assignments receive a penalty of **3 points per day**, up to one week. Later submissions will not be accepted.

Use of LLMs is allowed, but you must briefly state how you used them.

Good luck.
