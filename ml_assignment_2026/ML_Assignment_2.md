# Machine Learning – Assignment 2

**Topic:** Predicting whether an AI agent can answer a user's request with the tools it was given.

---

## Data

**Data source:** Berkeley Function-Calling Leaderboard (BFCL v4) — a benchmark used to evaluate how well large-language-model "agents" call external tools/functions. The original benchmark is published by the Gorilla LLM team (UC Berkeley).

You are given a single CSV file: **`agent_tasks.csv`** (≈ 3,500 rows).

Each row describes one **agent task**: a user message together with the list of tools/functions that the agent was offered. The columns are:

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `task_id` | string | Unique id of the task in the original benchmark. |
| 2 | `category` | string | The benchmark sub-category the task comes from. One of: `simple`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`, `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_irrelevance`, `live_relevance`. |
| 3 | `is_live_benchmark` | int (0/1) | 1 if the task comes from the *live* part of the benchmark (real user prompts collected in the wild), 0 if it comes from the *static* hand-curated part. |
| 4 | `query` | string | The raw natural-language request the user wrote. |
| 5 | `query_char_length` | int | Number of characters in `query`. |
| 6 | `query_word_count` | int | Number of words in `query`. |
| 7 | `query_question_marks` | int | Number of `?` characters in `query`. |
| 8 | `query_digit_count` | int | Number of digit characters (0–9) in `query`. |
| 9 | `query_uppercase_words` | int | Number of fully-uppercase words in `query` (e.g. `USA`, `API`). |
| 10 | `num_available_tools` | int | Number of tools/functions offered to the agent for this task. |
| 11 | `tool_names` | string | Pipe-separated (`|`) list of the tool names. May be empty (a few degenerate BFCL tasks have no tools). |
| 12 | `total_params` | int | Total number of parameters across all available tools. |
| 13 | `total_required_params` | int | Total number of *required* parameters across all available tools. |
| 14 | `num_string_params` | int | How many parameters are of type `string`. |
| 15 | `num_numeric_params` | int | How many parameters are numeric (`integer` or `float`). |
| 16 | `num_boolean_params` | int | How many parameters are of type `boolean`. |
| 17 | `num_array_params` | int | How many parameters are of type `array`. |
| 18 | `num_object_params` | int | How many parameters are of type `object`/`dict`. |
| 19 | `num_enum_params` | int | How many parameters are constrained to an `enum` (a fixed set of values). |
| 20 | `has_default_values` | int (0/1) | 1 if at least one parameter has a default value, else 0. |
| 21 | `avg_param_description_length` | float | Average length (in characters) of the parameter descriptions, across all tools of the task. **May be missing** when no tool has any parameters. |
| 22 | `tool_description_total_length` | int | Sum of the character lengths of the tool descriptions. |
| 23 | `label_is_relevant` | int (0/1) | **The label.** 1 if at least one of the available tools is relevant to the user's request, 0 if none of the tools can answer the request (the agent should refuse). |

> **Background — what is "relevance" in agent benchmarks?**
> A common failure mode of LLM agents is that they "hallucinate" a tool call even when none of the tools can actually fulfil the user's request. The `irrelevance` / `live_irrelevance` parts of BFCL contain user requests where **no available tool is suitable**, and the correct agent behaviour is to *not* call any tool. The `relevance` / `simple` / `multiple` / … parts contain requests where at least one tool *is* relevant.

**Your task** is to predict the binary label `label_is_relevant` from the other features (Sections C and the bonus). You will also explore the structure of the tasks via clustering (Section D).

> **Note:** The label counts are imbalanced (≈ 68% relevant / 32% irrelevant). You are expected to handle this as we learned in class, or — if you choose not to — provide a clear justification.

---

## Requirements

### Section A — Data Exploration & Visualisation (10 pts)

Explore the data using tables, visualisations, and other relevant methods.

- Plots must have an informative main title, axis labels, and a legend (when needed).
- For each plot or table, write a short description of the **key observations**. Only include content that is meaningful / informative.
- The visualisations should cover all relevant aspects of the data, including basic statistics (mean, median, mode, …) of each feature.
- Perform **at least 5 visualisations** of **at least 3 different plot types**.
- Include at least one visualisation that compares the two classes of `label_is_relevant`, and at least one that compares the static vs. live benchmark splits.

The goal is to get insights on the data which may or may not be useful for the next sections.

---

### Section B — Data Pre-processing (30 pts)

Apply pre-processing to prepare the data for the models in the next sections. The quality of this section directly affects your scores in Sections C and D.

#### B.1 Feature engineering

Add the following **5 mandatory engineered features**, and explain *why* each one might help the model:

1. **`required_params_ratio`** — `total_required_params / total_params`. If `total_params == 0`, set this to 0.
2. **`avg_params_per_tool`** — `total_params / num_available_tools`. If `num_available_tools == 0`, set this to 0.
3. **`query_avg_word_length`** — average word length in characters in the `query` field.
4. **`query_mentions_number`** — 1 if `query_digit_count > 0`, else 0.
5. **`tool_name_diversity`** — number of *unique tool name prefixes* in the task, where the prefix is everything before the first `.` of a tool name (e.g. `math.factorial` → `math`, `get_weather` → `get_weather`). Use the `tool_names` column.

In addition, design **at least 6 more features** of your own choosing. For each one, explain in one or two sentences what it captures and why you expect it to help the model.

> **Note:** Trivial variants of an existing feature (e.g. `query_char_length / query_word_count` if you already have `query_avg_word_length`) count as the same feature.

#### B.2 Cleaning, transformation and feature selection

Apply **at least one of each** of the following to the data:

- **One imputation method** (the column `avg_param_description_length` has missing values — you must handle them; explain your choice).
- **One transformation** (e.g. scaling, log-transform, one-hot/ordinal encoding of `category`, …).
- **One feature exclusion / selection** step (e.g. drop columns that should not be model inputs, drop highly correlated features, use `SelectKBest`, …).

For each step, briefly explain **what** you did and **why**. Your choices should reflect understanding of the method.

> Reminder: not every column in the CSV should be used as a model input. In particular, `task_id` and `query` are identifiers / raw text, not direct features. The column `tool_names` is also raw text — you may use it to derive features (as in B.1) but should not feed it as-is into a numerical model. The column `category` strongly leaks the label (the `*irrelevance` categories are exactly the irrelevant class) — you **must not** use `category` (or any feature directly derived from it) as a model input in Section C. You may, however, use `is_live_benchmark`.

> **IMPORTANT (next sections):** You do **not** have to implement the models yourself. You may use the implementations in `scikit-learn`. If you use a different library, briefly explain it.

---

### Section C — Classifying Relevant vs. Irrelevant Tasks (25 pts)

Train **at least three** different supervised models from class to predict `label_is_relevant`. At least one of them must be a **boosting** model (AdaBoost or Gradient Boosting) and at least one must be **k-Nearest-Neighbours (k-NN)**. The third model is up to you, from the supervised models we covered in class.

- Implementation **must include hyper-parameter tuning** (at least 2 hyper-parameters per model, with at least 3 values each).
- **Split the data into 80% / 10% / 10% train / validation / test**, stratified by `label_is_relevant`. Use `random_state=42` for reproducibility. Train only on the train set; tune hyper-parameters on the validation set; report final results on the test set.
- For k-NN, also report the result of using either an **LSH** approximate-nearest-neighbour index *or* a **kd-tree** index, and compare its inference-time / accuracy trade-off against brute-force k-NN. *Note: at the size and dimensionality of this dataset (d ≈ 27 features, n ≈ 3,000 train points) the index-based approach will not necessarily be faster than brute-force — your job is to **measure** and **explain** what you observe.*
- Choose and **justify a suitable evaluation metric** (remember the class imbalance!). Report it for all three models on the test set.
- Present the test-set results of the three models in a single comparison plot (e.g. a bar chart of the metric you chose).
- Discuss the results: which model worked best and why might that be?

#### Bonus — Feature Importance (10 pts)

Not all features affect model performance equally. Perform a feature-importance analysis using **at least one** of:

- correlation between each feature and the label,
- drop-column importance (re-train without each feature and check the change in score),
- correlation between features (to find redundant ones),
- a model-based importance (e.g. `feature_importances_` of a tree ensemble, permutation importance, SHAP),
- any other method of your choice (briefly explain it).

Report your findings, point out the 3 most useful and the 3 least useful features, and discuss.

---

### Section D — Clustering Agent Tasks (25 pts)

Apply **at least two** clustering algorithms covered in class to the data, in order to find natural groups of agent tasks. You may use **K-Means**, **agglomerative hierarchical clustering**, or **Gaussian Mixture Models (GMM)** — pick at least two.

Use the same engineered features as in Section C (you may add or remove features if you justify it). **Do not** use `label_is_relevant`, `category` or `task_id` as input features for clustering (they would either leak the label or are not informative).

- Tune at least one hyper-parameter per algorithm (e.g. number of clusters `k` for K-Means / GMM, the linkage method for agglomerative clustering, …).
- Use a clustering-quality measure of your choice (e.g. silhouette score, Davies-Bouldin, BIC for GMM, elbow plot, …) to pick the final hyper-parameters. Visualise the result of your tuning (e.g. silhouette vs. `k` plot).
- Identify the **most important features** that drive the differences between clusters, and find a way to visualise the clusters in 2D (e.g. by projecting onto the two most informative features, or onto two engineered features of your choice).
- Cross-tabulate your final cluster assignments with the `category` column from the original CSV. Are some clusters dominated by certain categories (e.g. `parallel*` tasks, `irrelevance` tasks)? Discuss.
- Try to **interpret each cluster** in plain English (e.g. "small math-style tasks with one tool and a numeric query", "live tasks with many tools and long descriptions", …).

#### Presentation (10 pts) — *mandatory, not a bonus!*

Create a short presentation (**no more than 6 slides**) with the most interesting findings of your work. A few presentations will be picked to be presented in front of the class.

---

### Section E — Exploring Tools — Bonus (15 pts)

In this section you will switch the unit of analysis from *tasks* to *tools*.

- From the `tool_names` column, build a **new dataset** where each row represents one **unique tool name** (use the *full* tool name, e.g. `math.factorial`, not just the prefix). Useful per-tool features include: number of tasks the tool appears in, average number of co-available tools when it is offered, fraction of tasks (containing this tool) whose label is `relevant`, the prefix / "domain" of the tool, average `query_word_length` of tasks where the tool appears, etc. Engineer at least 5 features.
- Formulate a clear question that can be asked about this per-tool dataset (a classification, regression *or* clustering task — your choice).
- Suggest and apply a machine-learning algorithm that answers your question. If you use a method we have not covered in the course, include a reference to where you studied it.
- Discuss the results and reflect on your question and your choice of method.

---

## Guidelines

### Coding guidelines

- Use familiar packages with explicit explanations.
- If you install any library beyond those used in the exercises, mention it in the report.
- The code should run without warnings or errors.
- Good documentation is critical.
- Indicate the section of the assignment in the code (e.g. as comments / markdown headers).
- Use meaningful variable names; do not use Python reserved words; use constants where appropriate.

### Submission guidelines

- The assignment should be submitted **in pairs** (only one submission per pair).
- You must submit **two files** containing all sections: one in `.ipynb` format and one in `.html` format. Both files must include the program's outputs. In addition, upload a **PDF of the presentation** (Section D).
- File names should be of the form: `ML_HW2_ID1_ID2.ipynb` / `.html` / `.pdf`.
- Late assignments will receive a penalty of **3 points per day**, up to one week. Later submissions will not be accepted.

### Grading

You can earn more than 100 points on this exercise. Grading is based on correctness, clarity, efficiency, and elegance of implementation.

### Self-learning, LLMs, and collaboration

Self-learning is an important part of the course — treat all sources critically. **Use of LLMs is allowed**, but you must reference where you used them. You are encouraged to discuss with other students, but each pair must write its own work. It is reasonable to assume that not all results and algorithms will be identical between submissions.

### Questions & reception hours

- Post questions on the exercise forum on Moodle (after reading previous posts). Professional questions sent by email will not be answered.
- For reception hours, email the instructor in advance with your questions.
- For personal matters (extension requests with justified reasons, etc.), email the instructor.

**Good luck!**
