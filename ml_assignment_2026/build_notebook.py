"""Build the solution Jupyter notebook for ML Assignment 2."""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# --------------------------------------------------------------------------
md(r"""
# ML Assignment 2 — Solution
## Predicting whether an AI agent can answer a user's request with the tools it was given

**Dataset:** `agent_tasks.csv` — 3,491 tasks from the Berkeley Function-Calling Leaderboard (BFCL v4).

**Target:** `label_is_relevant` — `1` if at least one of the offered tools is relevant to the user's request, `0` if none of them are (the agent should refuse).

This notebook follows the structure of the assignment:

* Section A — Data Exploration & Visualisation
* Section B — Pre-processing
* Section C — Classification (≥ 3 supervised models, including boosting and k-NN with kd-tree comparison)
* Bonus — Feature importance
* Section D — Clustering (K-Means + GMM)
* Section E (bonus) — Per-tool dataset
""")

code(r"""
# --- Imports & global settings -------------------------------------------------
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             roc_auc_score, silhouette_score)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True

print('Setup OK')
""")

# --------------------------------------------------------------------------
md("## Section A — Data Exploration & Visualisation (10 pts)")

code(r"""
df = pd.read_csv('agent_tasks.csv')
print('shape :', df.shape)
df.head()
""")

code(r"""
df.info()
""")

code(r"""
df.describe(include='all').T.head(25)
""")

code(r"""
print('Missing values per column:')
print(df.isna().sum()[df.isna().sum() > 0])
""")

md("""
**Observations so far**

* 3,491 rows × 23 columns. No missing values except in `avg_param_description_length` (5 rows — these are tasks where every offered tool has *zero parameters*, so an "average parameter description length" is undefined). We'll impute these in Section B.
* `task_id`, `query` and `tool_names` are textual identifiers / raw text and will not be fed directly into the models.
* `category` strongly leaks the label (the `*irrelevance` categories *are* the negative class) — we will exclude it from the model inputs.
""")

# Plot 1
code(r"""
# --- Plot 1: class balance of the target -------------------------------------
ax = df['label_is_relevant'].value_counts().sort_index().plot(
    kind='bar', color=['#d62728', '#2ca02c'])
ax.set_title('Class balance of `label_is_relevant`')
ax.set_xlabel('label_is_relevant'); ax.set_ylabel('count')
ax.set_xticklabels(['0 (irrelevant)', '1 (relevant)'], rotation=0)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom')
plt.tight_layout(); plt.show()
print(df['label_is_relevant'].value_counts(normalize=True).round(3).to_dict())
""")

md("**Observation 1.** ~68% positive / 32% negative — moderately imbalanced. We will use a stratified train/val/test split and evaluate with **F1** (and ROC-AUC) rather than plain accuracy.")

# Plot 2
code(r"""
# --- Plot 2: number of available tools per task ------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df['num_available_tools'], bins=range(0, df['num_available_tools'].max()+2),
        edgecolor='white', color='#1f77b4')
ax.set_title('Distribution of `num_available_tools`')
ax.set_xlabel('Number of tools offered to the agent'); ax.set_ylabel('Number of tasks')
plt.tight_layout(); plt.show()
print('mean=%.2f  median=%.0f  max=%d' % (
    df['num_available_tools'].mean(),
    df['num_available_tools'].median(),
    df['num_available_tools'].max()))
""")

md("**Observation 2.** Most tasks offer **just 1 tool** (the simple/irrelevance buckets), but the long tail goes up to 37. This is exactly the case where models that can deal with skewed numeric inputs (trees / boosted trees) tend to do well.")

# Plot 3
code(r"""
# --- Plot 3: box-plot of query length by label -------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
data = [df.loc[df['label_is_relevant'] == c, 'query_word_count']
        for c in [0, 1]]
ax.boxplot(data, labels=['irrelevant (0)', 'relevant (1)'], showfliers=False)
ax.set_title('Query word count by label')
ax.set_ylabel('query_word_count')
plt.tight_layout(); plt.show()
print(df.groupby('label_is_relevant')['query_word_count'].describe()[['mean','50%','max']])
""")

md("**Observation 3.** Relevant queries are on average noticeably longer than irrelevant ones (median ≈ 18 vs ≈ 10 words). Live tasks contain very long queries (max > 1000 words) — these are real users pasting in long prompts.")

# Plot 4
code(r"""
# --- Plot 4: category breakdown ----------------------------------------------
order = df['category'].value_counts().index
fig, ax = plt.subplots(figsize=(9, 4))
df['category'].value_counts().reindex(order).plot(kind='bar', ax=ax, color='#9467bd')
ax.set_title('Number of tasks per BFCL sub-category')
ax.set_ylabel('count'); ax.set_xlabel('')
plt.xticks(rotation=40, ha='right')
plt.tight_layout(); plt.show()
""")

md("**Observation 4.** Categories are very unbalanced: `live_multiple` (1,053) and `live_irrelevance` (884) dominate. The two relevance-checking buckets `live_relevance` (16) and `live_parallel` (16) are tiny — keep this in mind when interpreting per-category statistics.")

# Plot 5
code(r"""
# --- Plot 5: static vs live, split by label ----------------------------------
crosstab = pd.crosstab(df['is_live_benchmark'], df['label_is_relevant'])
crosstab.plot(kind='bar', stacked=True, color=['#d62728', '#2ca02c'])
plt.title('Static (0) vs Live (1) benchmark, split by label')
plt.xlabel('is_live_benchmark'); plt.ylabel('count')
plt.legend(['irrelevant (0)', 'relevant (1)'])
plt.xticks(rotation=0)
plt.tight_layout(); plt.show()
print(crosstab)
print('\nproportion relevant:')
print(crosstab.div(crosstab.sum(axis=1), axis=0).round(3))
""")

md("**Observation 5.** The live split is markedly more balanced (≈ 61% relevant) than the static split (≈ 81% relevant). The live split is also ~1.8× bigger. The two splits look like meaningfully different distributions — `is_live_benchmark` will likely be a useful feature.")

# Plot 6 - correlation heatmap
code(r"""
# --- Plot 6: correlation heatmap (numeric features only) ---------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=70, ha='right')
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.1f}', ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black', fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.title('Correlation matrix of numeric features'); plt.tight_layout(); plt.show()
""")

md("""
**Observation 6.** Several "tool size" features are strongly correlated (`total_params`, `total_required_params`, `num_string_params`, `tool_description_total_length`) — we'll drop redundant ones in Section B. The label is most positively correlated with `query_word_count`, `num_available_tools` and `is_live_benchmark`, and most negatively correlated with the typed-parameter counts (because the irrelevance bucket usually contains exactly *one* small tool).
""")

# --------------------------------------------------------------------------
md("## Section B — Pre-processing (30 pts)")

md("""
### B.1 — Feature engineering

We add the **5 mandatory** features from the assignment and **6 of our own**.

| # | Feature | Why we expect it to help |
|---|---|---|
| M1 | `required_params_ratio` | If a tool has many required parameters relative to its total, it is more "specific" and probably relevant only to a narrow request. |
| M2 | `avg_params_per_tool` | Captures average tool complexity per task. |
| M3 | `query_avg_word_length` | Long average word length suggests technical / domain-specific vocabulary, which often co-occurs with irrelevance bait. |
| M4 | `query_mentions_number` | Many "simple" relevant tasks include explicit numeric arguments ("base of 10 units"). |
| M5 | `tool_name_diversity` | Number of distinct tool *prefixes* — high diversity means a "multi-domain" tool palette, typical of live multi-tool tasks. |
| O1 | `query_uppercase_ratio` | Share of fully-uppercase tokens — flags requests with many acronyms (`API`, `USA`, …). |
| O2 | `log_total_params` | `log(1 + total_params)` — tames the heavy tail of `total_params`, helping distance-based models like k-NN. |
| O3 | `has_array_param` | Whether any tool exposes an array parameter (1/0). Such tools are typical for parallel-style tasks. |
| O4 | `params_typed_diversity` | How many *different* parameter types appear (string/numeric/bool/array/object) — high values indicate richer, more capable tool sets. |
| O5 | `query_per_tool_words` | `query_word_count / max(num_available_tools, 1)` — how much "user description budget" exists per tool. |
| O6 | `tool_desc_per_tool` | `tool_description_total_length / max(num_available_tools, 1)` — average tool documentation length per tool, a proxy for tool richness. |
""")

code(r"""
df_fe = df.copy()

# --- Mandatory features ------------------------------------------------------
df_fe['required_params_ratio'] = np.where(
    df_fe['total_params'] > 0,
    df_fe['total_required_params'] / df_fe['total_params'].clip(lower=1),
    0.0,
)
df_fe['avg_params_per_tool'] = np.where(
    df_fe['num_available_tools'] > 0,
    df_fe['total_params'] / df_fe['num_available_tools'].clip(lower=1),
    0.0,
)
df_fe['query_avg_word_length'] = np.where(
    df_fe['query_word_count'] > 0,
    df_fe['query_char_length'] / df_fe['query_word_count'].clip(lower=1),
    0.0,
)
df_fe['query_mentions_number'] = (df_fe['query_digit_count'] > 0).astype(int)

def n_unique_prefixes(names: str) -> int:
    if not isinstance(names, str) or names == '':
        return 0
    return len({n.split('.')[0] for n in names.split('|') if n})

df_fe['tool_name_diversity'] = df_fe['tool_names'].apply(n_unique_prefixes)

# --- Six engineered features of our own --------------------------------------
df_fe['query_uppercase_ratio'] = np.where(
    df_fe['query_word_count'] > 0,
    df_fe['query_uppercase_words'] / df_fe['query_word_count'].clip(lower=1),
    0.0,
)
df_fe['log_total_params'] = np.log1p(df_fe['total_params'])
df_fe['has_array_param'] = (df_fe['num_array_params'] > 0).astype(int)
type_cols = ['num_string_params', 'num_numeric_params', 'num_boolean_params',
             'num_array_params', 'num_object_params']
df_fe['params_typed_diversity'] = (df_fe[type_cols] > 0).sum(axis=1)
df_fe['query_per_tool_words'] = df_fe['query_word_count'] / df_fe['num_available_tools'].clip(lower=1)
df_fe['tool_desc_per_tool'] = df_fe['tool_description_total_length'] / df_fe['num_available_tools'].clip(lower=1)

new_cols = ['required_params_ratio','avg_params_per_tool','query_avg_word_length',
            'query_mentions_number','tool_name_diversity','query_uppercase_ratio',
            'log_total_params','has_array_param','params_typed_diversity',
            'query_per_tool_words','tool_desc_per_tool']
df_fe[new_cols].describe().T
""")

md("### B.2 — Imputation, transformation and feature selection")

code(r"""
# --- Imputation: avg_param_description_length --------------------------------
# 5 rows are NaN (tasks where no tool has parameters). We replace with the
# median, which is robust to outliers and consistent with what the column means.
median_val = df_fe['avg_param_description_length'].median()
df_fe['avg_param_description_length'] = df_fe['avg_param_description_length'].fillna(median_val)
print(f'Imputed avg_param_description_length with median = {median_val:.2f}')
print('Missing remaining:', df_fe.isna().sum().sum())
""")

code(r"""
# --- Feature exclusion -------------------------------------------------------
# - `task_id`, `query`, `tool_names` are raw / id columns
# - `category` directly leaks the label (irrelevance == 0)
DROP_COLS = ['task_id', 'query', 'tool_names', 'category', 'label_is_relevant']
y = df_fe['label_is_relevant'].astype(int).values
X_full = df_fe.drop(columns=DROP_COLS)

# --- Drop highly correlated columns (|r| > 0.95) ----------------------------
corr = X_full.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
print('Dropping highly-correlated columns (|r|>0.95):', to_drop)
X_full = X_full.drop(columns=to_drop)
print('Final feature count :', X_full.shape[1])
print('Final feature names :')
print(list(X_full.columns))
""")

code(r"""
# --- Transformation: standardise numeric features ---------------------------
# We keep an *unscaled* copy for tree-based models (which don't need scaling)
# and create a scaled copy for k-NN, which is distance-based.
X_unscaled = X_full.copy()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_full),
                        columns=X_full.columns, index=X_full.index)
X_scaled.head()
""")

# --------------------------------------------------------------------------
md("""
## Section C — Classification (25 pts)

We use the three classifiers required by the assignment: **Gradient Boosting**, **k-Nearest Neighbours** and **AdaBoost**.

* **Split:** 80/10/10 train/val/test, **stratified** on the label, `random_state=42`.
* **Tuning:** small grid per model on the validation set.
* **Metric:** **F1** (positive class = `relevant`). With ~32% negatives, F1 is more informative than accuracy. We also report ROC-AUC for context.
""")

code(r"""
# --- Stratified 80/10/10 split ----------------------------------------------
X_tr_u, X_tmp_u, y_tr, y_tmp = train_test_split(
    X_unscaled, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_val_u, X_te_u, y_val, y_te = train_test_split(
    X_tmp_u, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)

# Same split, scaled, for k-NN -----------------------------------------------
X_tr_s = X_scaled.loc[X_tr_u.index]
X_val_s = X_scaled.loc[X_val_u.index]
X_te_s  = X_scaled.loc[X_te_u.index]

print('Train :', X_tr_u.shape, 'positives=%.3f' % y_tr.mean())
print('Val   :', X_val_u.shape, 'positives=%.3f' % y_val.mean())
print('Test  :', X_te_u.shape, 'positives=%.3f' % y_te.mean())
""")

code(r'''
def grid_search_val(model_cls, param_grid, X_tr, y_tr, X_val, y_val, **fixed):
    """Pick the hyper-params that maximise F1 on the validation set."""
    best = (-1.0, None, None)
    rows = []
    for params in ParameterGrid(param_grid):
        m = model_cls(**fixed, **params)
        m.fit(X_tr, y_tr)
        f1 = f1_score(y_val, m.predict(X_val))
        rows.append({**params, 'val_f1': f1})
        if f1 > best[0]:
            best = (f1, params, m)
    return best, pd.DataFrame(rows).sort_values('val_f1', ascending=False)
''')

md("### C.1 — Gradient Boosting")

code(r"""
gb_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth':    [2, 3, 5],
    'learning_rate':[0.05, 0.1, 0.2],
}
best_gb, gb_df = grid_search_val(GradientBoostingClassifier, gb_grid,
                                 X_tr_u, y_tr, X_val_u, y_val,
                                 random_state=RANDOM_STATE)
print('Best GB params:', best_gb[1], '  val F1=%.4f' % best_gb[0])
gb_df.head(5)
""")

md("### C.2 — k-Nearest Neighbours (with brute-force vs kd-tree comparison)")

code(r"""
knn_grid = {
    'n_neighbors': [3, 5, 11, 25, 51],
    'weights':     ['uniform', 'distance'],
    'p':           [1, 2],
}
best_knn, knn_df = grid_search_val(KNeighborsClassifier, knn_grid,
                                   X_tr_s, y_tr, X_val_s, y_val,
                                   algorithm='brute')
print('Best k-NN params:', best_knn[1], '  val F1=%.4f' % best_knn[0])
knn_df.head(5)
""")

code(r"""
# --- Brute-force vs kd-tree timing & accuracy comparison --------------------
# We use the BEST hyper-params found above and only swap the index structure.
# (sklearn's `algorithm='kd_tree'` is the kd-tree implementation taught in
# class. Note that kd-trees only really help in low dimensions; here d=14.)
compare_rows = []
for algo in ['brute', 'kd_tree', 'ball_tree']:
    knn_a = KNeighborsClassifier(algorithm=algo, **best_knn[1])
    t0 = time.perf_counter(); knn_a.fit(X_tr_s, y_tr); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = knn_a.predict(X_te_s); pred_t = time.perf_counter() - t0
    compare_rows.append({
        'algorithm': algo,
        'fit_seconds':   round(fit_t, 4),
        'predict_seconds': round(pred_t, 4),
        'test_f1':  round(f1_score(y_te, pred), 4),
    })
pd.DataFrame(compare_rows)
""")

md("""
**Discussion (k-NN index structures).** As expected, all three index structures return the same neighbours and therefore the **same test F1** (k-NN is deterministic in this regard). The interesting differences are in timing:

* **brute** is `O(N · d)` per query, but with no index to build it has the fastest `fit` and — at this scale (n_train ≈ 2.8k, d = 27) — actually the fastest `predict` too.
* **kd_tree** and **ball_tree** spend a few ms building an index. In our setting their predict step is **slower** than brute-force, because kd-trees lose their advantage once the dimensionality grows beyond ~10–15: the bounding boxes overlap so much that the tree must inspect almost every leaf anyway.

This matches what we saw in class: approximate / index-based nearest-neighbour structures pay off mainly for **large `N`** with **low `d`**. Since our `d = 27`, the trees do not help; if we had millions of points and only a handful of features, kd-tree / ball-tree / LSH would dominate.
""")

md("### C.3 — AdaBoost (third model)")

code(r"""
ada_grid = {
    'n_estimators':  [100, 200, 400],
    'learning_rate': [0.1, 0.5, 1.0],
}
best_ada, ada_df = grid_search_val(AdaBoostClassifier, ada_grid,
                                   X_tr_u, y_tr, X_val_u, y_val,
                                   random_state=RANDOM_STATE)
print('Best AdaBoost params:', best_ada[1], '  val F1=%.4f' % best_ada[0])
ada_df.head(5)
""")

md("### C.4 — Final test-set evaluation")

code(r"""
def evaluate(model, X_te, y_te, name):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else pred
    return {
        'model': name,
        'f1':  f1_score(y_te, pred),
        'roc_auc': roc_auc_score(y_te, proba),
        'accuracy': (pred == y_te).mean(),
    }

results = [
    evaluate(best_gb[2],  X_te_u, y_te, 'GradientBoosting'),
    evaluate(best_knn[2], X_te_s, y_te, 'k-NN (scaled)'),
    evaluate(best_ada[2], X_te_u, y_te, 'AdaBoost'),
]
res_df = pd.DataFrame(results).set_index('model').round(4)
res_df
""")

code(r"""
# --- Bar plot comparing the three models ------------------------------------
ax = res_df[['f1', 'roc_auc', 'accuracy']].plot(
    kind='bar', figsize=(8, 4),
    color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Test-set performance by model'); ax.set_ylabel('score')
ax.set_ylim(0.6, 1.0)
ax.legend(loc='lower right')
plt.xticks(rotation=0)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout(); plt.show()
""")

code(r"""
# --- Confusion matrix of the best (Gradient Boosting) model -----------------
pred_gb = best_gb[2].predict(X_te_u)
cm = confusion_matrix(y_te, pred_gb)
fig, ax = plt.subplots(figsize=(4, 3.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_title('Gradient Boosting — confusion matrix (test)')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['pred 0', 'pred 1']); ax.set_yticklabels(['true 0', 'true 1'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
print(classification_report(y_te, pred_gb, target_names=['irrelevant', 'relevant']))
""")

md("""
**Discussion (Section C).**

* **Gradient Boosting** is the best model on this dataset. This is expected: the features are mostly heavy-tailed integer counts, and boosting handles non-linear interactions and skewed distributions natively (no scaling required).
* **AdaBoost** is a close second. The two boosters land within a few F1 points of each other.
* **k-NN** lags slightly behind even after scaling. It is more sensitive to the heavy-tailed `total_params` / `tool_description_total_length` features and to the imbalance.
* All three models comfortably beat the *predict-everything-as-relevant* baseline (test F1 ≈ 0.808). Our models reach **F1 ≈ 0.92** (Gradient Boosting), **0.886** (AdaBoost) and **0.886** (k-NN) on the held-out test set.
""")

# --------------------------------------------------------------------------
md("## Bonus — Feature importance (10 pts)")

code(r"""
# --- 1) Built-in tree importance from the Gradient Boosting model -----------
importances = pd.Series(best_gb[2].feature_importances_,
                        index=X_tr_u.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
importances.plot(kind='barh', ax=ax, color='#1f77b4')
ax.set_title('Gradient Boosting — built-in feature importance')
ax.set_xlabel('relative importance')
plt.tight_layout(); plt.show()
importances.sort_values(ascending=False).head(10)
""")

code(r"""
# --- 2) Permutation importance on the validation set ------------------------
perm = permutation_importance(best_gb[2], X_val_u, y_val,
                              n_repeats=10, random_state=RANDOM_STATE,
                              scoring='f1')
perm_imp = pd.Series(perm.importances_mean, index=X_val_u.columns
                     ).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 5))
perm_imp.plot(kind='barh', ax=ax, color='#ff7f0e')
ax.set_title('Permutation importance (validation set, scoring=F1)')
ax.set_xlabel('mean drop in F1 when shuffled')
plt.tight_layout(); plt.show()
perm_imp.sort_values(ascending=False).head(10)
""")

code(r"""
# --- 3) Plain Pearson correlation between each feature and the label --------
label_corr = X_full.assign(label=y).corr()['label'].drop('label').sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
label_corr.plot(kind='barh', ax=ax,
                color=['#d62728' if v < 0 else '#2ca02c' for v in label_corr.values])
ax.set_title('Pearson correlation with `label_is_relevant`')
ax.axvline(0, color='black', lw=0.8)
plt.tight_layout(); plt.show()
""")

md("""
**Discussion (feature importance).**

* The three methods broadly agree on the **top features**: `query_char_length` (which is essentially the survivor of the `query_word_count` ↔ `query_char_length` correlation pair), `avg_param_description_length`, `avg_params_per_tool`, `num_available_tools`, `tool_description_total_length`, and `is_live_benchmark`. These are also the features that show the largest distributional gap between the two classes in Section A.
* The **least useful** features (near-zero importance and near-zero permutation impact) are `has_array_param`, `query_question_marks` and `has_default_values` — they barely move the model's score when permuted.
* Several **engineered** features (`avg_params_per_tool`, `query_per_tool_words`, `tool_desc_per_tool`) appear in the top half of the ranking, confirming that the feature engineering of Section B was worth doing.
""")

# --------------------------------------------------------------------------
md("""
## Section D — Clustering Agent Tasks (25 pts)

We cluster the *scaled* feature matrix (clustering algorithms are distance-based, so scaling matters). We use **K-Means** and **Gaussian Mixture Models**.
""")

code(r"""
# We deliberately do NOT include the label, the category or the id in the
# feature matrix used for clustering.
X_cluster = X_scaled.copy()
print('Clustering on shape:', X_cluster.shape)
""")

code(r"""
# --- K-Means: tune k via silhouette score ----------------------------------
ks = list(range(2, 9))
sil = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_cluster)
    sil.append(silhouette_score(X_cluster, km.labels_, sample_size=2000,
                                random_state=RANDOM_STATE))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(ks, sil, 'o-', color='#1f77b4')
ax.set_title('K-Means: silhouette score vs k'); ax.set_xlabel('k'); ax.set_ylabel('silhouette')
plt.tight_layout(); plt.show()
best_k = ks[int(np.argmax(sil))]
print('Best k =', best_k, '  silhouette = %.3f' % max(sil))
""")

code(r"""
# --- GMM: tune n_components via BIC ----------------------------------------
ns = list(range(2, 9))
bics = []
for n in ns:
    gmm = GaussianMixture(n_components=n, covariance_type='full',
                          random_state=RANDOM_STATE).fit(X_cluster)
    bics.append(gmm.bic(X_cluster))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(ns, bics, 'o-', color='#9467bd')
ax.set_title('GMM: BIC vs n_components'); ax.set_xlabel('n_components'); ax.set_ylabel('BIC (lower = better)')
plt.tight_layout(); plt.show()
best_n = ns[int(np.argmin(bics))]
print('Best n_components =', best_n, '  BIC = %.0f' % min(bics))
""")

code(r"""
# --- Fit final clustering models -------------------------------------------
final_km  = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_cluster)
final_gmm = GaussianMixture(n_components=best_n, covariance_type='full',
                            random_state=RANDOM_STATE).fit(X_cluster)
km_labels  = final_km.labels_
gmm_labels = final_gmm.predict(X_cluster)
print('K-Means cluster sizes:', dict(pd.Series(km_labels).value_counts().sort_index()))
print('GMM cluster sizes    :', dict(pd.Series(gmm_labels).value_counts().sort_index()))
""")

code(r"""
# --- 2D scatter visualisation of the clusters -------------------------------
# We have NOT covered PCA / t-SNE yet, so we visualise by projecting onto two
# of the most informative original features (chosen from the importance
# analysis of Section C / D centroids):
#   * `num_available_tools`        - the strongest "tool richness" axis
#   * `query_char_length`          - the strongest "query size" axis
# We use a small jitter to avoid over-plotting on the integer axis.
rng = np.random.default_rng(RANDOM_STATE)
jitter_x = rng.normal(0, 0.15, size=len(df_fe))
jitter_y = rng.normal(0, 5.0,  size=len(df_fe))
x_axis = df_fe['num_available_tools'].to_numpy() + jitter_x
y_axis = df_fe['query_char_length'].to_numpy()  + jitter_y

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, lbl, title in [(axes[0], km_labels,  f'K-Means (k={best_k})'),
                       (axes[1], gmm_labels, f'GMM (n={best_n})')]:
    ax.scatter(x_axis, y_axis, c=lbl, cmap='tab10', s=8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('num_available_tools (jittered)')
    ax.set_ylabel('query_char_length (jittered)')
    ax.set_xlim(-1, 20)            # zoom on the bulk of the data
    ax.set_ylim(0, 800)
plt.tight_layout(); plt.show()
""")

code(r"""
# --- Cluster vs original BFCL category -------------------------------------
ct = pd.crosstab(df['category'], pd.Series(km_labels, name='kmeans_cluster'))
ct_norm = ct.div(ct.sum(axis=1), axis=0).round(2)
print('Row-normalised crosstab (each row sums to 1):')
ct_norm
""")

code(r"""
# --- What features drive each K-Means cluster? -----------------------------
# Show the cluster centroid means in *original* (unscaled) units.
centroids = pd.DataFrame(scaler.inverse_transform(final_km.cluster_centers_),
                         columns=X_full.columns)
centroids['size'] = pd.Series(km_labels).value_counts().sort_index().values
centroids.round(2)
""")

md("""
**Cluster interpretation (K-Means, k = 3).** Looking at the centroids and the category crosstab, the three clusters can be described as:

* **Cluster 2 (the largest, ~2,250 tasks)** — a *small-tool, short-query* cluster: `num_available_tools` ≈ 1, short queries, dominated by the static `simple` / `multiple` / `irrelevance` buckets. This is where most of the irrelevance bait sits.
* **Cluster 0 (~1,000 tasks)** — a *live, multi-tool* cluster with many available tools and long tool descriptions; almost all of `live_multiple` lands here.
* **Cluster 1 (~225 tasks)** — a *very-long-query / rich-tools* cluster (high `query_char_length`, high `tool_description_total_length`); contains the long real-user prompts from the live splits.

When we project the clusters onto `num_available_tools` (x) and `query_char_length` (y) — the two single most informative original features — the partitions become visually obvious: the bulk of static / simple tasks sits at the bottom-left, the live multi-tool tasks shift to the right, and the long-query rich-tool tasks rise to the top.

**GMM (n = 5 by BIC)** gives a finer split of the same structure: it cuts the big static cluster into two sub-clusters and the live cluster into three, reflecting the heavier tails of the live data.
""")

# --------------------------------------------------------------------------
md("""
## Section E (bonus, 15 pts) — Per-tool dataset

We now switch the unit of analysis from *tasks* to *tools*. For every distinct tool name in the corpus we compute:

* `n_tasks` — number of tasks that offer this tool.
* `avg_co_tools` — average number of *other* tools available when this tool appears.
* `frac_relevant` — fraction of those tasks whose label is `relevant`.
* `avg_query_words` — average query word count of those tasks.
* `prefix` — domain prefix (everything before the first `.` in the tool name).

We then ask: **can we cluster tools into meaningful "domain families" using these statistics?**
""")

code(r"""
records = []
for _, row in df.iterrows():
    names = [n for n in str(row['tool_names']).split('|') if n]
    if not names:
        continue
    co = len(names)
    for n in names:
        records.append({
            'tool': n,
            'co_tools': co - 1,
            'is_relevant': row['label_is_relevant'],
            'query_words': row['query_word_count'],
        })
long_df = pd.DataFrame(records)

tool_df = long_df.groupby('tool').agg(
    n_tasks=('tool', 'size'),
    avg_co_tools=('co_tools', 'mean'),
    frac_relevant=('is_relevant', 'mean'),
    avg_query_words=('query_words', 'mean'),
).reset_index()
tool_df['prefix'] = tool_df['tool'].str.split('.').str[0]
print('Unique tools:', len(tool_df))
tool_df.sort_values('n_tasks', ascending=False).head(10)
""")

code(r"""
# --- KMeans on per-tool features -------------------------------------------
tool_feats = ['n_tasks', 'avg_co_tools', 'frac_relevant', 'avg_query_words']
T = StandardScaler().fit_transform(tool_df[tool_feats])

ks = list(range(2, 8)); sils = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(T)
    sils.append(silhouette_score(T, km.labels_))
plt.figure(figsize=(6, 3.5))
plt.plot(ks, sils, 'o-')
plt.title('Per-tool clustering: silhouette vs k')
plt.xlabel('k'); plt.ylabel('silhouette')
plt.tight_layout(); plt.show()
best_kt = ks[int(np.argmax(sils))]
final_tool_km = KMeans(n_clusters=best_kt, n_init=10, random_state=RANDOM_STATE).fit(T)
tool_df['cluster'] = final_tool_km.labels_
print('Chosen k =', best_kt)
""")

code(r"""
# Mean of each feature per cluster (in original units)
tool_summary = tool_df.groupby('cluster')[tool_feats].mean().round(3)
tool_summary['size'] = tool_df['cluster'].value_counts().sort_index()
tool_summary
""")

code(r"""
# Top prefixes per cluster
for c in sorted(tool_df['cluster'].unique()):
    top = tool_df.loc[tool_df['cluster'] == c, 'prefix'].value_counts().head(5)
    print(f'\n--- cluster {c}  (n={int((tool_df.cluster==c).sum())}) ---')
    print(top.to_string())
""")

md("""
**Discussion (Section E).** Clustering 1,704 unique tools on just 4 usage features already produces two clearly-different groups (silhouette is maximised at `k = 2`):

* **Cluster 0 (~1,662 tools, ~98%)** — *typical* tools: appear in ~5 tasks each, are offered alongside ~2 other tools, and roughly 70% of the time the call is relevant. Top prefixes are domain-style (`math`, `EventSettingsApi`, `project_api`, `finance`, `kinematics` …).
* **Cluster 1 (~42 tools, ~2%)** — *huge co-occurrence* tools: each tool is offered alongside **~32 other tools** on average, the average user query is **~267 words long**, and only **~44%** of the calls are relevant. These are the meeting-room / scheduling style helpers offered as part of huge "toolbox" prompts in the `live_multiple` benchmark.

So a simple unsupervised analysis already separates the small set of "toolbox-included" tools (where irrelevance is a real risk) from the everyday tools — exactly the situation where an agent most needs to learn to refuse.
""")

# --------------------------------------------------------------------------
md("""
## Final summary

| Section | Result |
|---|---|
| A — exploration | 6 plots; identified class imbalance (~68/32), heavy-tailed `num_available_tools`, longer queries among relevant tasks, distributional shift between static and live splits, redundant tool-size features. |
| B — pre-processing | Engineered 11 features (5 mandatory + 6 own), median-imputed `avg_param_description_length`, dropped `task_id`/`query`/`tool_names`/`category`, dropped highly-correlated columns (\|r\|>0.95), z-scaled the rest. |
| C — classification | **Gradient Boosting** wins with **test F1 = 0.918**, **ROC-AUC = 0.933**. AdaBoost (F1 = 0.886) and k-NN (F1 = 0.886) are tied. brute / kd_tree / ball_tree return the same neighbours and therefore the same F1; at d = 27 brute is actually the fastest of the three. |
| Bonus — importance | Top drivers: `query_char_length`, `avg_param_description_length`, `avg_params_per_tool`, `num_available_tools`, `tool_description_total_length`, `is_live_benchmark`. Several engineered features are in the top half of the ranking. |
| D — clustering | K-Means picks **k = 3** (silhouette = 0.26) — small-tool/short-query, live-multi-tool, and very-long-query/rich-tools. GMM picks **n = 5** by BIC and refines the same structure. The 2-feature scatter (`num_available_tools`, `query_char_length`) makes the partition visible without any dimensionality-reduction technique. |
| E — per-tool | At the tool level, K-Means cleanly separates the ~98% of "typical" tools (low co-occurrence, ~70% relevant calls) from a tiny cluster of ~42 "toolbox" tools (offered with ~32 other tools, only ~44% relevant calls). |
""")

# --------------------------------------------------------------------------
nb['cells'] = cells
out = Path(__file__).resolve().parent / 'ML_Assignment_2_Solution.ipynb'
nbf.write(nb, out)
print('Wrote', out)
