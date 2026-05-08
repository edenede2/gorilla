"""Compare the old simple feature set with the new semantic/aspect feature set."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT.parent / "ml_assignment_2026"
RANDOM_STATE = 42


def minmax(s: pd.Series) -> pd.Series:
    low = float(s.min())
    high = float(s.max())
    if high == low:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - low) / (high - low)


def load_old_simple_dataset() -> pd.DataFrame:
    raw = pd.read_csv(RAW_ROOT / "agent_tasks.csv")
    df = raw.copy()
    df["tool_names"] = df["tool_names"].fillna("")
    df["avg_param_description_length"] = pd.to_numeric(
        df["avg_param_description_length"], errors="coerce"
    )
    complexity_score = (
        0.35 * minmax(df["query_word_count"])
        + 0.35 * minmax(df["num_available_tools"])
        + 0.30 * minmax(df["total_params"])
    )
    df["task_complexity"] = pd.qcut(
        complexity_score.rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)
    df = df.rename(columns={"label_is_relevant": "can_answer", "task_type": "task_domain"})
    df = df.drop(columns=["task_id", "category"])
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df.insert(0, "task_uid", [f"task_{i:05d}" for i in range(1, len(df) + 1)])
    return df[
        [
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
    ]


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def avg_word_length(text):
    tokens = TOKEN_RE.findall(str(text))
    if not tokens:
        return 0.0
    return float(np.mean([len(t) for t in tokens]))


def uppercase_token_ratio(text):
    tokens = re.findall(r"[A-Za-z]+", str(text))
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t.isupper() and len(t) > 1) / len(tokens)


def unique_tool_prefix_count(names):
    if not isinstance(names, str) or not names.strip():
        return 0
    prefixes = []
    for name in names.split("|"):
        name = name.strip()
        if not name:
            continue
        prefixes.append(name.split(".")[0])
    return len(set(prefixes))


def tool_name_token_count(names):
    if not isinstance(names, str) or not names.strip():
        return 0
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", names)
    normalized = re.sub(r"[_./|:;,-]+", " ", normalized)
    return len(re.findall(r"[A-Za-z0-9]+", normalized))


def add_assignment_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tool_names"] = out["tool_names"].fillna("")
    out["required_params_ratio"] = np.where(
        out["total_params"] > 0,
        out["total_required_params"] / out["total_params"],
        0.0,
    )
    out["avg_params_per_tool"] = np.where(
        out["num_available_tools"] > 0,
        out["total_params"] / out["num_available_tools"],
        0.0,
    )
    out["query_avg_word_length"] = out["query"].apply(avg_word_length)
    out["query_mentions_number"] = out["query"].str.contains(r"[0-9]", regex=True).astype(int)
    out["tool_name_diversity"] = out["tool_names"].apply(unique_tool_prefix_count)
    out["query_has_question"] = out["query"].str.contains("?", regex=False).astype(int)
    out["query_uppercase_ratio"] = out["query"].apply(uppercase_token_ratio)
    out["log_total_params"] = np.log1p(out["total_params"])
    out["no_available_tools_flag"] = (out["num_available_tools"] == 0).astype(int)
    out["required_params_per_tool"] = np.where(
        out["num_available_tools"] > 0,
        out["total_required_params"] / out["num_available_tools"],
        0.0,
    )
    out["quoted_string_count"] = out["query"].str.count("'") // 2 + out["query"].str.count('"') // 2
    long_query_cutoff = out["query_word_count"].quantile(0.75)
    out["long_query_flag"] = (out["query_word_count"] > long_query_cutoff).astype(int)
    out["tool_desc_per_tool"] = np.where(
        out["num_available_tools"] > 0,
        out["tool_description_total_length"] / out["num_available_tools"],
        0.0,
    )
    out["tool_name_token_count"] = out["tool_names"].apply(tool_name_token_count)
    return out


def tune_model(model_cls, param_grid, X_tr, y_tr, X_va, y_va, fixed=None):
    fixed = fixed or {}
    best = {"score": -np.inf, "params": None, "model": None}
    for params in ParameterGrid(param_grid):
        model = model_cls(**fixed, **params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        score = f1_score(y_va, pred, pos_label=0)
        if score > best["score"]:
            best = {"score": score, "params": params, "model": model}
    return best


def evaluate(name, model, X_test, y_test, feature_set):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return {
        "feature_set": feature_set,
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "refusal_f1": f1_score(y_test, pred, pos_label=0),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
    }


def run_feature_set(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    df = add_assignment_engineered_features(df)
    X_raw = df.drop(columns=["task_uid", "query", "tool_names", "can_answer"])
    y = df["can_answer"].astype(int)

    categorical_cols = X_raw.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X_raw, columns=categorical_cols, dtype=int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > 0.97).any()]
    X_train = X_train.drop(columns=to_drop)
    X_val = X_val.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    rows = []
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    rows.append(evaluate("Baseline majority", baseline, X_test, y_test, feature_set_name))

    knn = tune_model(
        KNeighborsClassifier,
        {"n_neighbors": [3, 7, 15], "weights": ["uniform", "distance"], "p": [1, 2, 3]},
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    )
    rows.append(evaluate("k-NN", knn["model"], X_test_scaled, y_test, feature_set_name))

    ada = tune_model(
        AdaBoostClassifier,
        {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.10, 0.50]},
        X_train,
        y_train,
        X_val,
        y_val,
        fixed={"random_state": RANDOM_STATE},
    )
    rows.append(evaluate("AdaBoost", ada["model"], X_test, y_test, feature_set_name))

    tree = tune_model(
        DecisionTreeClassifier,
        {
            "max_depth": [3, 6, 10],
            "min_samples_leaf": [1, 10, 30],
            "criterion": ["gini", "entropy", "log_loss"],
        },
        X_train,
        y_train,
        X_val,
        y_val,
        fixed={"random_state": RANDOM_STATE},
    )
    rows.append(evaluate("Decision Tree", tree["model"], X_test, y_test, feature_set_name))

    result = pd.DataFrame(rows)
    result["n_features_after_encoding"] = X_train.shape[1]
    result["dropped_correlated_features"] = len(to_drop)
    return result


def main() -> None:
    old_df = load_old_simple_dataset()
    new_df = pd.read_csv(ROOT / "agent_tool_tasks.csv")

    results = pd.concat(
        [
            run_feature_set(old_df, "old_simple_counts"),
            run_feature_set(new_df, "new_semantic_aspects"),
        ],
        ignore_index=True,
    )
    results = results.round(4)
    results.to_csv(ROOT / "feature_set_comparison.csv", index=False)
    print(results.to_string(index=False))
    print(f"\nWrote {ROOT / 'feature_set_comparison.csv'}")


if __name__ == "__main__":
    main()
