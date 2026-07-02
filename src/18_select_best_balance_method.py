# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:35:28 2026

@author: FarhanAli
"""

# 18_select_best_balance_method.py
# Select the best balancing method across benchmark views [A, B, AB]
#
# This script:
# 1. Loads balancing comparison outputs from:
#    - 17a_A_balance_method_comparison.py
#    - 17b_B_balance_method_comparison.py
#    - 17c_AB_balance_method_comparison.py
# 2. Combines all per-method VALID and TEST results
# 3. Computes mean performance across the benchmark views A, B, and AB
# 4. Compares balancing methods using:
#    - PR-AUC
#    - Recall
#    - F1
#    - FPR
#    - Precision
#    - ROC-AUC
#    - Accuracy
# 5. Selects the best method primarily from VALID mean results
# 6. Saves:
#    - raw combined benchmark metrics CSV
#    - mean summary CSV for VALID
#    - mean summary CSV for TEST
#    - best-method ranking CSVs
#    - comparison plots
#    - best balancing method JSON
#    - summary JSON

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "18_select_best_balance_method")

A_METRICS_PATH = os.path.join(
    RESULTS_ROOT, "17a_A_balance_method_comparison", "metrics_A_balance_method_comparison.csv"
)
B_METRICS_PATH = os.path.join(
    RESULTS_ROOT, "17b_B_balance_method_comparison", "metrics_B_balance_method_comparison.csv"
)
AB_METRICS_PATH = os.path.join(
    RESULTS_ROOT, "17c_AB_balance_method_comparison", "metrics_AB_balance_method_comparison.csv"
)

RAW_COMBINED_CSV_PATH = os.path.join(OUTPUT_DIR, "benchmark_views_combined_raw.csv")
VALID_MEAN_CSV_PATH = os.path.join(OUTPUT_DIR, "valid_mean_by_balance_method.csv")
TEST_MEAN_CSV_PATH = os.path.join(OUTPUT_DIR, "test_mean_by_balance_method.csv")

VALID_RANKING_CSV_PATH = os.path.join(OUTPUT_DIR, "valid_balance_method_ranking.csv")
TEST_RANKING_CSV_PATH = os.path.join(OUTPUT_DIR, "test_balance_method_ranking.csv")

VALID_PLOT_PATH = os.path.join(OUTPUT_DIR, "mean_metrics_valid_balance_methods.png")
TEST_PLOT_PATH = os.path.join(OUTPUT_DIR, "mean_metrics_test_balance_methods.png")
VALID_OPERATIONAL_PLOT_PATH = os.path.join(OUTPUT_DIR, "operational_metrics_valid_balance_methods.png")
TEST_OPERATIONAL_PLOT_PATH = os.path.join(OUTPUT_DIR, "operational_metrics_test_balance_methods.png")

BEST_METHOD_JSON_PATH = os.path.join(OUTPUT_DIR, "best_balance_method.json")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "balance_method_selection_summary.json")


# ============================================================
# Selection settings
# ============================================================
METHOD_ORDER = [
    "baseline",
    "sub_sampling",
    "smote",
    "sub_sampling_plus_smote"
]

# Selection priority for VALID ranking
# High is good for these metrics, except FPR where low is better.
SELECTION_PRIORITY = [
    ("mean_pr_auc", False),
    ("mean_recall", False),
    ("mean_f1", False),
    ("mean_fpr", True),
    ("mean_precision", False),
    ("mean_roc_auc", False),
    ("mean_accuracy", False),
]

MEAN_METRICS = [
    "pr_auc",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "tpr",
    "tnr",
    "positive_rate_predicted"
]

COUNT_METRICS = [
    "tn",
    "fp",
    "fn",
    "tp",
    "positive_predictions",
    "total_rows"
]


# ============================================================
# Helper functions
# ============================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

def load_metrics_file(path: str, benchmark_view: str) -> pd.DataFrame:
    """
    Load one benchmark-view metrics file and add the benchmark_view column.
    """
    assert_file_exists(path)
    df = pd.read_csv(path)

    if "method" not in df.columns or "split" not in df.columns:
        raise ValueError(f"'method' or 'split' column missing in: {path}")

    df["benchmark_view"] = benchmark_view
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure expected numeric columns exist and are numeric.
    """
    required_numeric = MEAN_METRICS + COUNT_METRICS
    for col in required_numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "benchmark_view" not in df.columns:
        df["benchmark_view"] = "unknown"

    df["method"] = df["method"].astype(str)
    df["split"] = df["split"].astype(str)

    return df


def combine_all_benchmark_views() -> pd.DataFrame:
    """
    Load and combine A, B, and AB balancing comparison results.
    """
    df_a = load_metrics_file(A_METRICS_PATH, "A")
    df_b = load_metrics_file(B_METRICS_PATH, "B")
    df_ab = load_metrics_file(AB_METRICS_PATH, "AB")

    combined = pd.concat([df_a, df_b, df_ab], axis=0, ignore_index=True)
    combined = standardize_columns(combined)

    combined["method"] = pd.Categorical(combined["method"], categories=METHOD_ORDER, ordered=True)
    combined = combined.sort_values(["split", "method", "benchmark_view"]).reset_index(drop=True)

    return combined


def summarize_by_method(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Compute mean metrics and aggregate counts across benchmark views for one split.
    """
    split_df = df[df["split"].str.lower() == split_name.lower()].copy()

    grouped = split_df.groupby("method", observed=True)

    summary = grouped.agg(
        benchmark_views_count=("benchmark_view", "nunique"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_accuracy=("accuracy", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_fpr=("fpr", "mean"),
        mean_fnr=("fnr", "mean"),
        mean_tpr=("tpr", "mean"),
        mean_tnr=("tnr", "mean"),
        mean_positive_rate_predicted=("positive_rate_predicted", "mean"),
        mean_fp=("fp", "mean"),
        mean_fn=("fn", "mean"),
        mean_tp=("tp", "mean"),
        mean_tn=("tn", "mean"),
        sum_fp=("fp", "sum"),
        sum_fn=("fn", "sum"),
        sum_tp=("tp", "sum"),
        sum_tn=("tn", "sum"),
        sum_positive_predictions=("positive_predictions", "sum"),
        sum_total_rows=("total_rows", "sum"),
    ).reset_index()

    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)

    return summary


def rank_methods(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank methods according to the predefined selection priority.
    """
    sort_cols = [item[0] for item in SELECTION_PRIORITY]
    ascending_flags = [item[1] for item in SELECTION_PRIORITY]

    ranked = summary_df.sort_values(
        by=sort_cols,
        ascending=ascending_flags
    ).reset_index(drop=True)

    ranked["rank"] = np.arange(1, len(ranked) + 1)

    cols_front = [
        "rank",
        "method",
        "benchmark_views_count",
        "mean_pr_auc",
        "mean_recall",
        "mean_f1",
        "mean_fpr",
        "mean_precision",
        "mean_roc_auc",
        "mean_accuracy",
        "mean_fnr",
        "mean_tpr",
        "mean_tnr",
        "mean_positive_rate_predicted",
        "mean_fp",
        "mean_fn",
        "mean_tp",
        "mean_tn",
        "sum_fp",
        "sum_fn",
        "sum_tp",
        "sum_tn",
        "sum_positive_predictions",
        "sum_total_rows",
    ]

    remaining = [c for c in ranked.columns if c not in cols_front]
    ranked = ranked[cols_front + remaining]

    return ranked


def plot_mean_metric_bars(summary_df: pd.DataFrame, output_path: str, title: str) -> None:
    """
    Plot grouped mean metrics across balancing methods.
    """
    plot_df = summary_df.copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values("method")

    metrics = [
        "mean_pr_auc",
        "mean_roc_auc",
        "mean_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1"
    ]

    x = np.arange(len(plot_df))
    width = 0.12

    plt.figure(figsize=(13, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, plot_df[metric].values, width=width, label=metric.replace("mean_", ""))

    plt.xticks(x + width * (len(metrics) - 1) / 2, plot_df["method"].tolist(), rotation=20)
    plt.ylabel("Mean Score Across Benchmark Views")
    plt.xlabel("Balancing Method")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_operational_metric_bars(summary_df: pd.DataFrame, output_path: str, title: str) -> None:
    """
    Plot operationally relevant mean metrics across balancing methods.
    """
    plot_df = summary_df.copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values("method")

    metrics = [
        "mean_recall",
        "mean_f1",
        "mean_precision",
        "mean_fpr",
        "mean_positive_rate_predicted"
    ]

    x = np.arange(len(plot_df))
    width = 0.15

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, plot_df[metric].values, width=width, label=metric.replace("mean_", ""))

    plt.xticks(x + width * (len(metrics) - 1) / 2, plot_df["method"].tolist(), rotation=20)
    plt.ylabel("Mean Score Across Benchmark Views")
    plt.xlabel("Balancing Method")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_summary_json(valid_ranked: pd.DataFrame, test_ranked: pd.DataFrame) -> dict:
    """
    Build a compact JSON summary of the selected best method.
    """
    summary = {
        "selection_priority": [
            "mean_pr_auc_high",
            "mean_recall_high",
            "mean_f1_high",
            "mean_fpr_low",
            "mean_precision_high",
            "mean_roc_auc_high",
            "mean_accuracy_high"
        ],
        "top_valid_method": {},
        "top_test_method": {}
    }

    if not valid_ranked.empty:
        top_valid = valid_ranked.iloc[0]
        summary["top_valid_method"] = {
            "method": str(top_valid["method"]),
            "rank": int(top_valid["rank"]),
            "benchmark_views_count": int(top_valid["benchmark_views_count"]),
            "mean_pr_auc": float(top_valid["mean_pr_auc"]),
            "mean_recall": float(top_valid["mean_recall"]),
            "mean_f1": float(top_valid["mean_f1"]),
            "mean_fpr": float(top_valid["mean_fpr"]),
            "mean_precision": float(top_valid["mean_precision"]),
            "mean_roc_auc": float(top_valid["mean_roc_auc"]),
            "mean_accuracy": float(top_valid["mean_accuracy"]),
            "sum_fp": float(top_valid["sum_fp"]),
            "sum_tp": float(top_valid["sum_tp"]),
            "sum_fn": float(top_valid["sum_fn"]),
            "sum_tn": float(top_valid["sum_tn"])
        }

    if not test_ranked.empty:
        top_test = test_ranked.iloc[0]
        summary["top_test_method"] = {
            "method": str(top_test["method"]),
            "rank": int(top_test["rank"]),
            "benchmark_views_count": int(top_test["benchmark_views_count"]),
            "mean_pr_auc": float(top_test["mean_pr_auc"]),
            "mean_recall": float(top_test["mean_recall"]),
            "mean_f1": float(top_test["mean_f1"]),
            "mean_fpr": float(top_test["mean_fpr"]),
            "mean_precision": float(top_test["mean_precision"]),
            "mean_roc_auc": float(top_test["mean_roc_auc"]),
            "mean_accuracy": float(top_test["mean_accuracy"]),
            "sum_fp": float(top_test["sum_fp"]),
            "sum_tp": float(top_test["sum_tp"]),
            "sum_fn": float(top_test["sum_fn"]),
            "sum_tn": float(top_test["sum_tn"])
        }

    return summary


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    print("=== Step 18: Select Best Balance Method ===")
    make_dir(OUTPUT_DIR)

    print("\n[1/6] Loading benchmark view results from A, B, and AB...")
    combined_df = combine_all_benchmark_views()

    print(f"Combined rows loaded: {combined_df.shape[0]}")
    print(f"Combined columns: {combined_df.shape[1]}")

    combined_df.to_csv(RAW_COMBINED_CSV_PATH, index=False)
    print(f"[OK] Raw combined CSV saved to: {RAW_COMBINED_CSV_PATH}")

    print("\n[2/6] Computing mean summary across benchmark views...")
    valid_summary = summarize_by_method(combined_df, "valid")
    test_summary = summarize_by_method(combined_df, "test")

    valid_summary.to_csv(VALID_MEAN_CSV_PATH, index=False)
    test_summary.to_csv(TEST_MEAN_CSV_PATH, index=False)

    print(f"[OK] VALID mean summary CSV saved to: {VALID_MEAN_CSV_PATH}")
    print(f"[OK] TEST mean summary CSV saved to: {TEST_MEAN_CSV_PATH}")

    print("\n[3/6] Ranking balancing methods...")
    valid_ranked = rank_methods(valid_summary)
    test_ranked = rank_methods(test_summary)

    valid_ranked.to_csv(VALID_RANKING_CSV_PATH, index=False)
    test_ranked.to_csv(TEST_RANKING_CSV_PATH, index=False)

    print(f"[OK] VALID ranking CSV saved to: {VALID_RANKING_CSV_PATH}")
    print(f"[OK] TEST ranking CSV saved to: {TEST_RANKING_CSV_PATH}")

    print("\n[4/6] Saving comparison plots...")
    plot_mean_metric_bars(
        valid_summary,
        VALID_PLOT_PATH,
        "Mean Metrics Across Benchmark Views (VALID) - Balance Method Selection"
    )
    plot_mean_metric_bars(
        test_summary,
        TEST_PLOT_PATH,
        "Mean Metrics Across Benchmark Views (TEST) - Balance Method Selection"
    )

    plot_operational_metric_bars(
        valid_summary,
        VALID_OPERATIONAL_PLOT_PATH,
        "Operational Metrics Across Benchmark Views (VALID) - Balance Method Selection"
    )
    plot_operational_metric_bars(
        test_summary,
        TEST_OPERATIONAL_PLOT_PATH,
        "Operational Metrics Across Benchmark Views (TEST) - Balance Method Selection"
    )

    print(f"[OK] VALID mean metric plot saved to: {VALID_PLOT_PATH}")
    print(f"[OK] TEST mean metric plot saved to: {TEST_PLOT_PATH}")
    print(f"[OK] VALID operational metric plot saved to: {VALID_OPERATIONAL_PLOT_PATH}")
    print(f"[OK] TEST operational metric plot saved to: {TEST_OPERATIONAL_PLOT_PATH}")

    print("\n[5/6] Saving best balance method JSON...")
    best_method_payload = {
        "selected_from_split": "valid",
        "selection_priority": [
            "mean_pr_auc_high",
            "mean_recall_high",
            "mean_f1_high",
            "mean_fpr_low",
            "mean_precision_high",
            "mean_roc_auc_high",
            "mean_accuracy_high"
        ]
    }

    if not valid_ranked.empty:
        best_valid = valid_ranked.iloc[0]
        best_method_payload["best_method"] = str(best_valid["method"])
        best_method_payload["details"] = {
            "benchmark_views_count": int(best_valid["benchmark_views_count"]),
            "mean_pr_auc": float(best_valid["mean_pr_auc"]),
            "mean_recall": float(best_valid["mean_recall"]),
            "mean_f1": float(best_valid["mean_f1"]),
            "mean_fpr": float(best_valid["mean_fpr"]),
            "mean_precision": float(best_valid["mean_precision"]),
            "mean_roc_auc": float(best_valid["mean_roc_auc"]),
            "mean_accuracy": float(best_valid["mean_accuracy"]),
            "sum_fp": float(best_valid["sum_fp"]),
            "sum_fn": float(best_valid["sum_fn"]),
            "sum_tp": float(best_valid["sum_tp"]),
            "sum_tn": float(best_valid["sum_tn"])
        }

    with open(BEST_METHOD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(best_method_payload, f, indent=2)

    print(f"[OK] Best balance method JSON saved to: {BEST_METHOD_JSON_PATH}")

    print("\n[6/6] Saving summary JSON...")
    summary = build_summary_json(valid_ranked, test_ranked)
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary JSON saved to: {SUMMARY_JSON_PATH}")

    print("\nVALID ranking:")
    print(valid_ranked[[
        "rank",
        "method",
        "mean_pr_auc",
        "mean_recall",
        "mean_f1",
        "mean_fpr",
        "mean_precision",
        "mean_roc_auc",
        "mean_accuracy",
        "sum_fp",
        "sum_tp"
    ]].to_string(index=False))

    print("\nTEST ranking:")
    print(test_ranked[[
        "rank",
        "method",
        "mean_pr_auc",
        "mean_recall",
        "mean_f1",
        "mean_fpr",
        "mean_precision",
        "mean_roc_auc",
        "mean_accuracy",
        "sum_fp",
        "sum_tp"
    ]].to_string(index=False))

    if not valid_ranked.empty:
        print("\nSelected best balancing method from VALID mean results:")
        print(valid_ranked.iloc[0][[
            "method",
            "mean_pr_auc",
            "mean_recall",
            "mean_f1",
            "mean_fpr",
            "mean_precision",
            "mean_roc_auc",
            "mean_accuracy"
        ]].to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)