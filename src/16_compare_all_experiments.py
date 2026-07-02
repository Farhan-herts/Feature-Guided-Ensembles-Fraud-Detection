# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:08:44 2026

@author: FarhanAli
"""

# 16_compare_all_experiments.py
# Final comparison script for all individual-view and multiview bagging experiments
#
# This script:
# 1. Loads the final metrics CSV from each completed experiment
# 2. Merges them into one comparison table
# 3. Creates separate VALID and TEST leaderboards
# 4. Ranks experiments by PR-AUC, ROC-AUC, F1, Recall, Precision, and Accuracy
# 5. Saves comparison CSV files
# 6. Saves comparison plots for VALID and TEST
#
# Expected prior experiment outputs:
# - 09a_individual_A_tuned/metrics_A_bagging_tuned.csv
# - 09b_individual_B_tuned/metrics_B_bagging_tuned.csv
# - 09c_individual_T_tuned/metrics_T_bagging_tuned.csv
# - 09d_individual_I_tuned/metrics_I_bagging_tuned.csv
# - 10_ab_multiview/metrics_AB_multiview.csv
# - 11_it_multiview/metrics_IT_multiview.csv
# - 12_tbi_multiview/metrics_TBI_multiview.csv
# - 13_abi_multiview/metrics_ABI_multiview.csv
# - 14_atb_multiview/metrics_ATB_multiview.csv
# - 15_atbi_multiview/metrics_ATBI_multiview.csv

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "16_compare_all_experiments")

COMBINED_CSV_PATH = os.path.join(OUTPUT_DIR, "all_experiments_combined.csv")
VALID_CSV_PATH = os.path.join(OUTPUT_DIR, "valid_experiments_only.csv")
TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "test_experiments_only.csv")

VALID_LEADERBOARD_CSV_PATH = os.path.join(OUTPUT_DIR, "valid_leaderboard_by_pr_auc.csv")
TEST_LEADERBOARD_CSV_PATH = os.path.join(OUTPUT_DIR, "test_leaderboard_by_pr_auc.csv")

BEST_BY_METRIC_VALID_CSV_PATH = os.path.join(OUTPUT_DIR, "best_experiment_by_metric_valid.csv")
BEST_BY_METRIC_TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "best_experiment_by_metric_test.csv")

VALID_METRICS_BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "comparison_metrics_valid.png")
TEST_METRICS_BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "comparison_metrics_test.png")
VALID_PR_AUC_PLOT_PATH = os.path.join(OUTPUT_DIR, "comparison_valid_pr_auc.png")
TEST_PR_AUC_PLOT_PATH = os.path.join(OUTPUT_DIR, "comparison_test_pr_auc.png")
VALID_F1_PLOT_PATH = os.path.join(OUTPUT_DIR, "comparison_valid_f1.png")
TEST_F1_PLOT_PATH = os.path.join(OUTPUT_DIR, "comparison_test_f1.png")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "comparison_summary.json")


# ============================================================
# Input experiment files
# ============================================================
EXPERIMENT_FILES = {
    "A": {
        "type": "individual",
        "file": os.path.join(RESULTS_ROOT, "09a_individual_A_tuned", "metrics_A_bagging_tuned.csv"),
        "display_name": "A",
        "component_views": "A"
    },
    "B": {
        "type": "individual",
        "file": os.path.join(RESULTS_ROOT, "09b_individual_B_tuned", "metrics_B_bagging_tuned.csv"),
        "display_name": "B",
        "component_views": "B"
    },
    "T": {
        "type": "individual",
        "file": os.path.join(RESULTS_ROOT, "09c_individual_T_tuned", "metrics_T_bagging_tuned.csv"),
        "display_name": "T",
        "component_views": "T"
    },
    "I": {
        "type": "individual",
        "file": os.path.join(RESULTS_ROOT, "09d_individual_I_tuned", "metrics_I_bagging_tuned.csv"),
        "display_name": "I",
        "component_views": "I"
    },
    "AB": {
        "type": "multiview",
        "file": os.path.join(RESULTS_ROOT, "10_ab_multiview", "metrics_AB_multiview.csv"),
        "display_name": "A+B",
        "component_views": "A+B"
    },
    "IT": {
        "type": "multiview",
        "file": os.path.join(RESULTS_ROOT, "11_it_multiview", "metrics_IT_multiview.csv"),
        "display_name": "I+T",
        "component_views": "I+T"
    },
    "TBI": {
        "type": "multiview",
        "file": os.path.join(RESULTS_ROOT, "12_tbi_multiview", "metrics_TBI_multiview.csv"),
        "display_name": "T+B+I",
        "component_views": "T+B+I"
    },
    "ABI": {
        "type": "multiview",
        "file": os.path.join(RESULTS_ROOT, "13_abi_multiview", "metrics_ABI_multiview.csv"),
        "display_name": "A+B+I",
        "component_views": "A+B+I"
    },
    "ATB": {
        "type": "multiview",
        "file": os.path.join(RESULTS_ROOT, "14_atb_multiview", "metrics_ATB_multiview.csv"),
        "display_name": "A+T+B",
        "component_views": "A+T+B"
    },
    "ATBI": {
        "type": "multiview",
        "file": os.path.join(RESULTS_ROOT, "15_atbi_multiview", "metrics_ATBI_multiview.csv"),
        "display_name": "A+T+B+I",
        "component_views": "A+T+B+I"
    }
}


# ============================================================
# Metrics to compare
# ============================================================
METRICS_FOR_RANKING = ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]
METRICS_FOR_BAR_PLOT = ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]


# ============================================================
# Helper functions
# ============================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def load_experiment_metrics(experiment_key: str, meta: dict) -> pd.DataFrame:
    """
    Load one experiment metrics CSV and standardize columns.
    """
    path = meta["file"]
    assert_file_exists(path)

    df = pd.read_csv(path)

    if "split" not in df.columns:
        raise ValueError(f"'split' column not found in {path}")

    df["experiment_key"] = experiment_key
    df["display_name"] = meta["display_name"]
    df["experiment_type"] = meta["type"]
    df["component_views_standardized"] = meta["component_views"]

    if "component_views" not in df.columns:
        df["component_views"] = meta["component_views"]

    if "combination_type" not in df.columns:
        df["combination_type"] = np.where(
            meta["type"] == "individual",
            "single_view",
            "multiview_soft_voting"
        )

    return df


def load_all_experiments() -> pd.DataFrame:
    """
    Load and concatenate all experiment metrics.
    """
    all_frames = []

    for exp_key, meta in EXPERIMENT_FILES.items():
        print(f"Loading metrics for {exp_key} from: {meta['file']}")
        df = load_experiment_metrics(exp_key, meta)
        all_frames.append(df)

    combined = pd.concat(all_frames, axis=0, ignore_index=True)
    return combined


def enforce_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure expected metric columns exist and are numeric where possible.
    """
    required_cols = [
        "display_name", "experiment_key", "experiment_type",
        "component_views_standardized", "split",
        "pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_leaderboard(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Build sorted leaderboard for a given split.
    """
    split_df = df[df["split"].str.lower() == split_name.lower()].copy()

    split_df = split_df.sort_values(
        by=["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"],
        ascending=[False, False, False, False, False, False]
    ).reset_index(drop=True)

    split_df["rank_by_pr_auc"] = np.arange(1, len(split_df) + 1)

    cols_front = [
        "rank_by_pr_auc", "display_name", "experiment_key", "experiment_type",
        "component_views_standardized", "split",
        "pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"
    ]

    remaining_cols = [c for c in split_df.columns if c not in cols_front]
    split_df = split_df[cols_front + remaining_cols]

    return split_df


def best_experiment_by_metric(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Return the best experiment for each selected metric.
    """
    split_df = df[df["split"].str.lower() == split_name.lower()].copy()
    rows = []

    for metric in METRICS_FOR_RANKING:
        metric_df = split_df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue

        best_row = metric_df.sort_values(by=metric, ascending=False).iloc[0]

        rows.append({
            "split": split_name,
            "metric": metric,
            "best_experiment": best_row["display_name"],
            "experiment_key": best_row["experiment_key"],
            "experiment_type": best_row["experiment_type"],
            "component_views": best_row["component_views_standardized"],
            "metric_value": float(best_row[metric]),
            "pr_auc": float(best_row["pr_auc"]) if pd.notna(best_row["pr_auc"]) else np.nan,
            "roc_auc": float(best_row["roc_auc"]) if pd.notna(best_row["roc_auc"]) else np.nan,
            "f1": float(best_row["f1"]) if pd.notna(best_row["f1"]) else np.nan,
            "recall": float(best_row["recall"]) if pd.notna(best_row["recall"]) else np.nan,
            "precision": float(best_row["precision"]) if pd.notna(best_row["precision"]) else np.nan,
            "accuracy": float(best_row["accuracy"]) if pd.notna(best_row["accuracy"]) else np.nan
        })

    return pd.DataFrame(rows)


def plot_grouped_metric_bars(df: pd.DataFrame, split_name: str, output_path: str, title: str) -> None:
    """
    Save grouped bar chart for all experiments across key metrics.
    """
    split_df = df[df["split"].str.lower() == split_name.lower()].copy()
    split_df = split_df.sort_values(by="pr_auc", ascending=False)

    labels = split_df["display_name"].tolist()
    metrics = METRICS_FOR_BAR_PLOT

    x = np.arange(len(labels))
    width = 0.12

    plt.figure(figsize=(16, 7))

    for i, metric in enumerate(metrics):
        values = split_df[metric].fillna(0).values
        plt.bar(x + i * width, values, width=width, label=metric)

    plt.xticks(x + width * (len(metrics) - 1) / 2, labels, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("Experiment")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_single_metric_ranking(df: pd.DataFrame, split_name: str, metric: str, output_path: str, title: str) -> None:
    """
    Save bar chart ranking for one metric.
    """
    split_df = df[df["split"].str.lower() == split_name.lower()].copy()
    split_df = split_df.sort_values(by=metric, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(split_df["display_name"], split_df[metric].fillna(0).values)
    plt.ylim(0, 1.05)
    plt.ylabel(metric.upper())
    plt.xlabel("Experiment")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_summary_json(valid_lb: pd.DataFrame, test_lb: pd.DataFrame) -> dict:
    """
    Build a short JSON summary of top performers.
    """
    summary = {
        "top_valid_by_pr_auc": {},
        "top_test_by_pr_auc": {},
        "num_experiments_loaded": int(len(EXPERIMENT_FILES))
    }

    if not valid_lb.empty:
        top_valid = valid_lb.iloc[0]
        summary["top_valid_by_pr_auc"] = {
            "display_name": top_valid["display_name"],
            "experiment_key": top_valid["experiment_key"],
            "component_views": top_valid["component_views_standardized"],
            "pr_auc": float(top_valid["pr_auc"]),
            "roc_auc": float(top_valid["roc_auc"]),
            "f1": float(top_valid["f1"]),
            "recall": float(top_valid["recall"]),
            "precision": float(top_valid["precision"]),
            "accuracy": float(top_valid["accuracy"])
        }

    if not test_lb.empty:
        top_test = test_lb.iloc[0]
        summary["top_test_by_pr_auc"] = {
            "display_name": top_test["display_name"],
            "experiment_key": top_test["experiment_key"],
            "component_views": top_test["component_views_standardized"],
            "pr_auc": float(top_test["pr_auc"]),
            "roc_auc": float(top_test["roc_auc"]),
            "f1": float(top_test["f1"]),
            "recall": float(top_test["recall"]),
            "precision": float(top_test["precision"]),
            "accuracy": float(top_test["accuracy"])
        }

    return summary


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    print("=== Step 16: Compare All Experiments ===")
    make_dir(OUTPUT_DIR)

    print("\n[1/6] Loading all experiment metrics...")
    combined_df = load_all_experiments()
    combined_df = enforce_metric_columns(combined_df)

    print(f"Combined rows loaded: {combined_df.shape[0]}")
    print(f"Combined columns: {combined_df.shape[1]}")

    print("\n[2/6] Saving combined and split-specific tables...")
    combined_df.to_csv(COMBINED_CSV_PATH, index=False)

    valid_df = combined_df[combined_df["split"].str.lower() == "valid"].copy()
    test_df = combined_df[combined_df["split"].str.lower() == "test"].copy()

    valid_df.to_csv(VALID_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    print(f"[OK] Combined CSV saved to: {COMBINED_CSV_PATH}")
    print(f"[OK] VALID-only CSV saved to: {VALID_CSV_PATH}")
    print(f"[OK] TEST-only CSV saved to: {TEST_CSV_PATH}")

    print("\n[3/6] Building leaderboards...")
    valid_leaderboard = build_leaderboard(combined_df, "valid")
    test_leaderboard = build_leaderboard(combined_df, "test")

    valid_leaderboard.to_csv(VALID_LEADERBOARD_CSV_PATH, index=False)
    test_leaderboard.to_csv(TEST_LEADERBOARD_CSV_PATH, index=False)

    print(f"[OK] VALID leaderboard saved to: {VALID_LEADERBOARD_CSV_PATH}")
    print(f"[OK] TEST leaderboard saved to: {TEST_LEADERBOARD_CSV_PATH}")

    print("\n[4/6] Finding best experiment by metric...")
    best_valid = best_experiment_by_metric(combined_df, "valid")
    best_test = best_experiment_by_metric(combined_df, "test")

    best_valid.to_csv(BEST_BY_METRIC_VALID_CSV_PATH, index=False)
    best_test.to_csv(BEST_BY_METRIC_TEST_CSV_PATH, index=False)

    print(f"[OK] Best-by-metric VALID CSV saved to: {BEST_BY_METRIC_VALID_CSV_PATH}")
    print(f"[OK] Best-by-metric TEST CSV saved to: {BEST_BY_METRIC_TEST_CSV_PATH}")

    print("\n[5/6] Saving comparison plots...")
    plot_grouped_metric_bars(
        combined_df,
        "valid",
        VALID_METRICS_BAR_PLOT_PATH,
        "All Experiment Metrics Comparison - VALID"
    )
    plot_grouped_metric_bars(
        combined_df,
        "test",
        TEST_METRICS_BAR_PLOT_PATH,
        "All Experiment Metrics Comparison - TEST"
    )

    plot_single_metric_ranking(
        combined_df,
        "valid",
        "pr_auc",
        VALID_PR_AUC_PLOT_PATH,
        "Experiment Ranking by PR-AUC - VALID"
    )
    plot_single_metric_ranking(
        combined_df,
        "test",
        "pr_auc",
        TEST_PR_AUC_PLOT_PATH,
        "Experiment Ranking by PR-AUC - TEST"
    )

    plot_single_metric_ranking(
        combined_df,
        "valid",
        "f1",
        VALID_F1_PLOT_PATH,
        "Experiment Ranking by F1 - VALID"
    )
    plot_single_metric_ranking(
        combined_df,
        "test",
        "f1",
        TEST_F1_PLOT_PATH,
        "Experiment Ranking by F1 - TEST"
    )

    print(f"[OK] VALID metrics comparison plot saved to: {VALID_METRICS_BAR_PLOT_PATH}")
    print(f"[OK] TEST metrics comparison plot saved to: {TEST_METRICS_BAR_PLOT_PATH}")
    print(f"[OK] VALID PR-AUC plot saved to: {VALID_PR_AUC_PLOT_PATH}")
    print(f"[OK] TEST PR-AUC plot saved to: {TEST_PR_AUC_PLOT_PATH}")
    print(f"[OK] VALID F1 plot saved to: {VALID_F1_PLOT_PATH}")
    print(f"[OK] TEST F1 plot saved to: {TEST_F1_PLOT_PATH}")

    print("\n[6/6] Saving summary JSON...")
    summary = build_summary_json(valid_leaderboard, test_leaderboard)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary JSON saved to: {SUMMARY_JSON_PATH}")

    print("\nTop VALID experiment by PR-AUC:")
    if not valid_leaderboard.empty:
        print(valid_leaderboard.head(1).to_string(index=False))

    print("\nTop TEST experiment by PR-AUC:")
    if not test_leaderboard.empty:
        print(test_leaderboard.head(1).to_string(index=False))

    print("\nBest experiment by metric on VALID:")
    print(best_valid.to_string(index=False))

    print("\nBest experiment by metric on TEST:")
    print(best_test.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)