# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:22:34 2026

@author: FarhanAli
"""
# 26_compare_balanced_vs_unbalanced.py
# Compare unbalanced pipeline results vs selected-best-method rerun results
#
# This script:
# 1. Loads the final metrics CSV from:
#    - unbalanced runs:
#         09a_individual_A_tuned
#         09b_individual_B_tuned
#         09c_individual_T_tuned
#         09d_individual_I_tuned
#         10_ab_multiview
#         11_it_multiview
#         12_tbi_multiview
#         13_abi_multiview
#         14_atb_multiview
#         15_atbi_multiview
#    - selected-best-method reruns:
#         19a_A_best_balanced
#         19b_B_best_balanced
#         19c_T_best_balanced
#         19d_I_best_balanced
#         20_ab_best_balanced
#         21_it_best_balanced
#         22_tbi_best_balanced
#         23_abi_best_balanced
#         24_atb_best_balanced
#         25_atbi_best_balanced
# 2. Standardizes one result row per experiment per split
# 3. Builds side-by-side comparison tables
# 4. Computes delta metrics:
#       balanced - unbalanced
# 5. Creates VALID and TEST comparison plots
# 6. Saves summary CSVs and summary JSON

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "26_compare_balanced_vs_unbalanced")

COMBINED_LONG_CSV_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_long.csv")
COMBINED_WIDE_CSV_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_wide.csv")

VALID_COMPARISON_CSV_PATH = os.path.join(OUTPUT_DIR, "valid_balanced_vs_unbalanced.csv")
TEST_COMPARISON_CSV_PATH = os.path.join(OUTPUT_DIR, "test_balanced_vs_unbalanced.csv")

VALID_DELTA_RANKING_CSV_PATH = os.path.join(OUTPUT_DIR, "valid_delta_ranking_by_pr_auc.csv")
TEST_DELTA_RANKING_CSV_PATH = os.path.join(OUTPUT_DIR, "test_delta_ranking_by_pr_auc.csv")

VALID_DELTA_PLOT_PATH = os.path.join(OUTPUT_DIR, "delta_metrics_valid.png")
TEST_DELTA_PLOT_PATH = os.path.join(OUTPUT_DIR, "delta_metrics_test.png")

VALID_PR_AUC_COMPARE_PLOT_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_pr_auc_valid.png")
TEST_PR_AUC_COMPARE_PLOT_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_pr_auc_test.png")

VALID_F1_COMPARE_PLOT_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_f1_valid.png")
TEST_F1_COMPARE_PLOT_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_f1_test.png")

SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "balanced_vs_unbalanced_summary.json")


# ============================================================
# Input files
# ============================================================
EXPERIMENT_MAP = {
    "A": {
        "display_name": "A",
        "component_views": "A",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "09a_individual_A_tuned", "metrics_A_bagging_tuned.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "19a_A_best_balanced", "metrics_A_best_balanced.csv")
    },
    "B": {
        "display_name": "B",
        "component_views": "B",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "09b_individual_B_tuned", "metrics_B_bagging_tuned.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "19b_B_best_balanced", "metrics_B_best_balanced.csv")
    },
    "T": {
        "display_name": "T",
        "component_views": "T",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "09c_individual_T_tuned", "metrics_T_bagging_tuned.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "19c_T_best_balanced", "metrics_T_best_balanced.csv")
    },
    "I": {
        "display_name": "I",
        "component_views": "I",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "09d_individual_I_tuned", "metrics_I_bagging_tuned.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "19d_I_best_balanced", "metrics_I_best_balanced.csv")
    },
    "AB": {
        "display_name": "A+B",
        "component_views": "A+B",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "10_ab_multiview", "metrics_AB_multiview.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "20_ab_best_balanced", "metrics_AB_best_balanced.csv")
    },
    "IT": {
        "display_name": "I+T",
        "component_views": "I+T",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "11_it_multiview", "metrics_IT_multiview.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "21_it_best_balanced", "metrics_IT_best_balanced.csv")
    },
    "TBI": {
        "display_name": "T+B+I",
        "component_views": "T+B+I",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "12_tbi_multiview", "metrics_TBI_multiview.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "22_tbi_best_balanced", "metrics_TBI_best_balanced.csv")
    },
    "ABI": {
        "display_name": "A+B+I",
        "component_views": "A+B+I",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "13_abi_multiview", "metrics_ABI_multiview.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "23_abi_best_balanced", "metrics_ABI_best_balanced.csv")
    },
    "ATB": {
        "display_name": "A+T+B",
        "component_views": "A+T+B",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "14_atb_multiview", "metrics_ATB_multiview.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "24_atb_best_balanced", "metrics_ATB_best_balanced.csv")
    },
    "ATBI": {
        "display_name": "A+T+B+I",
        "component_views": "A+T+B+I",
        "unbalanced_file": os.path.join(RESULTS_ROOT, "15_atbi_multiview", "metrics_ATBI_multiview.csv"),
        "balanced_file": os.path.join(RESULTS_ROOT, "25_atbi_best_balanced", "metrics_ATBI_best_balanced.csv")
    }
}


# ============================================================
# Metrics
# ============================================================
CORE_METRICS = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "fpr"]
DELTA_SORT_PRIORITY = ["delta_pr_auc", "delta_f1", "delta_recall", "delta_precision", "delta_roc_auc", "delta_accuracy", "delta_fpr"]
DELTA_SORT_ASCENDING = [False, False, False, False, False, False, True]


# ============================================================
# Helpers
# ============================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def read_metrics_file(path: str) -> pd.DataFrame:
    assert_file_exists(path)
    df = pd.read_csv(path)

    if "split" not in df.columns:
        raise ValueError(f"'split' column missing in: {path}")

    return df


def standardize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected_numeric = CORE_METRICS + ["threshold", "positive_predictions", "total_rows"]
    for col in expected_numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_one_row_per_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to one row per split.
    Priority:
    1. threshold_label contains 'default'
    2. threshold around 0.25
    3. otherwise first row per split
    """
    df = df.copy()
    df = standardize_numeric_columns(df)

    if "threshold_label" not in df.columns:
        df["threshold_label"] = ""

    df["threshold_label"] = df["threshold_label"].astype(str)

    def _priority(row):
        label = row["threshold_label"].lower()
        thr = row["threshold"]
        if "default" in label:
            return 0
        if pd.notna(thr) and abs(float(thr) - 0.25) < 1e-9:
            return 1
        return 2

    df["row_priority"] = df.apply(_priority, axis=1)
    df = df.sort_values(by=["split", "row_priority"]).reset_index(drop=True)
    reduced = df.groupby("split", as_index=False).first()
    reduced = reduced.drop(columns=["row_priority"], errors="ignore")
    return reduced


def load_one_experiment_pair(exp_key: str, meta: dict) -> pd.DataFrame:
    unbalanced_df = pick_one_row_per_split(read_metrics_file(meta["unbalanced_file"]))
    balanced_df = pick_one_row_per_split(read_metrics_file(meta["balanced_file"]))

    unbalanced_df["variant"] = "unbalanced"
    balanced_df["variant"] = "balanced"

    pair_df = pd.concat([unbalanced_df, balanced_df], axis=0, ignore_index=True)

    pair_df["experiment_key"] = exp_key
    pair_df["display_name"] = meta["display_name"]
    pair_df["component_views_standardized"] = meta["component_views"]

    return pair_df


def load_all_pairs() -> pd.DataFrame:
    frames = []

    for exp_key, meta in EXPERIMENT_MAP.items():
        print(f"Loading comparison pair for {exp_key}...")
        frames.append(load_one_experiment_pair(exp_key, meta))

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def build_wide_comparison(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for exp_key in long_df["experiment_key"].dropna().unique():
        exp_df = long_df[long_df["experiment_key"] == exp_key].copy()

        for split_name in ["valid", "test"]:
            split_df = exp_df[exp_df["split"].str.lower() == split_name].copy()

            row_bal = split_df[split_df["variant"] == "balanced"]
            row_unbal = split_df[split_df["variant"] == "unbalanced"]

            if row_bal.empty or row_unbal.empty:
                continue

            bal = row_bal.iloc[0]
            unbal = row_unbal.iloc[0]

            row = {
                "experiment_key": exp_key,
                "display_name": bal["display_name"],
                "component_views": bal["component_views_standardized"],
                "split": split_name
            }

            for metric in CORE_METRICS:
                row[f"unbalanced_{metric}"] = float(unbal[metric]) if pd.notna(unbal[metric]) else np.nan
                row[f"balanced_{metric}"] = float(bal[metric]) if pd.notna(bal[metric]) else np.nan
                row[f"delta_{metric}"] = (
                    float(bal[metric]) - float(unbal[metric])
                    if pd.notna(bal[metric]) and pd.notna(unbal[metric]) else np.nan
                )

            rows.append(row)

    wide_df = pd.DataFrame(rows)
    return wide_df


def build_delta_ranking(wide_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    split_df = wide_df[wide_df["split"].str.lower() == split_name.lower()].copy()

    split_df = split_df.sort_values(
        by=DELTA_SORT_PRIORITY,
        ascending=DELTA_SORT_ASCENDING
    ).reset_index(drop=True)

    split_df["rank_by_delta_pr_auc"] = np.arange(1, len(split_df) + 1)

    cols_front = [
        "rank_by_delta_pr_auc",
        "display_name",
        "experiment_key",
        "component_views",
        "split",
        "unbalanced_pr_auc",
        "balanced_pr_auc",
        "delta_pr_auc",
        "unbalanced_f1",
        "balanced_f1",
        "delta_f1",
        "unbalanced_recall",
        "balanced_recall",
        "delta_recall",
        "unbalanced_precision",
        "balanced_precision",
        "delta_precision",
        "unbalanced_fpr",
        "balanced_fpr",
        "delta_fpr",
        "unbalanced_roc_auc",
        "balanced_roc_auc",
        "delta_roc_auc",
        "unbalanced_accuracy",
        "balanced_accuracy",
        "delta_accuracy"
    ]

    remaining = [c for c in split_df.columns if c not in cols_front]
    split_df = split_df[cols_front + remaining]
    return split_df


def plot_delta_metrics(wide_df: pd.DataFrame, split_name: str, output_path: str, title: str) -> None:
    split_df = wide_df[wide_df["split"].str.lower() == split_name.lower()].copy()
    split_df = split_df.sort_values(by="delta_pr_auc", ascending=False)

    metrics = ["delta_pr_auc", "delta_f1", "delta_recall", "delta_precision", "delta_fpr"]
    x = np.arange(len(split_df))
    width = 0.15

    plt.figure(figsize=(14, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, split_df[metric].fillna(0).values, width=width, label=metric)

    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x + width * (len(metrics) - 1) / 2, split_df["display_name"].tolist(), rotation=35, ha="right")
    plt.ylabel("Balanced - Unbalanced")
    plt.xlabel("Experiment")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_balanced_vs_unbalanced_metric(wide_df: pd.DataFrame, split_name: str, metric: str, output_path: str, title: str) -> None:
    split_df = wide_df[wide_df["split"].str.lower() == split_name.lower()].copy()
    split_df = split_df.sort_values(by=f"balanced_{metric}", ascending=False)

    labels = split_df["display_name"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, split_df[f"unbalanced_{metric}"].fillna(0).values, width=width, label="unbalanced")
    plt.bar(x + width / 2, split_df[f"balanced_{metric}"].fillna(0).values, width=width, label="balanced")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(metric.upper())
    plt.xlabel("Experiment")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_summary_json(valid_ranked: pd.DataFrame, test_ranked: pd.DataFrame) -> dict:
    summary = {
        "best_valid_delta_pr_auc": {},
        "best_test_delta_pr_auc": {},
        "largest_valid_f1_gain": {},
        "largest_test_f1_gain": {}
    }

    if not valid_ranked.empty:
        top_valid = valid_ranked.iloc[0]
        summary["best_valid_delta_pr_auc"] = {
            "display_name": top_valid["display_name"],
            "experiment_key": top_valid["experiment_key"],
            "component_views": top_valid["component_views"],
            "delta_pr_auc": float(top_valid["delta_pr_auc"]),
            "delta_f1": float(top_valid["delta_f1"]),
            "delta_recall": float(top_valid["delta_recall"]),
            "delta_precision": float(top_valid["delta_precision"]),
            "delta_fpr": float(top_valid["delta_fpr"])
        }

        valid_f1 = valid_ranked.sort_values(by="delta_f1", ascending=False).iloc[0]
        summary["largest_valid_f1_gain"] = {
            "display_name": valid_f1["display_name"],
            "experiment_key": valid_f1["experiment_key"],
            "component_views": valid_f1["component_views"],
            "delta_pr_auc": float(valid_f1["delta_pr_auc"]),
            "delta_f1": float(valid_f1["delta_f1"]),
            "delta_recall": float(valid_f1["delta_recall"]),
            "delta_precision": float(valid_f1["delta_precision"]),
            "delta_fpr": float(valid_f1["delta_fpr"])
        }

    if not test_ranked.empty:
        top_test = test_ranked.iloc[0]
        summary["best_test_delta_pr_auc"] = {
            "display_name": top_test["display_name"],
            "experiment_key": top_test["experiment_key"],
            "component_views": top_test["component_views"],
            "delta_pr_auc": float(top_test["delta_pr_auc"]),
            "delta_f1": float(top_test["delta_f1"]),
            "delta_recall": float(top_test["delta_recall"]),
            "delta_precision": float(top_test["delta_precision"]),
            "delta_fpr": float(top_test["delta_fpr"])
        }

        test_f1 = test_ranked.sort_values(by="delta_f1", ascending=False).iloc[0]
        summary["largest_test_f1_gain"] = {
            "display_name": test_f1["display_name"],
            "experiment_key": test_f1["experiment_key"],
            "component_views": test_f1["component_views"],
            "delta_pr_auc": float(test_f1["delta_pr_auc"]),
            "delta_f1": float(test_f1["delta_f1"]),
            "delta_recall": float(test_f1["delta_recall"]),
            "delta_precision": float(test_f1["delta_precision"]),
            "delta_fpr": float(test_f1["delta_fpr"])
        }

    return summary

# ============================================================
# Main execution
# ============================================================
def main() -> None:
    print("=== Step 26: Compare Balanced vs Unbalanced ===")
    make_dir(OUTPUT_DIR)

    print("\n[1/6] Loading all balanced/unbalanced experiment pairs...")
    long_df = load_all_pairs()
    long_df.to_csv(COMBINED_LONG_CSV_PATH, index=False)

    print(f"Combined long rows loaded: {long_df.shape[0]}")
    print(f"Combined long columns: {long_df.shape[1]}")
    print(f"[OK] Long comparison CSV saved to: {COMBINED_LONG_CSV_PATH}")

    print("\n[2/6] Building wide balanced-vs-unbalanced comparison table...")
    wide_df = build_wide_comparison(long_df)
    wide_df.to_csv(COMBINED_WIDE_CSV_PATH, index=False)

    valid_df = wide_df[wide_df["split"].str.lower() == "valid"].copy()
    test_df = wide_df[wide_df["split"].str.lower() == "test"].copy()

    valid_df.to_csv(VALID_COMPARISON_CSV_PATH, index=False)
    test_df.to_csv(TEST_COMPARISON_CSV_PATH, index=False)

    print(f"[OK] Wide comparison CSV saved to: {COMBINED_WIDE_CSV_PATH}")
    print(f"[OK] VALID comparison CSV saved to: {VALID_COMPARISON_CSV_PATH}")
    print(f"[OK] TEST comparison CSV saved to: {TEST_COMPARISON_CSV_PATH}")

    print("\n[3/6] Ranking delta improvements...")
    valid_ranked = build_delta_ranking(wide_df, "valid")
    test_ranked = build_delta_ranking(wide_df, "test")

    valid_ranked.to_csv(VALID_DELTA_RANKING_CSV_PATH, index=False)
    test_ranked.to_csv(TEST_DELTA_RANKING_CSV_PATH, index=False)

    print(f"[OK] VALID delta ranking CSV saved to: {VALID_DELTA_RANKING_CSV_PATH}")
    print(f"[OK] TEST delta ranking CSV saved to: {TEST_DELTA_RANKING_CSV_PATH}")

    print("\n[4/6] Saving delta comparison plots...")
    plot_delta_metrics(
        wide_df,
        "valid",
        VALID_DELTA_PLOT_PATH,
        "Balanced vs Unbalanced Delta Metrics - VALID"
    )
    plot_delta_metrics(
        wide_df,
        "test",
        TEST_DELTA_PLOT_PATH,
        "Balanced vs Unbalanced Delta Metrics - TEST"
    )

    print(f"[OK] VALID delta plot saved to: {VALID_DELTA_PLOT_PATH}")
    print(f"[OK] TEST delta plot saved to: {TEST_DELTA_PLOT_PATH}")

    print("\n[5/6] Saving balanced-vs-unbalanced metric plots...")
    plot_balanced_vs_unbalanced_metric(
        wide_df,
        "valid",
        "pr_auc",
        VALID_PR_AUC_COMPARE_PLOT_PATH,
        "Balanced vs Unbalanced PR-AUC - VALID"
    )
    plot_balanced_vs_unbalanced_metric(
        wide_df,
        "test",
        "pr_auc",
        TEST_PR_AUC_COMPARE_PLOT_PATH,
        "Balanced vs Unbalanced PR-AUC - TEST"
    )

    plot_balanced_vs_unbalanced_metric(
        wide_df,
        "valid",
        "f1",
        VALID_F1_COMPARE_PLOT_PATH,
        "Balanced vs Unbalanced F1 - VALID"
    )
    plot_balanced_vs_unbalanced_metric(
        wide_df,
        "test",
        "f1",
        TEST_F1_COMPARE_PLOT_PATH,
        "Balanced vs Unbalanced F1 - TEST"
    )

    print(f"[OK] VALID PR-AUC comparison plot saved to: {VALID_PR_AUC_COMPARE_PLOT_PATH}")
    print(f"[OK] TEST PR-AUC comparison plot saved to: {TEST_PR_AUC_COMPARE_PLOT_PATH}")
    print(f"[OK] VALID F1 comparison plot saved to: {VALID_F1_COMPARE_PLOT_PATH}")
    print(f"[OK] TEST F1 comparison plot saved to: {TEST_F1_COMPARE_PLOT_PATH}")

    print("\n[6/6] Saving summary JSON...")
    summary = build_summary_json(valid_ranked, test_ranked)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary JSON saved to: {SUMMARY_JSON_PATH}")

    print("\nTop VALID experiment by delta PR-AUC:")
    if not valid_ranked.empty:
        print(valid_ranked.head(1).to_string(index=False))

    print("\nTop TEST experiment by delta PR-AUC:")
    if not test_ranked.empty:
        print(test_ranked.head(1).to_string(index=False))

    print("\nLargest VALID F1 gain:")
    if not valid_ranked.empty:
        print(valid_ranked.sort_values(by='delta_f1', ascending=False).head(1).to_string(index=False))

    print("\nLargest TEST F1 gain:")
    if not test_ranked.empty:
        print(test_ranked.sort_values(by='delta_f1', ascending=False).head(1).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)