# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:41:33 2026

@author: FarhanAli
"""
# 27_compare_standard_bagging_vs_multiview_ensemble.py
# Final comparison: standard single-view bagging vs multiview soft-voting ensemble
#
# This script:
# 1. Loads the balanced-vs-unbalanced comparison outputs from Step 26
# 2. Chooses the better variant (balanced or unbalanced) for each experiment
#    using VALID-first selection logic
# 3. Separates experiments into:
#       - standard bagging (single-view): A, B, T, I
#       - multiview ensemble: AB, IT, TBI, ABI, ATB, ATBI
# 4. Builds VALID and TEST leaderboards on the selected best variant per experiment
# 5. Finds:
#       - best standard bagging model
#       - best multiview ensemble
# 6. Saves:
#       - selected best-variant table
#       - standard-only leaderboards
#       - multiview-only leaderboards
#       - direct best-standard vs best-multiview comparison tables
#       - comparison plots
#       - summary JSON

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
STEP26_DIR = os.path.join(RESULTS_ROOT, "26_compare_balanced_vs_unbalanced")
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "27_compare_standard_bagging_vs_multiview_ensemble")

LONG_INPUT_CSV_PATH = os.path.join(STEP26_DIR, "balanced_vs_unbalanced_long.csv")
WIDE_INPUT_CSV_PATH = os.path.join(STEP26_DIR, "balanced_vs_unbalanced_wide.csv")

SELECTED_EXPERIMENTS_CSV_PATH = os.path.join(OUTPUT_DIR, "selected_best_variant_per_experiment.csv")
SELECTED_VALID_CSV_PATH = os.path.join(OUTPUT_DIR, "selected_best_variant_valid.csv")
SELECTED_TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "selected_best_variant_test.csv")

STANDARD_VALID_LEADERBOARD_CSV_PATH = os.path.join(OUTPUT_DIR, "standard_bagging_valid_leaderboard.csv")
STANDARD_TEST_LEADERBOARD_CSV_PATH = os.path.join(OUTPUT_DIR, "standard_bagging_test_leaderboard.csv")

MULTIVIEW_VALID_LEADERBOARD_CSV_PATH = os.path.join(OUTPUT_DIR, "multiview_valid_leaderboard.csv")
MULTIVIEW_TEST_LEADERBOARD_CSV_PATH = os.path.join(OUTPUT_DIR, "multiview_test_leaderboard.csv")

BEST_STANDARD_VS_MULTIVIEW_VALID_CSV_PATH = os.path.join(OUTPUT_DIR, "best_standard_vs_multiview_valid.csv")
BEST_STANDARD_VS_MULTIVIEW_TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "best_standard_vs_multiview_test.csv")

SELECTED_PR_AUC_VALID_PLOT_PATH = os.path.join(OUTPUT_DIR, "selected_models_pr_auc_valid.png")
SELECTED_PR_AUC_TEST_PLOT_PATH = os.path.join(OUTPUT_DIR, "selected_models_pr_auc_test.png")

SELECTED_F1_VALID_PLOT_PATH = os.path.join(OUTPUT_DIR, "selected_models_f1_valid.png")
SELECTED_F1_TEST_PLOT_PATH = os.path.join(OUTPUT_DIR, "selected_models_f1_test.png")

BEST_STANDARD_VS_MULTIVIEW_VALID_PLOT_PATH = os.path.join(OUTPUT_DIR, "best_standard_vs_multiview_valid.png")
BEST_STANDARD_VS_MULTIVIEW_TEST_PLOT_PATH = os.path.join(OUTPUT_DIR, "best_standard_vs_multiview_test.png")

SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "standard_vs_multiview_summary.json")


# ============================================================
# Experiment groups
# ============================================================
STANDARD_KEYS = ["A", "B", "T", "I"]
MULTIVIEW_KEYS = ["AB", "IT", "TBI", "ABI", "ATB", "ATBI"]

CORE_METRICS = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]


# ============================================================
# Helper functions
# ============================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def read_csv(path: str) -> pd.DataFrame:
    assert_file_exists(path)
    return pd.read_csv(path)


def safe_float(x):
    return float(x) if pd.notna(x) else np.nan


def standardize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1",
        "fpr", "fnr", "tpr", "tnr", "threshold",
        "positive_predictions", "tn", "fp", "fn", "tp", "total_rows"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def choose_best_variant_from_valid(wide_valid_row: pd.Series) -> str:
    """
    Choose best variant per experiment using VALID comparison row.
    Priority:
    PR-AUC -> Recall -> F1 -> Precision -> ROC-AUC -> Accuracy -> FPR(low)
    If still tied, prefer balanced.
    """
    b = {
        "variant": "balanced",
        "pr_auc": wide_valid_row.get("balanced_pr_auc", np.nan),
        "recall": wide_valid_row.get("balanced_recall", np.nan),
        "f1": wide_valid_row.get("balanced_f1", np.nan),
        "precision": wide_valid_row.get("balanced_precision", np.nan),
        "roc_auc": wide_valid_row.get("balanced_roc_auc", np.nan),
        "accuracy": wide_valid_row.get("balanced_accuracy", np.nan),
        "fpr": wide_valid_row.get("balanced_fpr", np.nan),
        "tie_pref": 0
    }

    u = {
        "variant": "unbalanced",
        "pr_auc": wide_valid_row.get("unbalanced_pr_auc", np.nan),
        "recall": wide_valid_row.get("unbalanced_recall", np.nan),
        "f1": wide_valid_row.get("unbalanced_f1", np.nan),
        "precision": wide_valid_row.get("unbalanced_precision", np.nan),
        "roc_auc": wide_valid_row.get("unbalanced_roc_auc", np.nan),
        "accuracy": wide_valid_row.get("unbalanced_accuracy", np.nan),
        "fpr": wide_valid_row.get("unbalanced_fpr", np.nan),
        "tie_pref": 1
    }

    candidates = [b, u]

    def sort_key(c):
        fpr_missing = 1 if pd.isna(c["fpr"]) else 0
        fpr_value = c["fpr"] if pd.notna(c["fpr"]) else 1e9

        return (
            -safe_float(c["pr_auc"]) if pd.notna(c["pr_auc"]) else 1e9,
            -safe_float(c["recall"]) if pd.notna(c["recall"]) else 1e9,
            -safe_float(c["f1"]) if pd.notna(c["f1"]) else 1e9,
            -safe_float(c["precision"]) if pd.notna(c["precision"]) else 1e9,
            -safe_float(c["roc_auc"]) if pd.notna(c["roc_auc"]) else 1e9,
            -safe_float(c["accuracy"]) if pd.notna(c["accuracy"]) else 1e9,
            fpr_missing,
            fpr_value,
            c["tie_pref"]
        )

    candidates_sorted = sorted(candidates, key=sort_key)
    return candidates_sorted[0]["variant"]


def build_selected_variant_table(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick one chosen variant per experiment using VALID row,
    then keep both VALID and TEST rows from that chosen variant.
    """
    rows = []

    valid_wide = wide_df[wide_df["split"].str.lower() == "valid"].copy()

    for _, valid_row in valid_wide.iterrows():
        exp_key = valid_row["experiment_key"]
        chosen_variant = choose_best_variant_from_valid(valid_row)

        selected_rows = long_df[
            (long_df["experiment_key"] == exp_key) &
            (long_df["variant"] == chosen_variant)
        ].copy()

        selected_rows["chosen_variant_from_valid"] = chosen_variant
        rows.append(selected_rows)

    selected_df = pd.concat(rows, axis=0, ignore_index=True)
    return selected_df


def build_leaderboard(df: pd.DataFrame, split_name: str, experiment_keys: list) -> pd.DataFrame:
    split_df = df[
        (df["split"].str.lower() == split_name.lower()) &
        (df["experiment_key"].isin(experiment_keys))
    ].copy()

    split_df = split_df.sort_values(
        by=["pr_auc", "recall", "f1", "precision", "roc_auc", "accuracy"],
        ascending=[False, False, False, False, False, False]
    ).reset_index(drop=True)

    split_df["rank_by_pr_auc"] = np.arange(1, len(split_df) + 1)

    front_cols = [
        "rank_by_pr_auc",
        "display_name",
        "experiment_key",
        "component_views_standardized",
        "split",
        "variant",
        "chosen_variant_from_valid",
        "pr_auc",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
    remaining = [c for c in split_df.columns if c not in front_cols]
    split_df = split_df[front_cols + remaining]

    return split_df


def build_best_standard_vs_multiview_table(standard_lb: pd.DataFrame, multiview_lb: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []

    if standard_lb.empty or multiview_lb.empty:
        return pd.DataFrame()

    std = standard_lb.iloc[0]
    mv = multiview_lb.iloc[0]

    for model_type, row in [("standard_bagging", std), ("multiview_ensemble", mv)]:
        rows.append({
            "split": split_name,
            "group_type": model_type,
            "display_name": row["display_name"],
            "experiment_key": row["experiment_key"],
            "component_views": row["component_views_standardized"],
            "variant": row["variant"],
            "pr_auc": safe_float(row["pr_auc"]),
            "roc_auc": safe_float(row["roc_auc"]),
            "accuracy": safe_float(row["accuracy"]),
            "precision": safe_float(row["precision"]),
            "recall": safe_float(row["recall"]),
            "f1": safe_float(row["f1"])
        })

    compare_df = pd.DataFrame(rows)

    if len(compare_df) == 2:
        std_row = compare_df.iloc[0]
        mv_row = compare_df.iloc[1]

        delta_row = {
            "split": split_name,
            "group_type": "delta_multiview_minus_standard",
            "display_name": "Multiview - Standard",
            "experiment_key": "DELTA",
            "component_views": f"{mv_row['component_views']} - {std_row['component_views']}",
            "variant": f"{mv_row['variant']} - {std_row['variant']}",
            "pr_auc": safe_float(mv_row["pr_auc"]) - safe_float(std_row["pr_auc"]),
            "roc_auc": safe_float(mv_row["roc_auc"]) - safe_float(std_row["roc_auc"]),
            "accuracy": safe_float(mv_row["accuracy"]) - safe_float(std_row["accuracy"]),
            "precision": safe_float(mv_row["precision"]) - safe_float(std_row["precision"]),
            "recall": safe_float(mv_row["recall"]) - safe_float(std_row["recall"]),
            "f1": safe_float(mv_row["f1"]) - safe_float(std_row["f1"])
        }
        compare_df = pd.concat([compare_df, pd.DataFrame([delta_row])], ignore_index=True)

    return compare_df


def plot_selected_metric(df: pd.DataFrame, split_name: str, metric: str, output_path: str, title: str) -> None:
    split_df = df[df["split"].str.lower() == split_name.lower()].copy()
    split_df = split_df.sort_values(by=metric, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(split_df["display_name"], split_df[metric].fillna(0).values)
    plt.ylim(0, 1.05)
    plt.ylabel(metric.upper())
    plt.xlabel("Selected experiment")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_best_standard_vs_multiview(compare_df: pd.DataFrame, output_path: str, title: str) -> None:
    plot_df = compare_df[compare_df["group_type"].isin(["standard_bagging", "multiview_ensemble"])].copy()

    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]
    plot_df = plot_df.set_index("group_type")[metrics].T

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_summary_json(
    standard_valid_lb: pd.DataFrame,
    standard_test_lb: pd.DataFrame,
    multiview_valid_lb: pd.DataFrame,
    multiview_test_lb: pd.DataFrame,
    compare_valid_df: pd.DataFrame,
    compare_test_df: pd.DataFrame
) -> dict:
    summary = {
        "best_standard_valid": {},
        "best_standard_test": {},
        "best_multiview_valid": {},
        "best_multiview_test": {},
        "valid_multiview_minus_standard": {},
        "test_multiview_minus_standard": {}
    }

    if not standard_valid_lb.empty:
        row = standard_valid_lb.iloc[0]
        summary["best_standard_valid"] = {
            "display_name": row["display_name"],
            "experiment_key": row["experiment_key"],
            "variant": row["variant"],
            "pr_auc": safe_float(row["pr_auc"]),
            "roc_auc": safe_float(row["roc_auc"]),
            "precision": safe_float(row["precision"]),
            "recall": safe_float(row["recall"]),
            "f1": safe_float(row["f1"]),
            "accuracy": safe_float(row["accuracy"])
        }

    if not standard_test_lb.empty:
        row = standard_test_lb.iloc[0]
        summary["best_standard_test"] = {
            "display_name": row["display_name"],
            "experiment_key": row["experiment_key"],
            "variant": row["variant"],
            "pr_auc": safe_float(row["pr_auc"]),
            "roc_auc": safe_float(row["roc_auc"]),
            "precision": safe_float(row["precision"]),
            "recall": safe_float(row["recall"]),
            "f1": safe_float(row["f1"]),
            "accuracy": safe_float(row["accuracy"])
        }

    if not multiview_valid_lb.empty:
        row = multiview_valid_lb.iloc[0]
        summary["best_multiview_valid"] = {
            "display_name": row["display_name"],
            "experiment_key": row["experiment_key"],
            "variant": row["variant"],
            "pr_auc": safe_float(row["pr_auc"]),
            "roc_auc": safe_float(row["roc_auc"]),
            "precision": safe_float(row["precision"]),
            "recall": safe_float(row["recall"]),
            "f1": safe_float(row["f1"]),
            "accuracy": safe_float(row["accuracy"])
        }

    if not multiview_test_lb.empty:
        row = multiview_test_lb.iloc[0]
        summary["best_multiview_test"] = {
            "display_name": row["display_name"],
            "experiment_key": row["experiment_key"],
            "variant": row["variant"],
            "pr_auc": safe_float(row["pr_auc"]),
            "roc_auc": safe_float(row["roc_auc"]),
            "precision": safe_float(row["precision"]),
            "recall": safe_float(row["recall"]),
            "f1": safe_float(row["f1"]),
            "accuracy": safe_float(row["accuracy"])
        }

    if not compare_valid_df.empty and len(compare_valid_df) >= 3:
        row = compare_valid_df[compare_valid_df["group_type"] == "delta_multiview_minus_standard"].iloc[0]
        summary["valid_multiview_minus_standard"] = {
            "delta_pr_auc": safe_float(row["pr_auc"]),
            "delta_roc_auc": safe_float(row["roc_auc"]),
            "delta_precision": safe_float(row["precision"]),
            "delta_recall": safe_float(row["recall"]),
            "delta_f1": safe_float(row["f1"]),
            "delta_accuracy": safe_float(row["accuracy"])
        }

    if not compare_test_df.empty and len(compare_test_df) >= 3:
        row = compare_test_df[compare_test_df["group_type"] == "delta_multiview_minus_standard"].iloc[0]
        summary["test_multiview_minus_standard"] = {
            "delta_pr_auc": safe_float(row["pr_auc"]),
            "delta_roc_auc": safe_float(row["roc_auc"]),
            "delta_precision": safe_float(row["precision"]),
            "delta_recall": safe_float(row["recall"]),
            "delta_f1": safe_float(row["f1"]),
            "delta_accuracy": safe_float(row["accuracy"])
        }

    return summary


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=== Step 27: Compare Standard Bagging vs Multiview Ensemble ===")
    make_dir(OUTPUT_DIR)

    print("\n[1/6] Loading Step 26 comparison outputs...")
    long_df = read_csv(LONG_INPUT_CSV_PATH)
    wide_df = read_csv(WIDE_INPUT_CSV_PATH)

    long_df = standardize_numeric_columns(long_df)

    print(f"Long rows loaded: {long_df.shape[0]}")
    print(f"Wide rows loaded: {wide_df.shape[0]}")

    print("\n[2/6] Selecting best variant per experiment using VALID-first logic...")
    selected_df = build_selected_variant_table(long_df, wide_df)
    selected_df.to_csv(SELECTED_EXPERIMENTS_CSV_PATH, index=False)

    selected_valid = selected_df[selected_df["split"].str.lower() == "valid"].copy()
    selected_test = selected_df[selected_df["split"].str.lower() == "test"].copy()

    selected_valid.to_csv(SELECTED_VALID_CSV_PATH, index=False)
    selected_test.to_csv(SELECTED_TEST_CSV_PATH, index=False)

    print(f"[OK] Selected experiment table saved to: {SELECTED_EXPERIMENTS_CSV_PATH}")
    print(f"[OK] Selected VALID table saved to: {SELECTED_VALID_CSV_PATH}")
    print(f"[OK] Selected TEST table saved to: {SELECTED_TEST_CSV_PATH}")

    print("\n[3/6] Building standard and multiview leaderboards...")
    standard_valid_lb = build_leaderboard(selected_df, "valid", STANDARD_KEYS)
    standard_test_lb = build_leaderboard(selected_df, "test", STANDARD_KEYS)

    multiview_valid_lb = build_leaderboard(selected_df, "valid", MULTIVIEW_KEYS)
    multiview_test_lb = build_leaderboard(selected_df, "test", MULTIVIEW_KEYS)

    standard_valid_lb.to_csv(STANDARD_VALID_LEADERBOARD_CSV_PATH, index=False)
    standard_test_lb.to_csv(STANDARD_TEST_LEADERBOARD_CSV_PATH, index=False)

    multiview_valid_lb.to_csv(MULTIVIEW_VALID_LEADERBOARD_CSV_PATH, index=False)
    multiview_test_lb.to_csv(MULTIVIEW_TEST_LEADERBOARD_CSV_PATH, index=False)

    print(f"[OK] Standard VALID leaderboard saved to: {STANDARD_VALID_LEADERBOARD_CSV_PATH}")
    print(f"[OK] Standard TEST leaderboard saved to: {STANDARD_TEST_LEADERBOARD_CSV_PATH}")
    print(f"[OK] Multiview VALID leaderboard saved to: {MULTIVIEW_VALID_LEADERBOARD_CSV_PATH}")
    print(f"[OK] Multiview TEST leaderboard saved to: {MULTIVIEW_TEST_LEADERBOARD_CSV_PATH}")

    print("\n[4/6] Comparing best standard vs best multiview...")
    compare_valid_df = build_best_standard_vs_multiview_table(standard_valid_lb, multiview_valid_lb, "valid")
    compare_test_df = build_best_standard_vs_multiview_table(standard_test_lb, multiview_test_lb, "test")

    compare_valid_df.to_csv(BEST_STANDARD_VS_MULTIVIEW_VALID_CSV_PATH, index=False)
    compare_test_df.to_csv(BEST_STANDARD_VS_MULTIVIEW_TEST_CSV_PATH, index=False)

    print(f"[OK] VALID best-standard-vs-multiview CSV saved to: {BEST_STANDARD_VS_MULTIVIEW_VALID_CSV_PATH}")
    print(f"[OK] TEST best-standard-vs-multiview CSV saved to: {BEST_STANDARD_VS_MULTIVIEW_TEST_CSV_PATH}")

    print("\n[5/6] Saving comparison plots...")
    plot_selected_metric(
        selected_df,
        "valid",
        "pr_auc",
        SELECTED_PR_AUC_VALID_PLOT_PATH,
        "Selected Best Variant per Experiment - PR-AUC (VALID)"
    )
    plot_selected_metric(
        selected_df,
        "test",
        "pr_auc",
        SELECTED_PR_AUC_TEST_PLOT_PATH,
        "Selected Best Variant per Experiment - PR-AUC (TEST)"
    )

    plot_selected_metric(
        selected_df,
        "valid",
        "f1",
        SELECTED_F1_VALID_PLOT_PATH,
        "Selected Best Variant per Experiment - F1 (VALID)"
    )
    plot_selected_metric(
        selected_df,
        "test",
        "f1",
        SELECTED_F1_TEST_PLOT_PATH,
        "Selected Best Variant per Experiment - F1 (TEST)"
    )

    if not compare_valid_df.empty:
        plot_best_standard_vs_multiview(
            compare_valid_df,
            BEST_STANDARD_VS_MULTIVIEW_VALID_PLOT_PATH,
            "Best Standard Bagging vs Best Multiview Ensemble (VALID)"
        )
    if not compare_test_df.empty:
        plot_best_standard_vs_multiview(
            compare_test_df,
            BEST_STANDARD_VS_MULTIVIEW_TEST_PLOT_PATH,
            "Best Standard Bagging vs Best Multiview Ensemble (TEST)"
        )

    print(f"[OK] Selected PR-AUC VALID plot saved to: {SELECTED_PR_AUC_VALID_PLOT_PATH}")
    print(f"[OK] Selected PR-AUC TEST plot saved to: {SELECTED_PR_AUC_TEST_PLOT_PATH}")
    print(f"[OK] Selected F1 VALID plot saved to: {SELECTED_F1_VALID_PLOT_PATH}")
    print(f"[OK] Selected F1 TEST plot saved to: {SELECTED_F1_TEST_PLOT_PATH}")
    print(f"[OK] Best standard vs multiview VALID plot saved to: {BEST_STANDARD_VS_MULTIVIEW_VALID_PLOT_PATH}")
    print(f"[OK] Best standard vs multiview TEST plot saved to: {BEST_STANDARD_VS_MULTIVIEW_TEST_PLOT_PATH}")

    print("\n[6/6] Saving summary JSON...")
    summary = build_summary_json(
        standard_valid_lb,
        standard_test_lb,
        multiview_valid_lb,
        multiview_test_lb,
        compare_valid_df,
        compare_test_df
    )

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary JSON saved to: {SUMMARY_JSON_PATH}")

    print("\nBest STANDARD bagging on VALID:")
    if not standard_valid_lb.empty:
        print(standard_valid_lb.head(1).to_string(index=False))

    print("\nBest MULTIVIEW ensemble on VALID:")
    if not multiview_valid_lb.empty:
        print(multiview_valid_lb.head(1).to_string(index=False))

    print("\nBest STANDARD bagging on TEST:")
    if not standard_test_lb.empty:
        print(standard_test_lb.head(1).to_string(index=False))

    print("\nBest MULTIVIEW ensemble on TEST:")
    if not multiview_test_lb.empty:
        print(multiview_test_lb.head(1).to_string(index=False))

    print("\nDirect comparison on VALID:")
    if not compare_valid_df.empty:
        print(compare_valid_df.to_string(index=False))

    print("\nDirect comparison on TEST:")
    if not compare_test_df.empty:
        print(compare_test_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)