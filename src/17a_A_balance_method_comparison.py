# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 02:30:28 2026

@author: FarhanAli
"""

# 17a_A_balance_method_comparison.py
# Balancing Method Comparison on Aggregation View [A]
#
# This script:
# 1. Loads the prepared Aggregation view data (train/valid/test)
# 2. Loads the best tuned model parameters from the earlier A-view experiment
# 3. Compares four balancing methods on TRAIN only:
#    - baseline
#    - sub_sampling (random undersampling)
#    - smote
#    - sub_sampling_plus_smote
# 4. Evaluates each method on VALID and TEST
# 5. Saves:
#    - combined metrics CSV
#    - validation predictions CSV
#    - test predictions CSV
#    - PR-curve comparison plots
#    - metrics comparison plots
#    - confusion matrix grids
#    - balancing summary JSON
#    - resampling distribution CSVs

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
except ImportError as err:
    raise ImportError(
        "This script requires imbalanced-learn. "
        "Install it with: pip install imbalanced-learn"
    ) from err


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "17a_A_balance_method_comparison")

TRAIN_X_PATH = os.path.join(PROCESSED_DIR, "X_train_view_A.csv")
VALID_X_PATH = os.path.join(PROCESSED_DIR, "X_valid_view_A.csv")
TEST_X_PATH = os.path.join(PROCESSED_DIR, "X_test_view_A.csv")

TRAIN_Y_PATH = os.path.join(PROCESSED_DIR, "y_train.csv")
VALID_Y_PATH = os.path.join(PROCESSED_DIR, "y_valid.csv")
TEST_Y_PATH = os.path.join(PROCESSED_DIR, "y_test.csv")

BEST_PARAMS_A_PATH = os.path.join(PROJECT_ROOT, "results", "09a_individual_A_tuned", "best_params_A.json")

METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_A_balance_method_comparison.csv")
VALID_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "valid_predictions_A_balance_methods.csv")
TEST_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions_A_balance_methods.csv")

VALID_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_valid_A_balance_methods.png")
TEST_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_test_A_balance_methods.png")

VALID_METRICS_PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_valid_A_balance_methods.png")
TEST_METRICS_PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_test_A_balance_methods.png")

VALID_CM_GRID_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_grid_valid_A_balance_methods.png")
TEST_CM_GRID_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_grid_test_A_balance_methods.png")

SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "summary_A_balance_methods.json")
RESAMPLING_DISTRIBUTION_CSV_PATH = os.path.join(RESULTS_DIR, "resampling_class_distributions_A.csv")
RUN_CONFIG_JSON_PATH = os.path.join(RESULTS_DIR, "run_config_A_balance_methods.json")


# ============================================================
# Experiment settings
# ============================================================
RANDOM_SEED = 42
#old settings
#DEFAULT_THRESHOLD = 0.50
#SUB_SAMPLING_STRATEGY = 0.10
#SMOTE_SAMPLING_STRATEGY = 0.10
#COMBINED_UNDER_STRATEGY = 0.05
#COMBINED_SMOTE_STRATEGY = 0.10
#SMOTE_K_NEIGHBORS = 5

#new settings
#DEFAULT_THRESHOLD = 0.25
#SUB_SAMPLING_STRATEGY = 0.10
#SMOTE_SAMPLING_STRATEGY = 0.10
#COMBINED_UNDER_STRATEGY = 0.05
#COMBINED_SMOTE_STRATEGY = 0.10
#SMOTE_K_NEIGHBORS = 5

#new settings 
#DEFAULT_THRESHOLD = 0.20
#SUB_SAMPLING_STRATEGY = 0.20
#SMOTE_SAMPLING_STRATEGY = 0.30
#COMBINED_UNDER_STRATEGY = 0.10
#COMBINED_SMOTE_STRATEGY = 0.30
#SMOTE_K_NEIGHBORS = 3

#further new settings for better result
DEFAULT_THRESHOLD = 0.25
SUB_SAMPLING_STRATEGY = 0.10
SMOTE_SAMPLING_STRATEGY = 0.20
COMBINED_UNDER_STRATEGY = 0.05
COMBINED_SMOTE_STRATEGY = 0.20
SMOTE_K_NEIGHBORS = 3


METHOD_ORDER = [
    "baseline",
    "sub_sampling",
    "smote",
    "sub_sampling_plus_smote"
]


# ============================================================
# Helper functions
# ============================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def read_feature_matrix(path: str) -> pd.DataFrame:
    """
    Load feature matrix and force numeric format.
    Any non-numeric values are coerced and filled with 0.
    """
    assert_file_exists(path)
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.astype(np.float32)


def read_binary_target(path: str) -> np.ndarray:
    """
    Robust target loader for CSV files with or without header.
    """
    assert_file_exists(path)
    raw = pd.read_csv(path, header=None).iloc[:, 0]
    y = pd.to_numeric(raw, errors="coerce").dropna().astype(int).values
    return y


def read_json_file(path: str) -> dict:
    assert_file_exists(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    X_train = read_feature_matrix(TRAIN_X_PATH)
    X_valid = read_feature_matrix(VALID_X_PATH)
    X_test = read_feature_matrix(TEST_X_PATH)

    y_train = read_binary_target(TRAIN_Y_PATH)
    y_valid = read_binary_target(VALID_Y_PATH)
    y_test = read_binary_target(TEST_Y_PATH)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def build_model(config: dict) -> BaggingClassifier:
    """
    Build Bagging classifier with a Decision Tree base learner.
    """
    base_tree = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=config.get("random_state", RANDOM_SEED)
    )

    model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        bootstrap=config.get("bootstrap", True),
        n_jobs=-1,
        random_state=config.get("random_state", RANDOM_SEED)
    )

    return model


def class_distribution_dict(y: np.ndarray, label: str) -> dict:
    """
    Return a simple class distribution summary.
    """
    unique, counts = np.unique(y, return_counts=True)
    mapping = {int(k): int(v) for k, v in zip(unique, counts)}

    count_0 = mapping.get(0, 0)
    count_1 = mapping.get(1, 0)
    total = count_0 + count_1
    minority_ratio = (count_1 / count_0) if count_0 > 0 else np.nan
    fraud_rate = (count_1 / total) if total > 0 else np.nan

    return {
        "label": label,
        "count_class_0": count_0,
        "count_class_1": count_1,
        "total_rows": total,
        "minority_over_majority": float(minority_ratio) if pd.notna(minority_ratio) else np.nan,
        "fraud_rate": float(fraud_rate) if pd.notna(fraud_rate) else np.nan
    }


def apply_balancing_method(method_name: str, X_train: pd.DataFrame, y_train: np.ndarray):
    """
    Apply balancing only on training data.
    """
    if method_name == "baseline":
        X_res = X_train.copy()
        y_res = y_train.copy()

    elif method_name == "sub_sampling":
        sampler = RandomUnderSampler(
            sampling_strategy=SUB_SAMPLING_STRATEGY,
            random_state=RANDOM_SEED
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    elif method_name == "smote":
        sampler = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=RANDOM_SEED
        )
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    elif method_name == "sub_sampling_plus_smote":
        under_sampler = RandomUnderSampler(
            sampling_strategy=COMBINED_UNDER_STRATEGY,
            random_state=RANDOM_SEED
        )
        X_temp, y_temp = under_sampler.fit_resample(X_train, y_train)

        over_sampler = SMOTE(
            sampling_strategy=COMBINED_SMOTE_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=RANDOM_SEED
        )
        X_res, y_res = over_sampler.fit_resample(X_temp, y_temp)

    else:
        raise ValueError(f"Unknown method_name: {method_name}")

    return X_res, y_res


def compute_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute threshold-independent metrics.
    """
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Compute threshold-dependent metrics including confusion-matrix-based rates.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_predictions": int(y_pred.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
        "tnr": float(tnr)
    }


def evaluate_one_split(
    method_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_config: dict
) -> dict:
    """
    Evaluate one method on one split.
    """
    prob_metrics = compute_probability_metrics(y_true, y_prob)
    cls_metrics = compute_threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)

    row = {
        "view": "A",
        "method": method_name,
        "model": "Bagging_DT_A",
        "split": split_name,
        "threshold": DEFAULT_THRESHOLD,
        "max_depth": model_config["max_depth"],
        "min_samples_split": model_config["min_samples_split"],
        "min_samples_leaf": model_config["min_samples_leaf"],
        "n_estimators": model_config["n_estimators"],
        "max_samples": model_config["max_samples"],
        "pr_auc": prob_metrics["pr_auc"],
        "roc_auc": prob_metrics["roc_auc"],
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "positive_predictions": cls_metrics["positive_predictions"],
        "tn": cls_metrics["tn"],
        "fp": cls_metrics["fp"],
        "fn": cls_metrics["fn"],
        "tp": cls_metrics["tp"],
        "fpr": cls_metrics["fpr"],
        "fnr": cls_metrics["fnr"],
        "tpr": cls_metrics["tpr"],
        "tnr": cls_metrics["tnr"],
        "total_rows": int(len(y_true)),
        "positive_rate_predicted": float(cls_metrics["positive_predictions"] / len(y_true))
    }

    return row


def save_predictions_wide_csv(
    path: str,
    y_true: np.ndarray,
    prob_dict: dict,
    threshold: float
) -> None:
    """
    Save row-level probabilities and predictions for all methods in one file.
    """
    out_df = pd.DataFrame({
        "row_id": np.arange(len(y_true)),
        "y_true": y_true
    })

    for method_name in METHOD_ORDER:
        y_prob = prob_dict[method_name]
        y_pred = (y_prob >= threshold).astype(int)
        out_df[f"y_prob_{method_name}"] = y_prob
        out_df[f"y_pred_{method_name}"] = y_pred

    out_df.to_csv(path, index=False)


def plot_pr_comparison(y_true: np.ndarray, prob_dict: dict, title: str, output_path: str) -> None:
    """
    Save PR-curve comparison plot across methods.
    """
    plt.figure(figsize=(9, 7))

    for method_name in METHOD_ORDER:
        y_prob = prob_dict[method_name]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{method_name} (AP={pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, split_name: str, output_path: str, title: str) -> None:
    """
    Save grouped bar chart for the selected split across methods.
    """
    split_df = metrics_df[metrics_df["split"].str.lower() == split_name.lower()].copy()
    split_df["method"] = pd.Categorical(split_df["method"], categories=METHOD_ORDER, ordered=True)
    split_df = split_df.sort_values("method")

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]

    x = np.arange(len(METHOD_ORDER))
    width = 0.11

    plt.figure(figsize=(13, 6))

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, split_df[metric].values, width=width, label=metric)

    plt.xticks(x + width * (len(metrics_to_plot) - 1) / 2, split_df["method"].tolist(), rotation=20)
    plt.ylabel("Score")
    plt.xlabel("Balancing Method")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix_grid(
    y_true: np.ndarray,
    prob_dict: dict,
    threshold: float,
    title: str,
    output_path: str
) -> None:
    """
    Save one 2x2 confusion matrix grid covering all four methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, method_name in zip(axes, METHOD_ORDER):
        y_prob = prob_dict[method_name]
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(method_name)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_summary(metrics_df: pd.DataFrame, distribution_df: pd.DataFrame) -> dict:
    """
    Build a compact summary JSON.
    """
    valid_df = metrics_df[metrics_df["split"] == "valid"].copy()
    test_df = metrics_df[metrics_df["split"] == "test"].copy()

    valid_rank = valid_df.sort_values(
        by=["pr_auc", "recall", "f1", "fpr", "precision", "roc_auc", "accuracy"],
        ascending=[False, False, False, True, False, False, False]
    )

    test_rank = test_df.sort_values(
        by=["pr_auc", "recall", "f1", "fpr", "precision", "roc_auc", "accuracy"],
        ascending=[False, False, False, True, False, False, False]
    )

    summary = {
        "selection_priority": ["pr_auc", "recall", "f1", "fpr_low", "precision", "roc_auc", "accuracy"],
        "top_valid_method": {},
        "top_test_method": {},
        "train_class_distributions": distribution_df.to_dict(orient="records")
    }

    if not valid_rank.empty:
        top_valid = valid_rank.iloc[0]
        summary["top_valid_method"] = {
            "method": top_valid["method"],
            "pr_auc": float(top_valid["pr_auc"]),
            "roc_auc": float(top_valid["roc_auc"]),
            "accuracy": float(top_valid["accuracy"]),
            "precision": float(top_valid["precision"]),
            "recall": float(top_valid["recall"]),
            "f1": float(top_valid["f1"]),
            "fp": int(top_valid["fp"]),
            "fn": int(top_valid["fn"]),
            "tp": int(top_valid["tp"]),
            "tn": int(top_valid["tn"]),
            "fpr": float(top_valid["fpr"]),
            "fnr": float(top_valid["fnr"])
        }

    if not test_rank.empty:
        top_test = test_rank.iloc[0]
        summary["top_test_method"] = {
            "method": top_test["method"],
            "pr_auc": float(top_test["pr_auc"]),
            "roc_auc": float(top_test["roc_auc"]),
            "accuracy": float(top_test["accuracy"]),
            "precision": float(top_test["precision"]),
            "recall": float(top_test["recall"]),
            "f1": float(top_test["f1"]),
            "fp": int(top_test["fp"]),
            "fn": int(top_test["fn"]),
            "tp": int(top_test["tp"]),
            "tn": int(top_test["tn"]),
            "fpr": float(top_test["fpr"]),
            "fnr": float(top_test["fnr"])
        }

    return summary


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    print("=== Step 17a: Balancing Method Comparison on A View ===")
    make_dir(RESULTS_DIR)

    print("\n[1/8] Loading A-view data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    print(f"X_train_A: {X_train.shape}")
    print(f"X_valid_A: {X_valid.shape}")
    print(f"X_test_A : {X_test.shape}")
    print(f"y_train  : {y_train.shape}")
    print(f"y_valid  : {y_valid.shape}")
    print(f"y_test   : {y_test.shape}")

    if len(X_train) != len(y_train):
        raise ValueError("Mismatch between X_train and y_train row counts.")
    if len(X_valid) != len(y_valid):
        raise ValueError("Mismatch between X_valid and y_valid row counts.")
    if len(X_test) != len(y_test):
        raise ValueError("Mismatch between X_test and y_test row counts.")

    print("\n[2/8] Loading best tuned A-view parameters...")
    best_params = read_json_file(BEST_PARAMS_A_PATH)
    best_params["bootstrap"] = best_params.get("bootstrap", True)
    best_params["random_state"] = best_params.get("random_state", RANDOM_SEED)

    with open(RUN_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "view_name": "A",
            "base_model": "Bagging_DT_A",
            "threshold": DEFAULT_THRESHOLD,
            "best_params_from_prior_tuned_run": best_params,
            "balancing_methods": {
                "baseline": {},
                "sub_sampling": {
                    "sampler": "RandomUnderSampler",
                    "sampling_strategy": SUB_SAMPLING_STRATEGY
                },
                "smote": {
                    "sampler": "SMOTE",
                    "sampling_strategy": SMOTE_SAMPLING_STRATEGY,
                    "k_neighbors": SMOTE_K_NEIGHBORS
                },
                "sub_sampling_plus_smote": {
                    "step_1": {
                        "sampler": "RandomUnderSampler",
                        "sampling_strategy": COMBINED_UNDER_STRATEGY
                    },
                    "step_2": {
                        "sampler": "SMOTE",
                        "sampling_strategy": COMBINED_SMOTE_STRATEGY,
                        "k_neighbors": SMOTE_K_NEIGHBORS
                    }
                }
            }
        }, f, indent=2)

    print(best_params)

    print("\n[3/8] Running balancing method comparison...")
    metric_rows = []
    distribution_rows = []
    valid_prob_dict = {}
    test_prob_dict = {}

    original_dist = class_distribution_dict(y_train, "original_train")
    distribution_rows.append({"method": "original_train", **original_dist})

    for method_name in METHOD_ORDER:
        print(f"\n--- Method: {method_name} ---")
        X_res, y_res = apply_balancing_method(method_name, X_train, y_train)

        dist = class_distribution_dict(y_res, f"{method_name}_resampled_train")
        distribution_rows.append({"method": method_name, **dist})

        print(
            f"Resampled train shape: {X_res.shape}, "
            f"class_0={dist['count_class_0']}, class_1={dist['count_class_1']}, "
            f"minority/majority={dist['minority_over_majority']:.4f}"
        )

        model = build_model(best_params)
        model.fit(X_res, y_res)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        valid_prob_dict[method_name] = valid_prob
        test_prob_dict[method_name] = test_prob

        valid_row = evaluate_one_split(method_name, "valid", y_valid, valid_prob, best_params)
        test_row = evaluate_one_split(method_name, "test", y_test, test_prob, best_params)

        metric_rows.append(valid_row)
        metric_rows.append(test_row)

        print(
            f"VALID -> PR-AUC={valid_row['pr_auc']:.6f}, "
            f"Recall={valid_row['recall']:.6f}, F1={valid_row['f1']:.6f}, "
            f"Precision={valid_row['precision']:.6f}, FPR={valid_row['fpr']:.6f}, "
            f"FP={valid_row['fp']}, TP={valid_row['tp']}"
        )
        print(
            f"TEST  -> PR-AUC={test_row['pr_auc']:.6f}, "
            f"Recall={test_row['recall']:.6f}, F1={test_row['f1']:.6f}, "
            f"Precision={test_row['precision']:.6f}, FPR={test_row['fpr']:.6f}, "
            f"FP={test_row['fp']}, TP={test_row['tp']}"
        )

    print("\n[4/8] Saving metrics and class distributions...")
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["method"] = pd.Categorical(metrics_df["method"], categories=METHOD_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values(by=["split", "method"]).reset_index(drop=True)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(RESAMPLING_DISTRIBUTION_CSV_PATH, index=False)

    print(f"[OK] Metrics CSV saved to: {METRICS_CSV_PATH}")
    print(f"[OK] Resampling distribution CSV saved to: {RESAMPLING_DISTRIBUTION_CSV_PATH}")

    print("\n[5/8] Saving wide prediction files...")
    save_predictions_wide_csv(VALID_PREDICTIONS_CSV_PATH, y_valid, valid_prob_dict, DEFAULT_THRESHOLD)
    save_predictions_wide_csv(TEST_PREDICTIONS_CSV_PATH, y_test, test_prob_dict, DEFAULT_THRESHOLD)

    print(f"[OK] VALID predictions CSV saved to: {VALID_PREDICTIONS_CSV_PATH}")
    print(f"[OK] TEST predictions CSV saved to: {TEST_PREDICTIONS_CSV_PATH}")

    print("\n[6/8] Saving PR comparison plots...")
    plot_pr_comparison(
        y_valid,
        valid_prob_dict,
        "PR Curve Comparison (VALID) - A View Balancing Methods",
        VALID_PR_PLOT_PATH
    )
    plot_pr_comparison(
        y_test,
        test_prob_dict,
        "PR Curve Comparison (TEST) - A View Balancing Methods",
        TEST_PR_PLOT_PATH
    )

    print(f"[OK] VALID PR comparison plot saved to: {VALID_PR_PLOT_PATH}")
    print(f"[OK] TEST PR comparison plot saved to: {TEST_PR_PLOT_PATH}")

    print("\n[7/8] Saving metric comparison plots and confusion matrix grids...")
    plot_metrics_comparison(
        metrics_df,
        "valid",
        VALID_METRICS_PLOT_PATH,
        "Metric Comparison (VALID) - A View Balancing Methods"
    )
    plot_metrics_comparison(
        metrics_df,
        "test",
        TEST_METRICS_PLOT_PATH,
        "Metric Comparison (TEST) - A View Balancing Methods"
    )

    plot_confusion_matrix_grid(
        y_valid,
        valid_prob_dict,
        DEFAULT_THRESHOLD,
        "Confusion Matrices (VALID) - A View Balancing Methods",
        VALID_CM_GRID_PATH
    )
    plot_confusion_matrix_grid(
        y_test,
        test_prob_dict,
        DEFAULT_THRESHOLD,
        "Confusion Matrices (TEST) - A View Balancing Methods",
        TEST_CM_GRID_PATH
    )

    print(f"[OK] VALID metrics comparison plot saved to: {VALID_METRICS_PLOT_PATH}")
    print(f"[OK] TEST metrics comparison plot saved to: {TEST_METRICS_PLOT_PATH}")
    print(f"[OK] VALID confusion matrix grid saved to: {VALID_CM_GRID_PATH}")
    print(f"[OK] TEST confusion matrix grid saved to: {TEST_CM_GRID_PATH}")

    print("\n[8/8] Saving summary JSON...")
    summary = build_summary(metrics_df, distribution_df)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary JSON saved to: {SUMMARY_JSON_PATH}")

    print("\nVALID ranking by PR-AUC:")
    valid_rank = metrics_df[metrics_df["split"] == "valid"].sort_values(
        by=["pr_auc", "recall", "f1", "fpr", "precision", "roc_auc", "accuracy"],
        ascending=[False, False, False, True, False, False, False]
    )
    print(valid_rank[[
        "method", "pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "fp", "tp", "fpr", "fnr"
    ]].to_string(index=False))

    print("\nTEST ranking by PR-AUC:")
    test_rank = metrics_df[metrics_df["split"] == "test"].sort_values(
        by=["pr_auc", "recall", "f1", "fpr", "precision", "roc_auc", "accuracy"],
        ascending=[False, False, False, True, False, False, False]
    )
    print(test_rank[[
        "method", "pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "fp", "tp", "fpr", "fnr"
    ]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)