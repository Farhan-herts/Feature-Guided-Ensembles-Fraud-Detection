# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:44:31 2026

@author: FarhanAli
"""


# 09c_individual_T_bagging.py
# Tuned Individual Bagging Experiment on Temporal View [T]
#
# This script:
# 1. Loads the prepared Temporal view data (train/valid/test)
# 2. Runs hyperparameter tuning on the VALID set
# 3. Selects the best configuration using VALID PR-AUC
# 4. Trains the final model with the best configuration
# 5. Evaluates on VALID and TEST sets
# 6. Saves:
#    - tuning results CSV
#    - best params JSON
#    - final metrics CSV
#    - validation predictions CSV
#    - test predictions CSV
#    - PR-curve plots
#    - metrics bar chart
#    - confusion matrix figures

import os
import sys
import json
import itertools
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


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "09c_individual_T_tuned")

TRAIN_X_PATH = os.path.join(PROCESSED_DIR, "X_train_view_T.csv")
VALID_X_PATH = os.path.join(PROCESSED_DIR, "X_valid_view_T.csv")
TEST_X_PATH = os.path.join(PROCESSED_DIR, "X_test_view_T.csv")

TRAIN_Y_PATH = os.path.join(PROCESSED_DIR, "y_train.csv")
VALID_Y_PATH = os.path.join(PROCESSED_DIR, "y_valid.csv")
TEST_Y_PATH = os.path.join(PROCESSED_DIR, "y_test.csv")

TUNING_CSV_PATH = os.path.join(RESULTS_DIR, "tuning_results_T.csv")
BEST_PARAMS_JSON_PATH = os.path.join(RESULTS_DIR, "best_params_T.json")
FINAL_METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_T_bagging_tuned.csv")
VALID_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "valid_predictions_T.csv")
TEST_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions_T.csv")
VALID_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_valid_T.png")
TEST_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_test_T.png")
METRICS_BAR_PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_bar_chart_T.png")
VALID_CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_valid_T.png")
TEST_CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_test_T.png")
RUN_CONFIG_JSON_PATH = os.path.join(RESULTS_DIR, "run_config_T.json")


# ============================================================
# Experiment settings
# ============================================================
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.50

# Slightly broader grid for the weak Temporal view
TREE_GRID = [
    {"max_depth": 2, "min_samples_split": 10, "min_samples_leaf": 4},
    {"max_depth": 3, "min_samples_split": 10, "min_samples_leaf": 4},
    {"max_depth": 4, "min_samples_split": 10, "min_samples_leaf": 4},
    {"max_depth": 6, "min_samples_split": 5,  "min_samples_leaf": 2},
    {"max_depth": 8, "min_samples_split": 10, "min_samples_leaf": 4},
]

BAGGING_GRID = [
    {"n_estimators": 50,  "max_samples": 0.8},
    {"n_estimators": 100, "max_samples": 0.8},
    {"n_estimators": 150, "max_samples": 1.0},
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
        random_state=config["random_state"]
    )

    model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        bootstrap=config["bootstrap"],
        n_jobs=-1,
        random_state=config["random_state"]
    )

    return model


def compute_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute threshold-independent probability metrics.
    """
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Compute threshold-based classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_predictions": int(y_pred.sum())
    }


def save_predictions_csv(path: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    """
    Save row-level probabilities and binary predictions.
    """
    y_pred = (y_prob >= threshold).astype(int)

    out_df = pd.DataFrame({
        "row_id": np.arange(len(y_true)),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred_default_0_5": y_pred
    })

    out_df.to_csv(path, index=False)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, output_path: str) -> None:
    """
    Save Precision-Recall curve plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metrics_bar_chart(metrics_df: pd.DataFrame, output_path: str, title: str) -> None:
    """
    Save a bar chart for key classification metrics on VALID and TEST.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    plot_df = metrics_df.set_index("split")[metrics_to_plot].T

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    title: str,
    output_path: str
) -> None:
    """
    Save confusion matrix figure.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_split(split_name: str, y_true: np.ndarray, y_prob: np.ndarray, config: dict) -> dict:
    """
    Evaluate one dataset split and return a flat dictionary of metrics.
    """
    prob_metrics = compute_probability_metrics(y_true, y_prob)
    cls_metrics = compute_threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)

    row = {
        "view": config["view_name"],
        "model": config["model_name"],
        "split": split_name,
        "threshold": DEFAULT_THRESHOLD,
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "n_estimators": config["n_estimators"],
        "max_samples": config["max_samples"],
        "pr_auc": prob_metrics["pr_auc"],
        "roc_auc": prob_metrics["roc_auc"],
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "positive_predictions": cls_metrics["positive_predictions"],
        "total_rows": int(len(y_true)),
        "positive_rate_predicted": float(cls_metrics["positive_predictions"] / len(y_true))
    }

    return row


def run_tuning(X_train, y_train, X_valid, y_valid) -> pd.DataFrame:
    """
    Run hyperparameter tuning using VALID PR-AUC as the main selection metric.
    """
    rows = []
    combo_num = 0

    for tree_cfg, bag_cfg in itertools.product(TREE_GRID, BAGGING_GRID):
        combo_num += 1

        config = {
            "view_name": "T",
            "model_name": "Bagging_DT_T",
            "max_depth": tree_cfg["max_depth"],
            "min_samples_split": tree_cfg["min_samples_split"],
            "min_samples_leaf": tree_cfg["min_samples_leaf"],
            "n_estimators": bag_cfg["n_estimators"],
            "max_samples": bag_cfg["max_samples"],
            "bootstrap": True,
            "random_state": RANDOM_SEED
        }

        print(f"\n[TUNE {combo_num}] {config}")

        model = build_model(config)
        model.fit(X_train, y_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        prob_metrics = compute_probability_metrics(y_valid, valid_prob)
        cls_metrics = compute_threshold_metrics(y_valid, valid_prob, DEFAULT_THRESHOLD)

        row = {
            "view": config["view_name"],
            "model": config["model_name"],
            "max_depth": config["max_depth"],
            "min_samples_split": config["min_samples_split"],
            "min_samples_leaf": config["min_samples_leaf"],
            "n_estimators": config["n_estimators"],
            "max_samples": config["max_samples"],
            "valid_pr_auc": prob_metrics["pr_auc"],
            "valid_roc_auc": prob_metrics["roc_auc"],
            "valid_accuracy_at_0_5": cls_metrics["accuracy"],
            "valid_precision_at_0_5": cls_metrics["precision"],
            "valid_recall_at_0_5": cls_metrics["recall"],
            "valid_f1_at_0_5": cls_metrics["f1"],
            "valid_positive_predictions": cls_metrics["positive_predictions"]
        }

        rows.append(row)

        print(
            f"    VALID PR-AUC={prob_metrics['pr_auc']:.6f} | "
            f"ROC-AUC={prob_metrics['roc_auc']:.6f} | "
            f"Accuracy@0.5={cls_metrics['accuracy']:.6f} | "
            f"Precision@0.5={cls_metrics['precision']:.6f} | "
            f"Recall@0.5={cls_metrics['recall']:.6f}"
        )

    tuning_df = pd.DataFrame(rows)

    tuning_df = tuning_df.sort_values(
        by=["valid_pr_auc", "valid_roc_auc", "valid_f1_at_0_5"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    tuning_df["rank"] = np.arange(1, len(tuning_df) + 1)

    ordered_cols = [
        "rank", "view", "model",
        "max_depth", "min_samples_split", "min_samples_leaf",
        "n_estimators", "max_samples",
        "valid_pr_auc", "valid_roc_auc",
        "valid_accuracy_at_0_5", "valid_precision_at_0_5",
        "valid_recall_at_0_5", "valid_f1_at_0_5",
        "valid_positive_predictions"
    ]
    tuning_df = tuning_df[ordered_cols]

    return tuning_df


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    print("=== Step 09c: Tuned Individual Bagging on Temporal View [T] ===")
    make_dir(RESULTS_DIR)

    print("\n[1/6] Loading data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    print(f"X_train_T: {X_train.shape}")
    print(f"X_valid_T: {X_valid.shape}")
    print(f"X_test_T : {X_test.shape}")
    print(f"y_train  : {y_train.shape}")
    print(f"y_valid  : {y_valid.shape}")
    print(f"y_test   : {y_test.shape}")

    if len(X_train) != len(y_train):
        raise ValueError("Mismatch between X_train and y_train row counts.")
    if len(X_valid) != len(y_valid):
        raise ValueError("Mismatch between X_valid and y_valid row counts.")
    if len(X_test) != len(y_test):
        raise ValueError("Mismatch between X_test and y_test row counts.")

    print("\n[2/6] Running hyperparameter tuning...")
    tuning_df = run_tuning(X_train, y_train, X_valid, y_valid)
    tuning_df.to_csv(TUNING_CSV_PATH, index=False)
    print(f"\n[OK] Tuning results saved to: {TUNING_CSV_PATH}")

    best_row = tuning_df.iloc[0]
    best_config = {
        "view_name": "T",
        "model_name": "Bagging_DT_T",
        "max_depth": int(best_row["max_depth"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "n_estimators": int(best_row["n_estimators"]),
        "max_samples": float(best_row["max_samples"]),
        "bootstrap": True,
        "random_state": RANDOM_SEED
    }

    with open(BEST_PARAMS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print("\nBest configuration selected:")
    print(best_config)
    print(f"[OK] Best params saved to: {BEST_PARAMS_JSON_PATH}")

    print("\n[3/6] Training final model with best configuration...")
    model = build_model(best_config)
    model.fit(X_train, y_train)

    print("\n[4/6] Generating probabilities...")
    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    print("\n[5/6] Evaluating final model...")
    valid_row = evaluate_split("valid", y_valid, valid_prob, best_config)
    test_row = evaluate_split("test", y_test, test_prob, best_config)

    metrics_df = pd.DataFrame([valid_row, test_row])
    metrics_df.to_csv(FINAL_METRICS_CSV_PATH, index=False)
    print(f"[OK] Final metrics saved to: {FINAL_METRICS_CSV_PATH}")

    print("\n[6/6] Saving predictions, plots, and run config...")
    save_predictions_csv(VALID_PREDICTIONS_CSV_PATH, y_valid, valid_prob, DEFAULT_THRESHOLD)
    save_predictions_csv(TEST_PREDICTIONS_CSV_PATH, y_test, test_prob, DEFAULT_THRESHOLD)

    plot_pr_curve(
        y_valid,
        valid_prob,
        "Precision-Recall Curve (VALID) - Tuned Individual T Bagging",
        VALID_PR_PLOT_PATH
    )

    plot_pr_curve(
        y_test,
        test_prob,
        "Precision-Recall Curve (TEST) - Tuned Individual T Bagging",
        TEST_PR_PLOT_PATH
    )

    plot_metrics_bar_chart(
        metrics_df,
        METRICS_BAR_PLOT_PATH,
        "Validation vs Test Metrics - Tuned Individual T Bagging"
    )

    plot_confusion_matrix_figure(
        y_valid,
        valid_prob,
        DEFAULT_THRESHOLD,
        "Confusion Matrix (VALID) - Tuned Individual T Bagging",
        VALID_CM_PLOT_PATH
    )

    plot_confusion_matrix_figure(
        y_test,
        test_prob,
        DEFAULT_THRESHOLD,
        "Confusion Matrix (TEST) - Tuned Individual T Bagging",
        TEST_CM_PLOT_PATH
    )

    with open(RUN_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print(f"[OK] VALID predictions saved to: {VALID_PREDICTIONS_CSV_PATH}")
    print(f"[OK] TEST predictions saved to: {TEST_PREDICTIONS_CSV_PATH}")
    print(f"[OK] VALID PR plot saved to: {VALID_PR_PLOT_PATH}")
    print(f"[OK] TEST PR plot saved to: {TEST_PR_PLOT_PATH}")
    print(f"[OK] Metrics bar chart saved to: {METRICS_BAR_PLOT_PATH}")
    print(f"[OK] VALID confusion matrix saved to: {VALID_CM_PLOT_PATH}")
    print(f"[OK] TEST confusion matrix saved to: {TEST_CM_PLOT_PATH}")
    print(f"[OK] Run config saved to: {RUN_CONFIG_JSON_PATH}")

    print("\nFinal metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)