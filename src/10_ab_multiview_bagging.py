# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:15:10 2026

@author: FarhanAli
"""

# 10_ab_multiview_bagging.py
# Multiview Bagging Experiment on Aggregation + Behaviour Views [A+B]
#
# This script:
# 1. Loads prepared A-view and B-view data (train/valid/test)
# 2. Loads best tuned parameters from the individual A and B experiments
# 3. Trains one bagging model for A and one bagging model for B
# 4. Combines A and B probabilities using soft voting (mean probability)
# 5. Evaluates the A+B ensemble on VALID and TEST sets
# 6. Saves:
#    - final metrics CSV
#    - validation predictions CSV
#    - test predictions CSV
#    - PR-curve plots
#    - metrics bar chart
#    - confusion matrix figures
#    - run configuration JSON
#    - component summary JSON

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


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "10_ab_multiview")

# A view files
TRAIN_X_A_PATH = os.path.join(PROCESSED_DIR, "X_train_view_A.csv")
VALID_X_A_PATH = os.path.join(PROCESSED_DIR, "X_valid_view_A.csv")
TEST_X_A_PATH = os.path.join(PROCESSED_DIR, "X_test_view_A.csv")

# B view files
TRAIN_X_B_PATH = os.path.join(PROCESSED_DIR, "X_train_view_B.csv")
VALID_X_B_PATH = os.path.join(PROCESSED_DIR, "X_valid_view_B.csv")
TEST_X_B_PATH = os.path.join(PROCESSED_DIR, "X_test_view_B.csv")

# Targets
TRAIN_Y_PATH = os.path.join(PROCESSED_DIR, "y_train.csv")
VALID_Y_PATH = os.path.join(PROCESSED_DIR, "y_valid.csv")
TEST_Y_PATH = os.path.join(PROCESSED_DIR, "y_test.csv")

# Tuned params from individual experiments
BEST_PARAMS_A_PATH = os.path.join(PROJECT_ROOT, "results", "09a_individual_A_tuned", "best_params_A.json")
BEST_PARAMS_B_PATH = os.path.join(PROJECT_ROOT, "results", "09b_individual_B_tuned", "best_params_B.json")

# Outputs
FINAL_METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_AB_multiview.csv")
VALID_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "valid_predictions_AB.csv")
TEST_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions_AB.csv")
VALID_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_valid_AB.png")
TEST_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_test_AB.png")
METRICS_BAR_PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_bar_chart_AB.png")
VALID_CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_valid_AB.png")
TEST_CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_test_AB.png")
RUN_CONFIG_JSON_PATH = os.path.join(RESULTS_DIR, "run_config_AB.json")
COMPONENT_SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "component_summary_AB.json")


# ============================================================
# Experiment settings
# ============================================================
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.50


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
    X_train_A = read_feature_matrix(TRAIN_X_A_PATH)
    X_valid_A = read_feature_matrix(VALID_X_A_PATH)
    X_test_A = read_feature_matrix(TEST_X_A_PATH)

    X_train_B = read_feature_matrix(TRAIN_X_B_PATH)
    X_valid_B = read_feature_matrix(VALID_X_B_PATH)
    X_test_B = read_feature_matrix(TEST_X_B_PATH)

    y_train = read_binary_target(TRAIN_Y_PATH)
    y_valid = read_binary_target(VALID_Y_PATH)
    y_test = read_binary_target(TEST_Y_PATH)

    return (
        X_train_A, X_valid_A, X_test_A,
        X_train_B, X_valid_B, X_test_B,
        y_train, y_valid, y_test
    )


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


def save_predictions_csv(
    path: str,
    y_true: np.ndarray,
    y_prob_ensemble: np.ndarray,
    threshold: float,
    y_prob_A: np.ndarray,
    y_prob_B: np.ndarray
) -> None:
    """
    Save row-level component and ensemble probabilities.
    """
    y_pred = (y_prob_ensemble >= threshold).astype(int)

    out_df = pd.DataFrame({
        "row_id": np.arange(len(y_true)),
        "y_true": y_true,
        "y_prob_A": y_prob_A,
        "y_prob_B": y_prob_B,
        "y_prob_AB_ensemble": y_prob_ensemble,
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


def evaluate_split(split_name: str, y_true: np.ndarray, y_prob: np.ndarray, run_config: dict) -> dict:
    """
    Evaluate one dataset split and return a flat dictionary of metrics.
    """
    prob_metrics = compute_probability_metrics(y_true, y_prob)
    cls_metrics = compute_threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)

    row = {
        "view": "AB",
        "model": "SoftVoting_Bagging_DT_AB",
        "split": split_name,
        "threshold": DEFAULT_THRESHOLD,
        "combination_type": "multiview_soft_voting",
        "component_views": "A+B",
        "pr_auc": prob_metrics["pr_auc"],
        "roc_auc": prob_metrics["roc_auc"],
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "positive_predictions": cls_metrics["positive_predictions"],
        "total_rows": int(len(y_true)),
        "positive_rate_predicted": float(cls_metrics["positive_predictions"] / len(y_true)),
        "A_max_depth": run_config["A_params"]["max_depth"],
        "A_min_samples_split": run_config["A_params"]["min_samples_split"],
        "A_min_samples_leaf": run_config["A_params"]["min_samples_leaf"],
        "A_n_estimators": run_config["A_params"]["n_estimators"],
        "A_max_samples": run_config["A_params"]["max_samples"],
        "B_max_depth": run_config["B_params"]["max_depth"],
        "B_min_samples_split": run_config["B_params"]["min_samples_split"],
        "B_min_samples_leaf": run_config["B_params"]["min_samples_leaf"],
        "B_n_estimators": run_config["B_params"]["n_estimators"],
        "B_max_samples": run_config["B_params"]["max_samples"]
    }

    return row


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    print("=== Step 10: A+B Multiview Bagging (Soft Voting) ===")
    make_dir(RESULTS_DIR)

    print("\n[1/7] Loading A-view and B-view data...")
    (
        X_train_A, X_valid_A, X_test_A,
        X_train_B, X_valid_B, X_test_B,
        y_train, y_valid, y_test
    ) = load_data()

    print(f"X_train_A: {X_train_A.shape}")
    print(f"X_valid_A: {X_valid_A.shape}")
    print(f"X_test_A : {X_test_A.shape}")
    print(f"X_train_B: {X_train_B.shape}")
    print(f"X_valid_B: {X_valid_B.shape}")
    print(f"X_test_B : {X_test_B.shape}")
    print(f"y_train  : {y_train.shape}")
    print(f"y_valid  : {y_valid.shape}")
    print(f"y_test   : {y_test.shape}")

    if len(X_train_A) != len(y_train) or len(X_train_B) != len(y_train):
        raise ValueError("Mismatch between training feature rows and y_train.")
    if len(X_valid_A) != len(y_valid) or len(X_valid_B) != len(y_valid):
        raise ValueError("Mismatch between validation feature rows and y_valid.")
    if len(X_test_A) != len(y_test) or len(X_test_B) != len(y_test):
        raise ValueError("Mismatch between test feature rows and y_test.")

    print("\n[2/7] Loading tuned best parameters from individual A and B experiments...")
    best_params_A = read_json_file(BEST_PARAMS_A_PATH)
    best_params_B = read_json_file(BEST_PARAMS_B_PATH)

    # Force consistent seed/bootstrap if not present
    best_params_A["bootstrap"] = best_params_A.get("bootstrap", True)
    best_params_A["random_state"] = best_params_A.get("random_state", RANDOM_SEED)

    best_params_B["bootstrap"] = best_params_B.get("bootstrap", True)
    best_params_B["random_state"] = best_params_B.get("random_state", RANDOM_SEED)

    print("Best A params:")
    print(best_params_A)
    print("Best B params:")
    print(best_params_B)

    print("\n[3/7] Training tuned A-view model...")
    model_A = build_model(best_params_A)
    model_A.fit(X_train_A, y_train)

    print("\n[4/7] Training tuned B-view model...")
    model_B = build_model(best_params_B)
    model_B.fit(X_train_B, y_train)

    print("\n[5/7] Generating component probabilities and soft-voting ensemble...")
    valid_prob_A = model_A.predict_proba(X_valid_A)[:, 1]
    test_prob_A = model_A.predict_proba(X_test_A)[:, 1]

    valid_prob_B = model_B.predict_proba(X_valid_B)[:, 1]
    test_prob_B = model_B.predict_proba(X_test_B)[:, 1]

    valid_prob_AB = (valid_prob_A + valid_prob_B) / 2.0
    test_prob_AB = (test_prob_A + test_prob_B) / 2.0

    valid_component_summary = {
        "A_valid_pr_auc": float(average_precision_score(y_valid, valid_prob_A)),
        "B_valid_pr_auc": float(average_precision_score(y_valid, valid_prob_B)),
        "AB_valid_pr_auc": float(average_precision_score(y_valid, valid_prob_AB)),
        "A_valid_roc_auc": float(roc_auc_score(y_valid, valid_prob_A)),
        "B_valid_roc_auc": float(roc_auc_score(y_valid, valid_prob_B)),
        "AB_valid_roc_auc": float(roc_auc_score(y_valid, valid_prob_AB))
    }

    test_component_summary = {
        "A_test_pr_auc": float(average_precision_score(y_test, test_prob_A)),
        "B_test_pr_auc": float(average_precision_score(y_test, test_prob_B)),
        "AB_test_pr_auc": float(average_precision_score(y_test, test_prob_AB)),
        "A_test_roc_auc": float(roc_auc_score(y_test, test_prob_A)),
        "B_test_roc_auc": float(roc_auc_score(y_test, test_prob_B)),
        "AB_test_roc_auc": float(roc_auc_score(y_test, test_prob_AB))
    }

    run_config = {
        "view_name": "AB",
        "model_name": "SoftVoting_Bagging_DT_AB",
        "ensemble_method": "simple_mean_probability",
        "threshold": DEFAULT_THRESHOLD,
        "A_params": best_params_A,
        "B_params": best_params_B
    }

    component_summary = {
        "valid_summary": valid_component_summary,
        "test_summary": test_component_summary
    }

    print("\n[6/7] Evaluating A+B multiview ensemble...")
    valid_row = evaluate_split("valid", y_valid, valid_prob_AB, run_config)
    test_row = evaluate_split("test", y_test, test_prob_AB, run_config)

    metrics_df = pd.DataFrame([valid_row, test_row])
    metrics_df.to_csv(FINAL_METRICS_CSV_PATH, index=False)

    with open(COMPONENT_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(component_summary, f, indent=2)

    print(f"[OK] Final metrics saved to: {FINAL_METRICS_CSV_PATH}")
    print(f"[OK] Component summary saved to: {COMPONENT_SUMMARY_JSON_PATH}")

    print("\n[7/7] Saving predictions, plots, and run config...")
    save_predictions_csv(
        VALID_PREDICTIONS_CSV_PATH,
        y_valid,
        valid_prob_AB,
        DEFAULT_THRESHOLD,
        valid_prob_A,
        valid_prob_B
    )
    save_predictions_csv(
        TEST_PREDICTIONS_CSV_PATH,
        y_test,
        test_prob_AB,
        DEFAULT_THRESHOLD,
        test_prob_A,
        test_prob_B
    )

    plot_pr_curve(
        y_valid,
        valid_prob_AB,
        "Precision-Recall Curve (VALID) - A+B Multiview Bagging",
        VALID_PR_PLOT_PATH
    )

    plot_pr_curve(
        y_test,
        test_prob_AB,
        "Precision-Recall Curve (TEST) - A+B Multiview Bagging",
        TEST_PR_PLOT_PATH
    )

    plot_metrics_bar_chart(
        metrics_df,
        METRICS_BAR_PLOT_PATH,
        "Validation vs Test Metrics - A+B Multiview Bagging"
    )

    plot_confusion_matrix_figure(
        y_valid,
        valid_prob_AB,
        DEFAULT_THRESHOLD,
        "Confusion Matrix (VALID) - A+B Multiview Bagging",
        VALID_CM_PLOT_PATH
    )

    plot_confusion_matrix_figure(
        y_test,
        test_prob_AB,
        DEFAULT_THRESHOLD,
        "Confusion Matrix (TEST) - A+B Multiview Bagging",
        TEST_CM_PLOT_PATH
    )

    with open(RUN_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[OK] VALID predictions saved to: {VALID_PREDICTIONS_CSV_PATH}")
    print(f"[OK] TEST predictions saved to: {TEST_PREDICTIONS_CSV_PATH}")
    print(f"[OK] VALID PR plot saved to: {VALID_PR_PLOT_PATH}")
    print(f"[OK] TEST PR plot saved to: {TEST_PR_PLOT_PATH}")
    print(f"[OK] Metrics bar chart saved to: {METRICS_BAR_PLOT_PATH}")
    print(f"[OK] VALID confusion matrix saved to: {VALID_CM_PLOT_PATH}")
    print(f"[OK] TEST confusion matrix saved to: {TEST_CM_PLOT_PATH}")
    print(f"[OK] Run config saved to: {RUN_CONFIG_JSON_PATH}")

    print("\nComponent summary:")
    print(json.dumps(component_summary, indent=2))

    print("\nFinal metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)