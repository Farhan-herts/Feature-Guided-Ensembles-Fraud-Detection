# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 20:04:16 2026

@author: FarhanAli
"""
# ============================================================
# 06_bagging_standard.py
# Standard Bagging (Baseline Ensemble)
# - Trains Bagging Classifier (base estimator: DecisionTreeClassifier)
# - Evaluates on PR-AUC, ROC-AUC, Recall@Precision
# - Saves results to results/metrics_bagging.csv
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score
)

import matplotlib.pyplot as plt


# ----------------------------
# Path Configuration
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset file paths
TRAIN_X = os.path.join(PROCESSED_DATA_PATH, "X_train_views.csv")
VALID_X = os.path.join(PROCESSED_DATA_PATH, "X_valid_views.csv")
TEST_X  = os.path.join(PROCESSED_DATA_PATH, "X_test_views.csv")

TRAIN_Y = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
VALID_Y = os.path.join(PROCESSED_DATA_PATH, "y_valid.csv")
TEST_Y  = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")

# Output paths for results
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "bagging_metrics.csv")
PR_CURVE_PLOT = os.path.join(RESULTS_DIR, "bagging_pr_curve_valid.png")


# ----------------------------
# Utility Functions
# ----------------------------
def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

def load_data():
    for path in [TRAIN_X, VALID_X, TEST_X, TRAIN_Y, VALID_Y, TEST_Y]:
        ensure_file_exists(path)

    X_train = pd.read_csv(TRAIN_X)
    X_valid = pd.read_csv(VALID_X)
    X_test  = pd.read_csv(TEST_X)

    y_train = pd.read_csv(TRAIN_Y).values.ravel().astype(int)
    y_valid = pd.read_csv(VALID_Y).values.ravel().astype(int)
    y_test  = pd.read_csv(TEST_Y).values.ravel().astype(int)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def calculate_basic_metrics(y_true, y_pred_prob):
    return {
        "pr_auc": float(average_precision_score(y_true, y_pred_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_prob))
    }

def evaluate_threshold(y_true, y_pred_prob, threshold: float):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

def select_threshold_for_precision(y_true, y_pred_prob, target_precision: float):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    
    precision = precision[:-1]
    recall = recall[:-1]

    valid_thresholds = precision >= target_precision
    if not np.any(valid_thresholds):
        return None, None, None

    best_idx = np.argmax(recall[valid_thresholds])
    chosen_threshold = thresholds[valid_thresholds][best_idx]
    return float(chosen_threshold), float(precision[valid_thresholds][best_idx]), float(recall[valid_thresholds][best_idx])


# ----------------------------
# Main Execution
# ----------------------------
def execute():
    create_directory(RESULTS_DIR)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    print("\nData Loaded:")
    print(f"X_train={X_train.shape}, X_valid={X_valid.shape}, X_test={X_test.shape}")

    # ----------------------------
    # Model: Bagging with Decision Tree
    # ----------------------------
    print("\n[1/1] Training Bagging Classifier (Decision Tree)...")

    base_tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    bagging_classifier = BaggingClassifier(
        estimator=base_tree,          
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    bagging_classifier.fit(X_train, y_train)

    valid_probs = bagging_classifier.predict_proba(X_valid)[:, 1]
    test_probs  = bagging_classifier.predict_proba(X_test)[:, 1]

    # Basic metrics
    valid_metrics = calculate_basic_metrics(y_valid, valid_probs)
    test_metrics  = calculate_basic_metrics(y_test, test_probs)

    result_rows = []

    # Default threshold 0.5
    valid_metrics_default = evaluate_threshold(y_valid, valid_probs, 0.5)
    test_metrics_default  = evaluate_threshold(y_test, test_probs, 0.5)

    result_rows.append({
        "model": "Bagging_DT",
        "threshold_type": "default_0.5",
        "precision_target": "",
        "threshold": 0.5,
        "valid_pr_auc": valid_metrics["pr_auc"],
        "valid_roc_auc": valid_metrics["roc_auc"],
        "valid_precision": valid_metrics_default["precision"],
        "valid_recall": valid_metrics_default["recall"],
        "valid_f1": valid_metrics_default["f1"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_precision": test_metrics_default["precision"],
        "test_recall": test_metrics_default["recall"],
        "test_f1": test_metrics_default["f1"],
    })

    # Precision-constrained thresholds (operational)
    for precision_target in [0.90, 0.95]:
        threshold, precision_val, recall_val = select_threshold_for_precision(y_valid, valid_probs, precision_target)

        if threshold is None:
            result_rows.append({
                "model": "Bagging_DT",
                "threshold_type": "precision_constrained",
                "precision_target": precision_target,
                "threshold": "",
                "valid_pr_auc": valid_metrics["pr_auc"],
                "valid_roc_auc": valid_metrics["roc_auc"],
                "valid_precision": "",
                "valid_recall": "",
                "valid_f1": "",
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_precision": "",
                "test_recall": "",
                "test_f1": "",
            })
            continue

        test_metrics_thr = evaluate_threshold(y_test, test_probs, threshold)
        result_rows.append({
            "model": "Bagging_DT",
            "threshold_type": "precision_constrained",
            "precision_target": precision_target,
            "threshold": threshold,
            "valid_pr_auc": valid_metrics["pr_auc"],
            "valid_roc_auc": valid_metrics["roc_auc"],
            "valid_precision": precision_val,
            "valid_recall": recall_val,
            "valid_f1": evaluate_threshold(y_valid, valid_probs, threshold)["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_precision": test_metrics_thr["precision"],
            "test_recall": test_metrics_thr["recall"],
            "test_f1": test_metrics_thr["f1"],
        })

    # Save results
    metrics_df = pd.DataFrame(result_rows)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"\n[OK] Metrics saved to: {OUTPUT_METRICS}")

    # PR curve plot on VALID
    precision, recall, _ = precision_recall_curve(y_valid, valid_probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (VALID) - Bagging (DT base)")
    plt.tight_layout()
    plt.savefig(PR_CURVE_PLOT, dpi=150)
    plt.close()

    print(f"[OK] PR curve plot saved to: {PR_CURVE_PLOT}")
   

if __name__ == "__main__":
    try:
        execute()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)