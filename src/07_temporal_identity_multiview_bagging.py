# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 11:21:58 2026

@author: FarhanAli
"""


# ============================================================
# 
# Multi-View Bagging Ensemble for fraud detection
# Trains separate Bagging classifiers for Temporal and Identity features
# and combines them using Soft Voting.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score
)

import matplotlib.pyplot as plt


# ----------------------------
# Define file paths
# ----------------------------
PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIRECTORY, "data", "processed")
RESULTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "results")

TRAIN_X_PATH = os.path.join(PROCESSED_DATA_PATH, "X_train_views.csv")
VALID_X_PATH = os.path.join(PROCESSED_DATA_PATH, "X_valid_views.csv")
TEST_X_PATH = os.path.join(PROCESSED_DATA_PATH, "X_test_views.csv")

TRAIN_Y_PATH = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
VALID_Y_PATH = os.path.join(PROCESSED_DATA_PATH, "y_valid.csv")
TEST_Y_PATH = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")

METRICS_OUTPUT_PATH = os.path.join(RESULTS_DIRECTORY, "metrics_multiview_bagging.csv")
PR_CURVE_OUTPUT_PATH = os.path.join(RESULTS_DIRECTORY, "pr_curve_multiview_bagging_valid.png")


# ----------------------------
# Helper Functions
# ----------------------------

def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

def load_datasets():
    for file_path in [TRAIN_X_PATH, VALID_X_PATH, TEST_X_PATH, TRAIN_Y_PATH, VALID_Y_PATH, TEST_Y_PATH]:
        check_file_exists(file_path)

    X_train = pd.read_csv(TRAIN_X_PATH)
    X_valid = pd.read_csv(VALID_X_PATH)
    X_test = pd.read_csv(TEST_X_PATH)

    y_train = pd.read_csv(TRAIN_Y_PATH).values.ravel().astype(int)
    y_valid = pd.read_csv(VALID_Y_PATH).values.ravel().astype(int)
    y_test = pd.read_csv(TEST_Y_PATH).values.ravel().astype(int)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def calculate_metrics(y_true, y_prob):
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

def evaluate_metrics_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

def find_threshold_for_precision(y_true, y_prob, target_precision: float):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]

    valid_thresholds = precision >= target_precision
    if not np.any(valid_thresholds):
        return None, None, None

    best_idx = np.argmax(recall[valid_thresholds])
    chosen_threshold = thresholds[valid_thresholds][best_idx]
    return float(chosen_threshold), float(precision[valid_thresholds][best_idx]), float(recall[valid_thresholds][best_idx])


# ----------------------------
# Main Process
# ----------------------------

def run_model_training():
    ensure_directory_exists(RESULTS_DIRECTORY)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_datasets()

    print("\nData Loaded:")
    print(f"Training data: X_train={X_train.shape}, X_valid={X_valid.shape}, X_test={X_test.shape}")

    # ----------------------------
    # Temporal Features Classifier (Bagging)
    # ----------------------------
    print("\n[1/3] Training Bagging Classifier for Temporal Features...")

    base_tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    temporal_bagging_model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    X_train_temp = X_train.filter(like="DT_")  # Temporal features
    X_valid_temp = X_valid.filter(like="DT_")
    X_test_temp = X_test.filter(like="DT_")

    temporal_bagging_model.fit(X_train_temp, y_train)

    # ----------------------------
    # Identity Features Classifier (Bagging)
    # ----------------------------
    print("\n[2/3] Training Bagging Classifier for Identity Features...")

    identity_bagging_model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    X_train_id = X_train.filter(like="Device")  # Identity features (DeviceType, DeviceInfo)
    X_valid_id = X_valid.filter(like="Device")
    X_test_id = X_test.filter(like="Device")

    identity_bagging_model.fit(X_train_id, y_train)

    # ----------------------------
    # Soft Voting Ensemble Model
    # ----------------------------
    print("\n[3/3] Combining models using Soft Voting...")

    ensemble_classifier = VotingClassifier(
        estimators=[
            ('temporal', temporal_bagging_model),
            ('identity', identity_bagging_model)
        ],
        voting='soft',
        n_jobs=-1
    )

    ensemble_classifier.fit(X_train, y_train)

    valid_prob = ensemble_classifier.predict_proba(X_valid)[:, 1]
    test_prob = ensemble_classifier.predict_proba(X_test)[:, 1]

    # Basic metrics
    valid_metrics = calculate_metrics(y_valid, valid_prob)
    test_metrics = calculate_metrics(y_test, test_prob)

    results = []

    # Default threshold (0.5)
    valid_default = evaluate_metrics_at_threshold(y_valid, valid_prob, 0.5)
    test_default = evaluate_metrics_at_threshold(y_test, test_prob, 0.5)

    results.append({
        "model": "MultiViewBagging",
        "threshold_type": "default_0.5",
        "precision_target": "",
        "threshold": 0.5,
        "valid_pr_auc": valid_metrics["pr_auc"],
        "valid_roc_auc": valid_metrics["roc_auc"],
        "valid_precision": valid_default["precision"],
        "valid_recall": valid_default["recall"],
        "valid_f1": valid_default["f1"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_precision": test_default["precision"],
        "test_recall": test_default["recall"],
        "test_f1": test_default["f1"],
    })

    # Precision-constrained thresholds (operational)
    for precision_target in [0.90, 0.95]:
        threshold, precision_val, recall_val = find_threshold_for_precision(y_valid, valid_prob, precision_target)

        if threshold is None:
            results.append({
                "model": "MultiViewBagging",
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

        test_threshold_metrics = evaluate_metrics_at_threshold(y_test, test_prob, threshold)
        results.append({
            "model": "MultiViewBagging",
            "threshold_type": "precision_constrained",
            "precision_target": precision_target,
            "threshold": threshold,
            "valid_pr_auc": valid_metrics["pr_auc"],
            "valid_roc_auc": valid_metrics["roc_auc"],
            "valid_precision": precision_val,
            "valid_recall": recall_val,
            "valid_f1": evaluate_metrics_at_threshold(y_valid, valid_prob, threshold)["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_precision": test_threshold_metrics["precision"],
            "test_recall": test_threshold_metrics["recall"],
            "test_f1": test_threshold_metrics["f1"],
        })

    # Save the results
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
    print(f"\n[OK] Saved metrics to: {METRICS_OUTPUT_PATH}")

    # Plot PR curve on VALID
    precision, recall, _ = precision_recall_curve(y_valid, valid_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (VALID) — Multi-View Bagging")
    plt.tight_layout()
    plt.savefig(PR_CURVE_OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"[OK] Saved PR curve plot to: {PR_CURVE_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        run_model_training()
    except Exception as error:
        print(f"\n[ERROR] {type(error).__name__}: {error}")
        sys.exit(1)