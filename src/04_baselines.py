# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:39:32 2026

@author: FarhanAli
"""

# ============================================================
# 04_baselines.py
# Baselines + metrics (PR-AUC + recall@precision)
# - Trains Logistic Regression + RandomForest
# - Selects threshold on VALID to meet precision targets (0.90, 0.95)
# - Evaluates on VALID and TEST
# - Saves results to results/metrics_baselines.csv
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score
)

import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

X_TRAIN_PATH = os.path.join(DATA_PROCESSED, "X_train.csv")
X_VALID_PATH = os.path.join(DATA_PROCESSED, "X_valid.csv")
X_TEST_PATH  = os.path.join(DATA_PROCESSED, "X_test.csv")

y_TRAIN_PATH = os.path.join(DATA_PROCESSED, "y_train.csv")
y_VALID_PATH = os.path.join(DATA_PROCESSED, "y_valid.csv")
y_TEST_PATH  = os.path.join(DATA_PROCESSED, "y_test.csv")

OUT_METRICS = os.path.join(RESULTS_DIR, "metrics_baselines.csv")
OUT_PRC_PLOT = os.path.join(RESULTS_DIR, "pr_curve_valid.png")

# ----------------------------
# Utils
# ----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

def load_xy():
    for p in [X_TRAIN_PATH, X_VALID_PATH, X_TEST_PATH, y_TRAIN_PATH, y_VALID_PATH, y_TEST_PATH]:
        require_file(p)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_valid = pd.read_csv(X_VALID_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(y_TRAIN_PATH).values.ravel().astype(int)
    y_valid = pd.read_csv(y_VALID_PATH).values.ravel().astype(int)
    y_test  = pd.read_csv(y_TEST_PATH).values.ravel().astype(int)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def choose_threshold_for_precision(y_true, y_prob, precision_target: float):
    """
    Choose the lowest threshold that achieves precision >= target,
    while maximizing recall among those thresholds.
    Uses precision-recall curve on VALID.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds length = len(precision)-1
    # Align arrays
    precision = precision[:-1]
    recall = recall[:-1]

    ok = precision >= precision_target
    if not np.any(ok):
        return None, None, None

    # among ok thresholds, pick one with max recall
    idx = np.argmax(recall[ok])
    chosen_threshold = thresholds[ok][idx]
    return chosen_threshold, precision[ok][idx], recall[ok][idx]

def eval_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

def compute_basic_metrics(y_true, y_prob):
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

# ----------------------------
# Main
# ----------------------------
def main():
   
    safe_mkdir(RESULTS_DIR)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_xy()

    print(f"\nLoaded:")
    print(f"X_train={X_train.shape}, X_valid={X_valid.shape}, X_test={X_test.shape}")

    results = []

    # ----------------------------
    # Model 1: Logistic Regression (balanced)
    # ----------------------------
    print("\n[1/2] Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)

    valid_prob = lr.predict_proba(X_valid)[:, 1]
    test_prob  = lr.predict_proba(X_test)[:, 1]

    # Basic metrics
    valid_basic = compute_basic_metrics(y_valid, valid_prob)
    test_basic  = compute_basic_metrics(y_test, test_prob)

    # Default threshold metrics (0.5)
    valid_default = eval_at_threshold(y_valid, valid_prob, 0.5)
    test_default  = eval_at_threshold(y_test, test_prob, 0.5)

    # Precision-constrained thresholds chosen on VALID
    for p_target in [0.90, 0.95]:
        thr, p_val, r_val = choose_threshold_for_precision(y_valid, valid_prob, p_target)

        if thr is None:
            # couldn't reach precision target
            results.append({
                "model": "LogReg_balanced",
                "precision_target": p_target,
                "threshold": None,
                "valid_pr_auc": valid_basic["pr_auc"],
                "valid_roc_auc": valid_basic["roc_auc"],
                "valid_precision@thr": None,
                "valid_recall@thr": None,
                "test_pr_auc": test_basic["pr_auc"],
                "test_roc_auc": test_basic["roc_auc"],
                "test_precision@thr": None,
                "test_recall@thr": None
            })
        else:
            # Evaluate same threshold on TEST
            test_thr_metrics = eval_at_threshold(y_test, test_prob, thr)

            results.append({
                "model": "LogReg_balanced",
                "precision_target": p_target,
                "threshold": float(thr),
                "valid_pr_auc": valid_basic["pr_auc"],
                "valid_roc_auc": valid_basic["roc_auc"],
                "valid_precision@thr": float(p_val),
                "valid_recall@thr": float(r_val),
                "test_pr_auc": test_basic["pr_auc"],
                "test_roc_auc": test_basic["roc_auc"],
                "test_precision@thr": test_thr_metrics["precision"],
                "test_recall@thr": test_thr_metrics["recall"]
            })

    # Add default threshold row too (useful for IPR)
    results.append({
        "model": "LogReg_balanced",
        "precision_target": "default_0.5",
        "threshold": 0.5,
        "valid_pr_auc": valid_basic["pr_auc"],
        "valid_roc_auc": valid_basic["roc_auc"],
        "valid_precision@thr": valid_default["precision"],
        "valid_recall@thr": valid_default["recall"],
        "test_pr_auc": test_basic["pr_auc"],
        "test_roc_auc": test_basic["roc_auc"],
        "test_precision@thr": test_default["precision"],
        "test_recall@thr": test_default["recall"]
    })

    # ----------------------------
    # Model 2: Random Forest (balanced_subsample)
    # ----------------------------
    print("\n[2/2] Training Random Forest (this may take a few minutes)...")
    rf = RandomForestClassifier(
        n_estimators=150, # 300 is good one but due to slow PC changed to 150
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=12, # none but changed to 12 due to slow PC
        min_samples_leaf=1
    )
    rf.fit(X_train, y_train)

    valid_prob = rf.predict_proba(X_valid)[:, 1]
    test_prob  = rf.predict_proba(X_test)[:, 1]

    valid_basic = compute_basic_metrics(y_valid, valid_prob)
    test_basic  = compute_basic_metrics(y_test, test_prob)

    valid_default = eval_at_threshold(y_valid, valid_prob, 0.5)
    test_default  = eval_at_threshold(y_test, test_prob, 0.5)

    for p_target in [0.90, 0.95]:
        thr, p_val, r_val = choose_threshold_for_precision(y_valid, valid_prob, p_target)

        if thr is None:
            results.append({
                "model": "RF_balanced_subsample",
                "precision_target": p_target,
                "threshold": None,
                "valid_pr_auc": valid_basic["pr_auc"],
                "valid_roc_auc": valid_basic["roc_auc"],
                "valid_precision@thr": None,
                "valid_recall@thr": None,
                "test_pr_auc": test_basic["pr_auc"],
                "test_roc_auc": test_basic["roc_auc"],
                "test_precision@thr": None,
                "test_recall@thr": None
            })
        else:
            test_thr_metrics = eval_at_threshold(y_test, test_prob, thr)

            results.append({
                "model": "RF_balanced_subsample",
                "precision_target": p_target,
                "threshold": float(thr),
                "valid_pr_auc": valid_basic["pr_auc"],
                "valid_roc_auc": valid_basic["roc_auc"],
                "valid_precision@thr": float(p_val),
                "valid_recall@thr": float(r_val),
                "test_pr_auc": test_basic["pr_auc"],
                "test_roc_auc": test_basic["roc_auc"],
                "test_precision@thr": test_thr_metrics["precision"],
                "test_recall@thr": test_thr_metrics["recall"]
            })

    results.append({
        "model": "RF_balanced_subsample",
        "precision_target": "default_0.5",
        "threshold": 0.5,
        "valid_pr_auc": valid_basic["pr_auc"],
        "valid_roc_auc": valid_basic["roc_auc"],
        "valid_precision@thr": valid_default["precision"],
        "valid_recall@thr": valid_default["recall"],
        "test_pr_auc": test_basic["pr_auc"],
        "test_roc_auc": test_basic["roc_auc"],
        "test_precision@thr": test_default["precision"],
        "test_recall@thr": test_default["recall"]
    })

    # ----------------------------
    # Save results
    # ----------------------------
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"\n[OK] Saved metrics to: {OUT_METRICS}")

    # Plot PR curve (VALID) for the last model trained (RF) as evidence
    precision, recall, _ = precision_recall_curve(y_valid, rf.predict_proba(X_valid)[:, 1])

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (VALID) — Random Forest")
    plt.tight_layout()
    plt.savefig(OUT_PRC_PLOT, dpi=150)
    plt.close()

    print(f"[OK] Saved PR curve plot to: {OUT_PRC_PLOT}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
