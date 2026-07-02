# 21_it_best_balanced.py

# This script:
# 1. Loads the prepared Identity + Temporal Multiview [I+T] data (train/valid/test)
# 2. Loads the best tuned model parameters from the earlier tuned runs
# 3. Loads the selected best balancing method from Step 18
# 4. Applies the selected balancing method on TRAIN only for each component view
# 5. Trains the final multiview bagging model
# 6. Combines probabilities using soft voting (mean probability)
# 7. Evaluates on VALID and TEST
# 8. Saves:
#    - final metrics CSV
#    - validation predictions CSV
#    - test predictions CSV
#    - PR-curve plots
#    - metrics bar chart
#    - confusion matrix figures
#    - selected-method summary JSON
#    - component summary JSON
#    - resampling distribution CSV
#    - run configuration JSON

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 2026

@author: FarhanAli
"""

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
# Experiment settings
# ============================================================
RANDOM_SEED = 42

# Selected settings from Step 17
DEFAULT_THRESHOLD = 0.25
SUB_SAMPLING_STRATEGY = 0.10
SMOTE_SAMPLING_STRATEGY = 0.20
COMBINED_UNDER_STRATEGY = 0.05
COMBINED_SMOTE_STRATEGY = 0.20
SMOTE_K_NEIGHBORS = 3


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


def load_selected_best_method(path: str) -> str:
    """
    Read selected balancing method from Step 18 JSON.
    """
    payload = read_json_file(path)

    if "best_method" in payload:
        return str(payload["best_method"])

    if "method" in payload:
        return str(payload["method"])

    if "top_valid_method" in payload and isinstance(payload["top_valid_method"], dict):
        if "method" in payload["top_valid_method"]:
            return str(payload["top_valid_method"]["method"])

    raise ValueError(
        "Could not find selected method in best_balance_method.json. "
        "Expected 'best_method' or another supported structure."
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
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }


def compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
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


def save_predictions_csv(path: str, y_true: np.ndarray, payload: dict, threshold: float) -> None:
    """
    Save row-level probabilities and predictions.
    payload keys should be component probability columns plus 'ensemble_prob_col' and 'ensemble_prob'
    """
    out_df = pd.DataFrame({
        "row_id": np.arange(len(y_true)),
        "y_true": y_true
    })

    for col_name, values in payload["component_probs"].items():
        out_df[col_name] = values

    out_df[payload["ensemble_prob_col"]] = payload["ensemble_prob"]
    out_df["y_pred"] = (payload["ensemble_prob"] >= threshold).astype(int)
    out_df.to_csv(path, index=False)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, output_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metrics_bar_chart(metrics_df: pd.DataFrame, output_path: str, title: str) -> None:
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]
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
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "21_it_best_balanced")
STEP18_DIR = os.path.join(PROJECT_ROOT, "results", "18_select_best_balance_method")

TRAIN_X_I_PATH = os.path.join(PROCESSED_DIR, "X_train_view_I.csv")
VALID_X_I_PATH = os.path.join(PROCESSED_DIR, "X_valid_view_I.csv")
TEST_X_I_PATH = os.path.join(PROCESSED_DIR, "X_test_view_I.csv")

TRAIN_X_T_PATH = os.path.join(PROCESSED_DIR, "X_train_view_T.csv")
VALID_X_T_PATH = os.path.join(PROCESSED_DIR, "X_valid_view_T.csv")
TEST_X_T_PATH = os.path.join(PROCESSED_DIR, "X_test_view_T.csv")

TRAIN_Y_PATH = os.path.join(PROCESSED_DIR, "y_train.csv")
VALID_Y_PATH = os.path.join(PROCESSED_DIR, "y_valid.csv")
TEST_Y_PATH = os.path.join(PROCESSED_DIR, "y_test.csv")

BEST_PARAMS_I_PATH = os.path.join(PROJECT_ROOT, "results", "09d_individual_I_tuned", "best_params_I.json")
BEST_PARAMS_T_PATH = os.path.join(PROJECT_ROOT, "results", "09c_individual_T_tuned", "best_params_T.json")
BEST_BALANCE_JSON_PATH = os.path.join(STEP18_DIR, "best_balance_method.json")

FINAL_METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics_IT_best_balanced.csv")
VALID_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "valid_predictions_IT_best_balanced.csv")
TEST_PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "test_predictions_IT_best_balanced.csv")

VALID_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_valid_IT_best_balanced.png")
TEST_PR_PLOT_PATH = os.path.join(RESULTS_DIR, "pr_curve_test_IT_best_balanced.png")

METRICS_BAR_PLOT_PATH = os.path.join(RESULTS_DIR, "metrics_bar_chart_IT_best_balanced.png")
VALID_CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_valid_IT_best_balanced.png")
TEST_CM_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_test_IT_best_balanced.png")

SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "summary_IT_best_balanced.json")
COMPONENT_SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "component_summary_IT_best_balanced.json")
RESAMPLING_DISTRIBUTION_CSV_PATH = os.path.join(RESULTS_DIR, "resampling_class_distribution_IT_best_balanced.csv")
RUN_CONFIG_JSON_PATH = os.path.join(RESULTS_DIR, "run_config_IT_best_balanced.json")


def load_data():
    X_train_I = read_feature_matrix(TRAIN_X_I_PATH)
    X_valid_I = read_feature_matrix(VALID_X_I_PATH)
    X_test_I = read_feature_matrix(TEST_X_I_PATH)
    X_train_T = read_feature_matrix(TRAIN_X_T_PATH)
    X_valid_T = read_feature_matrix(VALID_X_T_PATH)
    X_test_T = read_feature_matrix(TEST_X_T_PATH)
    y_train = read_binary_target(TRAIN_Y_PATH)
    y_valid = read_binary_target(VALID_Y_PATH)
    y_test = read_binary_target(TEST_Y_PATH)

    return (
        X_train_I,
        X_valid_I,
        X_test_I,
        X_train_T,
        X_valid_T,
        X_test_T,
        y_train,
        y_valid,
        y_test
    )


def evaluate_one_split(
    method_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    run_config: dict
) -> dict:
    """
    Evaluate one method on one split.
    """
    prob_metrics = compute_probability_metrics(y_true, y_prob)
    cls_metrics = compute_threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)

    row = {
        "view": "IT",
        "method": method_name,
        "model": "SoftVoting_Bagging_DT_IT_best_balanced",
        "split": split_name,
        "threshold": DEFAULT_THRESHOLD,
        "combination_type": "multiview_soft_voting",
        "component_views": "I+T",
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
        "positive_rate_predicted": float(cls_metrics["positive_predictions"] / len(y_true)),
        "I_max_depth": run_config["I_params"]["max_depth"],
        "I_min_samples_split": run_config["I_params"]["min_samples_split"],
        "I_min_samples_leaf": run_config["I_params"]["min_samples_leaf"],
        "I_n_estimators": run_config["I_params"]["n_estimators"],
        "I_max_samples": run_config["I_params"]["max_samples"],
        "T_max_depth": run_config["T_params"]["max_depth"],
        "T_min_samples_split": run_config["T_params"]["min_samples_split"],
        "T_min_samples_leaf": run_config["T_params"]["min_samples_leaf"],
        "T_n_estimators": run_config["T_params"]["n_estimators"],
        "T_max_samples": run_config["T_params"]["max_samples"],
    }

    return row


def build_summary(metrics_df: pd.DataFrame, distribution_df: pd.DataFrame, selected_method: str) -> dict:
    valid_df = metrics_df[metrics_df["split"] == "valid"].copy()
    test_df = metrics_df[metrics_df["split"] == "test"].copy()

    summary = {
        "selected_best_method_from_step18": selected_method,
        "view_name": "IT",
        "component_views": "I+T",
        "valid_metrics": {},
        "test_metrics": {},
        "resampled_train_distributions": distribution_df.to_dict(orient="records")
    }

    if not valid_df.empty:
        row = valid_df.iloc[0]
        summary["valid_metrics"] = {
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "tp": int(row["tp"]),
            "tn": int(row["tn"]),
            "fpr": float(row["fpr"]),
            "fnr": float(row["fnr"])
        }

    if not test_df.empty:
        row = test_df.iloc[0]
        summary["test_metrics"] = {
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
            "accuracy": float(row["accuracy"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "tp": int(row["tp"]),
            "tn": int(row["tn"]),
            "fpr": float(row["fpr"]),
            "fnr": float(row["fnr"])
        }

    return summary


def main() -> None:
    print("=== Step 21: IT Multiview with Selected Best Balance Method ===")
    make_dir(RESULTS_DIR)

    print("\n[1/8] Loading data...")
    (
        X_train_I, X_valid_I, X_test_I, X_train_T, X_valid_T, X_test_T,
        y_train, y_valid, y_test
    ) = load_data()

    print(f"X_train_I: {X_train_I.shape}")
    print(f"X_valid_I: {X_valid_I.shape}")
    print(f"X_test_I : {X_test_I.shape}")
    print(f"X_train_T: {X_train_T.shape}")
    print(f"X_valid_T: {X_valid_T.shape}")
    print(f"X_test_T : {X_test_T.shape}")
    print(f"y_train  : {y_train.shape}")
    print(f"y_valid  : {y_valid.shape}")
    print(f"y_test   : {y_test.shape}")

    if len(X_train_I) != len(y_train) or len(X_train_T) != len(y_train):
        raise ValueError("Mismatch between training feature rows and y_train.")
    if len(X_valid_I) != len(y_valid) or len(X_valid_T) != len(y_valid):
        raise ValueError("Mismatch between validation feature rows and y_valid.")
    if len(X_test_I) != len(y_test) or len(X_test_T) != len(y_test):
        raise ValueError("Mismatch between test feature rows and y_test.")

    print("\n[2/8] Loading best tuned parameters and selected best method...")
    best_params_I = read_json_file(BEST_PARAMS_I_PATH)
    best_params_I["bootstrap"] = best_params_I.get("bootstrap", True)
    best_params_I["random_state"] = best_params_I.get("random_state", RANDOM_SEED)
    best_params_T = read_json_file(BEST_PARAMS_T_PATH)
    best_params_T["bootstrap"] = best_params_T.get("bootstrap", True)
    best_params_T["random_state"] = best_params_T.get("random_state", RANDOM_SEED)
    selected_method = load_selected_best_method(BEST_BALANCE_JSON_PATH)

    print("Best I params:")
    print(best_params_I)
    print("Best T params:")
    print(best_params_T)
    print(f"Selected best balance method: {selected_method}")

    with open(RUN_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "view_name": "IT",
            "base_model": "SoftVoting_Bagging_DT_IT_best_balanced",
            "threshold": DEFAULT_THRESHOLD,
            "ensemble_method": "simple_mean_probability",
            "component_views": "I+T",
            "selected_best_method_from_step18": selected_method,
            "best_params_from_prior_tuned_run": {
                "I_params": best_params_I,
                "T_params": best_params_T,
            },
            "balancing_settings": {
                "sub_sampling_strategy": SUB_SAMPLING_STRATEGY,
                "smote_sampling_strategy": SMOTE_SAMPLING_STRATEGY,
                "combined_under_strategy": COMBINED_UNDER_STRATEGY,
                "combined_smote_strategy": COMBINED_SMOTE_STRATEGY,
                "smote_k_neighbors": SMOTE_K_NEIGHBORS
            }
        }, f, indent=2)

    print("\n[3/8] Applying selected balancing method on TRAIN only and training component models...")
    original_dist = class_distribution_dict(y_train, "original_train")
    distribution_rows = [
        {"method": "original_train", **original_dist}
    ]

    X_res_I, y_res_I = apply_balancing_method(selected_method, X_train_I, y_train)
    dist_I = class_distribution_dict(y_res_I, f"{selected_method}_resampled_train_I")
    distribution_rows.append({"method": f"{selected_method}_I", **dist_I})

    print(
        f"I-resampled train shape: {X_res_I.shape}, "
        f"class_0={dist_I['count_class_0']}, "
        f"class_1={dist_I['count_class_1']}, "
        f"minority/majority={dist_I['minority_over_majority']:.4f}"
    )

    model_I = build_model(best_params_I)
    model_I.fit(X_res_I, y_res_I)
    X_res_T, y_res_T = apply_balancing_method(selected_method, X_train_T, y_train)
    dist_T = class_distribution_dict(y_res_T, f"{selected_method}_resampled_train_T")
    distribution_rows.append({"method": f"{selected_method}_T", **dist_T})

    print(
        f"T-resampled train shape: {X_res_T.shape}, "
        f"class_0={dist_T['count_class_0']}, "
        f"class_1={dist_T['count_class_1']}, "
        f"minority/majority={dist_T['minority_over_majority']:.4f}"
    )

    model_T = build_model(best_params_T)
    model_T.fit(X_res_T, y_res_T)
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(RESAMPLING_DISTRIBUTION_CSV_PATH, index=False)

    print("\n[4/8] Generating component probabilities and soft-voting ensemble...")
    valid_prob_I = model_I.predict_proba(X_valid_I)[:, 1]
    valid_prob_T = model_T.predict_proba(X_valid_T)[:, 1]
    test_prob_I = model_I.predict_proba(X_test_I)[:, 1]
    test_prob_T = model_T.predict_proba(X_test_T)[:, 1]

    valid_prob_ensemble = (valid_prob_I + valid_prob_T) / 2.0
    test_prob_ensemble = (test_prob_I + test_prob_T) / 2.0

    valid_component_summary = {
        "I_valid_pr_auc": float(average_precision_score(y_valid, valid_prob_I)),
        "I_valid_roc_auc": float(roc_auc_score(y_valid, valid_prob_I)),
        "T_valid_pr_auc": float(average_precision_score(y_valid, valid_prob_T)),
        "T_valid_roc_auc": float(roc_auc_score(y_valid, valid_prob_T)),
        "IT_valid_pr_auc": float(average_precision_score(y_valid, valid_prob_ensemble)),
        "IT_valid_roc_auc": float(roc_auc_score(y_valid, valid_prob_ensemble))
    }

    test_component_summary = {
        "I_test_pr_auc": float(average_precision_score(y_test, test_prob_I)),
        "I_test_roc_auc": float(roc_auc_score(y_test, test_prob_I)),
        "T_test_pr_auc": float(average_precision_score(y_test, test_prob_T)),
        "T_test_roc_auc": float(roc_auc_score(y_test, test_prob_T)),
        "IT_test_pr_auc": float(average_precision_score(y_test, test_prob_ensemble)),
        "IT_test_roc_auc": float(roc_auc_score(y_test, test_prob_ensemble))
    }

    with open(COMPONENT_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "selected_best_method_from_step18": selected_method,
            "valid_summary": valid_component_summary,
            "test_summary": test_component_summary
        }, f, indent=2)

    print("\n[5/8] Evaluating ensemble...")
    run_config = {
        "I_params": best_params_I,
        "T_params": best_params_T,
    }

    valid_row = evaluate_one_split(selected_method, "valid", y_valid, valid_prob_ensemble, run_config)
    test_row = evaluate_one_split(selected_method, "test", y_test, test_prob_ensemble, run_config)

    metrics_df = pd.DataFrame([valid_row, test_row])
    metrics_df.to_csv(FINAL_METRICS_CSV_PATH, index=False)

    print(f"[OK] Final metrics saved to: {FINAL_METRICS_CSV_PATH}")

    save_predictions_csv(
        VALID_PREDICTIONS_CSV_PATH,
        y_valid,
        {
            "component_probs": {
                "y_prob_I": valid_prob_I,
                "y_prob_T": valid_prob_T,
            },
            "ensemble_prob_col": "y_prob_IT_ensemble",
            "ensemble_prob": valid_prob_ensemble
        },
        DEFAULT_THRESHOLD
    )
    save_predictions_csv(
        TEST_PREDICTIONS_CSV_PATH,
        y_test,
        {
            "component_probs": {
                "y_prob_I": test_prob_I,
                "y_prob_T": test_prob_T,
            },
            "ensemble_prob_col": "y_prob_IT_ensemble",
            "ensemble_prob": test_prob_ensemble
        },
        DEFAULT_THRESHOLD
    )

    print(f"[OK] VALID predictions saved to: {VALID_PREDICTIONS_CSV_PATH}")
    print(f"[OK] TEST predictions saved to: {TEST_PREDICTIONS_CSV_PATH}")

    print("\n[6/8] Saving PR curves...")
    plot_pr_curve(
        y_valid,
        valid_prob_ensemble,
        "Precision-Recall Curve (VALID) - IT Best Balanced [" + selected_method + "]",
        VALID_PR_PLOT_PATH
    )
    plot_pr_curve(
        y_test,
        test_prob_ensemble,
        "Precision-Recall Curve (TEST) - IT Best Balanced [" + selected_method + "]",
        TEST_PR_PLOT_PATH
    )

    print(f"[OK] VALID PR plot saved to: {VALID_PR_PLOT_PATH}")
    print(f"[OK] TEST PR plot saved to: {TEST_PR_PLOT_PATH}")

    print("\n[7/8] Saving plots...")
    plot_metrics_bar_chart(
        metrics_df,
        METRICS_BAR_PLOT_PATH,
        "Validation vs Test Metrics - IT Best Balanced [" + selected_method + "]"
    )

    plot_confusion_matrix_figure(
        y_valid,
        valid_prob_ensemble,
        DEFAULT_THRESHOLD,
        "Confusion Matrix (VALID) - IT Best Balanced [" + selected_method + "]",
        VALID_CM_PLOT_PATH
    )
    plot_confusion_matrix_figure(
        y_test,
        test_prob_ensemble,
        DEFAULT_THRESHOLD,
        "Confusion Matrix (TEST) - IT Best Balanced [" + selected_method + "]",
        TEST_CM_PLOT_PATH
    )

    print(f"[OK] Metrics bar chart saved to: {METRICS_BAR_PLOT_PATH}")
    print(f"[OK] VALID confusion matrix saved to: {VALID_CM_PLOT_PATH}")
    print(f"[OK] TEST confusion matrix saved to: {TEST_CM_PLOT_PATH}")
    print(f"[OK] Resampling distribution CSV saved to: {RESAMPLING_DISTRIBUTION_CSV_PATH}")
    print(f"[OK] Component summary JSON saved to: {COMPONENT_SUMMARY_JSON_PATH}")

    print("\n[8/8] Saving summary JSON...")
    summary = build_summary(metrics_df, distribution_df, selected_method)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Summary JSON saved to: {SUMMARY_JSON_PATH}")

    print("\nFinal metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[ERROR] {type(err).__name__}: {err}")
        sys.exit(1)
