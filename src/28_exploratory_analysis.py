# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:47:28 2026

@author: FarhanAli
"""

# ============================================================
# 28_exploratory_analysis.py
#
# This script:
# 1. Loads raw transaction + identity data from data/raw/
# 2. Uses train_merged.csv from data/interim/ if available
#    (otherwise merges on the fly)
# 3. Creates exploratory plots:
#       - class distribution
#       - identity coverage after merging
#       - average missingness by feature group
#       - top missing features
#       - transaction amount distribution (log-scaled)
#       - hourly fraud-rate pattern
# 4. Saves:
#       - PNG plots
#       - summary CSV
#       - missingness CSVs
#       - summary JSON
#       - figure manifest CSV
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "28_exploratory_analysis")

TRAIN_TRANS_PATH = os.path.join(DATA_RAW, "train_transaction.csv")
TRAIN_ID_PATH = os.path.join(DATA_RAW, "train_identity.csv")
TRAIN_MERGED_PATH = os.path.join(DATA_INTERIM, "train_merged.csv")

# Output files
CLASS_DIST_PLOT_PATH = os.path.join(RESULTS_DIR, "class_distribution.png")
IDENTITY_COVERAGE_PLOT_PATH = os.path.join(RESULTS_DIR, "identity_coverage.png")
MISSING_GROUP_PLOT_PATH = os.path.join(RESULTS_DIR, "missingness_by_feature_group.png")
TOP_MISSING_PLOT_PATH = os.path.join(RESULTS_DIR, "top_missing_features.png")
AMOUNT_DIST_PLOT_PATH = os.path.join(RESULTS_DIR, "transaction_amount_log_distribution.png")
HOURLY_FRAUD_PLOT_PATH = os.path.join(RESULTS_DIR, "hourly_fraud_rate.png")

SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "dataset_summary.csv")
MISSING_GROUP_CSV_PATH = os.path.join(RESULTS_DIR, "missingness_by_group.csv")
TOP_MISSING_CSV_PATH = os.path.join(RESULTS_DIR, "top_missing_features.csv")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "summary.json")
FIGURE_MANIFEST_CSV_PATH = os.path.join(RESULTS_DIR, "figure_manifest.csv")


# ============================================================
# Utilities
# ============================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def assert_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def assign_feature_group(col_name: str) -> str:
    """
    Broad feature groups for missingness summary.
    """
    if col_name in ["TransactionID", "isFraud"]:
        return "Target / ID"

    if col_name in ["TransactionDT"]:
        return "Temporal"

    if col_name in ["TransactionAmt", "ProductCD"]:
        return "Core Transaction"

    if col_name.startswith("card"):
        return "Card"

    if col_name.startswith("addr"):
        return "Address"

    if col_name.startswith("dist"):
        return "Distance"

    if col_name.startswith("P_emaildomain") or col_name.startswith("R_emaildomain"):
        return "Email"

    if col_name.startswith("C"):
        return "Aggregation / Counts"

    if col_name.startswith("D"):
        return "Delay / Time-related"

    if col_name.startswith("M"):
        return "Matching / Behaviour"

    if col_name.startswith("V"):
        return "Anonymised Engineered"

    if col_name.startswith("id_"):
        return "Identity"

    if col_name.startswith("Device"):
        return "Device"

    return "Other"


def annotate_bars(ax, values, decimal_places: int = 0) -> None:
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        if decimal_places == 0:
            label = f"{int(v):,}"
        else:
            label = f"{v:.{decimal_places}f}"
        ax.text(i, v, label, ha="center", va="bottom", fontsize=8)


# ============================================================
# Data loading
# ============================================================
def load_data():
    assert_file_exists(TRAIN_TRANS_PATH)
    assert_file_exists(TRAIN_ID_PATH)

    print("Loading raw training files...")
    train_trans = pd.read_csv(TRAIN_TRANS_PATH, low_memory=False)
    train_id = pd.read_csv(TRAIN_ID_PATH, low_memory=False)

    if os.path.exists(TRAIN_MERGED_PATH):
        print("Loading existing merged file from data/interim/ ...")
        train_merged = pd.read_csv(TRAIN_MERGED_PATH, low_memory=False)
    else:
        print("Merged file not found. Creating merge on the fly...")
        train_merged = train_trans.merge(train_id, on="TransactionID", how="left")

    return train_trans, train_id, train_merged


# ============================================================
# Plotting
# ============================================================
def plot_class_distribution(train_trans: pd.DataFrame) -> dict:
    if "isFraud" not in train_trans.columns:
        raise ValueError("'isFraud' not found in train_transaction.csv")

    counts = train_trans["isFraud"].value_counts().sort_index()
    counts = counts.reindex([0, 1], fill_value=0)

    labels = ["Non-Fraud (0)", "Fraud (1)"]
    values = counts.values

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar(labels, values)
    ax.set_title("Class Distribution of Fraud and Non-Fraud Transactions")
    ax.set_ylabel("Number of Transactions")
    annotate_bars(ax, values, decimal_places=0)
    plt.tight_layout()
    plt.savefig(CLASS_DIST_PLOT_PATH, dpi=150)
    plt.close()

    fraud_rate = float(train_trans["isFraud"].mean())
    return {
        "non_fraud_count": int(counts.loc[0]),
        "fraud_count": int(counts.loc[1]),
        "fraud_rate": fraud_rate
    }


def plot_identity_coverage(train_merged: pd.DataFrame, train_id: pd.DataFrame) -> dict:
    identity_cols = [c for c in train_id.columns if c != "TransactionID"]

    if len(identity_cols) == 0:
        raise ValueError("No identity columns found in train_identity.csv")

    has_identity = train_merged[identity_cols].notna().any(axis=1)

    counts = has_identity.value_counts().reindex([False, True], fill_value=0)
    labels = ["No Identity Info", "Has Identity Info"]
    values = counts.values

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar(labels, values)
    ax.set_title("Coverage of Identity Information After Merging")
    ax.set_ylabel("Number of Transactions")
    annotate_bars(ax, values, decimal_places=0)
    plt.tight_layout()
    plt.savefig(IDENTITY_COVERAGE_PLOT_PATH, dpi=150)
    plt.close()
 
    return {
        "rows_without_identity_info": int(counts.loc[False]),
        "rows_with_identity_info": int(counts.loc[True]),
        "identity_coverage_rate": float(has_identity.mean())
    }


def plot_missingness_by_group(train_merged: pd.DataFrame) -> pd.DataFrame:
    missing_pct = train_merged.isna().mean() * 100.0

    missing_df = pd.DataFrame({
        "feature": missing_pct.index,
        "missing_pct": missing_pct.values
    })

    missing_df["feature_group"] = missing_df["feature"].apply(assign_feature_group)

    group_df = (
        missing_df.groupby("feature_group", as_index=False)["missing_pct"]
        .mean()
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(group_df["feature_group"], group_df["missing_pct"])
    ax.set_title("Average Missingness by Feature Group")
    ax.set_ylabel("Average Missing Values (%)")
    ax.set_xlabel("Feature Group")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(MISSING_GROUP_PLOT_PATH, dpi=150)
    plt.close()

    group_df.to_csv(MISSING_GROUP_CSV_PATH, index=False)
    return group_df


def plot_top_missing_features(train_merged: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    missing_pct = (train_merged.isna().mean() * 100.0).sort_values(ascending=False)
    top_missing = missing_pct.head(top_n)

    top_df = pd.DataFrame({
        "feature": top_missing.index,
        "missing_pct": top_missing.values,
        "dtype": train_merged[top_missing.index].dtypes.astype(str).values
    })

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.barh(top_df["feature"][::-1], top_df["missing_pct"][::-1])
    ax.set_title(f"Top {top_n} Features by Missingness")
    ax.set_xlabel("Missing Values (%)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(TOP_MISSING_PLOT_PATH, dpi=150)
    plt.close()

    top_df.to_csv(TOP_MISSING_CSV_PATH, index=False)
    return top_df


def plot_transaction_amount_distribution(train_trans: pd.DataFrame) -> dict:
    if "TransactionAmt" not in train_trans.columns:
        raise ValueError("'TransactionAmt' not found in train_transaction.csv")

    if "isFraud" not in train_trans.columns:
        raise ValueError("'isFraud' not found in train_transaction.csv")

    non_fraud_amt = train_trans.loc[train_trans["isFraud"] == 0, "TransactionAmt"].dropna()
    fraud_amt = train_trans.loc[train_trans["isFraud"] == 1, "TransactionAmt"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(non_fraud_amt), bins=70, alpha=0.7, label="Non-Fraud")
    plt.hist(np.log1p(fraud_amt), bins=70, alpha=0.7, label="Fraud")
    plt.title("Log-Scaled Transaction Amount Distribution")
    plt.xlabel("log(1 + TransactionAmt)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(AMOUNT_DIST_PLOT_PATH, dpi=150)
    plt.close()

    return {
        "transaction_amount_mean": float(train_trans["TransactionAmt"].mean()),
        "transaction_amount_median": float(train_trans["TransactionAmt"].median()),
        "transaction_amount_std": float(train_trans["TransactionAmt"].std(ddof=0))
    }


def plot_hourly_fraud_rate(train_trans: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in train_trans.columns:
        raise ValueError("'TransactionDT' not found in train_transaction.csv")

    if "isFraud" not in train_trans.columns:
        raise ValueError("'isFraud' not found in train_transaction.csv")

    temp = train_trans[["TransactionDT", "isFraud"]].copy()
    temp["DT_hour"] = ((pd.to_numeric(temp["TransactionDT"], errors="coerce").fillna(0) // 3600) % 24).astype(int)

    hourly_df = (
        temp.groupby("DT_hour", as_index=False)
        .agg(
            transaction_count=("isFraud", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean")
        )
        .sort_values("DT_hour")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(8, 5))
    plt.plot(hourly_df["DT_hour"], hourly_df["fraud_rate"])
    plt.title("Hourly Fraud Rate Pattern")
    plt.xlabel("Derived Hour from TransactionDT")
    plt.ylabel("Fraud Rate")
    plt.xticks(range(0, 24, 1))
    plt.tight_layout()
    plt.savefig(HOURLY_FRAUD_PLOT_PATH, dpi=150)
    plt.close()

    return hourly_df


# ============================================================
# Reporting outputs
# ============================================================
def build_summary_table(
    train_trans: pd.DataFrame,
    train_id: pd.DataFrame,
    train_merged: pd.DataFrame,
    class_summary: dict,
    identity_summary: dict,
    amount_summary: dict
) -> pd.DataFrame:
    rows = [
        {"metric": "train_transaction_rows", "value": int(train_trans.shape[0])},
        {"metric": "train_transaction_columns", "value": int(train_trans.shape[1])},
        {"metric": "train_identity_rows", "value": int(train_id.shape[0])},
        {"metric": "train_identity_columns", "value": int(train_id.shape[1])},
        {"metric": "train_merged_rows", "value": int(train_merged.shape[0])},
        {"metric": "train_merged_columns", "value": int(train_merged.shape[1])},
        {"metric": "train_transaction_memory_mb", "value": round(mem_mb(train_trans), 2)},
        {"metric": "train_identity_memory_mb", "value": round(mem_mb(train_id), 2)},
        {"metric": "train_merged_memory_mb", "value": round(mem_mb(train_merged), 2)},
        {"metric": "non_fraud_count", "value": class_summary["non_fraud_count"]},
        {"metric": "fraud_count", "value": class_summary["fraud_count"]},
        {"metric": "fraud_rate_percent", "value": round(class_summary["fraud_rate"] * 100.0, 6)},
        {"metric": "identity_rows_with_info", "value": identity_summary["rows_with_identity_info"]},
        {"metric": "identity_rows_without_info", "value": identity_summary["rows_without_identity_info"]},
        {"metric": "identity_coverage_percent", "value": round(identity_summary["identity_coverage_rate"] * 100.0, 6)},
        {"metric": "transaction_amount_mean", "value": round(amount_summary["transaction_amount_mean"], 6)},
        {"metric": "transaction_amount_median", "value": round(amount_summary["transaction_amount_median"], 6)},
        {"metric": "transaction_amount_std", "value": round(amount_summary["transaction_amount_std"], 6)},
    ]

    return pd.DataFrame(rows)


def save_figure_manifest() -> None:
    figure_rows = [
        {
            "figure_id": "Figure B1",
            "filename": os.path.basename(CLASS_DIST_PLOT_PATH),
            "suggested_caption": "Class distribution of fraud and non-fraud transactions in the IEEE-CIS training data."
        },
        {
            "figure_id": "Figure B2",
            "filename": os.path.basename(IDENTITY_COVERAGE_PLOT_PATH),
            "suggested_caption": "Coverage of identity-linked information after merging transaction and identity tables."
        },
        {
            "figure_id": "Figure B3",
            "filename": os.path.basename(MISSING_GROUP_PLOT_PATH),
            "suggested_caption": "Average missingness across major feature groups in the merged dataset."
        },
        {
            "figure_id": "Figure B4",
            "filename": os.path.basename(TOP_MISSING_PLOT_PATH),
            "suggested_caption": "Top features with the highest proportion of missing values in the merged dataset."
        },
        {
            "figure_id": "Figure B5",
            "filename": os.path.basename(AMOUNT_DIST_PLOT_PATH),
            "suggested_caption": "Log-scaled transaction amount distribution for fraud and non-fraud transactions."
        },
        {
            "figure_id": "Figure B6",
            "filename": os.path.basename(HOURLY_FRAUD_PLOT_PATH),
            "suggested_caption": "Hourly fraud-rate pattern derived from TransactionDT in the training data."
        },
    ]

    pd.DataFrame(figure_rows).to_csv(FIGURE_MANIFEST_CSV_PATH, index=False)


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=== Step 28: Exploratory Analysis ===")
    make_dir(RESULTS_DIR)

    print("\n[1/6] Loading data...")
    train_trans, train_id, train_merged = load_data()

    print(f"train_transaction shape: {train_trans.shape}")
    print(f"train_identity shape:    {train_id.shape}")
    print(f"train_merged shape:      {train_merged.shape}")

    print("\n[2/6] Creating exploratory plots...")
    class_summary = plot_class_distribution(train_trans)
    identity_summary = plot_identity_coverage(train_merged, train_id)
    group_missing_df = plot_missingness_by_group(train_merged)
    top_missing_df = plot_top_missing_features(train_merged, top_n=15)
    amount_summary = plot_transaction_amount_distribution(train_trans)
    hourly_df = plot_hourly_fraud_rate(train_trans)

    hourly_df.to_csv(os.path.join(RESULTS_DIR, "hourly_fraud_rate_table.csv"), index=False)

    print("\n[3/6] Saving summary table...")
    summary_df = build_summary_table(
        train_trans=train_trans,
        train_id=train_id,
        train_merged=train_merged,
        class_summary=class_summary,
        identity_summary=identity_summary,
        amount_summary=amount_summary
    )
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print("\n[4/6] Saving figure manifest...")
    save_figure_manifest()

    print("\n[5/6] Saving summary JSON...")
    summary_payload = {
        "dataset_files_used": {
            "train_transaction": TRAIN_TRANS_PATH,
            "train_identity": TRAIN_ID_PATH,
            "train_merged": TRAIN_MERGED_PATH if os.path.exists(TRAIN_MERGED_PATH) else "created_on_the_fly"
        },
        "dataset_shapes": {
            "train_transaction": list(train_trans.shape),
            "train_identity": list(train_id.shape),
            "train_merged": list(train_merged.shape)
        },
        "class_summary": class_summary,
        "identity_summary": identity_summary,
        "amount_summary": amount_summary,
        "missingness_by_group_top5": group_missing_df.head(5).to_dict(orient="records"),
        "top_missing_features_top10": top_missing_df.head(10).to_dict(orient="records")
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n[6/6] Done.")
    print(f"[OK] Results folder: {RESULTS_DIR}")
    print(f"[OK] Saved plot: {CLASS_DIST_PLOT_PATH}")
    print(f"[OK] Saved plot: {IDENTITY_COVERAGE_PLOT_PATH}")
    print(f"[OK] Saved plot: {MISSING_GROUP_PLOT_PATH}")
    print(f"[OK] Saved plot: {TOP_MISSING_PLOT_PATH}")
    print(f"[OK] Saved plot: {AMOUNT_DIST_PLOT_PATH}")
    print(f"[OK] Saved plot: {HOURLY_FRAUD_PLOT_PATH}")
    print(f"[OK] Saved summary CSV: {SUMMARY_CSV_PATH}")
    print(f"[OK] Saved summary JSON: {SUMMARY_JSON_PATH}")
    print(f"[OK] Saved figure manifest CSV: {FIGURE_MANIFEST_CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)