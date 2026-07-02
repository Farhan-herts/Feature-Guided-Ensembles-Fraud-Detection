# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 23:42:44 2026

@author: FarhanAli
"""

# ============================================================
# 03_preprocess.py
#   Preprocessing + leakage-safe encoding
# - Loads train_merged.csv and saved split indices
# - Splits into Train/Valid/Test using indices
# - Identifies numeric/categorical columns
# - Missing value handling:
#     numeric -> median (fit on train only)
#     categorical -> "__MISSING__"
# - Frequency encoding for categorical features (fit on train only)
# - Saves processed datasets for modeling
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "results", "splits")

TRAIN_MERGED_PATH = os.path.join(DATA_INTERIM, "train_merged.csv")

TRAIN_IDX_PATH = os.path.join(SPLITS_DIR, "train_idx.npy")
VALID_IDX_PATH = os.path.join(SPLITS_DIR, "valid_idx.npy")
TEST_IDX_PATH  = os.path.join(SPLITS_DIR, "test_idx.npy")

# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

def frequency_encode(train_series: pd.Series, other_series: pd.Series) -> pd.Series:
    """
    Leakage-safe frequency encoding:
    - Compute value counts from TRAIN only
    - Map to other split
    """
    vc = train_series.value_counts(dropna=False)
    return other_series.map(vc).fillna(0).astype(np.float32)

# ----------------------------
# Main
# ----------------------------
def main():

    # Check files
    for p in [TRAIN_MERGED_PATH, TRAIN_IDX_PATH, VALID_IDX_PATH, TEST_IDX_PATH]:
        require_file(p)

    safe_mkdir(DATA_PROCESSED)

    # Load merged data
    print("\nLoading merged train data...")
    df = pd.read_csv(TRAIN_MERGED_PATH, low_memory=False)

    # Load splits
    idx_train = np.load(TRAIN_IDX_PATH)
    idx_valid = np.load(VALID_IDX_PATH)
    idx_test  = np.load(TEST_IDX_PATH)

    # Target
    y = df["isFraud"].astype(int)

    # Features (drop label)
    X = df.drop(columns=["isFraud"])

    # Remove ID column from features (keep separately if you want)
    if "TransactionID" in X.columns:
        transaction_id = X["TransactionID"].copy()
        X = X.drop(columns=["TransactionID"])
    else:
        transaction_id = None

    # Split
    X_train = X.iloc[idx_train].copy()
    X_valid = X.iloc[idx_valid].copy()
    X_test  = X.iloc[idx_test].copy()

    y_train = y.iloc[idx_train].copy()
    y_valid = y.iloc[idx_valid].copy()
    y_test  = y.iloc[idx_test].copy()

    print(f"\nRaw split shapes:")
    print(f"X_train={X_train.shape}, X_valid={X_valid.shape}, X_test={X_test.shape}")

    # Identify column types
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    print(f"\nDetected columns: numeric={len(num_cols)}, categorical={len(cat_cols)}")

    # ----------------------------
    # 1) Numeric imputation (median fit on train only)
    # ----------------------------
    print("\n[1/3] Numeric imputation (median from TRAIN)...")
    train_medians = X_train[num_cols].median(numeric_only=True)

    X_train[num_cols] = X_train[num_cols].fillna(train_medians)
    X_valid[num_cols] = X_valid[num_cols].fillna(train_medians)
    X_test[num_cols]  = X_test[num_cols].fillna(train_medians)

    # ----------------------------
    # 2) Categorical missing handling
    # ----------------------------
    print("[2/3] Categorical missing -> '__MISSING__'...")
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("__MISSING__").astype(str)
        X_valid[c] = X_valid[c].fillna("__MISSING__").astype(str)
        X_test[c]  = X_test[c].fillna("__MISSING__").astype(str)

    # ----------------------------
    # 3) Frequency encoding (fit on train only)
    # ----------------------------
    print("[3/3] Frequency encoding categoricals (fit on TRAIN only)...")
    for c in cat_cols:
        X_train[c] = frequency_encode(X_train[c], X_train[c])
        X_valid[c] = frequency_encode(X_train[c], X_valid[c])  # mapping based on TRAIN counts
        X_test[c]  = frequency_encode(X_train[c], X_test[c])

    # Convert numeric to float32 to reduce memory
    X_train[num_cols] = X_train[num_cols].astype(np.float32)
    X_valid[num_cols] = X_valid[num_cols].astype(np.float32)
    X_test[num_cols]  = X_test[num_cols].astype(np.float32)

    # Final check
    print("\nProcessed split shapes:")
    print(f"X_train={X_train.shape}, X_valid={X_valid.shape}, X_test={X_test.shape}")

    # Save outputs
    print("\nSaving processed datasets to data/processed/ ...")
    X_train.to_csv(os.path.join(DATA_PROCESSED, "X_train.csv"), index=False)
    X_valid.to_csv(os.path.join(DATA_PROCESSED, "X_valid.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_PROCESSED, "X_test.csv"), index=False)

    y_train.to_csv(os.path.join(DATA_PROCESSED, "y_train.csv"), index=False)
    y_valid.to_csv(os.path.join(DATA_PROCESSED, "y_valid.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_PROCESSED, "y_test.csv"), index=False)

    # Save columns list (for reproducibility)
    pd.Series(X_train.columns).to_csv(os.path.join(DATA_PROCESSED, "feature_columns.csv"),
                                      index=False, header=["feature_name"])

    print("[OK] Saved X_train/X_valid/X_test + y_train/y_valid/y_test + feature_columns.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
