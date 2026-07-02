# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 23:18:48 2026

@author: FarhanAli
"""

# ============================================================
# 02_split.py
# : Leakage-safe splits
# - Loads merged training data (train_merged.csv)
# - Creates stratified Train/Valid/Test split
# - Saves indices for reproducibility
# - Prints class balance in each split
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "splits")

TRAIN_MERGED_PATH = os.path.join(DATA_INTERIM, "train_merged.csv")

# ----------------------------
# Utilities
# ----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

def print_split_stats(name: str, y: pd.Series) -> None:
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    rate = pos / n if n > 0 else 0
    print(f"{name:>10} | n={n:,} | fraud={pos:,} | non-fraud={neg:,} | fraud_rate={rate:.6f}")

# ----------------------------
# Main
# ----------------------------
def main():
    print("=== Step 2: Leakage-safe split (stratified) ===")

    require_file(TRAIN_MERGED_PATH)
    safe_mkdir(RESULTS_DIR)

    print("\nLoading merged train data...")
    df = pd.read_csv(TRAIN_MERGED_PATH, low_memory=False)

    # Basic checks
    if "isFraud" not in df.columns:
        raise ValueError("Column 'isFraud' not found in train_merged.csv")

    y = df["isFraud"].astype(int)

    print("\nOverall class balance:")
    print_split_stats("FULL", y)

    # ----------------------------
    # Split configuration
    # ----------------------------
    RANDOM_STATE = 42
    TEST_SIZE = 0.20      # 20% final test set
    VALID_SIZE = 0.20     # 20% of remaining (i.e., 16% of full)

    # First split: train+valid vs test
    idx_full = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        idx_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Second split: train vs valid (on trainval portion)
    y_trainval = y.iloc[idx_trainval]
    idx_train, idx_valid = train_test_split(
        idx_trainval,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    # Convert to sorted arrays (nice for reproducibility)
    idx_train = np.sort(idx_train)
    idx_valid = np.sort(idx_valid)
    idx_test = np.sort(idx_test)

    # ----------------------------
    # Save indices
    # ----------------------------
    np.save(os.path.join(RESULTS_DIR, "train_idx.npy"), idx_train)
    np.save(os.path.join(RESULTS_DIR, "valid_idx.npy"), idx_valid)
    np.save(os.path.join(RESULTS_DIR, "test_idx.npy"), idx_test)

    # Also save a readable CSV for quick reference
    split_df = pd.DataFrame({
        "index": np.concatenate([idx_train, idx_valid, idx_test]),
        "split": (["train"] * len(idx_train)) + (["valid"] * len(idx_valid)) + (["test"] * len(idx_test))
    })
    split_df.to_csv(os.path.join(RESULTS_DIR, "splits.csv"), index=False)

    # ----------------------------
    # Print split stats
    # ----------------------------
    print("\nSplit sizes and class balance:")
    print_split_stats("TRAIN", y.iloc[idx_train])
    print_split_stats("VALID", y.iloc[idx_valid])
    print_split_stats("TEST",  y.iloc[idx_test])

    print("\nSaved split indices to:", RESULTS_DIR)
    print("Files: train_idx.npy, valid_idx.npy, test_idx.npy, splits.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
