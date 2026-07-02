# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 19:51:59 2026

@author: FarhanAli
"""

# ============================================================
# 05_feature_views.py
#  Feature Views (Temporal + Identity)
# - Temporal (time-based features) + Identity (device-related features)
# - Leakage-safe encoding for categoricals: frequency encoding
# - Save to data/processed/ directory
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

X_TRAIN_PATH = os.path.join(DATA_PROCESSED, "X_train.csv")
X_VALID_PATH = os.path.join(DATA_PROCESSED, "X_valid.csv")
X_TEST_PATH  = os.path.join(DATA_PROCESSED, "X_test.csv")

# ----------------------------
# Utilities
# ----------------------------
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
# Temporal Features (TransactionDT)
# ----------------------------
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from 'TransactionDT' (timedelta)
    - Hour, day of week, and day of the month
    """
    df["DT_hour"] = (df["TransactionDT"] // 3600) % 24
    df["DT_day"] = (df["TransactionDT"] // 86400)
    df["DT_week"] = (df["TransactionDT"] // (86400 * 7))
    
    return df

# ----------------------------
# Identity Features (DeviceType, DeviceInfo)
# ----------------------------
def create_identity_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Create identity features: frequency encoding for DeviceType and DeviceInfo
    """
    # Apply frequency encoding (Leakage-safe on Train only)
    for c in cat_cols:
        df[c] = frequency_encode(df[c], df[c])  # Train based frequency encoding

    return df

# ----------------------------
# Main
# ----------------------------
def main():


    # Load the data
    for p in [X_TRAIN_PATH, X_VALID_PATH, X_TEST_PATH]:
        require_file(p)
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_valid = pd.read_csv(X_VALID_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)

    print(f"\nLoaded X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")

    # 1) Temporal View (time features from 'TransactionDT')
    print("\n[1/2] Creating Temporal Features (DT_hour, DT_day, DT_week)...")
    X_train = create_temporal_features(X_train)
    X_valid = create_temporal_features(X_valid)
    X_test  = create_temporal_features(X_test)

    # 2) Identity/Device View (frequency encoding)
    print("\n[2/2] Creating Identity/Device Features (DeviceType, DeviceInfo)...")
    cat_cols = ['DeviceType', 'DeviceInfo']  # You can add more if needed
    X_train = create_identity_features(X_train, cat_cols)
    X_valid = create_identity_features(X_valid, cat_cols)
    X_test  = create_identity_features(X_test, cat_cols)

    # Save processed feature sets
    print("\nSaving processed feature views to data/processed/ ...")
    X_train.to_csv(os.path.join(DATA_PROCESSED, "X_train_views.csv"), index=False)
    X_valid.to_csv(os.path.join(DATA_PROCESSED, "X_valid_views.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_PROCESSED, "X_test_views.csv"), index=False)

    print("[OK] Saved X_train_views.csv, X_valid_views.csv, X_test_views.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
