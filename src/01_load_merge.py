# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 22:49:38 2026

@author: FarhanAli
"""

# ============================================================
# 01_load_merge.py
#   Load + Merge IEEE-CIS Dataset
# - Loads train_transaction, train_identity, test_transaction, test_identity
# - Merges on TransactionID
# - Prints shapes, key-column checks, and missingness summary
# - Saves merged files to data/interim/
# ============================================================

import os
import sys
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")

TRAIN_TRANS_PATH = os.path.join(DATA_RAW, "train_transaction.csv")
TRAIN_ID_PATH = os.path.join(DATA_RAW, "train_identity.csv")
TEST_TRANS_PATH = os.path.join(DATA_RAW, "test_transaction.csv")
TEST_ID_PATH = os.path.join(DATA_RAW, "test_identity.csv")

OUT_TRAIN_MERGED = os.path.join(DATA_INTERIM, "train_merged.csv")
OUT_TEST_MERGED = os.path.join(DATA_INTERIM, "test_merged.csv")

# ----------------------------
# Utilities
# ----------------------------
def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)

def missingness_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({
        "missing_rate": miss,
        "missing_count": df.isna().sum()[miss.index],
        "dtype": df.dtypes[miss.index].astype(str)
    })
    return out.head(top_n)

def check_columns(df: pd.DataFrame, cols: list, df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] {df_name}: missing expected columns: {missing}")
    else:
        print(f"[OK] {df_name}: all expected columns present.")

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("=== IEEE-CIS: Step 1 Load + Merge ===")

    # 1) Validate file existence
    for p in [TRAIN_TRANS_PATH, TRAIN_ID_PATH, TEST_TRANS_PATH, TEST_ID_PATH]:
        require_file(p)

    safe_mkdir(DATA_INTERIM)

    # 2) Load CSVs (low_memory=False to reduce dtype issues)
    print("\n[1/4] Loading CSV files...")
    train_trans = pd.read_csv(TRAIN_TRANS_PATH, low_memory=False)
    train_id = pd.read_csv(TRAIN_ID_PATH, low_memory=False)
    test_trans = pd.read_csv(TEST_TRANS_PATH, low_memory=False)
    test_id = pd.read_csv(TEST_ID_PATH, low_memory=False)

    # 3) Basic checks
    print("\n[2/4] Basic checks...")
    print(f"train_transaction: shape={train_trans.shape}, mem={mem_mb(train_trans):.2f} MB")
    print(f"train_identity:    shape={train_id.shape}, mem={mem_mb(train_id):.2f} MB")
    print(f"test_transaction:  shape={test_trans.shape}, mem={mem_mb(test_trans):.2f} MB")
    print(f"test_identity:     shape={test_id.shape}, mem={mem_mb(test_id):.2f} MB")

    # Required join key and label check
    required_trans_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]
    required_label = ["isFraud"]

    check_columns(train_trans, required_trans_cols + required_label, "train_transaction")
    check_columns(test_trans, required_trans_cols, "test_transaction")

    # Identity evidence columns (may not all exist in every version, but typical)
    required_id_cols = ["TransactionID", "DeviceType", "DeviceInfo"]
    check_columns(train_id, required_id_cols, "train_identity")
    check_columns(test_id, required_id_cols, "test_identity")

    # 4) Merge on TransactionID (left join keeps all transactions)
    print("\n[3/4] Merging transaction + identity on TransactionID (left join)...")
    train_merged = train_trans.merge(train_id, on="TransactionID", how="left")
    test_merged = test_trans.merge(test_id, on="TransactionID", how="left")

    print(f"train_merged: shape={train_merged.shape}, mem={mem_mb(train_merged):.2f} MB")
    print(f"test_merged:  shape={test_merged.shape}, mem={mem_mb(test_merged):.2f} MB")

    # Identity coverage (how many rows have any identity info)
    if "DeviceType" in train_merged.columns:
        identity_present = train_merged["DeviceType"].notna().mean()
        print(f"Identity coverage (train): {identity_present*100:.2f}% rows have DeviceType (proxy for identity availability)")
    else:
        # fallback: estimate using intersection of TransactionIDs
        identity_present = train_trans["TransactionID"].isin(train_id["TransactionID"]).mean()
        print(f"Identity coverage (train) via TransactionID intersection: {identity_present*100:.2f}%")

    # 5) Missingness summary for evidence
    print("\n[4/4] Missingness summary (top 15 columns by missing rate) — train_merged:")
    print(missingness_summary(train_merged, top_n=15).to_string())

    # 6) Save merged outputs
    print("\nSaving merged datasets to data/interim/ ...")
    train_merged.to_csv(OUT_TRAIN_MERGED, index=False)
    test_merged.to_csv(OUT_TEST_MERGED, index=False)
    print(f"[OK] Saved: {OUT_TRAIN_MERGED}")
    print(f"[OK] Saved: {OUT_TEST_MERGED}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
