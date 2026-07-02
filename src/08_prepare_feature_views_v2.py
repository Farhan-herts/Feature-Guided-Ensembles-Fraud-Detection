# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 01:25:54 2026

@author: FarhanAli
"""
# 08_prepare_feature_views_v2.py
# Revised feature-view builder for the updated experiment plan
#
# It creates:
# [T] Temporal view
# [I] Identity view
# [A] Aggregation view
# [B] Behaviour view
#
# It is leakage-safe for A/B view creation because group statistics
# are fitted on TRAIN only, then mapped to VALID and TEST.
#
# Outputs:
# data/processed/X_train_view_T.csv
# data/processed/X_valid_view_T.csv
# data/processed/X_test_view_T.csv
#
# data/processed/X_train_view_I.csv
# data/processed/X_valid_view_I.csv
# data/processed/X_test_view_I.csv
#
# data/processed/X_train_view_A.csv
# data/processed/X_valid_view_A.csv
# data/processed/X_test_view_A.csv
#
# data/processed/X_train_view_B.csv
# data/processed/X_valid_view_B.csv
# data/processed/X_test_view_B.csv
#
# data/processed/X_train_views_ATBI.csv
# data/processed/X_valid_views_ATBI.csv
# data/processed/X_test_views_ATBI.csv
#
# data/processed/view_manifest_ATBI.json

import os
import sys
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

X_TRAIN_PATH = os.path.join(DATA_PROCESSED, "X_train.csv")
X_VALID_PATH = os.path.join(DATA_PROCESSED, "X_valid.csv")
X_TEST_PATH = os.path.join(DATA_PROCESSED, "X_test.csv")

OUTPUTS = {
    "T": {
        "train": os.path.join(DATA_PROCESSED, "X_train_view_T.csv"),
        "valid": os.path.join(DATA_PROCESSED, "X_valid_view_T.csv"),
        "test": os.path.join(DATA_PROCESSED, "X_test_view_T.csv"),
    },
    "I": {
        "train": os.path.join(DATA_PROCESSED, "X_train_view_I.csv"),
        "valid": os.path.join(DATA_PROCESSED, "X_valid_view_I.csv"),
        "test": os.path.join(DATA_PROCESSED, "X_test_view_I.csv"),
    },
    "A": {
        "train": os.path.join(DATA_PROCESSED, "X_train_view_A.csv"),
        "valid": os.path.join(DATA_PROCESSED, "X_valid_view_A.csv"),
        "test": os.path.join(DATA_PROCESSED, "X_test_view_A.csv"),
    },
    "B": {
        "train": os.path.join(DATA_PROCESSED, "X_train_view_B.csv"),
        "valid": os.path.join(DATA_PROCESSED, "X_valid_view_B.csv"),
        "test": os.path.join(DATA_PROCESSED, "X_test_view_B.csv"),
    },
    "ATBI": {
        "train": os.path.join(DATA_PROCESSED, "X_train_views_ATBI.csv"),
        "valid": os.path.join(DATA_PROCESSED, "X_valid_views_ATBI.csv"),
        "test": os.path.join(DATA_PROCESSED, "X_test_views_ATBI.csv"),
    },
}

MANIFEST_PATH = os.path.join(DATA_PROCESSED, "view_manifest_ATBI.json")

AMOUNT_COL = "TransactionAmt"
TIME_COL = "TransactionDT"

# These are common IEEE-CIS columns. The script uses whichever exist.
IDENTITY_CANDIDATES = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]
UID_CANDIDATES = ["card1", "card2", "card3", "card5"]
GROUP_KEY_CANDIDATES = ["uid", "card1", "card2", "addr1"]


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_datasets():
    for p in [X_TRAIN_PATH, X_VALID_PATH, X_TEST_PATH]:
        require_file(p)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_valid = pd.read_csv(X_VALID_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    return X_train, X_valid, X_test


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    out = numerator / denom
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


def build_uid_proxy(df: pd.DataFrame) -> pd.Series:
    available = [c for c in UID_CANDIDATES if c in df.columns]
    if not available:
        raise KeyError(
            f"None of the UID candidate columns were found. "
            f"Expected one or more of: {UID_CANDIDATES}"
        )

    uid = df[available[0]].astype(str)
    for c in available[1:]:
        uid = uid + "_" + df[c].astype(str)

    return uid


def create_temporal_view(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COL not in df.columns:
        raise KeyError(f"'{TIME_COL}' not found. Cannot build temporal view.")

    out = pd.DataFrame(index=df.index)

    dt = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(0).astype(np.float64)

    out["T_DT_hour"] = ((dt // 3600) % 24).astype(np.float32)
    out["T_DT_day"] = (dt // 86400).astype(np.float32)
    out["T_DT_week"] = (dt // (86400 * 7)).astype(np.float32)

    return out


def create_identity_view(df: pd.DataFrame) -> pd.DataFrame:
    use_cols = [c for c in IDENTITY_CANDIDATES if c in df.columns]
    if not use_cols:
        raise KeyError(
            f"No identity columns found. Expected one or more of: {IDENTITY_CANDIDATES}"
        )

    out = pd.DataFrame(index=df.index)
    for c in use_cols:
        out[f"I_{c}"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    return out


def fit_group_amount_stats(train_df: pd.DataFrame, key_col: str) -> dict:
    grouped = train_df.groupby(key_col)[AMOUNT_COL].agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    )

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["range"] = grouped["max"] - grouped["min"]

    stats = {
        "count": grouped["count"].to_dict(),
        "mean": grouped["mean"].to_dict(),
        "std": grouped["std"].to_dict(),
        "median": grouped["median"].to_dict(),
        "min": grouped["min"].to_dict(),
        "max": grouped["max"].to_dict(),
        "range": grouped["range"].to_dict(),
    }
    return stats


def add_mapped_aggregation_features(
    df: pd.DataFrame,
    key_col: str,
    key_name: str,
    stats: dict,
    global_stats: dict,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    key_series = df[key_col]

    out[f"A_{key_name}_txn_count"] = key_series.map(stats["count"]).fillna(0).astype(np.float32)
    out[f"A_{key_name}_amt_mean"] = key_series.map(stats["mean"]).fillna(global_stats["mean"]).astype(np.float32)
    out[f"A_{key_name}_amt_std"] = key_series.map(stats["std"]).fillna(global_stats["std"]).astype(np.float32)
    out[f"A_{key_name}_amt_median"] = key_series.map(stats["median"]).fillna(global_stats["median"]).astype(np.float32)
    out[f"A_{key_name}_amt_min"] = key_series.map(stats["min"]).fillna(global_stats["min"]).astype(np.float32)
    out[f"A_{key_name}_amt_max"] = key_series.map(stats["max"]).fillna(global_stats["max"]).astype(np.float32)
    out[f"A_{key_name}_amt_range"] = key_series.map(stats["range"]).fillna(global_stats["range"]).astype(np.float32)

    return out


def create_aggregation_views(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    if AMOUNT_COL not in train_df.columns:
        raise KeyError(f"'{AMOUNT_COL}' not found. Cannot build aggregation/behaviour views.")

    global_stats = {
        "mean": float(train_df[AMOUNT_COL].mean()),
        "std": float(max(train_df[AMOUNT_COL].std(ddof=0), 1e-6)),
        "median": float(train_df[AMOUNT_COL].median()),
        "min": float(train_df[AMOUNT_COL].min()),
        "max": float(train_df[AMOUNT_COL].max()),
        "range": float(train_df[AMOUNT_COL].max() - train_df[AMOUNT_COL].min()),
    }

    fitted_stats = {}
    group_keys = []

    for candidate in GROUP_KEY_CANDIDATES:
        if candidate in train_df.columns:
            group_keys.append(candidate)

    if not group_keys:
        raise KeyError("No aggregation group keys found.")

    A_train_parts = []
    A_valid_parts = []
    A_test_parts = []

    for key in group_keys:
        key_name = key
        stats = fit_group_amount_stats(train_df, key)
        fitted_stats[key_name] = stats

        A_train_parts.append(add_mapped_aggregation_features(train_df, key, key_name, stats, global_stats))
        A_valid_parts.append(add_mapped_aggregation_features(valid_df, key, key_name, stats, global_stats))
        A_test_parts.append(add_mapped_aggregation_features(test_df, key, key_name, stats, global_stats))

    A_train = pd.concat(A_train_parts, axis=1)
    A_valid = pd.concat(A_valid_parts, axis=1)
    A_test = pd.concat(A_test_parts, axis=1)

    return A_train, A_valid, A_test, fitted_stats, global_stats, group_keys


def create_behaviour_view(
    base_df: pd.DataFrame,
    A_df: pd.DataFrame,
    fitted_group_names: list,
    global_stats: dict,
) -> pd.DataFrame:
    out = pd.DataFrame(index=base_df.index)
    amt = pd.to_numeric(base_df[AMOUNT_COL], errors="coerce").fillna(global_stats["mean"]).astype(np.float32)

    for name in fitted_group_names:
        count_col = f"A_{name}_txn_count"
        mean_col = f"A_{name}_amt_mean"
        std_col = f"A_{name}_amt_std"
        median_col = f"A_{name}_amt_median"

        out[f"B_{name}_amt_minus_mean"] = (amt - A_df[mean_col]).astype(np.float32)
        out[f"B_{name}_amt_minus_median"] = (amt - A_df[median_col]).astype(np.float32)
        out[f"B_{name}_amt_over_mean"] = safe_divide(amt, A_df[mean_col])
        out[f"B_{name}_amt_over_median"] = safe_divide(amt, A_df[median_col])
        out[f"B_{name}_amt_zscore"] = safe_divide((amt - A_df[mean_col]), A_df[std_col])

        out[f"B_{name}_is_unseen_group"] = (A_df[count_col] == 0).astype(np.float32)
        out[f"B_{name}_is_rare_group"] = (A_df[count_col] <= 2).astype(np.float32)

    # Global behaviour signals
    out["B_global_amt_minus_mean"] = (amt - global_stats["mean"]).astype(np.float32)
    out["B_global_amt_over_mean"] = safe_divide(amt, pd.Series(global_stats["mean"], index=base_df.index))
    out["B_global_amt_zscore"] = safe_divide(
        (amt - global_stats["mean"]),
        pd.Series(global_stats["std"], index=base_df.index)
    )

    # Cross-group behaviour comparisons where possible
    if "uid" in fitted_group_names and "card1" in fitted_group_names:
        out["B_uid_count_over_card1_count"] = safe_divide(
            A_df["A_uid_txn_count"],
            A_df["A_card1_txn_count"]
        )

    if "uid" in fitted_group_names and "addr1" in fitted_group_names:
        out["B_uid_count_over_addr1_count"] = safe_divide(
            A_df["A_uid_txn_count"],
            A_df["A_addr1_txn_count"]
        )

    return out.astype(np.float32)


def save_triplet(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, paths: dict) -> None:
    train_df.to_csv(paths["train"], index=False)
    valid_df.to_csv(paths["valid"], index=False)
    test_df.to_csv(paths["test"], index=False)


def main() -> None:
    print("=== IEEE-CIS: Step 8 Revised View Builder (T / I / A / B) ===")
    safe_mkdir(DATA_PROCESSED)

    X_train, X_valid, X_test = load_datasets()

    print("\nLoaded processed datasets:")
    print(f"X_train={X_train.shape}")
    print(f"X_valid={X_valid.shape}")
    print(f"X_test={X_test.shape}")

    # Work on copies
    train_df = X_train.copy()
    valid_df = X_valid.copy()
    test_df = X_test.copy()

    # Build UID proxy for aggregation/behaviour
    print("\n[1/5] Building UID proxy from available card columns...")
    train_df["uid"] = build_uid_proxy(train_df)
    valid_df["uid"] = build_uid_proxy(valid_df)
    test_df["uid"] = build_uid_proxy(test_df)

    # T view
    print("[2/5] Creating Temporal view...")
    T_train = create_temporal_view(train_df)
    T_valid = create_temporal_view(valid_df)
    T_test = create_temporal_view(test_df)

    # I view
    print("[3/5] Creating Identity view...")
    I_train = create_identity_view(train_df)
    I_valid = create_identity_view(valid_df)
    I_test = create_identity_view(test_df)

    # A view
    print("[4/5] Creating Aggregation view (TRAIN-fit, VALID/TEST-map)...")
    A_train, A_valid, A_test, fitted_stats, global_stats, group_keys = create_aggregation_views(
        train_df, valid_df, test_df
    )

    # B view
    print("[5/5] Creating Behaviour view from aggregation baselines...")
    fitted_group_names = list(fitted_stats.keys())
    B_train = create_behaviour_view(train_df, A_train, fitted_group_names, global_stats)
    B_valid = create_behaviour_view(valid_df, A_valid, fitted_group_names, global_stats)
    B_test = create_behaviour_view(test_df, A_test, fitted_group_names, global_stats)

    # Combined ATBI
    ATBI_train = pd.concat([A_train, T_train, B_train, I_train], axis=1)
    ATBI_valid = pd.concat([A_valid, T_valid, B_valid, I_valid], axis=1)
    ATBI_test = pd.concat([A_test, T_test, B_test, I_test], axis=1)

    # Save everything
    print("\nSaving outputs...")
    save_triplet(T_train, T_valid, T_test, OUTPUTS["T"])
    save_triplet(I_train, I_valid, I_test, OUTPUTS["I"])
    save_triplet(A_train, A_valid, A_test, OUTPUTS["A"])
    save_triplet(B_train, B_valid, B_test, OUTPUTS["B"])
    save_triplet(ATBI_train, ATBI_valid, ATBI_test, OUTPUTS["ATBI"])

    manifest = {
        "base_input_files": {
            "train": X_TRAIN_PATH,
            "valid": X_VALID_PATH,
            "test": X_TEST_PATH,
        },
        "amount_column": AMOUNT_COL,
        "time_column": TIME_COL,
        "uid_candidates_used_from": UID_CANDIDATES,
        "group_keys_used": group_keys,
        "identity_columns_found": [c for c in IDENTITY_CANDIDATES if c in X_train.columns],
        "view_columns": {
            "A": A_train.columns.tolist(),
            "T": T_train.columns.tolist(),
            "B": B_train.columns.tolist(),
            "I": I_train.columns.tolist(),
            "ATBI": ATBI_train.columns.tolist(),
        },
        "output_files": OUTPUTS,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[OK] Saved view files:")
    for view_name, paths in OUTPUTS.items():
        print(f"\n{view_name} view:")
        for split_name, p in paths.items():
            print(f"  {split_name}: {p}")

    print(f"\n[OK] Manifest saved to: {MANIFEST_PATH}")

    print("\nOutput shapes:")
    print(f"T view   -> train={T_train.shape}, valid={T_valid.shape}, test={T_test.shape}")
    print(f"I view   -> train={I_train.shape}, valid={I_valid.shape}, test={I_test.shape}")
    print(f"A view   -> train={A_train.shape}, valid={A_valid.shape}, test={A_test.shape}")
    print(f"B view   -> train={B_train.shape}, valid={B_valid.shape}, test={B_test.shape}")
    print(f"ATBI all -> train={ATBI_train.shape}, valid={ATBI_valid.shape}, test={ATBI_test.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)