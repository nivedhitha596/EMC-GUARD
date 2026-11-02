"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 1 | Script 2: Data Inspection and Quality Reporting
File   : data_cleaning.py
Author : EMC-GUARD Development Team

This script:
  1. Loads the raw synthetic dataset.
  2. Performs thorough data inspection (shape, dtypes, stats, nulls, dups).
  3. Identifies invalid values and outliers using the IQR method.
  4. Applies a documented cleaning policy.
  5. Saves the cleaned dataset and a human-readable quality report.

IMPORTANT — Cleaning Policy
----------------------------
Step | Action
-----|---------------------------------------------------------------
  1  | Remove rows where a physically-required column is negative or zero.
  2  | Impute remaining NaN values with column median (robust to outliers).
  3  | Remove EXACT duplicate rows (same input feature values), keeping first.
  4  | Flag IQR outliers; remove only those that are physically implausible;
     |   retain genuine extreme observations (marked with an Outlier_Flag).

Configuration_ID is preserved throughout for full traceability.
Original units are never altered.

Usage
-----
    python src/data_cleaning.py

Outputs
-------
    dataset/EMC_GUARD_Physics_Dataset.csv   — cleaned dataset
    reports/data_quality_report.txt         — quality analysis report
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "dataset", "dummy_simulation_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
REPORT_FILE= os.path.join(REPORT_DIR, "data_quality_report.txt")

# Columns that must be strictly positive (physically)
POSITIVE_REQUIRED = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm",
    "Frequency_Hz", "Voltage_V", "Conductor_Radius_mm",
    "Dielectric_Constant", "MutualInductance_nH",
    "MutualCapacitance_pF"
]

# Input feature columns (used to detect duplicate configurations)
INPUT_FEATURE_COLS = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
    "Shield_Type", "Cable_Configuration"
]

# All numerical columns for outlier analysis
NUMERICAL_COLS = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V"
]

IQR_MULTIPLIER = 3.0   # Conservative: only flag very extreme values

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV, skipping comment lines that start with '#'."""
    try:
        df = pd.read_csv(path, comment="#")
        print(f"  Loaded: {os.path.relpath(path, BASE_DIR)}")
        print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at:\n  {path}")
        print("  Run generate_dummy_data.py first.")
        sys.exit(1)


def inspect_dataset(df: pd.DataFrame, log: list) -> None:
    """Perform full inspection and append results to log list."""

    sep = "=" * 65

    log.append(sep)
    log.append("EMC-GUARD — DATA QUALITY INSPECTION REPORT")
    log.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append("Data      : SYNTHETIC DUMMY DATASET (testing only)")
    log.append(sep)

    # 1. Shape
    log.append(f"\n[1] Dataset Shape\n    {df.shape[0]} rows × {df.shape[1]} columns")

    # 2. Column names
    log.append(f"\n[2] Columns ({len(df.columns)})\n    " + ", ".join(df.columns.tolist()))

    # 3. Data types
    log.append("\n[3] Data Types")
    for col, dtype in df.dtypes.items():
        log.append(f"    {col:<25s} {str(dtype)}")

    # 4. First five records
    log.append("\n[4] First 5 Records")
    log.append(df.head().to_string(index=False))

    # 5. Last five records
    log.append("\n[5] Last 5 Records")
    log.append(df.tail().to_string(index=False))

    # 6. Descriptive statistics
    log.append("\n[6] Descriptive Statistics")
    log.append(df.describe(include="all").to_string())

    # 7. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    mv_df = pd.DataFrame({"Missing": missing, "Percent(%)": missing_pct})
    mv_df = mv_df[mv_df["Missing"] > 0]
    log.append(f"\n[7] Missing Values  (Total: {missing.sum()})")
    if mv_df.empty:
        log.append("    No missing values found.")
    else:
        log.append(mv_df.to_string())

    # 8. Duplicate configurations
    dup_mask = df.duplicated(subset=INPUT_FEATURE_COLS, keep=False)
    n_dups = dup_mask.sum()
    log.append(f"\n[8] Duplicate Configurations (same input features)  : {n_dups} rows")
    if n_dups > 0:
        log.append("    Sample duplicate IDs:")
        log.append("    " + ", ".join(df[dup_mask]["Configuration_ID"].tolist()[:10]))

    # 9. Invalid numerical values (negative where physically impossible)
    log.append("\n[9] Invalid Numerical Values (negative in positive-required columns)")
    any_invalid = False
    for col in POSITIVE_REQUIRED:
        if col not in df.columns:
            continue
        n_neg = (df[col] < 0).sum()
        n_zero = (df[col] == 0).sum()
        if n_neg > 0 or n_zero > 0:
            log.append(f"    {col:<25s}  negatives={n_neg}  zeros={n_zero}")
            any_invalid = True
    if not any_invalid:
        log.append("    No invalid values detected.")

    # 10. Outlier investigation (IQR method)
    log.append(f"\n[10] Outlier Investigation (IQR × {IQR_MULTIPLIER})")
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR
        n_out = ((series < lower) | (series > upper)).sum()
        if n_out > 0:
            log.append(
                f"    {col:<25s}  bounds=[{lower:.3f}, {upper:.3f}]  "
                f"outliers={n_out}"
            )


def remove_invalid_records(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Step 1: Drop rows with physically impossible (negative) values."""
    before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    for col in POSITIVE_REQUIRED:
        if col in df.columns:
            mask &= ~(df[col] < 0).fillna(False)
    df = df[mask].copy()
    removed = before - len(df)
    log.append(f"\n[CLEAN-1] Remove invalid (negative) values")
    log.append(f"          Records before : {before}")
    log.append(f"          Records removed: {removed}")
    log.append(f"          Records after  : {len(df)}")
    return df


def impute_missing_values(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Step 2: Fill remaining NaN with column median (robust imputation)."""
    before_nulls = df.isnull().sum().sum()
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    after_nulls = df.isnull().sum().sum()
    log.append(f"\n[CLEAN-2] Impute missing values (median strategy)")
    log.append(f"          Missing before : {before_nulls}")
    log.append(f"          Missing after  : {after_nulls}")
    return df


def remove_duplicates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Step 3: Remove exact duplicate input-feature combinations (keep first)."""
    before = len(df)
    df = df.drop_duplicates(subset=INPUT_FEATURE_COLS, keep="first").copy()
    removed = before - len(df)
    log.append(f"\n[CLEAN-3] Remove duplicate configurations (keep first)")
    log.append(f"          Records before : {before}")
    log.append(f"          Records removed: {removed}")
    log.append(f"          Records after  : {len(df)}")
    return df


def flag_outliers(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """
    Step 4: IQR outlier investigation.

    Policy
    ------
    - Values far outside IQR bounds AND physically implausible → remove.
    - Values far outside IQR bounds but physically plausible  → keep,
      mark with Outlier_Flag = 1 for downstream awareness.
    """
    before = len(df)
    df["Outlier_Flag"] = 0

    # Define "implausible" as > 1 000 000 for any electrical quantity
    IMPLAUSIBLE_THRESHOLD = 1_000_000.0

    rows_to_drop = set()
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - IQR_MULTIPLIER * IQR
        upper = Q3 + IQR_MULTIPLIER * IQR

        outlier_idx = df.index[(df[col] < lower) | (df[col] > upper)]
        for idx in outlier_idx:
            val = df.at[idx, col]
            if abs(val) > IMPLAUSIBLE_THRESHOLD:
                rows_to_drop.add(idx)
            else:
                df.at[idx, "Outlier_Flag"] = 1

    df = df.drop(index=list(rows_to_drop)).copy()
    removed = before - len(df)
    n_flagged = df["Outlier_Flag"].sum()

    log.append(f"\n[CLEAN-4] Outlier handling (IQR × {IQR_MULTIPLIER})")
    log.append(f"          Records before         : {before}")
    log.append(f"          Implausible → removed  : {removed}")
    log.append(f"          Extreme but valid → kept (Outlier_Flag=1): {n_flagged}")
    log.append(f"          Records after          : {len(df)}")
    return df


def save_cleaned_dataset(df: pd.DataFrame, log: list) -> None:
    """Save cleaned dataset and append summary to log."""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log.append(f"\n[SAVE] Cleaned dataset saved to:")
    log.append(f"       {os.path.relpath(OUTPUT_CSV, BASE_DIR)}")
    log.append(f"       Final shape: {df.shape[0]} rows × {df.shape[1]} columns")


def write_report(log: list) -> None:
    """Write the full inspection + cleaning log to the report file."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print(f"\n  Quality report saved: {os.path.relpath(REPORT_FILE, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("EMC-GUARD  |  Module 1  |  Data Inspection & Cleaning")
    print("=" * 65)

    log = []

    print("\n[1/6] Loading raw dataset …")
    df = load_dataset(INPUT_CSV)

    print("[2/6] Inspecting dataset …")
    inspect_dataset(df, log)

    print("[3/6] Removing invalid (negative) records …")
    df = remove_invalid_records(df, log)

    print("[4/6] Imputing missing values …")
    df = impute_missing_values(df, log)

    print("[5/6] Removing duplicate configurations …")
    df = remove_duplicates(df, log)

    print("[6/6] Flagging / removing outliers …")
    df = flag_outliers(df, log)

    save_cleaned_dataset(df, log)
    write_report(log)

    print("\n" + "=" * 65)
    print("Cleaning Summary")
    print("-" * 65)
    print(f"  Final dataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Outlier_Flag rows   : {df['Outlier_Flag'].sum()}")
    print(f"  Output CSV          : {os.path.relpath(OUTPUT_CSV, BASE_DIR)}")
    print(f"  Quality report      : {os.path.relpath(REPORT_FILE, BASE_DIR)}")
    print("=" * 65)
    print("\n[DONE] Data cleaning complete.")


if __name__ == "__main__":
    main()
