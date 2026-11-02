"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 1 | Script 3: Feature Engineering, Encoding, and Train/Test Split
File   : data_preprocessing.py
Author : EMC-GUARD Development Team

This script:
  1. Loads the cleaned dataset.
  2. Separates Configuration_ID, input features (X), and targets (Y).
  3. Builds a Scikit-learn ColumnTransformer pipeline:
       - Numerical : StandardScaler (with median imputation as safety net)
       - Categorical: OneHotEncoder (drop='first' to avoid multicollinearity)
  4. Splits into 80 / 20 train / test with fixed random_state.
  5. Fits the preprocessor on training data only (no leakage).
  6. Transforms train and test sets.
  7. Saves X_train, X_test, Y_train, Y_test as CSV files.
  8. Saves the fitted preprocessor as emc_preprocessor.pkl.
  9. Writes preprocessing_summary.txt and feature_encoding_details.txt.

Usage
-----
    python src/data_preprocessing.py

Outputs (in dataset/ unless otherwise noted)
--------------------------------------------
    dataset/X_train.csv
    dataset/X_test.csv
    dataset/Y_train.csv
    dataset/Y_test.csv
    dataset/emc_preprocessor.pkl
    reports/preprocessing_summary.txt
    reports/feature_encoding_details.txt
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIGURATION — update INPUT_CSV to point to the real MATLAB dataset when
# it is available.  Nothing else needs changing.
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV     = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
PKL_FILE      = os.path.join(OUTPUT_DIR, "emc_preprocessor.pkl")

RANDOM_STATE  = 42
TEST_SIZE     = 0.20       # 80 / 20 split

# Column definitions
ID_COL = "Configuration_ID"

NUMERICAL_FEATURES = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
]

CATEGORICAL_FEATURES = [
    "Shield_Type", "Cable_Configuration",
]

INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_COLS = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_cleaned_dataset(path: str) -> pd.DataFrame:
    """Load the cleaned dataset, ensuring required columns are present."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] Cleaned dataset not found: {path}")
        print("  Run data_cleaning.py first.")
        sys.exit(1)

    required = [ID_COL] + INPUT_FEATURES + TARGET_COLS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing expected columns: {missing_cols}")
        sys.exit(1)

    print(f"  Loaded: {os.path.relpath(path, BASE_DIR)}")
    print(f"  Shape : {df.shape}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer preprocessing pipeline.

    Numerical pipeline:
      1. SimpleImputer  (median) — safety net for any residual NaN
      2. StandardScaler          — zero mean, unit variance

    Categorical pipeline:
      1. SimpleImputer  (most_frequent) — safety net for any residual NaN
      2. OneHotEncoder (drop='first')   — avoid perfect multicollinearity
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numerical",    numerical_pipeline,    NUMERICAL_FEATURES),
        ("categorical",  categorical_pipeline,  CATEGORICAL_FEATURES),
    ], remainder="drop")   # Drop Outlier_Flag and any other extra columns

    return preprocessor


def get_feature_names_after_encoding(preprocessor: ColumnTransformer) -> list:
    """Extract ordered output feature names from the fitted ColumnTransformer."""
    ohe = preprocessor.named_transformers_["categorical"]["encoder"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERICAL_FEATURES + cat_feature_names


def write_encoding_details(preprocessor: ColumnTransformer,
                            feature_names: list) -> None:
    """Save a human-readable description of the encoding to a text file."""
    ohe = preprocessor.named_transformers_["categorical"]["encoder"]
    scaler = preprocessor.named_transformers_["numerical"]["scaler"]

    lines = [
        "=" * 65,
        "EMC-GUARD — FEATURE ENCODING DETAILS",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Data      : SYNTHETIC DUMMY DATASET (testing only)",
        "=" * 65,
        "",
        "[NUMERICAL FEATURES — StandardScaler]",
        f"  Strategy : zero mean, unit variance",
        f"  Imputer  : median (safety net for residual NaN)",
        "",
    ]
    for col, mean, scale in zip(NUMERICAL_FEATURES, scaler.mean_, scaler.scale_):
        lines.append(f"  {col:<25s}  mean={mean:.4f}  std={scale:.4f}")

    lines += [
        "",
        "[CATEGORICAL FEATURES — OneHotEncoder (drop='first')]",
        "  Note: The first category of each feature is dropped to avoid",
        "        perfect multicollinearity (reference-category encoding).",
        "",
    ]
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        kept = cats[1:]   # first is dropped
        lines.append(f"  {col}")
        lines.append(f"    All categories : {cats.tolist()}")
        lines.append(f"    Reference (dropped): {cats[0]}")
        lines.append(f"    Encoded categories : {kept.tolist()}")
        lines.append("")

    lines += [
        "[FULL ENCODED FEATURE LIST]",
        f"  Total features after encoding: {len(feature_names)}",
        "",
    ]
    for i, name in enumerate(feature_names, 1):
        lines.append(f"  {i:>3d}. {name}")

    path = os.path.join(REPORTS_DIR, "feature_encoding_details.txt")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Encoding details  : {os.path.relpath(path, BASE_DIR)}")


def write_preprocessing_summary(
    n_clean: int, n_train: int, n_test: int,
    n_features_raw: int, n_features_encoded: int,
    feature_names: list
) -> None:
    """Write preprocessing_summary.txt."""
    lines = [
        "=" * 65,
        "EMC-GUARD — PREPROCESSING SUMMARY",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Data      : SYNTHETIC DUMMY DATASET (testing only)",
        "=" * 65,
        "",
        "[DATASET SPLIT]",
        f"  Total records (after cleaning)  : {n_clean}",
        f"  Training records (80%)          : {n_train}",
        f"  Testing  records (20%)          : {n_test}",
        f"  Random state                    : {RANDOM_STATE}",
        "",
        "[FEATURES]",
        f"  Raw input features              : {n_features_raw}",
        f"  Encoded/transformed features    : {n_features_encoded}",
        f"  Numerical (StandardScaler)      : {len(NUMERICAL_FEATURES)}",
        f"  Categorical (OneHotEncoder)     : {len(CATEGORICAL_FEATURES)}",
        "",
        "[TARGETS (not scaled)]",
    ]
    for t in TARGET_COLS:
        lines.append(f"  {t}")
    lines += [
        "",
        "[LEAKAGE PREVENTION]",
        "  Preprocessor fitted on TRAINING data only.",
        "  Same fitted preprocessor applied to TEST data.",
        "  No test-set statistics used for imputation or scaling.",
        "",
        "[OUTPUT FILES]",
        "  dataset/X_train.csv               — encoded training features",
        "  dataset/X_test.csv                — encoded testing features",
        "  dataset/Y_train.csv               — training targets (original units)",
        "  dataset/Y_test.csv                — testing targets (original units)",
        "  dataset/emc_preprocessor.pkl      — fitted ColumnTransformer",
        "  reports/feature_encoding_details.txt",
        "  reports/preprocessing_summary.txt",
        "",
        "[REPLACING WITH REAL MATLAB DATA]",
        "  Update INPUT_CSV at the top of each script to point to the",
        "  real MATLAB simulation CSV.  The pipeline requires the same",
        "  column names and units; no other code changes are needed.",
    ]

    path = os.path.join(REPORTS_DIR, "preprocessing_summary.txt")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Preprocessing summary: {os.path.relpath(path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("EMC-GUARD  |  Module 1  |  Feature Engineering & Splitting")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load cleaned data
    print("\n[1/7] Loading cleaned dataset …")
    df = load_cleaned_dataset(INPUT_CSV)

    # 2. Separate ID, X, Y
    print("[2/7] Separating ID / features / targets …")
    ids   = df[ID_COL].reset_index(drop=True)
    X_raw = df[INPUT_FEATURES].reset_index(drop=True)
    Y     = df[TARGET_COLS].reset_index(drop=True)
    print(f"  X shape (raw) : {X_raw.shape}")
    print(f"  Y shape       : {Y.shape}")

    # 3. Train / test split (on raw data before encoding — prevents leakage)
    print("[3/7] Splitting into train/test sets …")
    (X_train_raw, X_test_raw,
     Y_train, Y_test,
     ids_train, ids_test) = train_test_split(
        X_raw, Y, ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"  Training rows : {len(X_train_raw)}")
    print(f"  Testing  rows : {len(X_test_raw)}")

    # 4. Build and fit preprocessor on training data only
    print("[4/7] Building and fitting preprocessor (train set only) …")
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train_raw)

    # 5. Apply fitted preprocessor to test data (no re-fitting)
    print("[5/7] Transforming test set with fitted preprocessor …")
    X_test_enc = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names_after_encoding(preprocessor)
    print(f"  Encoded feature count: {len(feature_names)}")

    # 6. Convert to DataFrames with meaningful column names
    X_train_df = pd.DataFrame(X_train_enc, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_enc,  columns=feature_names)

    # Re-attach Configuration_ID as first column for traceability
    X_train_df.insert(0, ID_COL, ids_train.values)
    X_test_df.insert(0,  ID_COL, ids_test.values)

    Y_train_df = Y_train.copy().reset_index(drop=True)
    Y_test_df  = Y_test.copy().reset_index(drop=True)
    Y_train_df.insert(0, ID_COL, ids_train.values)
    Y_test_df.insert(0,  ID_COL, ids_test.values)

    # 7. Save all outputs
    print("[6/7] Saving split datasets …")
    X_train_df.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test_df.to_csv( os.path.join(OUTPUT_DIR, "X_test.csv"),  index=False)
    Y_train_df.to_csv(os.path.join(OUTPUT_DIR, "Y_train.csv"), index=False)
    Y_test_df.to_csv( os.path.join(OUTPUT_DIR, "Y_test.csv"),  index=False)

    print("[7/7] Saving fitted preprocessor (Joblib) …")
    joblib.dump(preprocessor, PKL_FILE)
    print(f"  Saved: {os.path.relpath(PKL_FILE, BASE_DIR)}")

    # Write text reports
    write_encoding_details(preprocessor, feature_names)
    write_preprocessing_summary(
        n_clean=len(df),
        n_train=len(X_train_df),
        n_test=len(X_test_df),
        n_features_raw=len(INPUT_FEATURES),
        n_features_encoded=len(feature_names),
        feature_names=feature_names
    )

    print("\n" + "=" * 65)
    print("Preprocessing Summary")
    print("-" * 65)
    print(f"  Total records       : {len(df)}")
    print(f"  Training records    : {len(X_train_df)}")
    print(f"  Testing records     : {len(X_test_df)}")
    print(f"  Raw features        : {len(INPUT_FEATURES)}")
    print(f"  Encoded features    : {len(feature_names)}")
    print(f"  Targets             : {len(TARGET_COLS)}")
    print("=" * 65)
    print("\n[DONE] Feature engineering and train/test split complete.")


if __name__ == "__main__":
    main()
