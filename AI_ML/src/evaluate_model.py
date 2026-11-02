"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 2 | Script 2: Model Evaluation Utilities
File   : evaluate_model.py
Author : EMC-GUARD Development Team

This module provides standalone evaluation utilities that can be imported
by train_model.py or called independently to re-evaluate saved models
against a fresh test set without re-training.

Usage (standalone)
------------------
    python src/evaluate_model.py

    Loads the saved emc_surrogate_model.pkl and emc_preprocessor.pkl,
    evaluates on the test partition of EMC_GUARD_Physics_Dataset.csv,
    and prints a formatted report.

DISCLAIMER
----------
Results produced from the SYNTHETIC DUMMY DATASET are labelled
experimental and do NOT represent validated electromagnetic physics.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET       = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
MODEL_PKL     = os.path.join(BASE_DIR, "models",  "emc_surrogate_model.pkl")
PREPROC_PKL   = os.path.join(BASE_DIR, "models",  "emc_preprocessor.pkl")
METADATA_JSON = os.path.join(BASE_DIR, "models",  "target_model_metadata.json")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")

# Same column definitions as train_model.py
NUMERICAL_FEATURES   = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
]
CATEGORICAL_FEATURES = ["Shield_Type", "Cable_Configuration"]
INPUT_FEATURES       = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLS          = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]
RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_artefacts():
    """Load and return (preprocessor, surrogate_bundle, metadata)."""
    for path, label in [(MODEL_PKL, "surrogate model"),
                        (PREPROC_PKL, "preprocessor"),
                        (METADATA_JSON, "metadata")]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
            print("  Run train_model.py first.")
            sys.exit(1)

    preprocessor      = joblib.load(PREPROC_PKL)
    surrogate_bundle  = joblib.load(MODEL_PKL)
    with open(METADATA_JSON) as f:
        metadata = json.load(f)
    return preprocessor, surrogate_bundle, metadata


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, R2 rounded to 6 dp."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        "MAE":  round(float(mae),  6),
        "RMSE": round(float(rmse), 6),
        "R2":   round(float(r2),   6),
    }


def interpret_r2(r2: float) -> str:
    """Return a plain-English interpretation of an R2 value."""
    if r2 >= 0.9:
        return "Excellent fit (R2 >= 0.90)"
    elif r2 >= 0.7:
        return "Good fit (R2 0.70-0.89)"
    elif r2 >= 0.5:
        return "Moderate fit (R2 0.50-0.69)"
    elif r2 >= 0.0:
        return "Poor fit (R2 0.00-0.49) — model explains little variance"
    else:
        return "Negative R2 — model is worse than predicting the mean"


def print_separator(char: str = "-", width: int = 68):
    print(char * width)


# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def evaluate_saved_models(verbose: bool = True) -> dict:
    """
    Load saved artefacts, re-evaluate on the test split of the cleaned
    dataset, and return a results dictionary.

    Parameters
    ----------
    verbose : bool
        If True, print a formatted report to stdout.

    Returns
    -------
    dict
        {target: {"algorithm": str, "MAE": float, "RMSE": float, "R2": float,
                  "interpretation": str}}
    """
    if verbose:
        print("=" * 68)
        print("EMC-GUARD  |  Module 2  |  Standalone Model Evaluation")
        print("=" * 68)

    preprocessor, surrogate_bundle, metadata = load_artefacts()

    if not os.path.exists(DATASET):
        print(f"[ERROR] Dataset not found: {DATASET}")
        sys.exit(1)

    df = pd.read_csv(DATASET)
    active_targets = [t for t in TARGET_COLS if t in surrogate_bundle]

    results = {}
    rows    = []

    for tgt in active_targets:
        sub      = df.dropna(subset=[tgt])
        X_sub    = sub[INPUT_FEATURES].reset_index(drop=True)
        y_sub    = sub[tgt].reset_index(drop=True)
        _, Xte, _, yte = train_test_split(
            X_sub, y_sub, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        Xte_enc  = preprocessor.transform(Xte)

        bundle   = surrogate_bundle[tgt]
        algo     = bundle["algorithm"]
        model    = bundle["model"]
        y_pred   = model.predict(Xte_enc)
        metrics  = compute_metrics(np.array(yte), y_pred)
        interp   = interpret_r2(metrics["R2"])

        results[tgt] = {
            "algorithm":     algo,
            "MAE":           metrics["MAE"],
            "RMSE":          metrics["RMSE"],
            "R2":            metrics["R2"],
            "interpretation": interp,
        }
        rows.append({
            "Target": tgt, "Algorithm": algo,
            "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"],
        })

    if verbose:
        print("\n[DISCLAIMER] Results based on SYNTHETIC DUMMY DATA.")
        print("             Not validated electromagnetic physics.\n")
        print_separator("=")
        print(f"{'Target':<26} {'Algorithm':<16} {'MAE':>10} {'RMSE':>10} {'R2':>8}  Interpretation")
        print_separator()
        for tgt, r in results.items():
            print(f"{tgt:<26} {r['algorithm']:<16} {r['MAE']:>10.4f} "
                  f"{r['RMSE']:>10.4f} {r['R2']:>8.4f}  {r['interpretation']}")
        print_separator("=")
        print("\nR2 Interpretation Guide:")
        print("  >= 0.90  Excellent    |  0.70-0.89  Good    |  0.50-0.69  Moderate")
        print("   0.00-0.49  Poor      |  < 0.00  Worse than mean predictor")
        print()
        print("[DONE] Standalone evaluation complete.")

    return results


# ---------------------------------------------------------------------------
# CROSS-VALIDATION HELPER (optional deeper analysis)
# ---------------------------------------------------------------------------

def cross_validate_target(df: pd.DataFrame, target: str,
                           preprocessor, model,
                           cv: int = 5) -> dict:
    """
    Run k-fold cross-validation for a single target on the TRAINING portion
    of the dataset only. Returns mean and std of MAE, RMSE, R2.

    NOTE: This is a diagnostic utility for assessing stability.
    Do NOT use the test split inside this function.
    """
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline as SKPipeline

    sub  = df.dropna(subset=[target])
    X    = sub[INPUT_FEATURES].reset_index(drop=True)
    y    = sub[target].reset_index(drop=True)
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    maes, rmses, r2s = [], [], []

    for train_idx, val_idx in kf.split(Xtr):
        Xf = preprocessor.transform(Xtr.iloc[train_idx])
        Xv = preprocessor.transform(Xtr.iloc[val_idx])
        yf = ytr.iloc[train_idx]
        yv = ytr.iloc[val_idx]

        m = type(model)(**model.get_params())
        m.fit(Xf, yf)
        y_pred = m.predict(Xv)

        maes.append(mean_absolute_error(yv, y_pred))
        rmses.append(np.sqrt(mean_squared_error(yv, y_pred)))
        r2s.append(r2_score(yv, y_pred))

    return {
        "target": target,
        "cv_folds": cv,
        "MAE_mean":  round(float(np.mean(maes)),  6),
        "MAE_std":   round(float(np.std(maes)),   6),
        "RMSE_mean": round(float(np.mean(rmses)), 6),
        "RMSE_std":  round(float(np.std(rmses)),  6),
        "R2_mean":   round(float(np.mean(r2s)),   6),
        "R2_std":    round(float(np.std(r2s)),    6),
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = evaluate_saved_models(verbose=True)

