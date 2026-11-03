"""
feature_importance.py
---------------------
Feature Importance analysis for the KONE EMC-GUARD AI_ML system.

Analyses the Stage 14A Extra Trees models to identify the contribution
of each input feature to the model's predictions.

Methods used:
  1. ExtraTrees native impurity-based feature importance (mean decrease impurity)
  2. Permutation importance (sklearn) — more robust, model-agnostic measure

For capacitance:   PairSeparation_mm, PairPitch_mm, GroundClearance_mm
For mutual ind.:   PairSeparation_mm, PairPitch_mm, Overlap_mm

IMPORTANT: Feature importance describes model behaviour, NOT physical causation.
Correlation between feature importance rank and physical influence is plausible
but must not be stated as causal without physics-based justification.

DO NOT retrain or overwrite production models. This module is read-only.

Usage (from AI_ML/ directory):
    py src/feature_importance.py

Usage (from any directory):
    py C:/path/to/kone/AI_ML/src/feature_importance.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    CAP_FEATURES,
    CAP_JOBLIB_PATH,
    MUT_FEATURES,
    MUT_JOBLIB_PATH,
    AIML_REPORTS_DIR,
)
from dataset import load_dataset

warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)

import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAP_TARGET = "TotalCapacitance_pF_per_m"
MUT_TARGET = "AbsMutualInductance_nH"
CAP_UNIT   = "pF/m"
MUT_UNIT   = "nH"

PERM_IMPORTANCE_REPEATS = 30   # permutation repeats for stable estimates
PERM_IMPORTANCE_SEED    = 42


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> Tuple:
    """Load both Extra Trees models (read-only — no retraining)."""
    if not CAP_JOBLIB_PATH.exists():
        raise FileNotFoundError(f"Capacitance model not found: {CAP_JOBLIB_PATH}")
    if not MUT_JOBLIB_PATH.exists():
        raise FileNotFoundError(f"Mutual inductance model not found: {MUT_JOBLIB_PATH}")
    cap_model = joblib.load(CAP_JOBLIB_PATH)
    mut_model = joblib.load(MUT_JOBLIB_PATH)
    logger.info("Models loaded (read-only).")
    return cap_model, mut_model


# ---------------------------------------------------------------------------
# Model wrapper helper
# ---------------------------------------------------------------------------

def _get_inner_regressor(model):
    """
    Unwrap TransformedTargetRegressor to access the inner ExtraTreesRegressor.
    The wrapper's predict() works fine; feature_importances_ requires the inner.
    """
    if hasattr(model, "regressor_"):
        return model.regressor_
    if hasattr(model, "estimator"):
        return model.estimator
    return model


# ---------------------------------------------------------------------------
# Native impurity-based importance
# ---------------------------------------------------------------------------

def native_feature_importance(
    model,
    feature_names: List[str],
    target_name: str,
    unit: str,
) -> pd.DataFrame:
    """
    Extract mean decrease in impurity (MDI) importance from ExtraTrees.

    Each value is the normalised fraction of total impurity reduction
    attributable to that feature, averaged across all trees.
    Range: [0, 1], sums to 1.

    NOTE: MDI can overestimate importance of high-cardinality features.
    Use permutation importance for a more robust estimate.
    """
    importances = _get_inner_regressor(model).feature_importances_

    df_imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
        "Method":     "MDI (native ExtraTrees)",
        "Target":     target_name,
        "Unit":       unit,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    df_imp["Rank"] = range(1, len(df_imp) + 1)
    df_imp["Importance_Pct"] = df_imp["Importance"] * 100.0

    logger.info("  Native MDI importance for %s:", target_name)
    for _, row in df_imp.iterrows():
        logger.info("    Rank %d: %-24s  %.4f  (%.1f%%)",
                    row["Rank"], row["Feature"], row["Importance"], row["Importance_Pct"])

    return df_imp


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    target_name: str,
    unit: str,
    n_repeats: int = PERM_IMPORTANCE_REPEATS,
    seed: int = PERM_IMPORTANCE_SEED,
) -> pd.DataFrame:
    """
    Compute permutation feature importance (PFI).

    For each feature:
      1. Randomly shuffle that feature column across all samples.
      2. Measure the drop in R² score.
      3. Average over n_repeats shuffles.

    A large drop → the feature is important.
    A near-zero drop → the model barely uses that feature.

    NOTE: PFI on training data measures reliance (possible overfitting).
    On a deterministic dataset with no hold-out set, this is the best
    available approximation.
    """
    logger.info("  Computing permutation importance for %s (%d repeats) …",
                target_name, n_repeats)

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="r2",
        n_jobs=-1,
    )

    df_pfi = pd.DataFrame({
        "Feature":              feature_names,
        "Importance_Mean":      result.importances_mean,
        "Importance_Std":       result.importances_std,
        "Importance_Mean_Pct":  result.importances_mean / result.importances_mean.sum() * 100
                                if result.importances_mean.sum() > 0 else np.zeros(len(feature_names)),
        "Method":               "Permutation Importance (R²-drop)",
        "Target":               target_name,
        "Unit":                 unit,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

    df_pfi["Rank"] = range(1, len(df_pfi) + 1)

    logger.info("  Permutation importance for %s:", target_name)
    for _, row in df_pfi.iterrows():
        logger.info("    Rank %d: %-24s  mean=%.6g ± %.6g",
                    row["Rank"], row["Feature"],
                    row["Importance_Mean"], row["Importance_Std"])

    return df_pfi


# ---------------------------------------------------------------------------
# Full feature importance analysis
# ---------------------------------------------------------------------------

def analyse_feature_importance(
    df: pd.DataFrame,
    cap_model,
    mut_model,
) -> Dict:
    """
    Run both MDI and permutation importance for both targets.

    Returns a dict with:
      capacitance:
        mdi_df, pfi_df, combined_df
      mutual_inductance:
        mdi_df, pfi_df, combined_df
    """
    logger.info("=" * 60)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 60)

    results = {}

    for target_name, target_col, features, model, unit in [
        (CAP_TARGET, CAP_TARGET, CAP_FEATURES, cap_model, CAP_UNIT),
        (MUT_TARGET, MUT_TARGET, MUT_FEATURES, mut_model, MUT_UNIT),
    ]:
        target_key = "capacitance" if target_col == CAP_TARGET else "mutual_inductance"
        logger.info("\nTarget: %s [%s]", target_col, unit)

        X = df[features].values
        y = df[target_col].values

        # 1. Native MDI
        mdi_df = native_feature_importance(model, features, target_name, unit)

        # 2. Permutation importance
        pfi_df = compute_permutation_importance(model, X, y, features, target_name, unit)

        # 3. Combined ranked table
        combined = mdi_df[["Feature", "Importance", "Importance_Pct"]].copy()
        combined.columns = ["Feature", "MDI_Importance", "MDI_Pct"]
        pfi_sub = pfi_df[["Feature", "Importance_Mean", "Importance_Std", "Importance_Mean_Pct"]].copy()
        pfi_sub.columns = ["Feature", "PFI_Mean", "PFI_Std", "PFI_Pct"]
        combined = combined.merge(pfi_sub, on="Feature")
        combined["MDI_Rank"] = combined["MDI_Importance"].rank(ascending=False).astype(int)
        combined["PFI_Rank"] = combined["PFI_Mean"].rank(ascending=False).astype(int)
        combined["Target"] = target_name
        combined["Unit"]   = unit

        results[target_key] = {
            "target_name": target_name,
            "features":    features,
            "unit":        unit,
            "mdi_df":      mdi_df,
            "pfi_df":      pfi_df,
            "combined_df": combined,
        }

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_importance_results(
    importance: Dict,
    out_dir: Path,
) -> Dict:
    """Save feature importance tables to CSV and summary to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved = {}

    for target_key, data in importance.items():
        # MDI
        mdi_fname = out_dir / f"feature_importance_mdi_{target_key}_{timestamp}.csv"
        data["mdi_df"].to_csv(mdi_fname, index=False)
        saved[f"mdi_{target_key}"] = str(mdi_fname)
        logger.info("  Saved: %s", mdi_fname.name)

        # PFI
        pfi_fname = out_dir / f"feature_importance_pfi_{target_key}_{timestamp}.csv"
        data["pfi_df"].to_csv(pfi_fname, index=False)
        saved[f"pfi_{target_key}"] = str(pfi_fname)
        logger.info("  Saved: %s", pfi_fname.name)

        # Combined
        comb_fname = out_dir / f"feature_importance_combined_{target_key}_{timestamp}.csv"
        data["combined_df"].to_csv(comb_fname, index=False)
        saved[f"combined_{target_key}"] = str(comb_fname)
        logger.info("  Saved: %s", comb_fname.name)

    # JSON summary
    summary = {
        "generated_at": timestamp,
        "module": "feature_importance.py",
        "methods": {
            "MDI": (
                "Mean Decrease Impurity — native ExtraTrees importance. "
                "Fast and consistent but can overestimate high-cardinality features."
            ),
            "PFI": (
                "Permutation Feature Importance — R² score drop after column shuffle. "
                "More robust; uses training set (no independent test set available)."
            ),
        },
        "disclaimer": (
            "Feature importance describes model behaviour, not physical causation. "
            "Correlation between model importance rank and physical influence is plausible "
            "but is not proven by these results alone."
        ),
        "results": {
            tkey: {
                "target": data["target_name"],
                "unit": data["unit"],
                "features": data["features"],
                "mdi_ranking": data["mdi_df"][["Rank", "Feature", "Importance_Pct"]].to_dict(orient="records"),
                "pfi_ranking": data["pfi_df"][["Rank", "Feature", "Importance_Mean", "Importance_Std", "Importance_Mean_Pct"]].to_dict(orient="records"),
            }
            for tkey, data in importance.items()
        },
        "files_saved": saved,
    }

    json_fname = out_dir / f"feature_importance_summary_{timestamp}.json"
    with open(json_fname, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    saved["summary_json"] = str(json_fname)
    logger.info("  Saved: %s", json_fname.name)

    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_importance(save_results: bool = True) -> Dict:
    """Execute the full feature importance analysis pipeline."""
    logger.info("KONE EMC-GUARD — Feature Importance Analysis")

    df = load_dataset()
    cap_model, mut_model = _load_models()
    importance = analyse_feature_importance(df, cap_model, mut_model)

    files_saved = {}
    if save_results:
        files_saved = save_importance_results(importance, AIML_REPORTS_DIR)

    importance["files_saved"] = files_saved
    return importance


if __name__ == "__main__":
    import sys as _sys
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() != 'utf-8':
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    results = run_feature_importance(save_results=True)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)

    for tkey, label in [
        ("capacitance",      "Capacitance (pF/m)"),
        ("mutual_inductance", "Mutual Inductance (nH)"),
    ]:
        data = results[tkey]
        print(f"\n{label}")
        print(f"  Features: {data['features']}")
        print("\n  MDI (Mean Decrease Impurity):")
        for _, row in data["mdi_df"].iterrows():
            bar = "#" * int(row["Importance_Pct"] / 2)
            print(f"    [{row['Rank']}] {row['Feature']:<24}  {row['Importance_Pct']:5.1f}%  {bar}")
        print("\n  Permutation Importance (R2-drop):")
        for _, row in data["pfi_df"].iterrows():
            print(f"    [{row['Rank']}] {row['Feature']:<24}  {row['Importance_Mean']:+.6f} +/- {row['Importance_Std']:.6f}  ({row['Importance_Mean_Pct']:.1f}%)")

    if "files_saved" in results:
        print("\nFiles saved:")
        for k, v in results["files_saved"].items():
            print(f"  {k}: {Path(v).name}")

