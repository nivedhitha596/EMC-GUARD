"""
evaluate.py
-----------
Model Evaluation module for the KONE EMC-GUARD AI_ML system.

Evaluates the Stage 14A Extra Trees surrogate models using:
  - In-sample reproduction metrics (clearly labelled as such)
  - Leave-One-Level-Out (LOLO) cross-validation for each input feature
  - Residual analysis and error distribution
  - Separate evaluation for capacitance and mutual inductance targets

IMPORTANT SCIENTIFIC CONTEXT
-----------------------------
The dataset contains 216 deterministic MATLAB analytical configurations,
not experimental measurements. The MATLAB outputs are computed from physics
equations and are therefore deterministic (no measurement noise).

In-sample R² ≈ 1.0 is expected and correct — the Extra Trees models
reproduce MATLAB-computed values. This does NOT imply experimental
validation, real-world accuracy, or safety certification.

LOLO validation reveals true generalisation capability: how well the
model predicts configurations at unseen feature levels.

DO NOT retrain or overwrite production models. This module is read-only.

Usage (from AI_ML/ directory):
    py src/evaluate.py

Usage (from any directory):
    py C:/path/to/kone/AI_ML/src/evaluate.py
"""

from __future__ import annotations

import json
import logging
import sys
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Ensure AI_ML/src is on the Python path
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
    SUPPORTED_SEPARATIONS,
    SUPPORTED_PITCHES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_CLEARANCES,
)
from dataset import load_dataset

# Suppress sklearn version mismatch warnings (1.8.0 models loaded under 1.9.1)
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
CAP_TARGET  = "TotalCapacitance_pF_per_m"
MUT_TARGET  = "AbsMutualInductance_nH"
CAP_UNIT    = "pF/m"
MUT_UNIT    = "nH"

LOLO_FEATURES = {
    CAP_TARGET: {
        "PairSeparation_mm":  SUPPORTED_SEPARATIONS,
        "PairPitch_mm":       SUPPORTED_PITCHES,
        "GroundClearance_mm": SUPPORTED_CLEARANCES,
    },
    MUT_TARGET: {
        "PairSeparation_mm": SUPPORTED_SEPARATIONS,
        "PairPitch_mm":      SUPPORTED_PITCHES,
        "Overlap_mm":        SUPPORTED_OVERLAPS,
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> Tuple:
    """Load both Extra Trees models. Does NOT retrain."""
    logger.info("Loading Extra Trees models …")
    if not CAP_JOBLIB_PATH.exists():
        raise FileNotFoundError(f"Capacitance model not found: {CAP_JOBLIB_PATH}")
    if not MUT_JOBLIB_PATH.exists():
        raise FileNotFoundError(f"Mutual inductance model not found: {MUT_JOBLIB_PATH}")

    cap_model = joblib.load(CAP_JOBLIB_PATH)
    mut_model = joblib.load(MUT_JOBLIB_PATH)
    logger.info("  Capacitance model    : %s", CAP_JOBLIB_PATH.name)
    logger.info("  Mutual inductance model: %s", MUT_JOBLIB_PATH.name)
    return cap_model, mut_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict:
    """Compute MAE, MSE, RMSE, R² for a set of predictions."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))

    if label:
        logger.info(
            "  [%s]  MAE=%.6g  RMSE=%.6g  R²=%.8f  n=%d",
            label, mae, rmse, r2, len(y_true),
        )
    return {"label": label, "n": int(len(y_true)),
            "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# In-sample evaluation (clearly labelled)
# ---------------------------------------------------------------------------

def in_sample_evaluation(
    df: pd.DataFrame,
    cap_model,
    mut_model,
) -> Dict:
    """
    Compute in-sample reproduction metrics.

    WARNING: These are NOT independent test metrics. The Extra Trees
    models were trained on all 216 configurations. In-sample R² ≈ 1.0
    is expected because the dataset is deterministic. These results
    reflect how well the model memorises MATLAB outputs, not how well
    it generalises.
    """
    logger.info("--- In-sample evaluation (training set reproduction) ---")
    logger.info("NOTE: These are IN-SAMPLE results, not independent test performance.")

    X_cap = df[CAP_FEATURES].values
    X_mut = df[MUT_FEATURES].values
    y_cap = df[CAP_TARGET].values
    y_mut = df[MUT_TARGET].values

    cap_pred = cap_model.predict(X_cap)
    mut_pred = mut_model.predict(X_mut)

    cap_metrics = _compute_metrics(y_cap, cap_pred, "Cap in-sample")
    mut_metrics = _compute_metrics(y_mut, mut_pred, "Mut in-sample")

    # Build detailed comparison table
    comparison = df[["Config_ID"] + CAP_FEATURES + [CAP_TARGET]].copy()
    comparison["Cap_Predicted"] = cap_pred
    comparison["Cap_Residual"]  = y_cap - cap_pred
    comparison["Cap_AbsError"]  = np.abs(y_cap - cap_pred)

    mut_comparison = df[["Config_ID"] + MUT_FEATURES + [MUT_TARGET]].copy()
    mut_comparison["Mut_Predicted"] = mut_pred
    mut_comparison["Mut_Residual"]  = y_mut - mut_pred
    mut_comparison["Mut_AbsError"]  = np.abs(y_mut - mut_pred)

    return {
        "evaluation_type": "in_sample",
        "warning": (
            "In-sample results. Dataset is deterministic MATLAB analytical output. "
            "R² near 1.0 is expected and does NOT imply experimental validation."
        ),
        "capacitance": {
            "metrics": cap_metrics,
            "unit": CAP_UNIT,
            "comparison_df": comparison,
        },
        "mutual_inductance": {
            "metrics": mut_metrics,
            "unit": MUT_UNIT,
            "comparison_df": mut_comparison,
        },
    }


# ---------------------------------------------------------------------------
# LOLO cross-validation
# ---------------------------------------------------------------------------

def lolo_validation(
    df: pd.DataFrame,
    cap_model,
    mut_model,
) -> Dict:
    """
    Leave-One-Level-Out (LOLO) cross-validation.

    For each feature and each unique level of that feature:
      1. Hold out all rows where the feature equals that level.
      2. Train a NEW temporary model on the remaining rows.
      3. Predict the held-out rows.
      4. Record metrics.

    This reveals generalisation to unseen feature levels, which is
    the most relevant validation for this discrete-grid dataset.

    NOTE: Temporary models are trained in memory only and are discarded
    immediately. The production models are never overwritten.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    logger.info("--- LOLO cross-validation ---")
    logger.info("Training temporary models for each held-out level (not overwriting production).")

    # Re-read the production model hyperparameters for reproducibility
    cap_params = _get_model_params(cap_model)
    mut_params = _get_model_params(mut_model)

    all_results = []

    for target, feature_map in LOLO_FEATURES.items():
        model_params = cap_params if target == CAP_TARGET else mut_params
        features     = CAP_FEATURES if target == CAP_TARGET else MUT_FEATURES
        y_col        = target
        unit         = CAP_UNIT if target == CAP_TARGET else MUT_UNIT

        logger.info("  Target: %s (%s)", target, unit)

        for feat, levels in feature_map.items():
            fold_metrics = []
            fold_details = []

            logger.info("    LOLO feature: %s  (%d levels)", feat, len(levels))

            for level in levels:
                # Split
                test_mask  = np.isclose(df[feat].values, level, atol=1e-6)
                train_mask = ~test_mask

                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    logger.warning("      Skipping level %s=%s (empty split)", feat, level)
                    continue

                X_train = df.loc[train_mask, features].values
                y_train = df.loc[train_mask, y_col].values
                X_test  = df.loc[test_mask,  features].values
                y_test  = df.loc[test_mask,  y_col].values

                # Baseline: predict mean of training set
                baseline_pred = np.full_like(y_test, y_train.mean())
                baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))

                # Fit temporary model (discarded after this fold)
                tmp_model = ExtraTreesRegressor(**model_params)
                tmp_model.fit(X_train, y_train)
                y_pred = tmp_model.predict(X_test)
                del tmp_model  # explicitly discard

                m = _compute_metrics(y_test, y_pred,
                                     label=f"{target[:3]}|{feat}={level}")
                m["held_out_feature"]  = feat
                m["held_out_level"]    = float(level)
                m["n_train"]           = int(train_mask.sum())
                m["n_test"]            = int(test_mask.sum())
                m["baseline_rmse"]     = baseline_rmse
                m["rmse_improvement_pct"] = (
                    (baseline_rmse - m["RMSE"]) / baseline_rmse * 100
                    if baseline_rmse > 0 else 0.0
                )
                m["target"] = target
                m["unit"]   = unit

                fold_metrics.append(m)
                fold_details.append({
                    "held_out_level": float(level),
                    "y_true":  y_test.tolist(),
                    "y_pred":  y_pred.tolist(),
                    "residuals": (y_test - y_pred).tolist(),
                })

            # Aggregate across all levels of this feature
            if fold_metrics:
                mean_mae  = float(np.mean([f["MAE"]  for f in fold_metrics]))
                mean_rmse = float(np.mean([f["RMSE"] for f in fold_metrics]))
                mean_r2   = float(np.mean([f["R2"]   for f in fold_metrics]))
                mean_imp  = float(np.mean([f["rmse_improvement_pct"] for f in fold_metrics]))

                all_results.append({
                    "target":            target,
                    "unit":              unit,
                    "held_out_feature":  feat,
                    "n_levels":          len(fold_metrics),
                    "mean_MAE":          mean_mae,
                    "mean_RMSE":         mean_rmse,
                    "mean_R2":           mean_r2,
                    "mean_rmse_imp_pct": mean_imp,
                    "per_level":         fold_metrics,
                    "fold_details":      fold_details,
                })
                logger.info(
                    "    → Mean RMSE=%.4g  Mean R²=%.4f  RMSE-imp=%.1f%%",
                    mean_rmse, mean_r2, mean_imp,
                )

    return {"lolo_results": all_results}


def _get_inner_regressor(model):
    """
    Unwrap TransformedTargetRegressor or similar sklearn wrappers.
    Returns the inner ExtraTreesRegressor.
    """
    if hasattr(model, "regressor_"):
        return model.regressor_
    if hasattr(model, "estimator"):
        return model.estimator
    return model


def _get_model_params(model) -> Dict:
    """Extract key hyperparameters from an ExtraTrees model (handles wrapper)."""
    inner = _get_inner_regressor(model)
    try:
        return {
            "n_estimators":      inner.n_estimators,
            "max_features":      inner.max_features,
            "min_samples_split": inner.min_samples_split,
            "min_samples_leaf":  inner.min_samples_leaf,
            "random_state":      42,  # reproducible LOLO folds
            "n_jobs":            -1,
        }
    except AttributeError:
        return {"n_estimators": 300, "random_state": 42, "n_jobs": -1}


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def residual_analysis(
    in_sample: Dict,
    lolo: Dict,
) -> Dict:
    """
    Compile residual statistics for both targets.

    Returns a structured dict describing:
      - In-sample residual distribution (mean, std, skewness, kurtosis)
      - LOLO aggregated residuals per feature per target
    """
    results = {}

    for target_key, data in [
        ("capacitance",      in_sample["capacitance"]),
        ("mutual_inductance", in_sample["mutual_inductance"]),
    ]:
        comp_df = data["comparison_df"]
        res_col = "Cap_Residual" if target_key == "capacitance" else "Mut_Residual"
        residuals = comp_df[res_col].values

        results[target_key] = {
            "in_sample_residuals": {
                "mean":      float(np.mean(residuals)),
                "std":       float(np.std(residuals)),
                "min":       float(np.min(residuals)),
                "max":       float(np.max(residuals)),
                "skewness":  float(_safe_skewness(residuals)),
                "kurtosis":  float(_safe_kurtosis(residuals)),
                "values":    residuals.tolist(),
            },
        }

    # LOLO residuals by feature
    for entry in lolo["lolo_results"]:
        target = entry["target"]
        feat   = entry["held_out_feature"]
        tkey   = "capacitance" if target == CAP_TARGET else "mutual_inductance"

        all_lolo_res = []
        for fold in entry["fold_details"]:
            all_lolo_res.extend(fold["residuals"])

        all_lolo_res = np.array(all_lolo_res)
        results.setdefault(tkey, {}).setdefault("lolo_residuals", {})[feat] = {
            "mean":  float(np.mean(all_lolo_res)),
            "std":   float(np.std(all_lolo_res)),
            "min":   float(np.min(all_lolo_res)),
            "max":   float(np.max(all_lolo_res)),
            "values": all_lolo_res.tolist(),
        }

    return results


def _safe_skewness(x: np.ndarray) -> float:
    from scipy.stats import skew
    try:
        return float(skew(x))
    except Exception:
        return float("nan")


def _safe_kurtosis(x: np.ndarray) -> float:
    from scipy.stats import kurtosis
    try:
        return float(kurtosis(x))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Save results to CSV / JSON
# ---------------------------------------------------------------------------

def save_evaluation_results(
    in_sample: Dict,
    lolo: Dict,
    residuals: Dict,
    out_dir: Path,
) -> Dict:
    """Save all evaluation results to CSV and JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved = {}

    # 1. In-sample comparison tables
    for target_key, data in [
        ("capacitance",      in_sample["capacitance"]),
        ("mutual_inductance", in_sample["mutual_inductance"]),
    ]:
        fname = out_dir / f"insample_{target_key}_{timestamp}.csv"
        data["comparison_df"].to_csv(fname, index=False)
        saved[f"insample_{target_key}"] = str(fname)
        logger.info("  Saved: %s", fname.name)

    # 2. LOLO summary CSV
    lolo_rows = []
    for entry in lolo["lolo_results"]:
        for level_m in entry["per_level"]:
            lolo_rows.append({
                "Target":             entry["target"],
                "Unit":               entry["unit"],
                "HeldOutFeature":     entry["held_out_feature"],
                "HeldOutLevel":       level_m["held_out_level"],
                "n_train":            level_m["n_train"],
                "n_test":             level_m["n_test"],
                "MAE":                level_m["MAE"],
                "MSE":                level_m["MSE"],
                "RMSE":               level_m["RMSE"],
                "R2":                 level_m["R2"],
                "Baseline_RMSE":      level_m["baseline_rmse"],
                "RMSE_Improvement_%": level_m["rmse_improvement_pct"],
            })
    lolo_df = pd.DataFrame(lolo_rows)
    lolo_fname = out_dir / f"lolo_validation_{timestamp}.csv"
    lolo_df.to_csv(lolo_fname, index=False)
    saved["lolo_validation"] = str(lolo_fname)
    logger.info("  Saved: %s", lolo_fname.name)

    # 3. LOLO aggregate summary CSV
    agg_rows = []
    for entry in lolo["lolo_results"]:
        agg_rows.append({
            "Target":           entry["target"],
            "Unit":             entry["unit"],
            "HeldOutFeature":   entry["held_out_feature"],
            "N_Levels":         entry["n_levels"],
            "Mean_MAE":         entry["mean_MAE"],
            "Mean_RMSE":        entry["mean_RMSE"],
            "Mean_R2":          entry["mean_R2"],
            "Mean_RMSE_Imp_%":  entry["mean_rmse_imp_pct"],
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_fname = out_dir / f"lolo_aggregate_{timestamp}.csv"
    agg_df.to_csv(agg_fname, index=False)
    saved["lolo_aggregate"] = str(agg_fname)
    logger.info("  Saved: %s", agg_fname.name)

    # 4. Summary JSON
    summary = {
        "generated_at": timestamp,
        "module": "evaluate.py",
        "dataset_source": "MATLAB 216-configuration physics dataset (Stage 13A)",
        "warning": (
            "All results are based on deterministic MATLAB analytical data. "
            "In-sample R²≈1.0 is expected. LOLO results reveal generalisation. "
            "No experimental validation, real-world field accuracy, or safety "
            "certification is implied."
        ),
        "in_sample": {
            "capacitance": {
                **in_sample["capacitance"]["metrics"],
                "unit": in_sample["capacitance"]["unit"],
            },
            "mutual_inductance": {
                **in_sample["mutual_inductance"]["metrics"],
                "unit": in_sample["mutual_inductance"]["unit"],
            },
        },
        "lolo_aggregate": agg_rows,
        "files_saved": saved,
    }
    json_fname = out_dir / f"evaluation_summary_{timestamp}.json"
    with open(json_fname, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    saved["summary_json"] = str(json_fname)
    logger.info("  Saved: %s", json_fname.name)

    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evaluation(save_results: bool = True) -> Dict:
    """
    Execute the full model evaluation pipeline.

    Returns a dict containing:
      - in_sample   : in-sample metrics and comparison tables
      - lolo        : LOLO validation results per feature per target
      - residuals   : residual distribution statistics
      - files_saved : paths to all output files
    """
    logger.info("=" * 60)
    logger.info("KONE EMC-GUARD — Model Evaluation")
    logger.info("Stage 14A Extra Trees surrogate models")
    logger.info("=" * 60)

    df            = load_dataset()
    cap_model, mut_model = _load_models()

    in_sample = in_sample_evaluation(df, cap_model, mut_model)
    lolo      = lolo_validation(df, cap_model, mut_model)
    residuals = residual_analysis(in_sample, lolo)

    files_saved = {}
    if save_results:
        files_saved = save_evaluation_results(
            in_sample, lolo, residuals,
            AIML_REPORTS_DIR,
        )

    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("=" * 60)

    return {
        "in_sample":   in_sample,
        "lolo":        lolo,
        "residuals":   residuals,
        "files_saved": files_saved,
    }


if __name__ == "__main__":
    import sys as _sys
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() != 'utf-8':
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    results = run_evaluation(save_results=True)

    # Print compact summary to terminal
    print("\n" + "=" * 60)
    print("IN-SAMPLE METRICS (MATLAB dataset reproduction)")
    print("WARNING: Not independent test performance")
    print("=" * 60)
    for tname, tkey in [("Capacitance (pF/m)", "capacitance"),
                         ("Mutual Inductance (nH)", "mutual_inductance")]:
        m = results["in_sample"][tkey]["metrics"]
        print(f"\n  {tname}")
        print(f"    MAE  = {m['MAE']:.6g}")
        print(f"    RMSE = {m['RMSE']:.6g}")
        print(f"    R²   = {m['R2']:.8f}")

    print("\n" + "=" * 60)
    print("LOLO VALIDATION SUMMARY")
    print("=" * 60)
    for entry in results["lolo"]["lolo_results"]:
        print(
            f"\n  Target={entry['target'][:20]:<20}  "
            f"Feature={entry['held_out_feature']:<22}  "
            f"MeanR²={entry['mean_R2']:+.4f}  "
            f"MeanRMSE={entry['mean_RMSE']:.4g}  "
            f"ImpvBaseline={entry['mean_rmse_imp_pct']:.1f}%"
        )

    if results["files_saved"]:
        print("\nFiles saved:")
        for k, v in results["files_saved"].items():
            print(f"  {k}: {Path(v).name}")

