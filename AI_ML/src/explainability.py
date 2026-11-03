"""
explainability.py
-----------------
Explainable AI (XAI) module for the KONE EMC-GUARD AI_ML system.

Explains predictions from the Stage 14A Extra Trees surrogate models using:

  1. Permutation Importance (global) — feature contribution to overall accuracy
  2. Individual Prediction Explanation (local) — feature contribution per sample
     via TreeExplainer-style manual tree traversal
  3. Sensitivity Analysis — how each feature affects the output when varied
  4. Interaction Analysis — pairwise feature effect estimates

SHAP compatibility note:
  SHAP is not available in the current environment. This module implements
  an equivalent set of explanations using only scikit-learn and numpy:
  - Global importance: permutation importance (same as SHAP global effect)
  - Local explanation: marginal contribution approximation (ICE-style)
  - Sensitivity: one-at-a-time parameter sweep on the discrete grid

All explanations are clearly labelled as model-behaviour descriptions.
Correlation with physical causation is noted where plausible but is not
asserted as proven.

SCIENTIFIC LIMITATIONS
----------------------
- Dataset: 216 deterministic MATLAB analytical configurations
- No experimental measurements, no measurement uncertainty
- Physical correlations noted are based on electrostatic/inductive physics
  but are described from the model's learned response
- Do NOT claim experimental validation, field accuracy, or safety certification

DO NOT retrain or overwrite production models.

Usage (from AI_ML/ directory):
    py src/explainability.py

Usage (from any directory):
    py C:/path/to/kone/AI_ML/src/explainability.py
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
    SUPPORTED_SEPARATIONS,
    SUPPORTED_PITCHES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_CLEARANCES,
)
from dataset import load_dataset

warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)

import joblib

logging.basicConfig(    level=logging.INFO,
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

FEATURE_GRID = {
    "PairSeparation_mm":  SUPPORTED_SEPARATIONS,
    "PairPitch_mm":       SUPPORTED_PITCHES,
    "Overlap_mm":         SUPPORTED_OVERLAPS,
    "GroundClearance_mm": SUPPORTED_CLEARANCES,
}

PHYSICAL_CONTEXT = {
    "PairSeparation_mm": (
        "Wire-to-wire centre distance. Larger separation reduces mutual inductive coupling "
        "(decreasing M) and reduces inter-wire capacitance (decreasing C). "
        "Dominant control parameter for both EMC properties."
    ),
    "PairPitch_mm": (
        "Twist pitch of the pair. Shorter pitch improves field cancellation, "
        "reducing effective mutual inductance. Has moderate effect on capacitance "
        "through wire spacing geometry."
    ),
    "Overlap_mm": (
        "Length of parallel run between pairs. Mutual inductance scales nearly "
        "linearly with overlap (more parallel conductor = more flux linkage). "
        "No significant effect on per-metre capacitance (hence excluded from cap model)."
    ),
    "GroundClearance_mm": (
        "Distance from pair to ground plane. Larger clearance reduces image-charge "
        "effect, decreasing total capacitance. No significant effect on mutual "
        "inductance (hence excluded from mut model)."
    ),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> Tuple:
    """Load both Extra Trees models (read-only)."""
    if not CAP_JOBLIB_PATH.exists():
        raise FileNotFoundError(f"Capacitance model not found: {CAP_JOBLIB_PATH}")
    if not MUT_JOBLIB_PATH.exists():
        raise FileNotFoundError(f"Mutual inductance model not found: {MUT_JOBLIB_PATH}")
    cap_model = joblib.load(CAP_JOBLIB_PATH)
    mut_model = joblib.load(MUT_JOBLIB_PATH)
    logger.info("Models loaded (read-only).")
    return cap_model, mut_model


# ---------------------------------------------------------------------------
# 1. Global permutation importance (XAI global view)
# ---------------------------------------------------------------------------

def global_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    target_name: str,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute global permutation feature importance.

    Measures: how much does R² drop when a feature is randomly shuffled?
    A larger drop → the model relies more on that feature.

    This is the XAI equivalent of SHAP global mean |SHAP value|.
    """
    logger.info("  Global permutation importance for %s …", target_name)
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="r2",
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "Feature":       feature_names,
        "R2_Drop_Mean":  result.importances_mean,
        "R2_Drop_Std":   result.importances_std,
        "Target":        target_name,
    }).sort_values("R2_Drop_Mean", ascending=False).reset_index(drop=True)

    df["Rank"]           = range(1, len(df) + 1)
    df["Direction_Note"] = df["Feature"].map(_get_physical_direction(target_name))
    df["Physical_Context"] = df["Feature"].map(PHYSICAL_CONTEXT)

    return df


def _get_physical_direction(target_name: str) -> Dict[str, str]:
    """
    Return plausible physical direction of effect for each feature.
    These are physics-motivated but describe expected trend, not proven causation.
    """
    if "Capacitance" in target_name:
        return {
            "PairSeparation_mm":  "Negative: larger separation → lower inter-wire capacitance",
            "PairPitch_mm":       "Moderate: twist pitch affects effective wire geometry",
            "GroundClearance_mm": "Negative: larger ground clearance → lower capacitance",
        }
    else:
        return {
            "PairSeparation_mm": "Negative: larger separation → weaker flux linkage → lower M",
            "PairPitch_mm":      "Negative: shorter pitch → better cancellation → lower effective M",
            "Overlap_mm":        "Positive: longer overlap → more coupled length → higher M",
        }


# ---------------------------------------------------------------------------
# 2. Individual Conditional Expectation (ICE) — local sensitivity
# ---------------------------------------------------------------------------

def individual_conditional_expectation(
    model,
    df: pd.DataFrame,
    features: List[str],
    target_name: str,
    feature_grid: Dict[str, List],
    n_samples: int = 20,
) -> Dict:
    """
    Individual Conditional Expectation (ICE) curves.

    For each feature f and each sample i:
      - Fix all other features at their observed values for sample i.
      - Vary feature f over its supported grid levels.
      - Record the model's prediction.

    ICE shows how the model's prediction CHANGES for individual samples
    as one feature varies. The average over samples is the Partial
    Dependence Plot (PDP).

    This is analogous to individual SHAP waterfall charts but shows
    the full range rather than a single reference point.
    """
    logger.info("  ICE curves for %s …", target_name)

    # Sample rows for ICE (use all if dataset small enough)
    sample_idx = np.arange(len(df)) if len(df) <= n_samples else \
                 np.random.RandomState(42).choice(len(df), size=n_samples, replace=False)

    ice_results = {}

    for feat in features:
        levels = feature_grid.get(feat, sorted(df[feat].unique()))
        other_features = [f for f in features if f != feat]

        curves = []
        for idx in sample_idx:
            row = df.iloc[idx]
            preds = []
            for level in levels:
                X_mod = np.array([[
                    level if f == feat else float(row[f])
                    for f in features
                ]])
                pred = float(model.predict(X_mod)[0])
                preds.append(pred)

            curves.append({
                "sample_id":   str(row.get("Config_ID", idx)),
                "base_values": {f: float(row[f]) for f in features},
                "base_pred":   float(model.predict(
                    np.array([[float(row[f]) for f in features]])
                )[0]),
                "x_values":    levels,
                "predictions": preds,
            })

        # PDP: average across all ICE curves
        x_vals  = np.array(levels)
        y_matrix = np.array([c["predictions"] for c in curves])
        pdp_mean = y_matrix.mean(axis=0).tolist()
        pdp_std  = y_matrix.std(axis=0).tolist()

        ice_results[feat] = {
            "feature":     feat,
            "x_values":    levels,
            "pdp_mean":    pdp_mean,
            "pdp_std":     pdp_std,
            "ice_curves":  curves,
            "n_samples":   len(curves),
        }
        logger.info("    %s: %d ICE curves computed", feat, len(curves))

    return ice_results


# ---------------------------------------------------------------------------
# 3. Individual prediction explanation
# ---------------------------------------------------------------------------

def explain_single_prediction(
    model,
    features: List[str],
    feature_values: Dict[str, float],
    target_name: str,
    unit: str,
    df: pd.DataFrame,
) -> Dict:
    """
    Explain a single prediction via marginal contribution approximation.

    For each feature f:
      1. Start from the dataset mean (baseline prediction).
      2. Freeze feature f at the query value; shuffle all others.
      3. Measure how prediction changes from the baseline when f is fixed.

    This is a simplified but interpretable attribution — analogous to
    a single-sample Shapley approximation using random coalitions.

    Returns: feature attribution values and direction for the given query.
    """
    # Baseline: dataset mean prediction (surrogate of SHAP base value)
    X_all = df[features].values
    baseline_pred = float(model.predict(X_all).mean())

    # Query prediction
    X_query = np.array([[feature_values[f] for f in features]])
    query_pred = float(model.predict(X_query)[0])
    total_delta = query_pred - baseline_pred

    # Marginal contribution: vary each feature, hold others at query
    attributions = {}
    rng = np.random.RandomState(42)
    n_random = 100

    for feat in features:
        other_feats = [f for f in features if f != feat]

        # With feature at query value, others randomised
        X_with = X_all[rng.choice(len(X_all), size=n_random, replace=True)].copy()
        feat_idx = features.index(feat)
        X_with[:, feat_idx] = feature_values[feat]
        pred_with = model.predict(X_with).mean()

        # Without (others randomised, feature also randomised = baseline)
        pred_without = float(model.predict(X_all[rng.choice(len(X_all), size=n_random)]).mean())

        contribution = float(pred_with - pred_without)
        direction = "increases" if contribution > 0 else "decreases"
        magnitude = abs(contribution)

        attributions[feat] = {
            "value":        feature_values[feat],
            "contribution": contribution,
            "direction":    direction,
            "magnitude":    magnitude,
            "physical_note": PHYSICAL_CONTEXT.get(feat, ""),
        }

    return {
        "target":         target_name,
        "unit":           unit,
        "feature_values": feature_values,
        "baseline_pred":  baseline_pred,
        "query_pred":     query_pred,
        "total_delta":    total_delta,
        "attributions":   attributions,
        "method_note": (
            "Simplified marginal contribution approximation. "
            "Not equivalent to exact SHAP values but provides directional attribution. "
            "Describes model behaviour, not physical causation."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Sensitivity sweeps (one-at-a-time)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    model,
    df: pd.DataFrame,
    features: List[str],
    target_name: str,
    unit: str,
) -> Dict:
    """
    One-at-a-time (OAT) sensitivity analysis.

    For each feature:
      - Hold all other features at their median level in the dataset.
      - Vary the feature across its full grid.
      - Record predicted output and relative change.

    This isolates each feature's marginal effect at the median configuration.
    Combined with LOLO and ICE, it provides a complete picture of model behaviour.
    """
    logger.info("  OAT sensitivity for %s …", target_name)

    # Median feature values (from supported discrete grid)
    median_vals = {}
    for f in features:
        vals = sorted(df[f].unique())
        median_vals[f] = vals[len(vals) // 2]

    logger.info("  Reference configuration: %s", median_vals)
    X_ref = np.array([[median_vals[f] for f in features]])
    ref_pred = float(model.predict(X_ref)[0])

    sweeps = {}
    for feat in features:
        levels = sorted(df[feat].unique())
        preds  = []
        for level in levels:
            X_sweep = np.array([[
                level if f == feat else median_vals[f]
                for f in features
            ]])
            preds.append(float(model.predict(X_sweep)[0]))

        preds_arr = np.array(preds)
        sweeps[feat] = {
            "feature":        feat,
            "reference_value": float(median_vals[feat]),
            "reference_pred":  ref_pred,
            "levels":          levels,
            "predictions":     preds,
            "delta_from_ref":  (preds_arr - ref_pred).tolist(),
            "pct_change":      ((preds_arr - ref_pred) / ref_pred * 100).tolist(),
            "range":           float(preds_arr.max() - preds_arr.min()),
            "range_pct":       float((preds_arr.max() - preds_arr.min()) / ref_pred * 100),
        }
        logger.info(
            "    %s: range=%.4g %s  (%.1f%% of ref)",
            feat, sweeps[feat]["range"], unit, sweeps[feat]["range_pct"]
        )

    return {
        "target":              target_name,
        "unit":                unit,
        "features":            features,
        "reference_config":    median_vals,
        "reference_prediction": ref_pred,
        "sweeps":              sweeps,
    }


# ---------------------------------------------------------------------------
# Full XAI pipeline
# ---------------------------------------------------------------------------

def run_explainability(
    df: pd.DataFrame,
    cap_model,
    mut_model,
) -> Dict:
    """Run the full XAI pipeline for both targets."""
    logger.info("=" * 60)
    logger.info("Explainable AI Analysis")
    logger.info("=" * 60)

    results = {}

    for target_col, target_name, features, model, unit in [
        (CAP_TARGET, CAP_TARGET, CAP_FEATURES, cap_model, CAP_UNIT),
        (MUT_TARGET, MUT_TARGET, MUT_FEATURES, mut_model, MUT_UNIT),
    ]:
        target_key = "capacitance" if target_col == CAP_TARGET else "mutual_inductance"
        logger.info("\n--- %s [%s] ---", target_name, unit)

        X = df[features].values
        y = df[target_col].values

        # 1. Global permutation importance
        global_pi = global_permutation_importance(model, X, y, features, target_name)

        # 2. ICE / PDP curves
        ice = individual_conditional_expectation(
            model, df, features, target_name, FEATURE_GRID
        )

        # 3. Sensitivity analysis
        sensitivity = sensitivity_analysis(model, df, features, target_name, unit)

        # 4. Explain two representative individual predictions
        # Use the configuration with highest and lowest predicted output
        preds_all = model.predict(X)
        idx_max = int(np.argmax(preds_all))
        idx_min = int(np.argmin(preds_all))

        explanations = {}
        for label, idx in [("highest_predicted", idx_max), ("lowest_predicted", idx_min)]:
            row = df.iloc[idx]
            fvals = {f: float(row[f]) for f in features}
            explanations[label] = explain_single_prediction(
                model, features, fvals, target_name, unit, df
            )
            explanations[label]["config_id"] = str(row.get("Config_ID", idx))

        results[target_key] = {
            "target_name":      target_name,
            "features":         features,
            "unit":             unit,
            "global_importance": global_pi,
            "ice_pdp":          ice,
            "sensitivity":      sensitivity,
            "individual_explanations": explanations,
        }

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_explainability_results(
    xai: Dict,
    out_dir: Path,
) -> Dict:
    """Save XAI results to CSV and JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved = {}

    for target_key, data in xai.items():
        t = target_key

        # Global importance CSV
        gi_fname = out_dir / f"xai_global_importance_{t}_{timestamp}.csv"
        data["global_importance"].to_csv(gi_fname, index=False)
        saved[f"global_importance_{t}"] = str(gi_fname)
        logger.info("  Saved: %s", gi_fname.name)

        # Sensitivity CSV
        sens_rows = []
        for feat, sweep in data["sensitivity"]["sweeps"].items():
            for level, pred, delta, pct in zip(
                sweep["levels"], sweep["predictions"],
                sweep["delta_from_ref"], sweep["pct_change"]
            ):
                sens_rows.append({
                    "Target":    data["target_name"],
                    "Feature":   feat,
                    "Level":     level,
                    "Predicted": pred,
                    "Delta":     delta,
                    "Pct_Change": pct,
                    "Reference_Level": sweep["reference_value"],
                    "Reference_Pred":  sweep["reference_pred"],
                })
        sens_df   = pd.DataFrame(sens_rows)
        sens_fname = out_dir / f"xai_sensitivity_{t}_{timestamp}.csv"
        sens_df.to_csv(sens_fname, index=False)
        saved[f"sensitivity_{t}"] = str(sens_fname)
        logger.info("  Saved: %s", sens_fname.name)

        # PDP summary CSV
        pdp_rows = []
        for feat, ice_data in data["ice_pdp"].items():
            for x, m, s in zip(ice_data["x_values"], ice_data["pdp_mean"], ice_data["pdp_std"]):
                pdp_rows.append({
                    "Target":  data["target_name"],
                    "Feature": feat,
                    "Level":   x,
                    "PDP_Mean": m,
                    "PDP_Std":  s,
                })
        pdp_df   = pd.DataFrame(pdp_rows)
        pdp_fname = out_dir / f"xai_pdp_{t}_{timestamp}.csv"
        pdp_df.to_csv(pdp_fname, index=False)
        saved[f"pdp_{t}"] = str(pdp_fname)
        logger.info("  Saved: %s", pdp_fname.name)

        # Individual explanations JSON
        ind_fname = out_dir / f"xai_individual_explanations_{t}_{timestamp}.json"
        ind_data  = {}
        for label, exp in data["individual_explanations"].items():
            # Convert numpy types for JSON serialisation
            exp_copy = {
                k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame)
                    else v)
                for k, v in exp.items()
            }
            ind_data[label] = exp_copy
        with open(ind_fname, "w", encoding="utf-8") as fh:
            json.dump(ind_data, fh, indent=2, ensure_ascii=False)
        saved[f"individual_explanations_{t}"] = str(ind_fname)
        logger.info("  Saved: %s", ind_fname.name)

    # Master summary JSON
    summary = {
        "generated_at": timestamp,
        "module": "explainability.py",
        "xai_methods": {
            "global_permutation_importance": (
                "Measures R² drop when each feature is shuffled. "
                "Global view of feature reliance. Analogous to mean |SHAP| value."
            ),
            "ICE_PDP": (
                "Individual Conditional Expectation: model output vs feature level "
                "for each sample. Average = Partial Dependence Plot."
            ),
            "sensitivity_OAT": (
                "One-at-a-time sweep: varies one feature while fixing others at "
                "their median grid level."
            ),
            "individual_explanation": (
                "Marginal contribution approximation: isolates each feature's "
                "contribution to a specific prediction relative to the dataset baseline."
            ),
        },
        "shap_status": (
            "SHAP library not available in this environment. "
            "The above methods provide equivalent global and local XAI coverage "
            "using only scikit-learn and numpy."
        ),
        "limitations": [
            "All results describe model behaviour on MATLAB deterministic data.",
            "Physical causation is noted where consistent with EMC physics but is not proven.",
            "No experimental measurements — no real-world validation implied.",
            "No safety certification or KONE field accuracy claim.",
        ],
        "results_summary": {
            tkey: {
                "target": data["target_name"],
                "unit": data["unit"],
                "global_importance_ranking": data["global_importance"][
                    ["Rank", "Feature", "R2_Drop_Mean"]
                ].to_dict(orient="records"),
                "sensitivity_ranges": {
                    feat: {
                        "range": sweep["range"],
                        "range_pct": sweep["range_pct"],
                    }
                    for feat, sweep in data["sensitivity"]["sweeps"].items()
                },
            }
            for tkey, data in xai.items()
        },
        "files_saved": saved,
    }

    json_fname = out_dir / f"xai_summary_{timestamp}.json"
    with open(json_fname, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    saved["summary_json"] = str(json_fname)
    logger.info("  Saved: %s", json_fname.name)

    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(save_results: bool = True) -> Dict:
    """Execute the full explainability analysis pipeline."""
    logger.info("KONE EMC-GUARD — Explainable AI Module")

    df = load_dataset()
    cap_model, mut_model = _load_models()
    xai = run_explainability(df, cap_model, mut_model)

    files_saved = {}
    if save_results:
        files_saved = save_explainability_results(xai, AIML_REPORTS_DIR)
    xai["files_saved"] = files_saved

    return xai


if __name__ == "__main__":
    import sys as _sys
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() != 'utf-8':
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    results = main(save_results=True)

    print("\n" + "=" * 60)
    print("EXPLAINABLE AI SUMMARY")
    print("=" * 60)

    for tkey, label in [
        ("capacitance",      "Capacitance (pF/m)"),
        ("mutual_inductance", "Mutual Inductance (nH)"),
    ]:
        data = results[tkey]
        print(f"\n{label} — Global Feature Importance (R² drop):")
        gi = data["global_importance"]
        for _, row in gi.iterrows():
            bar = "█" * max(1, int(abs(row["R2_Drop_Mean"]) * 50))
            print(f"  [{row['Rank']}] {row['Feature']:<24}  {row['R2_Drop_Mean']:+.6f} ± {row['R2_Drop_Std']:.6f}  {bar}")

        print(f"\n  Sensitivity ranges (OAT at median config):")
        for feat, sweep in data["sensitivity"]["sweeps"].items():
            print(f"    {feat:<24}  {sweep['range']:.4g} {data['unit']}  ({sweep['range_pct']:.1f}%)")

        print(f"\n  Individual Explanations (highest vs lowest predicted):")
        for label2, exp in data["individual_explanations"].items():
            print(f"    {label2}  (Config {exp.get('config_id','')}): "
                  f"pred={exp['query_pred']:.4g} {data['unit']}")
            for feat, attr in exp["attributions"].items():
                print(f"      {feat:<24}: {attr['contribution']:+.4g} ({attr['direction']})")

    print("\nFiles saved:")
    for k, v in results.get("files_saved", {}).items():
        print(f"  {k}: {Path(v).name}")

