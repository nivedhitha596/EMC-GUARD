"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 3 | Risk Analysis Engine
File   : risk_analysis.py
Author : EMC-GUARD Development Team

PUBLIC API
----------
    analyze_configuration(configuration)          -> dict
    compare_configurations(configurations_dict)   -> dict

DISCLAIMER — Risk Classification
---------------------------------
Risk categories produced by this module are DATASET-RELATIVE, meaning
they reflect where a configuration's predicted electromagnetic outputs fall
relative to the 33rd and 67th percentiles of the training dataset.

They are NOT:
  - Certified EMC compliance limits.
  - Validated safety thresholds.
  - Substitutes for a formal EMC test or MATLAB simulation.

They are decision-support indicators to help engineers PRIORITISE
configurations for further analysis.

When validated engineering thresholds are supplied by the project team,
update THRESHOLDS_JSON and set "threshold_method": "engineering_validated".

Usage
-----
    python src/risk_analysis.py

Outputs
-------
    reports/risk_analysis_results.csv
    reports/risk_assessment_report.txt
    reports/risk_thresholds.json   (written once; reused on subsequent runs)
"""

import os
import sys
import json
import warnings
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR         = os.path.join(BASE_DIR, "src")
DATASET_CSV     = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
MODEL_PKL       = os.path.join(BASE_DIR, "models",  "emc_surrogate_model.pkl")
PREPROC_PKL     = os.path.join(BASE_DIR, "models",  "emc_preprocessor.pkl")
META_JSON       = os.path.join(BASE_DIR, "models",  "target_model_metadata.json")
PERF_JSON       = os.path.join(BASE_DIR, "reports", "model_performance.json")
THRESHOLDS_JSON = os.path.join(BASE_DIR, "reports", "risk_thresholds.json")
RESULTS_CSV     = os.path.join(BASE_DIR, "reports", "risk_analysis_results.csv")
REPORT_TXT      = os.path.join(BASE_DIR, "reports", "risk_assessment_report.txt")
VIS_DIR         = os.path.join(BASE_DIR, "reports", "visualisations")

# Add src/ to path so we can import predict_emc directly
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# RISK CATEGORY LABELS
# ---------------------------------------------------------------------------
RISK_LOWER        = "LOWER PREDICTED RISK"
RISK_INTERMEDIATE = "INTERMEDIATE PREDICTED RISK"
RISK_HIGHER       = "HIGHER PREDICTED RISK"
RISK_UNKNOWN      = "UNDETERMINED RISK"

# Primary target used as the leading indicator for risk (victim-side voltage)
PRIMARY_TARGET    = "Victim_Voltage_V"

# All targets expected from Module 2
ALL_TARGETS = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]

# ============================================================
# SECTION 1 — LOAD ARTEFACTS
# ============================================================

def _load_thresholds() -> dict:
    """Load risk threshold configuration from JSON."""
    if not os.path.exists(THRESHOLDS_JSON):
        raise FileNotFoundError(
            f"risk_thresholds.json not found at {THRESHOLDS_JSON}.\n"
            "Run the project setup or check Module 3 installation."
        )
    with open(THRESHOLDS_JSON) as f:
        return json.load(f)


def _load_model_metadata() -> dict:
    """Load per-target model performance metadata from Module 2."""
    if not os.path.exists(META_JSON):
        raise FileNotFoundError(
            f"target_model_metadata.json not found: {META_JSON}\n"
            "Run train_model.py (Module 2) first."
        )
    with open(META_JSON) as f:
        return json.load(f)


def _load_model_performance() -> dict:
    """Load full model performance report from Module 2."""
    if not os.path.exists(PERF_JSON):
        raise FileNotFoundError(
            f"model_performance.json not found: {PERF_JSON}\n"
            "Run train_model.py (Module 2) first."
        )
    with open(PERF_JSON) as f:
        return json.load(f)


def _load_prediction_models():
    """
    Load preprocessor and surrogate bundle from Module 2.
    Returns (preprocessor, surrogate_bundle).
    """
    for path, label in [(MODEL_PKL, "emc_surrogate_model.pkl"),
                        (PREPROC_PKL, "emc_preprocessor.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "Run train_model.py (Module 2) first."
            )
    return joblib.load(PREPROC_PKL), joblib.load(MODEL_PKL)


# ============================================================
# SECTION 2 — PREDICTION INTERFACE (wraps predict_emc.py API)
# ============================================================

def _get_predictions(configuration: dict) -> dict:
    """
    Call the Module 2 prediction engine.

    Returns the full result dict from predict_interference():
        {
          "predictions": {target: float},
          "warnings":    [str],
          "model_info":  {target: {"algorithm": str, "R2": float}}
        }

    Raises ValidationError (from predict_emc) on invalid inputs.
    """
    # Import here (not at module level) to keep risk_analysis.py self-contained
    try:
        from predict_emc import predict_interference, ValidationError
    except ImportError as e:
        raise ImportError(
            f"Cannot import predict_emc: {e}\n"
            "Ensure predict_emc.py is in the src/ directory."
        ) from e

    return predict_interference(configuration, verbose=False)


# ============================================================
# SECTION 3 — RISK CLASSIFICATION
# ============================================================

def _classify_single_target(predicted_value: float,
                             target: str,
                             thresholds: dict) -> str:
    """
    Classify one predicted target value into a risk category.

    Logic (in priority order):
      1. Engineering-validated thresholds  — if threshold_method == 'engineering_validated'
      2. Dataset-relative percentile bounds — default

    Returns one of: RISK_LOWER, RISK_INTERMEDIATE, RISK_HIGHER, RISK_UNKNOWN
    """
    tgt_cfg = thresholds.get("thresholds", {}).get(target)
    if tgt_cfg is None:
        return RISK_UNKNOWN

    lower_upper = tgt_cfg["lower_upper"]   # P33 boundary
    upper_lower = tgt_cfg["upper_lower"]   # P67 boundary

    if predicted_value <= lower_upper:
        return RISK_LOWER
    elif predicted_value <= upper_lower:
        return RISK_INTERMEDIATE
    else:
        return RISK_HIGHER


def _overall_risk_category(per_target_categories: dict,
                            predictions: dict) -> str:
    """
    Determine the overall configuration risk category.

    Strategy
    --------
    Use the PRIMARY_TARGET (Victim_Voltage_V) as the lead indicator.
    If it is unavailable, fall back to the worst category across
    all victim-side targets (Victim_Voltage_V, Victim_RMS_V,
    Victim_PeakToPeak_V).
    If those are also unavailable, take the worst of all targets.

    This avoids inventing a composite score while still producing
    a single summary category for comparison tables.
    """
    # Priority order for the summary
    priority_targets = [
        "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
        "MutualInductance_nH", "MutualCapacitance_pF",
    ]
    rank = {RISK_LOWER: 0, RISK_INTERMEDIATE: 1, RISK_HIGHER: 2, RISK_UNKNOWN: -1}

    overall_rank = -1
    for tgt in priority_targets:
        cat = per_target_categories.get(tgt, RISK_UNKNOWN)
        if cat in rank:
            overall_rank = max(overall_rank, rank[cat])

    reverse_rank = {0: RISK_LOWER, 1: RISK_INTERMEDIATE, 2: RISK_HIGHER}
    return reverse_rank.get(overall_rank, RISK_UNKNOWN)


# ============================================================
# SECTION 4 — NORMALISED RISK INDICATORS
# ============================================================

def _compute_risk_indicators(predictions: dict,
                              per_target_categories: dict,
                              thresholds: dict) -> dict:
    """
    Compute a 0–1 normalised indicator for each predicted target.

    Normalisation formula:
        indicator = clip((value - min_val) / (max_val - min_val), 0, 1)

    Where min_val and max_val are the dataset min and max for that target
    (loaded from the threshold config as a percentile spread proxy).

    IMPORTANT: This indicator shows RELATIVE position within the training
    dataset range. It is NOT a calibrated probability or physical intensity.

    Returns dict: {target: float (0-1)}
    """
    indicators = {}
    tgt_cfgs = thresholds.get("thresholds", {})

    # Approximate dataset range from threshold config + known min/max
    # We use P33 and P67 as anchors; for full normalisation we need
    # the actual dataset min/max. Load them on demand.
    _dataset_ranges = _get_dataset_ranges()

    for target, pred_val in predictions.items():
        if target not in _dataset_ranges:
            indicators[target] = None
            continue
        lo, hi = _dataset_ranges[target]
        span = hi - lo
        if span <= 0:
            indicators[target] = 0.0
        else:
            indicators[target] = float(np.clip((pred_val - lo) / span, 0.0, 1.0))

    return indicators


_DATASET_RANGES_CACHE: dict = {}


def _get_dataset_ranges() -> dict:
    """Cache and return {target: (min, max)} from the cleaned dataset."""
    global _DATASET_RANGES_CACHE
    if _DATASET_RANGES_CACHE:
        return _DATASET_RANGES_CACHE

    if not os.path.exists(DATASET_CSV):
        # Fallback: derive from thresholds JSON (less accurate)
        return {}

    df = pd.read_csv(DATASET_CSV)
    for tgt in ALL_TARGETS:
        if tgt in df.columns:
            _DATASET_RANGES_CACHE[tgt] = (float(df[tgt].min()),
                                           float(df[tgt].max()))
    return _DATASET_RANGES_CACHE


# ============================================================
# SECTION 5 — PREDICTION RELIABILITY ASSESSMENT
# ============================================================

def _assess_reliability(prediction_result: dict,
                         configuration: dict,
                         thresholds: dict,
                         metadata: dict) -> dict:
    """
    Assess prediction reliability on three dimensions:

    1. Out-of-domain detection — any numeric feature outside training range.
    2. Unseen categorical value — Shield_Type or Cable_Configuration not in
       training data.
    3. Model R² adequacy — flag targets where R² < acceptable threshold.

    Returns:
        {
          "overall_reliability": "ACCEPTABLE" | "CAUTION" | "LOW",
          "ood_features": [str],
          "unseen_categories": [str],
          "low_r2_targets": [str],
          "rmse_relative_warnings": [str],
          "calibration_note": str,
          "warnings": [str]
        }
    """
    reliability_cfg  = thresholds.get("model_reliability", {})
    r2_threshold     = reliability_cfg.get("R2_acceptable_threshold", 0.60)
    rmse_warn_frac   = reliability_cfg.get("RMSE_relative_warning_fraction", 0.30)
    training_ranges  = thresholds.get("training_ranges", {})
    valid_cats       = thresholds.get("valid_categorical_values", {})

    reliability_warnings = list(prediction_result.get("warnings", []))
    ood_features         = []
    unseen_categories    = []
    low_r2_targets       = []
    rmse_relative_warns  = []

    # 1. Out-of-domain numeric features
    for feat, (lo, hi) in training_ranges.items():
        if feat in configuration:
            val = float(configuration[feat])
            if val < lo or val > hi:
                ood_features.append(feat)

    # 2. Unseen categorical values
    for cat_feat, allowed in valid_cats.items():
        val = str(configuration.get(cat_feat, ""))
        if val not in allowed:
            unseen_categories.append(f"{cat_feat}='{val}'")

    # 3. Model R² per target
    for tgt, m in metadata.items():
        r2 = m.get("R2", 1.0)
        if r2 < r2_threshold:
            low_r2_targets.append(f"{tgt} (R2={r2:.4f})")

    # 4. RMSE relative to prediction value
    predictions = prediction_result.get("predictions", {})
    for tgt, pred_val in predictions.items():
        rmse = metadata.get(tgt, {}).get("RMSE", None)
        if rmse is not None and pred_val > 0:
            rel = rmse / pred_val
            if rel > rmse_warn_frac:
                rmse_relative_warns.append(
                    f"{tgt}: RMSE={rmse:.4f} is "
                    f"{rel*100:.1f}% of predicted value {pred_val:.4f}"
                )

    # Determine overall reliability level
    if ood_features or unseen_categories:
        overall = "CAUTION"
    elif low_r2_targets:
        overall = "CAUTION"
    else:
        overall = "ACCEPTABLE"

    if len(ood_features) > 2 or len(unseen_categories) > 0:
        overall = "LOW"

    return {
        "overall_reliability": overall,
        "ood_features":             ood_features,
        "unseen_categories":        unseen_categories,
        "low_r2_targets":           low_r2_targets,
        "rmse_relative_warnings":   rmse_relative_warns,
        "calibration_note":         (
            "Prediction uncertainty is not yet calibrated. "
            "The RMSE from the test set is the best available proxy "
            "but is not a formal prediction interval."
        ),
        "warnings": reliability_warnings,
    }


# ============================================================
# SECTION 6 — CABLE GEOMETRY ANALYSIS
# ============================================================

def _analyse_geometry(configuration: dict,
                       predictions: dict,
                       thresholds: dict) -> List[str]:
    """
    Generate rule-based geometry observations.

    Each observation is only generated when the relevant condition is
    actually present in the configuration. No blanket statements are made.

    Returns a list of plain-English observation strings.
    """
    observations = []
    geo_cfg      = thresholds.get("risk_scoring", {}).get("geometry_modifiers", {})
    sep_low      = geo_cfg.get("separation_low_threshold_mm", 30.0)
    ovl_high     = geo_cfg.get("overlap_high_threshold_mm",  350.0)

    sep    = float(configuration.get("Separation_mm",      999))
    ovl    = float(configuration.get("Overlap_mm",           0))
    length = float(configuration.get("Cable_Length_mm",      0))
    shield = str(configuration.get("Shield_Type",           ""))
    config = str(configuration.get("Cable_Configuration",   ""))
    radius = float(configuration.get("Conductor_Radius_mm",  0))
    eps_r  = float(configuration.get("Dielectric_Constant",  1))

    # Separation
    if sep < sep_low:
        observations.append(
            f"Cable separation is {sep:.1f} mm, which is below the "
            f"dataset-context compact threshold ({sep_low:.0f} mm). "
            "Close proximity can increase electromagnetic coupling."
        )
    elif sep > 150:
        observations.append(
            f"Cable separation is {sep:.1f} mm. Larger separation "
            "generally reduces inductive and capacitive coupling "
            "in the dataset-relative context."
        )

    # Overlap
    if ovl > ovl_high:
        observations.append(
            f"Parallel overlap length is {ovl:.1f} mm, exceeding the "
            f"dataset-context extensive threshold ({ovl_high:.0f} mm). "
            "Longer parallel runs may expose more cable length "
            "to electromagnetic coupling."
        )
    elif ovl < 80:
        observations.append(
            f"Parallel overlap is {ovl:.1f} mm, which is relatively "
            "short. Shorter overlap typically limits the total coupling "
            "exposure within this dataset context."
        )

    # Cable configuration
    if config == "Parallel":
        observations.append(
            "Cable arrangement is Parallel. In this dataset context, "
            "parallel arrangements tend to exhibit higher predicted "
            "coupling than perpendicular or crossed arrangements."
        )
    elif config == "Perpendicular":
        observations.append(
            "Cable arrangement is Perpendicular. The model was trained "
            "with configurations where perpendicular arrangements showed "
            "reduced coupling compared to parallel."
        )
    elif config == "Crossed":
        observations.append(
            "Cable arrangement is Crossed. Crossed configurations in "
            "this dataset showed intermediate coupling compared to parallel."
        )

    # Shielding
    if shield == "Shielded":
        observations.append(
            "Shielded cables are used. The surrogate model was trained "
            "with shielded cables assigned a reduced coupling factor "
            "in the synthetic data generator."
        )
    elif shield == "Unshielded":
        observations.append(
            "Unshielded cables are used. In the training dataset, "
            "unshielded configurations showed higher predicted coupling "
            "than equivalent shielded arrangements."
        )

    # Combined compact + long overlap interaction
    if sep < sep_low and ovl > ovl_high:
        observations.append(
            f"The combination of reduced separation ({sep:.1f} mm) and "
            f"extended overlap ({ovl:.1f} mm) places this configuration "
            "in a geometrically high-exposure region of the dataset."
        )

    # Dielectric constant
    if eps_r > 7.0:
        observations.append(
            f"The dielectric constant is {eps_r:.2f}, which is above the "
            "dataset median. Higher permittivity is associated with "
            "increased predicted capacitive coupling in the surrogate model."
        )

    return observations


# ============================================================
# SECTION 7 — EXPLANATION GENERATOR
# ============================================================

def _generate_explanation(overall_risk: str,
                            per_target_categories: dict,
                            risk_indicators: dict,
                            geometry_observations: List[str],
                            reliability: dict,
                            configuration: dict,
                            predictions: dict,
                            thresholds: dict) -> str:
    """
    Compose a structured engineering explanation for the risk assessment.
    All statements refer to actual configuration values and model predictions.
    """
    lines = []

    # Opening summary
    if overall_risk == RISK_LOWER:
        lines.append(
            "This configuration has relatively LOWER predicted electromagnetic "
            "coupling compared with the intermediate and upper ranges of the "
            "training dataset."
        )
    elif overall_risk == RISK_INTERMEDIATE:
        lines.append(
            "This configuration shows INTERMEDIATE predicted electromagnetic "
            "coupling. Its predicted outputs fall within the middle range "
            "(between the 33rd and 67th percentiles) of the training dataset."
        )
    elif overall_risk == RISK_HIGHER:
        lines.append(
            "This configuration has relatively HIGHER predicted electromagnetic "
            "coupling. Its predicted outputs exceed the 67th percentile of the "
            "training dataset distribution."
        )
    else:
        lines.append(
            "The overall risk category could not be determined from available predictions."
        )

    # Per-target summary
    lines.append("")
    lines.append("Predicted electromagnetic outputs:")
    for tgt in ALL_TARGETS:
        if tgt in predictions and tgt in per_target_categories:
            pred_val = predictions[tgt]
            cat      = per_target_categories[tgt]
            ind      = risk_indicators.get(tgt)
            ind_str  = f"  (indicator={ind:.2f}/1.00)" if ind is not None else ""
            lines.append(f"  {tgt}: {pred_val:.6f}  -> {cat}{ind_str}")

    # Geometry observations
    if geometry_observations:
        lines.append("")
        lines.append("Cable geometry observations:")
        for obs in geometry_observations:
            lines.append(f"  - {obs}")

    # Reliability
    lines.append("")
    rel = reliability["overall_reliability"]
    lines.append(f"Prediction reliability: {rel}")
    if reliability["ood_features"]:
        lines.append(
            f"  Out-of-training-range features: {reliability['ood_features']}\n"
            "  The model is extrapolating outside its validated input space.\n"
            "  Verify this prediction with MATLAB simulation."
        )
    if reliability["unseen_categories"]:
        lines.append(
            f"  Unseen categorical values: {reliability['unseen_categories']}\n"
            "  The OneHotEncoder maps these to zero-columns. "
            "Prediction may not reflect the intended configuration."
        )
    if reliability["low_r2_targets"]:
        lines.append(
            f"  Targets with R2 below acceptable threshold: "
            f"{reliability['low_r2_targets']}"
        )
    if reliability["rmse_relative_warnings"]:
        lines.append("  RMSE relative warnings:")
        for w in reliability["rmse_relative_warnings"]:
            lines.append(f"    {w}")
    lines.append(f"  {reliability['calibration_note']}")

    # Threshold disclosure
    method = thresholds.get("threshold_method", "dataset_relative_percentile")
    lines.append("")
    lines.append(
        f"Threshold method: {method} "
        f"(P{thresholds.get('percentile_lower_boundary',33)} / "
        f"P{thresholds.get('percentile_upper_boundary',67)} of "
        f"{thresholds.get('source_dataset','dataset')} "
        f"[{thresholds.get('dataset_type','UNKNOWN')}])."
    )
    lines.append(
        "IMPORTANT: This is a PREDICTIVE DECISION-SUPPORT PROTOTYPE. "
        "Risk categories are dataset-relative and are NOT certified EMC "
        "safety thresholds or compliance assessments."
    )

    return "\n".join(lines)


# ============================================================
# SECTION 8 — MAIN PUBLIC API: analyze_configuration()
# ============================================================

def analyze_configuration(configuration: dict,
                            config_id: str = "CONF_USER") -> dict:
    """
    Analyse a single cable configuration for electromagnetic interference risk.

    Parameters
    ----------
    configuration : dict
        Dictionary with all required input features (same as predict_interference).
    config_id : str
        Human-readable identifier for this configuration (used in reports).

    Returns
    -------
    dict with keys:
        config_id               : str
        configuration           : dict  (input copy)
        predictions             : dict  {target: float}
        per_target_risk         : dict  {target: risk_category_str}
        per_target_indicators   : dict  {target: float 0-1}
        overall_risk_category   : str
        geometry_observations   : list[str]
        reliability             : dict
        explanation             : str
        timestamp               : str

    Raises
    ------
    ValidationError (from predict_emc) — on invalid configuration inputs.
    FileNotFoundError — if model artefacts are missing.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load artefacts
    thresholds = _load_thresholds()
    metadata   = _load_model_metadata()

    # Get predictions from Module 2
    pred_result = _get_predictions(configuration)
    predictions = pred_result["predictions"]

    # Risk classification — per target
    per_target_risk = {}
    for tgt in ALL_TARGETS:
        if tgt in predictions:
            per_target_risk[tgt] = _classify_single_target(
                predictions[tgt], tgt, thresholds
            )

    # Normalised risk indicators (0-1 position within dataset range)
    risk_indicators = _compute_risk_indicators(predictions, per_target_risk, thresholds)

    # Overall risk category
    overall_risk = _overall_risk_category(per_target_risk, predictions)

    # Geometry analysis
    geo_observations = _analyse_geometry(configuration, predictions, thresholds)

    # Reliability assessment
    reliability = _assess_reliability(pred_result, configuration, thresholds, metadata)

    # Natural-language explanation
    explanation = _generate_explanation(
        overall_risk, per_target_risk, risk_indicators,
        geo_observations, reliability, configuration, predictions, thresholds
    )

    return {
        "config_id":             config_id,
        "configuration":         copy.deepcopy(configuration),
        "predictions":           predictions,
        "per_target_risk":       per_target_risk,
        "per_target_indicators": risk_indicators,
        "overall_risk_category": overall_risk,
        "geometry_observations": geo_observations,
        "reliability":           reliability,
        "explanation":           explanation,
        "timestamp":             timestamp,
    }


# ============================================================
# SECTION 9 — MAIN PUBLIC API: compare_configurations()
# ============================================================

def compare_configurations(configurations: Dict[str, dict]) -> dict:
    """
    Analyse and compare multiple cable configurations.

    Parameters
    ----------
    configurations : dict
        {config_id: configuration_dict, ...}

    Returns
    -------
    dict with keys:
        results         : {config_id: analyze_configuration() output}
        ranked_by_risk  : list[config_id] — ordered worst-to-best by
                          Victim_Voltage_V prediction (highest first)
        comparison_df   : pd.DataFrame   — tabular summary
        summary_text    : str            — human-readable comparison narrative
    """
    results   = {}
    rank_data = []

    print(f"\n  Analysing {len(configurations)} configurations ...")
    for cfg_id, cfg in configurations.items():
        print(f"    [{cfg_id}] ...", end=" ", flush=True)
        try:
            result = analyze_configuration(cfg, config_id=cfg_id)
            results[cfg_id] = result
            vv = result["predictions"].get("Victim_Voltage_V", 0.0)
            rank_data.append((cfg_id, vv, result["overall_risk_category"]))
            print(f"-> {result['overall_risk_category']}")
        except Exception as e:
            results[cfg_id] = {"error": str(e), "config_id": cfg_id}
            print(f"-> ERROR: {e}")

    # Rank by Victim_Voltage_V descending (highest interference first)
    rank_data.sort(key=lambda x: x[1], reverse=True)
    ranked_by_risk = [r[0] for r in rank_data]

    # Build comparison DataFrame
    rows = []
    for cfg_id, result in results.items():
        if "error" in result:
            rows.append({"Configuration_ID": cfg_id, "Error": result["error"]})
            continue
        preds   = result["predictions"]
        per_tgt = result["per_target_risk"]
        indics  = result["per_target_indicators"]
        rel     = result["reliability"]["overall_reliability"]
        row = {
            "Configuration_ID":    cfg_id,
            "Separation_mm":       result["configuration"].get("Separation_mm"),
            "Overlap_mm":          result["configuration"].get("Overlap_mm"),
            "Cable_Length_mm":     result["configuration"].get("Cable_Length_mm"),
            "Shield_Type":         result["configuration"].get("Shield_Type"),
            "Cable_Configuration": result["configuration"].get("Cable_Configuration"),
        }
        for tgt in ALL_TARGETS:
            row[f"Pred_{tgt}"]   = round(preds.get(tgt, float("nan")), 6)
            row[f"Risk_{tgt}"]   = per_tgt.get(tgt, RISK_UNKNOWN)
            row[f"Ind_{tgt}"]    = round(indics.get(tgt) or float("nan"), 4)
        row["Overall_Risk_Category"] = result["overall_risk_category"]
        row["Prediction_Reliability"] = rel
        row["Engineering_Explanation"] = result["explanation"].replace("\n", " | ")
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    # Comparison narrative
    summary_lines = [
        "=" * 68,
        "CONFIGURATION COMPARISON SUMMARY",
        "[SYNTHETIC DATA — dataset-relative risk categories]",
        "=" * 68,
        "",
        f"Configurations analysed: {len(configurations)}",
        f"Ranked by predicted Victim_Voltage_V (highest to lowest):",
        "",
    ]
    for rank_pos, (cfg_id, vv, cat) in enumerate(rank_data, 1):
        if cfg_id not in results or "error" in results[cfg_id]:
            summary_lines.append(f"  {rank_pos}. {cfg_id}: ERROR")
            continue
        r     = results[cfg_id]
        sep   = r["configuration"].get("Separation_mm", "?")
        ovl   = r["configuration"].get("Overlap_mm", "?")
        shld  = r["configuration"].get("Shield_Type", "?")
        cfg   = r["configuration"].get("Cable_Configuration", "?")
        rel   = r["reliability"]["overall_reliability"]
        summary_lines.append(
            f"  {rank_pos}. {cfg_id:<14} | Victim_V={vv:>10.4f} V "
            f"| {cat:<30} | Sep={sep} mm, Ovl={ovl} mm "
            f"| {shld}/{cfg} | Reliability={rel}"
        )

    summary_lines += [
        "",
        "Threshold basis: dataset-relative (P33/P67 of training dataset).",
        "NOT certified EMC safety thresholds.",
        "=" * 68,
    ]
    summary_text = "\n".join(summary_lines)

    return {
        "results":        results,
        "ranked_by_risk": ranked_by_risk,
        "comparison_df":  comparison_df,
        "summary_text":   summary_text,
    }


# ============================================================
# SECTION 10 — OUTPUT WRITERS
# ============================================================

def save_results_csv(comparison_df: pd.DataFrame) -> None:
    """Write the risk analysis comparison table to CSV."""
    comparison_df.to_csv(RESULTS_CSV, index=False)
    print(f"  Saved: reports/risk_analysis_results.csv")


def save_assessment_report(compare_result: dict,
                            test_results: List[dict],
                            dataset_type: str) -> None:
    """Write the full human-readable risk assessment report."""
    lines = [
        "=" * 68,
        "EMC-GUARD — MODULE 3 RISK ASSESSMENT REPORT",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset    : {dataset_type}",
        "=" * 68,
        "",
        "DISCLAIMER",
        "-" * 68,
        "This report is produced by a PREDICTIVE DECISION-SUPPORT PROTOTYPE.",
        "Risk categories are DATASET-RELATIVE, derived from percentile",
        "boundaries of the training dataset. They are NOT certified EMC",
        "safety limits or compliance assessments.",
        "All predictions are based on SYNTHETIC DUMMY DATA and do not",
        "represent real electromagnetic measurements.",
        "",
        compare_result["summary_text"],
        "",
        "=" * 68,
        "DETAILED CONFIGURATION ANALYSES",
        "=" * 68,
    ]

    for result in test_results:
        if "error" in result:
            lines += [
                "",
                f"--- {result.get('config_id', 'UNKNOWN')} ---",
                f"ERROR: {result['error']}",
            ]
            continue

        lines += [
            "",
            f"--- {result['config_id']} ---",
            f"Timestamp   : {result['timestamp']}",
            f"Overall Risk: {result['overall_risk_category']}",
            f"Reliability : {result['reliability']['overall_reliability']}",
            "",
            "Input Configuration:",
        ]
        for k, v in result["configuration"].items():
            lines.append(f"  {k:<26}: {v}")

        lines += ["", "Explanation:"]
        for ln in result["explanation"].split("\n"):
            lines.append(f"  {ln}")
        lines.append("-" * 68)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: reports/risk_assessment_report.txt")


# ============================================================
# SECTION 11 — VISUALISATIONS
# ============================================================

def _save_fig(fig, name: str) -> None:
    os.makedirs(VIS_DIR, exist_ok=True)
    fig.savefig(os.path.join(VIS_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reports/visualisations/{name}")


def plot_risk_comparison(compare_result: dict) -> None:
    """
    Grouped bar chart: predicted Victim_Voltage_V per configuration,
    colour-coded by overall risk category.
    """
    df   = compare_result["comparison_df"]
    if "Pred_Victim_Voltage_V" not in df.columns:
        return

    df_valid = df.dropna(subset=["Pred_Victim_Voltage_V"]).copy()
    if df_valid.empty:
        return

    palette = {
        RISK_LOWER:        "#4CAF50",
        RISK_INTERMEDIATE: "#FF9800",
        RISK_HIGHER:       "#F44336",
        RISK_UNKNOWN:      "#9E9E9E",
    }
    colours = [palette.get(r, "#9E9E9E")
               for r in df_valid["Overall_Risk_Category"]]

    fig, ax = plt.subplots(figsize=(max(8, len(df_valid) * 1.4), 5))
    bars = ax.bar(df_valid["Configuration_ID"],
                  df_valid["Pred_Victim_Voltage_V"],
                  color=colours, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, df_valid["Pred_Victim_Voltage_V"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r)
                        for r, c in palette.items()
                        if r != RISK_UNKNOWN]
    ax.legend(handles=legend_elements, title="Risk Category",
              fontsize=8, title_fontsize=9)

    ax.set_title(
        "Predicted Victim_Voltage_V by Configuration\n"
        "[SYNTHETIC DATA — dataset-relative risk categories]",
        fontsize=11
    )
    ax.set_xlabel("Configuration ID")
    ax.set_ylabel("Predicted Victim_Voltage_V (V)")
    fig.tight_layout()
    _save_fig(fig, "17_risk_comparison_victim_voltage.png")


def plot_risk_indicators_heatmap(compare_result: dict) -> None:
    """Heatmap of 0-1 normalised risk indicators across configurations."""
    df = compare_result["comparison_df"]
    ind_cols = [c for c in df.columns if c.startswith("Ind_")]
    if not ind_cols:
        return

    df_valid = df.dropna(subset=ind_cols, how="all").copy()
    if df_valid.empty:
        return

    hmap_data = df_valid.set_index("Configuration_ID")[ind_cols].astype(float)
    hmap_data.columns = [c.replace("Ind_", "") for c in hmap_data.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(ind_cols) * 2), max(4, len(df_valid) * 0.8)))
    sns.heatmap(hmap_data, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"label": "Normalised indicator (0=low, 1=high)"})
    ax.set_title(
        "Normalised Risk Indicators per Configuration and Target\n"
        "[SYNTHETIC DATA — 0=dataset min, 1=dataset max]",
        fontsize=10
    )
    ax.set_xlabel("EMC Target")
    ax.set_ylabel("Configuration")
    fig.tight_layout()
    _save_fig(fig, "18_risk_indicators_heatmap.png")


def plot_risk_radar(compare_result: dict) -> None:
    """
    Radar / spider chart: per-target normalised indicators for each configuration.
    Uses only the indicators (not raw predictions) so different-unit targets
    can be compared on the same axis.
    """
    df      = compare_result["comparison_df"]
    ind_cols = [c for c in df.columns if c.startswith("Ind_")]
    if len(ind_cols) < 3:
        return

    df_valid = df.dropna(subset=ind_cols, how="all").copy()
    if df_valid.empty:
        return

    categories  = [c.replace("Ind_", "") for c in ind_cols]
    N           = len(categories)
    angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles     += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap    = plt.get_cmap("tab10")

    for i, row in df_valid.iterrows():
        cfg_id = row["Configuration_ID"]
        vals   = [float(row[c]) for c in ind_cols]
        vals  += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, linestyle="solid",
                label=cfg_id, color=cmap(i % 10))
        ax.fill(angles, vals, alpha=0.1, color=cmap(i % 10))

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Risk Indicator Radar\n[Normalised 0-1; SYNTHETIC DATA]",
        pad=20, fontsize=10
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "19_risk_radar_chart.png")


def plot_separation_vs_victim_voltage(compare_result: dict) -> None:
    """Scatter: Separation_mm vs Predicted Victim_Voltage_V, coloured by risk."""
    df = compare_result["comparison_df"]
    if "Pred_Victim_Voltage_V" not in df.columns or "Separation_mm" not in df.columns:
        return

    df_valid = df.dropna(subset=["Pred_Victim_Voltage_V", "Separation_mm"]).copy()
    if df_valid.empty:
        return

    palette = {
        RISK_LOWER:        "#4CAF50",
        RISK_INTERMEDIATE: "#FF9800",
        RISK_HIGHER:       "#F44336",
        RISK_UNKNOWN:      "#9E9E9E",
    }
    colours = [palette.get(r, "#9E9E9E")
               for r in df_valid["Overall_Risk_Category"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df_valid["Separation_mm"], df_valid["Pred_Victim_Voltage_V"],
                    c=colours, s=120, edgecolors="black", linewidths=0.5, zorder=3)
    for _, row in df_valid.iterrows():
        ax.annotate(row["Configuration_ID"],
                    (row["Separation_mm"], row["Pred_Victim_Voltage_V"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Separation_mm")
    ax.set_ylabel("Predicted Victim_Voltage_V (V)")
    ax.set_title(
        "Separation vs Predicted Victim Voltage\n"
        "[SYNTHETIC DATA — coloured by overall risk]",
        fontsize=10
    )
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r)
                        for r, c in palette.items() if r != RISK_UNKNOWN]
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "20_separation_vs_victim_voltage.png")


# ============================================================
# SECTION 12 — TASK 11 TEST SUITE
# ============================================================

def run_test_suite() -> Tuple[Dict[str, dict], dict]:
    """
    Task 11: Run the 7 test configurations and return results.

    Test configurations:
      TC1 — Large separation, short overlap                 (geometry → likely lower coupling)
      TC2 — Moderate separation, moderate overlap           (middle geometry)
      TC3 — Small separation, long overlap                  (geometry → likely higher coupling)
      TC4 — Shielded cables, moderate geometry
      TC5 — Unshielded cables, moderate geometry
      TC6 — Out-of-domain numerical input (Separation_mm=5000 mm)
      TC7 — Missing required input key (should raise error)
    """
    test_configs = {
        "TC1_LargeSep_ShortOvl": {
            "Separation_mm":       150.0,
            "Overlap_mm":          50.0,
            "Cable_Length_mm":     1000.0,
            "Frequency_Hz":        500_000.0,
            "Voltage_V":           48.0,
            "Conductor_Radius_mm": 1.5,
            "Dielectric_Constant": 3.5,
            "Shield_Type":         "Unshielded",
            "Cable_Configuration": "Parallel",
        },
        "TC2_ModerateSep_ModerateOvl": {
            "Separation_mm":       60.0,
            "Overlap_mm":          200.0,
            "Cable_Length_mm":     2000.0,
            "Frequency_Hz":        1_000_000.0,
            "Voltage_V":           110.0,
            "Conductor_Radius_mm": 2.5,
            "Dielectric_Constant": 5.0,
            "Shield_Type":         "Unshielded",
            "Cable_Configuration": "Parallel",
        },
        "TC3_SmallSep_LongOvl": {
            "Separation_mm":       15.0,
            "Overlap_mm":          420.0,
            "Cable_Length_mm":     3500.0,
            "Frequency_Hz":        5_000_000.0,
            "Voltage_V":           230.0,
            "Conductor_Radius_mm": 3.5,
            "Dielectric_Constant": 7.5,
            "Shield_Type":         "Unshielded",
            "Cable_Configuration": "Parallel",
        },
        "TC4_Shielded": {
            "Separation_mm":       40.0,
            "Overlap_mm":          250.0,
            "Cable_Length_mm":     2500.0,
            "Frequency_Hz":        2_000_000.0,
            "Voltage_V":           110.0,
            "Conductor_Radius_mm": 2.0,
            "Dielectric_Constant": 4.5,
            "Shield_Type":         "Shielded",
            "Cable_Configuration": "Perpendicular",
        },
        "TC5_Unshielded": {
            "Separation_mm":       40.0,
            "Overlap_mm":          250.0,
            "Cable_Length_mm":     2500.0,
            "Frequency_Hz":        2_000_000.0,
            "Voltage_V":           110.0,
            "Conductor_Radius_mm": 2.0,
            "Dielectric_Constant": 4.5,
            "Shield_Type":         "Unshielded",
            "Cable_Configuration": "Perpendicular",
        },
        "TC6_OutOfDomain": {
            "Separation_mm":       5000.0,    # far outside training range [5.24, 199.8]
            "Overlap_mm":          250.0,
            "Cable_Length_mm":     2500.0,
            "Frequency_Hz":        2_000_000.0,
            "Voltage_V":           110.0,
            "Conductor_Radius_mm": 2.0,
            "Dielectric_Constant": 4.5,
            "Shield_Type":         "Unshielded",
            "Cable_Configuration": "Parallel",
        },
    }

    # TC7 — Missing key (should raise; captured separately)
    tc7_config = {
        "Separation_mm":       40.0,
        # Frequency_Hz intentionally omitted
        "Overlap_mm":          200.0,
        "Cable_Length_mm":     2000.0,
        "Voltage_V":           110.0,
        "Conductor_Radius_mm": 2.0,
        "Dielectric_Constant": 4.5,
        "Shield_Type":         "Unshielded",
        "Cable_Configuration": "Parallel",
    }

    # Run comparison for TC1–TC6
    compare_result = compare_configurations(test_configs)

    # TC7 — test error handling
    print("\n  [TC7] Missing required input key test ...")
    tc7_result = {"config_id": "TC7_MissingKey", "error": None, "caught": False}
    try:
        analyze_configuration(tc7_config, config_id="TC7_MissingKey")
        tc7_result["error"] = "No exception raised — UNEXPECTED"
    except Exception as e:
        tc7_result["caught"] = True
        tc7_result["error"] = str(e)
        print(f"    [PASS] Exception caught as expected: {type(e).__name__}")

    return compare_result, tc7_result


# ============================================================
# SECTION 13 — MAIN ENTRY POINT
# ============================================================

def main():
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    print("=" * 68)
    print("EMC-GUARD  |  Module 3  |  Intelligent Configuration Risk Analysis")
    print("=" * 68)
    print("[SYNTHETIC DATA — dataset-relative risk assessment prototype]")

    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Load performance metadata to embed in report
    perf     = _load_model_performance()
    ds_type  = perf.get("dataset_type", "UNKNOWN")

    # Run test suite
    print("\n[1/4] Running test configuration suite ...")
    compare_result, tc7_result = run_test_suite()

    # Collect all test results into a flat list for the report
    all_results = [v for v in compare_result["results"].values()]
    all_results.append(tc7_result)   # TC7 has no full result dict, shown as error entry

    # Save CSV
    print("\n[2/4] Saving output files ...")
    save_results_csv(compare_result["comparison_df"])
    save_assessment_report(compare_result, all_results, ds_type)

    # Visualisations
    print("\n[3/4] Generating visualisations ...")
    plot_risk_comparison(compare_result)
    plot_risk_indicators_heatmap(compare_result)
    plot_risk_radar(compare_result)
    plot_separation_vs_victim_voltage(compare_result)

    # Print final comparison summary
    print("\n[4/4] Final risk comparison table:")
    print(compare_result["summary_text"])

    # TC7 result
    print("\n[TC7 — Missing Input Key Test]")
    status = "[PASS]" if tc7_result.get("caught") else "[FAIL]"
    print(f"  {status} Exception caught: {tc7_result['error'][:100]}")

    print("\n" + "=" * 68)
    print("MODULE 3 COMPLETE")
    print("-" * 68)
    print("  Output files:")
    print("    reports/risk_analysis_results.csv")
    print("    reports/risk_assessment_report.txt")
    print("    reports/risk_thresholds.json")
    print("    reports/visualisations/17_risk_comparison_victim_voltage.png")
    print("    reports/visualisations/18_risk_indicators_heatmap.png")
    print("    reports/visualisations/19_risk_radar_chart.png")
    print("    reports/visualisations/20_separation_vs_victim_voltage.png")
    print("=" * 68)
    print("[DONE] Risk analysis engine complete.")
    print("  REMINDER: All risk categories are DATASET-RELATIVE.")
    print("  They are NOT certified EMC safety thresholds.")


if __name__ == "__main__":
    main()

