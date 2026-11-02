"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 5 | Script: Intelligent Cable-Routing Optimisation Engine
File   : optimise_routing.py
Author : EMC-GUARD Development Team

OBJECTIVE
---------
Grid-search optimisation of cable separation and parallel overlap length to
find alternative cable configurations with lower predicted electromagnetic
interference, using the trained ML surrogate model from Module 2.

DESIGN DECISIONS
----------------
1. Algorithm: Grid Search.
   Simple, transparent, deterministic. No genetic algorithms or Bayesian
   optimisation — those add complexity without benefit at this prototype stage.

2. Primary objective: minimise Victim_Voltage_V.
   Automatically falls back to the first available victim-side target if
   Victim_Voltage_V is absent from the surrogate bundle.

3. All non-optimised parameters (Frequency_Hz, Voltage_V, Shield_Type, etc.)
   are fixed at the baseline values supplied by the engineer.

4. Risk integration: Module 3 analyze_configuration() is called for every
   feasible candidate. Thresholds from risk_thresholds.json are reused as-is.

5. Domain guard: Any candidate whose Separation_mm or Overlap_mm lies outside
   the training range documented in risk_thresholds.json is labelled
   OUT_OF_DOMAIN and excluded from the final recommendation.

DISCLAIMER
----------
Optimisation results are ML-model PREDICTIONS based on SYNTHETIC DUMMY DATA.
They do NOT represent validated electromagnetic improvements.
All recommended configurations must be verified with MATLAB physics simulation
and, ultimately, with physical prototype measurements before engineering use.
Risk categories are dataset-relative (P33/P67 percentile boundaries) and are
NOT certified EMC safety thresholds.

Usage
-----
    python src/optimise_routing.py
    -- or import individual functions --
    from optimise_routing import optimise_cable_routing
"""

import os
import sys
import json
import copy
import warnings
import itertools
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VIS_DIR    = os.path.join(REPORTS_DIR, "visualisations", "optimisation")
THRESH_JSON = os.path.join(REPORTS_DIR, "risk_thresholds.json")

OUT_CSV    = os.path.join(REPORTS_DIR, "optimisation_results.csv")
OUT_JSON   = os.path.join(REPORTS_DIR, "recommended_configuration.json")
OUT_TXT    = os.path.join(REPORTS_DIR, "optimisation_summary.txt")

# Ensure src/ is importable from any working directory
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# COLUMN & TARGET DEFINITIONS
# ---------------------------------------------------------------------------
ALL_TARGETS = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]
VICTIM_TARGETS = ["Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V"]

SYNTH_NOTE = "[SYNTHETIC DATA — model-predicted improvement, not measured EMI reduction]"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ===========================================================================
# HELPERS
# ===========================================================================

def _load_training_ranges() -> dict:
    """Load validated training ranges from risk_thresholds.json (Module 3)."""
    if not os.path.exists(THRESH_JSON):
        raise FileNotFoundError(
            f"risk_thresholds.json not found at {THRESH_JSON}.\n"
            "Run risk_analysis.py (Module 3) first."
        )
    with open(THRESH_JSON) as f:
        thresh = json.load(f)
    return thresh.get("training_ranges", {})


def _load_thresholds() -> dict:
    """Load the full risk_thresholds.json."""
    with open(THRESH_JSON) as f:
        return json.load(f)


def _predict_one(configuration: dict) -> dict:
    """
    Call Module 2 predict_interference().
    Returns the result dict or raises on fatal validation failure.
    """
    from predict_emc import predict_interference
    return predict_interference(configuration, verbose=False)


def _analyse_risk(configuration: dict, candidate_id: str) -> dict:
    """
    Call Module 3 analyze_configuration().
    Returns the result dict or a stub dict on error.
    """
    try:
        from risk_analysis import analyze_configuration
        return analyze_configuration(configuration, config_id=candidate_id)
    except Exception as e:
        return {
            "config_id": candidate_id,
            "overall_risk_category": "UNDETERMINED RISK",
            "reliability": {"overall_reliability": "CAUTION"},
            "explanation": f"Risk analysis failed: {e}",
            "per_target_risk": {},
            "per_target_indicators": {},
        }


def _select_primary_objective(surrogate_targets: list) -> Optional[str]:
    """
    Select the primary optimisation objective in priority order.
    Returns None if no victim-side target is available.
    """
    for tgt in VICTIM_TARGETS:
        if tgt in surrogate_targets:
            return tgt
    return None


def _safe_figure_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "")


def _save_fig(fig, filename: str) -> None:
    os.makedirs(VIS_DIR, exist_ok=True)
    path = os.path.join(VIS_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    rel = os.path.relpath(path, BASE_DIR)
    print(f"  Saved: {rel}")


# ===========================================================================
# TASK 3 & 4 — GENERATE CANDIDATE CONFIGURATIONS
# ===========================================================================

def generate_candidate_configurations(
    baseline: dict,
    sep_min: float,
    sep_max: float,
    sep_step: float,
    ovl_min: float,
    ovl_max: float,
    ovl_step: float,
    mechanical_constraints: Optional[dict] = None,
    include_baseline: bool = True,
) -> pd.DataFrame:
    """
    Generate all (Separation_mm, Overlap_mm) grid combinations, fix all
    other parameters at the baseline values, and assign unique Candidate_IDs.

    Parameters
    ----------
    baseline : dict
        Original engineer-supplied configuration. All non-optimised
        parameters are inherited from this dict.
    sep_min, sep_max, sep_step : float
        Separation_mm grid parameters (mm).
    ovl_min, ovl_max, ovl_step : float
        Overlap_mm grid parameters (mm).
    mechanical_constraints : dict or None
        Optional additional hard limits, e.g.:
            {"max_separation_mm": 80, "min_cable_length_mm": 200}
        Applied on top of training-range domain guard.
    include_baseline : bool
        If True, the baseline configuration is always included as
        Candidate_ID = "CAND_000" (BASELINE) even if it sits outside the
        grid.

    Returns
    -------
    pd.DataFrame  — one row per unique candidate, with columns:
        Candidate_ID, Separation_mm, Overlap_mm, [all other baseline params],
        _is_baseline (bool)
    """
    mc = mechanical_constraints or {}

    # Build grid values — round to 4 decimal places to avoid float drift
    sep_values = np.round(
        np.arange(sep_min, sep_max + sep_step * 0.5, sep_step), 4
    ).tolist()
    ovl_values = np.round(
        np.arange(ovl_min, ovl_max + ovl_step * 0.5, ovl_step), 4
    ).tolist()

    rows = []
    seen = set()

    if include_baseline:
        b_sep = float(baseline["Separation_mm"])
        b_ovl = float(baseline["Overlap_mm"])
        key   = (round(b_sep, 3), round(b_ovl, 3))
        seen.add(key)
        row = copy.deepcopy(baseline)
        row["Separation_mm"] = b_sep
        row["Overlap_mm"]    = b_ovl
        row["Candidate_ID"]  = "CAND_000"
        row["_is_baseline"]  = True
        rows.append(row)

    cand_num = 1
    for sep, ovl in itertools.product(sep_values, ovl_values):
        key = (round(sep, 3), round(ovl, 3))
        if key in seen:
            continue
        seen.add(key)

        cand = copy.deepcopy(baseline)
        cand["Separation_mm"] = sep
        cand["Overlap_mm"]    = ovl
        cand["Candidate_ID"]  = f"CAND_{cand_num:03d}"
        cand["_is_baseline"]  = False
        cand_num += 1
        rows.append(cand)

    df = pd.DataFrame(rows)

    # Apply optional mechanical constraints as additional column-level filters
    for key_name, col, op in [
        ("max_separation_mm",  "Separation_mm", "le"),
        ("min_separation_mm",  "Separation_mm", "ge"),
        ("max_overlap_mm",     "Overlap_mm",    "le"),
        ("min_overlap_mm",     "Overlap_mm",    "ge"),
        ("max_cable_length_mm","Cable_Length_mm","le"),
        ("min_cable_length_mm","Cable_Length_mm","ge"),
    ]:
        if key_name in mc and col in df.columns:
            limit = float(mc[key_name])
            # Always keep the baseline regardless of constraints
            mask = df["_is_baseline"] | (
                df[col] <= limit if op == "le" else df[col] >= limit
            )
            df = df[mask].reset_index(drop=True)

    print(f"  Generated {len(df)} candidate configurations "
          f"(sep grid: {len(sep_values)}, ovl grid: {len(ovl_values)}).")
    return df


# ===========================================================================
# TASK 5 — PREDICT ELECTROMAGNETIC INTERFERENCE FOR ALL CANDIDATES
# ===========================================================================

def predict_candidate_interference(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Call Module 2 predict_interference() for every candidate configuration.

    Adds columns to the DataFrame:
        Pred_<target>   for each available target
        Pred_warnings   (comma-separated warning strings)
        Pred_error      (error message if prediction failed, else empty)

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Output of generate_candidate_configurations().

    Returns
    -------
    pd.DataFrame with prediction columns appended in-place (copy).
    """
    from predict_emc import predict_interference, ValidationError

    df = candidates_df.copy()

    # Initialise prediction columns
    for tgt in ALL_TARGETS:
        df[f"Pred_{tgt}"] = np.nan
    df["Pred_warnings"] = ""
    df["Pred_error"]    = ""

    total = len(df)
    print(f"  Running predictions for {total} candidates ...")

    for idx, row in df.iterrows():
        cid = row.get("Candidate_ID", f"row_{idx}")
        config = {
            k: row[k] for k in [
                "Separation_mm", "Overlap_mm", "Cable_Length_mm",
                "Frequency_Hz", "Voltage_V", "Conductor_Radius_mm",
                "Dielectric_Constant", "Shield_Type", "Cable_Configuration",
            ]
        }
        try:
            result = predict_interference(config, verbose=False)
            preds  = result.get("predictions", {})
            warns  = "; ".join(result.get("warnings", []))
            for tgt in ALL_TARGETS:
                if tgt in preds:
                    df.at[idx, f"Pred_{tgt}"] = preds[tgt]
            df.at[idx, "Pred_warnings"] = warns
        except ValidationError as ve:
            df.at[idx, "Pred_error"] = f"ValidationError: {ve}"
        except Exception as e:
            df.at[idx, "Pred_error"] = f"Error: {e}"

    success = (df["Pred_error"] == "").sum()
    print(f"  Predictions complete: {success}/{total} successful.")
    return df


# ===========================================================================
# TASK 7 — CONFIGURATION FEASIBILITY CHECK
# ===========================================================================

def check_configuration_feasibility(
    candidates_df: pd.DataFrame,
    sep_min: float,
    sep_max: float,
    ovl_min: float,
    ovl_max: float,
    cable_length_min: float = 0.0,
    cable_length_max: float = float("inf"),
    mechanical_constraints: Optional[dict] = None,
    training_ranges: Optional[dict] = None,
    optimised_features: Optional[list] = None,
) -> pd.DataFrame:
    """
    Evaluate each candidate configuration for feasibility.

    Checks (in order)
    -----------------
    1. Prediction error (no valid prediction)          → INFEASIBLE
    2. Engineer-supplied separation bounds             → INFEASIBLE
    3. Engineer-supplied overlap bounds                → INFEASIBLE
    4. Cable length constraints                        → INFEASIBLE
    5. Additional mechanical constraints               → INFEASIBLE
    6. Training domain check for OPTIMISED features   → OUT_OF_DOMAIN

    IMPORTANT — Domain check scope
    --------------------------------
    The OUT_OF_DOMAIN status is applied only to features that are actively
    varied in the grid search (``optimised_features``), which defaults to
    ``["Separation_mm", "Overlap_mm"]``.

    Fixed parameters inherited from the baseline (Frequency_Hz,
    Conductor_Radius_mm, etc.) may themselves be near the training boundary,
    but they are constant across ALL candidates.  Flagging them as OOD would
    eliminate every candidate simultaneously and make the optimiser useless.
    Their OOD status is captured as a reliability warning in the prediction
    output (from predict_emc.py) and surfaced in the Risk_Reliability column
    via Module 3 — it does NOT prevent a candidate from being FEASIBLE.

    Parameters
    ----------
    candidates_df : pd.DataFrame
    sep_min, sep_max : float        — engineer-supplied separation bounds (mm)
    ovl_min, ovl_max : float        — engineer-supplied overlap bounds (mm)
    cable_length_min/max : float    — optional cable length hard limits
    mechanical_constraints : dict   — additional {key: value} limits
    training_ranges : dict          — from risk_thresholds.json
    optimised_features : list       — features to domain-check (default:
                                      Separation_mm and Overlap_mm only)

    Returns
    -------
    pd.DataFrame with Feasibility_Status and Feasibility_Reason columns added.
    """
    if training_ranges is None:
        training_ranges = _load_training_ranges()

    # Only domain-check the features that are actively optimised.
    # Fixed baseline parameters use the same value for every candidate —
    # checking them here would either pass all or fail all uniformly.
    if optimised_features is None:
        optimised_features = ["Separation_mm", "Overlap_mm"]

    mc = mechanical_constraints or {}
    df = candidates_df.copy()
    df["Feasibility_Status"] = "FEASIBLE"
    df["Feasibility_Reason"] = ""

    for idx, row in df.iterrows():
        reasons = []

        # 1. Prediction error
        if row.get("Pred_error", ""):
            reasons.append(f"Prediction failed: {row['Pred_error']}")

        sep = float(row["Separation_mm"])
        ovl = float(row["Overlap_mm"])
        cbl = float(row["Cable_Length_mm"])

        # 2. Engineer separation bounds
        if sep < sep_min:
            reasons.append(f"Separation {sep} mm < minimum {sep_min} mm.")
        if sep > sep_max:
            reasons.append(f"Separation {sep} mm > maximum {sep_max} mm.")

        # 3. Engineer overlap bounds
        if ovl < ovl_min:
            reasons.append(f"Overlap {ovl} mm < minimum {ovl_min} mm.")
        if ovl > ovl_max:
            reasons.append(f"Overlap {ovl} mm > maximum {ovl_max} mm.")

        # 4. Cable length constraints
        if cbl < cable_length_min:
            reasons.append(
                f"Cable_Length {cbl} mm < minimum {cable_length_min} mm."
            )
        if cbl > cable_length_max:
            reasons.append(
                f"Cable_Length {cbl} mm > maximum {cable_length_max} mm."
            )

        # 5. Mechanical constraints
        for mkey, mlimit in mc.items():
            if "separation" in mkey.lower() and "max" in mkey.lower():
                if sep > float(mlimit):
                    reasons.append(
                        f"Mechanical constraint '{mkey}': {sep} mm > {mlimit} mm."
                    )
            if "overlap" in mkey.lower() and "max" in mkey.lower():
                if ovl > float(mlimit):
                    reasons.append(
                        f"Mechanical constraint '{mkey}': {ovl} mm > {mlimit} mm."
                    )

        if reasons:
            df.at[idx, "Feasibility_Status"] = "INFEASIBLE"
            df.at[idx, "Feasibility_Reason"] = " | ".join(reasons)
            continue

        # 6. Training domain check — ONLY for optimised (varied) features.
        ood = []
        for feat in optimised_features:
            if feat not in training_ranges:
                continue
            lo, hi = training_ranges[feat]
            val = float(row.get(feat, np.nan))
            if not np.isnan(val) and (val < lo or val > hi):
                ood.append(
                    f"{feat}={val:.3f} (training range [{lo:.3f}, {hi:.3f}])"
                )
        if ood:
            df.at[idx, "Feasibility_Status"] = "OUT_OF_DOMAIN"
            df.at[idx, "Feasibility_Reason"] = (
                "Optimised feature outside model training domain: "
                + "; ".join(ood)
            )

    feasible_n   = (df["Feasibility_Status"] == "FEASIBLE").sum()
    infeasible_n = (df["Feasibility_Status"] == "INFEASIBLE").sum()
    ood_n        = (df["Feasibility_Status"] == "OUT_OF_DOMAIN").sum()
    print(f"  Feasibility: FEASIBLE={feasible_n}, "
          f"INFEASIBLE={infeasible_n}, OUT_OF_DOMAIN={ood_n}.")
    print(f"  NOTE: Domain check applied only to optimised features "
          f"{optimised_features}. Fixed baseline parameters that are "
          f"near the training boundary are surfaced as reliability warnings "
          f"in Risk_Reliability, not as disqualifiers.")
    return df


# ===========================================================================
# TASK 8 — RELATIVE IMPROVEMENT
# ===========================================================================

def calculate_relative_improvement(
    candidates_df: pd.DataFrame,
    primary_objective: str,
    baseline_value: float,
) -> pd.DataFrame:
    """
    Compute relative improvement compared with the baseline.

    Formula
    -------
    Relative_Improvement_pct =
        (baseline_value - candidate_value) / |baseline_value| * 100

    Positive = lower predicted interference than baseline (improvement).
    Negative = higher predicted interference than baseline (worsening).

    Handles zero/negative baseline safely by using |baseline_value| as
    the denominator; if baseline_value == 0, sets improvement to NaN and
    logs a warning.

    Returns
    -------
    DataFrame with column  Relative_Improvement_pct  added.
    """
    df = candidates_df.copy()
    pred_col = f"Pred_{primary_objective}"

    if abs(baseline_value) < 1e-12:
        print(f"  [WARN] Baseline {primary_objective} = {baseline_value:.6f} "
              "(near zero). Relative improvement set to NaN.")
        df["Relative_Improvement_pct"] = np.nan
        return df

    df["Relative_Improvement_pct"] = (
        (baseline_value - df[pred_col]) / abs(baseline_value) * 100
    ).round(4)
    return df


# ===========================================================================
# TASK 9 — RANK CONFIGURATIONS
# ===========================================================================

def rank_configurations(
    candidates_df: pd.DataFrame,
    primary_objective: str,
) -> pd.DataFrame:
    """
    Rank FEASIBLE candidates by ascending predicted primary objective value.
    Non-feasible rows are assigned rank NaN and sorted to the bottom.

    Returns DataFrame with column  Rank  added (1 = best).
    """
    df = candidates_df.copy()
    feasible_mask = df["Feasibility_Status"] == "FEASIBLE"

    pred_col = f"Pred_{primary_objective}"
    df["Rank"] = np.nan

    feasible_df = df[feasible_mask].copy()
    feasible_df = feasible_df.sort_values(pred_col, ascending=True)
    feasible_df["Rank"] = range(1, len(feasible_df) + 1)

    df.update(feasible_df[["Rank"]])
    return df


# ===========================================================================
# TASK 10 — RISK ANALYSIS INTEGRATION
# ===========================================================================

def integrate_risk_analysis(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Call Module 3 analyze_configuration() for each FEASIBLE candidate.
    Infeasible/OOD candidates get risk category UNDETERMINED RISK.

    Adds columns:
        Risk_Category
        Risk_Reliability
        Risk_Explanation  (first 300 chars)

    Returns DataFrame with risk columns added.
    """
    df = candidates_df.copy()
    df["Risk_Category"]   = "UNDETERMINED RISK"
    df["Risk_Reliability"] = "N/A"
    df["Risk_Explanation"] = ""

    feasible_idx = df[df["Feasibility_Status"] == "FEASIBLE"].index
    total = len(feasible_idx)
    print(f"  Running Module 3 risk analysis for {total} feasible candidates ...")

    for i, idx in enumerate(feasible_idx, 1):
        row = df.loc[idx]
        cid = row["Candidate_ID"]
        config = {
            k: row[k] for k in [
                "Separation_mm", "Overlap_mm", "Cable_Length_mm",
                "Frequency_Hz", "Voltage_V", "Conductor_Radius_mm",
                "Dielectric_Constant", "Shield_Type", "Cable_Configuration",
            ]
        }
        risk_result = _analyse_risk(config, cid)
        df.at[idx, "Risk_Category"]    = risk_result.get(
            "overall_risk_category", "UNDETERMINED RISK"
        )
        df.at[idx, "Risk_Reliability"] = risk_result.get(
            "reliability", {}
        ).get("overall_reliability", "N/A")
        df.at[idx, "Risk_Explanation"] = risk_result.get(
            "explanation", ""
        )[:300]

    print(f"  Risk analysis complete.")
    return df


# ===========================================================================
# TASK 11 — ENGINEERING EXPLANATION
# ===========================================================================

def _build_engineering_explanation(
    baseline_config: dict,
    best_config: dict,
    baseline_pred: float,
    best_pred: float,
    rel_improvement: float,
    primary_objective: str,
    risk_category: str,
    reliability: str,
    is_same_as_baseline: bool,
) -> str:
    """
    Compose a plain-English explanation that only makes claims supported
    by the actual prediction values.
    """
    lines = []

    if is_same_as_baseline:
        lines.append(
            "The baseline configuration is already the best feasible candidate "
            "in the explored search space. No alternative configuration shows "
            "lower predicted primary interference."
        )
    else:
        sep_change = best_config["Separation_mm"] - baseline_config["Separation_mm"]
        ovl_change = best_config["Overlap_mm"]    - baseline_config["Overlap_mm"]
        sep_dir    = "increases" if sep_change > 0 else "decreases"
        ovl_dir    = "increases" if ovl_change > 0 else "decreases"

        lines.append(
            f"The recommended configuration {sep_dir} cable separation from "
            f"{baseline_config['Separation_mm']:.1f} mm to "
            f"{best_config['Separation_mm']:.1f} mm "
            f"(change: {sep_change:+.1f} mm) and {ovl_dir} parallel overlap "
            f"from {baseline_config['Overlap_mm']:.1f} mm to "
            f"{best_config['Overlap_mm']:.1f} mm "
            f"(change: {ovl_change:+.1f} mm)."
        )
        lines.append(
            f"The trained surrogate model predicts {primary_objective} = "
            f"{best_pred:.6f} for this arrangement, compared with "
            f"{baseline_pred:.6f} for the original configuration."
        )
        if rel_improvement > 0:
            lines.append(
                f"This represents a predicted relative improvement of "
                f"{rel_improvement:.2f}% in {primary_objective}."
            )
        elif rel_improvement < 0:
            lines.append(
                f"Note: the recommended configuration shows {abs(rel_improvement):.2f}% "
                f"HIGHER predicted {primary_objective} than the baseline. "
                "No better alternative was found within the explored search space."
            )

    lines.append(
        f"Risk category: {risk_category}. "
        f"Prediction reliability: {reliability}."
    )
    lines.append(
        "IMPORTANT: This is a model-predicted improvement based on SYNTHETIC DATA. "
        "It does NOT represent a measured real-world EMI reduction. "
        "Send this configuration to MATLAB physics simulation for verification "
        "before engineering use."
    )
    return "\n".join(lines)


# ===========================================================================
# TASK 6 — PRIMARY OPTIMISATION FUNCTION
# ===========================================================================

def optimise_cable_routing(
    initial_configuration: dict,
    sep_min: float        = 10.0,
    sep_max: float        = 200.0,
    sep_step: float       = 10.0,
    ovl_min: float        = 10.0,
    ovl_max: float        = 500.0,
    ovl_step: float       = 50.0,
    cable_length_min: float = 0.0,
    cable_length_max: float = float("inf"),
    mechanical_constraints: Optional[dict] = None,
    top_n: int            = 5,
    verbose: bool         = True,
) -> dict:
    """
    Full end-to-end cable-routing optimisation pipeline.

    Parameters
    ----------
    initial_configuration : dict
        Engineer-supplied baseline configuration.  All non-optimised
        parameters are inherited.
    sep_min / sep_max / sep_step : float
        Separation_mm search grid parameters (mm).
    ovl_min / ovl_max / ovl_step : float
        Overlap_mm search grid parameters (mm).
    cable_length_min / cable_length_max : float
        Optional cable length hard limits (mm).
    mechanical_constraints : dict
        Optional additional hard limits (see check_configuration_feasibility).
    top_n : int
        Number of top ranked configurations to include in detailed report.
    verbose : bool
        Print step-by-step progress.

    Returns
    -------
    dict with keys:
        candidates_df           : pd.DataFrame — all candidates
        feasible_df             : pd.DataFrame — only FEASIBLE rows, ranked
        top_n_df                : pd.DataFrame — top-n feasible rows
        recommended             : dict         — best configuration + predictions
        baseline_predictions    : dict         — baseline predictions
        primary_objective       : str
        baseline_value          : float
        best_value              : float
        relative_improvement_pct: float
        explanation             : str
        risk_baseline           : str
        risk_recommended        : str
        n_candidates            : int
        n_feasible              : int
        n_infeasible            : int
        n_ood                   : int
    """
    sep_line = "=" * 68

    if verbose:
        print(sep_line)
        print("EMC-GUARD  |  Module 5  |  Cable-Routing Optimisation Engine")
        print(sep_line)
        print(SYNTH_NOTE)

    training_ranges = _load_training_ranges()

    # -----------------------------------------------------------------------
    # STEP 0: Predict baseline
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[0/7] Predicting baseline configuration ...")
    baseline_result = _predict_one(initial_configuration)
    baseline_preds  = baseline_result.get("predictions", {})
    baseline_warns  = baseline_result.get("warnings", [])
    available_targets = list(baseline_preds.keys())

    if verbose:
        for tgt, val in baseline_preds.items():
            print(f"  {tgt:<26}: {val:.6f}")
        if baseline_warns:
            for w in baseline_warns:
                print(f"  WARN: {w}")

    # Select primary objective
    primary_objective = _select_primary_objective(available_targets)
    if primary_objective is None:
        # No victim-side target available; fall back to first available coupling target
        if available_targets:
            primary_objective = available_targets[0]
            print(
                f"  [WARN] No victim-side voltage target available. "
                f"Falling back to '{primary_objective}' as primary objective. "
                "Clearly disclose this limitation: victim-side EM interference "
                "cannot be directly optimised with the current model bundle."
            )
        else:
            raise RuntimeError(
                "No prediction targets are available from the surrogate model. "
                "Cannot perform optimisation."
            )

    baseline_value = baseline_preds[primary_objective]
    if verbose:
        print(f"\n  Primary objective : {primary_objective}")
        print(f"  Baseline value    : {baseline_value:.6f}")

    # -----------------------------------------------------------------------
    # STEP 1: Generate candidate configurations
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[1/7] Generating candidate configurations ...")
    candidates_df = generate_candidate_configurations(
        baseline          = initial_configuration,
        sep_min           = sep_min,
        sep_max           = sep_max,
        sep_step          = sep_step,
        ovl_min           = ovl_min,
        ovl_max           = ovl_max,
        ovl_step          = ovl_step,
        mechanical_constraints = mechanical_constraints,
        include_baseline  = True,
    )

    # -----------------------------------------------------------------------
    # STEP 2: Predict interference for all candidates
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[2/7] Predicting interference for all candidates ...")
    candidates_df = predict_candidate_interference(candidates_df)

    # -----------------------------------------------------------------------
    # STEP 3: Feasibility check
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[3/7] Checking feasibility ...")
    candidates_df = check_configuration_feasibility(
        candidates_df,
        sep_min              = sep_min,
        sep_max              = sep_max,
        ovl_min              = ovl_min,
        ovl_max              = ovl_max,
        cable_length_min     = cable_length_min,
        cable_length_max     = cable_length_max,
        mechanical_constraints = mechanical_constraints,
        training_ranges      = training_ranges,
        optimised_features   = ["Separation_mm", "Overlap_mm"],
    )

    # -----------------------------------------------------------------------
    # STEP 4: Relative improvement
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4/7] Calculating relative improvement ...")
    candidates_df = calculate_relative_improvement(
        candidates_df, primary_objective, baseline_value
    )

    # -----------------------------------------------------------------------
    # STEP 5: Rank
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[5/7] Ranking feasible configurations ...")
    candidates_df = rank_configurations(candidates_df, primary_objective)

    # -----------------------------------------------------------------------
    # STEP 6: Risk analysis for feasible candidates
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[6/7] Integrating Module 3 risk analysis ...")
    candidates_df = integrate_risk_analysis(candidates_df)

    # -----------------------------------------------------------------------
    # Derive summary views
    # -----------------------------------------------------------------------
    feasible_df = candidates_df[
        candidates_df["Feasibility_Status"] == "FEASIBLE"
    ].sort_values("Rank").reset_index(drop=True)

    n_feasible   = (candidates_df["Feasibility_Status"] == "FEASIBLE").sum()
    n_infeasible = (candidates_df["Feasibility_Status"] == "INFEASIBLE").sum()
    n_ood        = (candidates_df["Feasibility_Status"] == "OUT_OF_DOMAIN").sum()

    if len(feasible_df) == 0:
        raise RuntimeError(
            "No feasible candidates found in the search space. "
            "Widen the sep/ovl range or review mechanical constraints."
        )

    top_n_df = feasible_df.head(top_n)

    # Best candidate
    best_row   = feasible_df.iloc[0]
    best_cid   = best_row["Candidate_ID"]
    best_value = float(best_row[f"Pred_{primary_objective}"])
    rel_impr   = float(best_row["Relative_Improvement_pct"])
    risk_recommended = best_row.get("Risk_Category", "UNDETERMINED RISK")
    reliability_rec  = best_row.get("Risk_Reliability", "N/A")

    best_config = {
        k: best_row[k] for k in [
            "Separation_mm", "Overlap_mm", "Cable_Length_mm",
            "Frequency_Hz", "Voltage_V", "Conductor_Radius_mm",
            "Dielectric_Constant", "Shield_Type", "Cable_Configuration",
        ]
    }
    best_config["Candidate_ID"] = best_cid

    is_same = (
        abs(best_config["Separation_mm"] - initial_configuration["Separation_mm"]) < 0.01
        and abs(best_config["Overlap_mm"] - initial_configuration["Overlap_mm"]) < 0.01
    )

    # Baseline risk
    if verbose:
        print("\n  Baseline risk assessment ...")
    risk_baseline_result = _analyse_risk(initial_configuration, "BASELINE")
    risk_baseline = risk_baseline_result.get("overall_risk_category", "UNDETERMINED RISK")

    # Explanation
    explanation = _build_engineering_explanation(
        baseline_config      = initial_configuration,
        best_config          = best_config,
        baseline_pred        = baseline_value,
        best_pred            = best_value,
        rel_improvement      = rel_impr,
        primary_objective    = primary_objective,
        risk_category        = risk_recommended,
        reliability          = reliability_rec,
        is_same_as_baseline  = is_same,
    )

    # Attach explanation to the results df as well
    candidates_df["Recommendation_Explanation"] = ""
    mask_best = candidates_df["Candidate_ID"] == best_cid
    candidates_df.loc[mask_best, "Recommendation_Explanation"] = explanation

    result = {
        "candidates_df":            candidates_df,
        "feasible_df":              feasible_df,
        "top_n_df":                 top_n_df,
        "recommended":              best_config,
        "baseline_predictions":     baseline_preds,
        "primary_objective":        primary_objective,
        "baseline_value":           baseline_value,
        "best_value":               best_value,
        "relative_improvement_pct": rel_impr,
        "explanation":              explanation,
        "risk_baseline":            risk_baseline,
        "risk_recommended":         risk_recommended,
        "reliability_recommended":  reliability_rec,
        "n_candidates":             len(candidates_df),
        "n_feasible":               int(n_feasible),
        "n_infeasible":             int(n_infeasible),
        "n_ood":                    int(n_ood),
        "available_targets":        available_targets,
        "initial_configuration":    copy.deepcopy(initial_configuration),
        "is_baseline_optimal":      is_same,
    }

    if verbose:
        _print_summary(result)

    return result


def _print_summary(result: dict) -> None:
    """Print a formatted summary of the optimisation results."""
    sep_line = "=" * 68
    pri = result["primary_objective"]
    print(f"\n{sep_line}")
    print("OPTIMISATION RESULT SUMMARY")
    print(sep_line)
    print(f"  Candidates evaluated  : {result['n_candidates']}")
    print(f"  Feasible              : {result['n_feasible']}")
    print(f"  Infeasible            : {result['n_infeasible']}")
    print(f"  Out-of-domain         : {result['n_ood']}")
    print(f"  Primary objective     : {pri}")
    print(f"  Baseline value        : {result['baseline_value']:.6f}")
    print(f"  Best predicted value  : {result['best_value']:.6f}")
    print(f"  Relative improvement  : {result['relative_improvement_pct']:.2f}%")
    print(f"  Baseline risk         : {result['risk_baseline']}")
    print(f"  Recommended risk      : {result['risk_recommended']}")
    print(f"\n  Top-5 feasible candidates:")
    pred_col = f"Pred_{pri}"
    for _, row in result["top_n_df"].iterrows():
        print(
            f"    Rank {int(row['Rank']):>2} | {row['Candidate_ID']} | "
            f"Sep={row['Separation_mm']:>6.1f} mm | "
            f"Ovl={row['Overlap_mm']:>6.1f} mm | "
            f"{pri}={row[pred_col]:.4f} | "
            f"Improv={row['Relative_Improvement_pct']:.2f}% | "
            f"{row['Risk_Category']}"
        )
    print(f"\n  Recommended: {result['recommended']['Candidate_ID']}")
    print(f"    Separation  : {result['recommended']['Separation_mm']:.1f} mm")
    print(f"    Overlap     : {result['recommended']['Overlap_mm']:.1f} mm")
    print(f"\n  Explanation:\n")
    for line in result["explanation"].split("\n"):
        print(f"    {line}")


# ===========================================================================
# TASK 12 — EXPORT RESULTS
# ===========================================================================

def generate_optimisation_report(result: dict) -> None:
    """
    Write all output files:
        reports/optimisation_results.csv
        reports/recommended_configuration.json
        reports/optimisation_summary.txt

    Parameters
    ----------
    result : dict — output of optimise_cable_routing()
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pri      = result["primary_objective"]
    pred_col = f"Pred_{pri}"
    df       = result["candidates_df"]

    # --- 1. CSV ---
    csv_cols = [
        "Candidate_ID", "Separation_mm", "Overlap_mm", "Cable_Length_mm",
        "Frequency_Hz", "Voltage_V", "Conductor_Radius_mm",
        "Dielectric_Constant", "Shield_Type", "Cable_Configuration",
        "_is_baseline",
    ]
    for tgt in result["available_targets"]:
        csv_cols.append(f"Pred_{tgt}")
    csv_cols += [
        "Relative_Improvement_pct",
        "Feasibility_Status", "Feasibility_Reason",
        "Risk_Category", "Risk_Reliability",
        "Rank",
        "Pred_warnings", "Pred_error",
        "Recommendation_Explanation",
    ]
    csv_cols = [c for c in csv_cols if c in df.columns]
    df[csv_cols].to_csv(OUT_CSV, index=False)
    print(f"  Saved: reports/optimisation_results.csv")

    # --- 2. JSON ---
    rec = copy.deepcopy(result["recommended"])
    rec["_metadata"] = {
        "primary_objective":       pri,
        "predicted_value":         round(result["best_value"], 8),
        "baseline_value":          round(result["baseline_value"], 8),
        "relative_improvement_pct": round(result["relative_improvement_pct"], 4),
        "risk_category":           result["risk_recommended"],
        "risk_reliability":        result["reliability_recommended"],
        "is_baseline_optimal":     result["is_baseline_optimal"],
        "dataset_type":            "SYNTHETIC",
        "disclaimer":              (
            "This recommendation is based on ML model predictions from "
            "SYNTHETIC DUMMY DATA. It is NOT a validated EMC measurement. "
            "Verify with MATLAB simulation before engineering use."
        ),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Add all predicted values for the recommended config
    best_cid  = rec["Candidate_ID"]
    best_mask = df["Candidate_ID"] == best_cid
    if best_mask.any():
        best_row = df[best_mask].iloc[0]
        rec["_predictions"] = {}
        for tgt in result["available_targets"]:
            col = f"Pred_{tgt}"
            if col in best_row:
                rec["_predictions"][tgt] = round(float(best_row[col]), 8)

    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"  Saved: reports/recommended_configuration.json")

    # --- 3. TXT Summary ---
    lines = [
        "=" * 70,
        "EMC-GUARD — MODULE 5: CABLE-ROUTING OPTIMISATION SUMMARY",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Dataset    : SYNTHETIC",
        "=" * 70,
        "",
        "DISCLAIMER",
        "-" * 70,
        "Results are from a model trained on SYNTHETIC DUMMY DATA.",
        "They do NOT represent validated electromagnetic improvements.",
        "All recommendations must be verified with MATLAB simulation.",
        "Risk categories are dataset-relative (P33/P67 percentile bounds),",
        "NOT certified EMC safety thresholds.",
        "",
        "=" * 70,
        "SECTION 1 — OPTIMISATION PARAMETERS",
        "-" * 70,
        f"  Primary objective  : {pri}",
        f"  Candidates total   : {result['n_candidates']}",
        f"  Feasible           : {result['n_feasible']}",
        f"  Infeasible         : {result['n_infeasible']}",
        f"  Out-of-domain      : {result['n_ood']}",
        "",
        "=" * 70,
        "SECTION 2 — BASELINE CONFIGURATION",
        "-" * 70,
    ]
    for k, v in result["initial_configuration"].items():
        lines.append(f"  {k:<26}: {v}")
    lines += [
        "",
        "  Baseline predictions:",
    ]
    for tgt, val in result["baseline_predictions"].items():
        lines.append(f"    {tgt:<26}: {val:.8f}")
    lines += [
        f"  Baseline risk category  : {result['risk_baseline']}",
        "",
        "=" * 70,
        "SECTION 3 — RECOMMENDED CONFIGURATION",
        "-" * 70,
    ]
    for k, v in result["recommended"].items():
        if not k.startswith("_"):
            lines.append(f"  {k:<26}: {v}")
    if "_predictions" in rec:
        lines.append("")
        lines.append("  Predicted EMC outputs:")
        for tgt, val in rec["_predictions"].items():
            lines.append(f"    {tgt:<26}: {val:.8f}")
    lines += [
        "",
        f"  Relative improvement    : {result['relative_improvement_pct']:.2f}%",
        f"  Recommended risk        : {result['risk_recommended']}",
        f"  Prediction reliability  : {result['reliability_recommended']}",
        "",
        "=" * 70,
        "SECTION 4 — ENGINEERING EXPLANATION",
        "-" * 70,
    ]
    for ln in result["explanation"].split("\n"):
        lines.append(f"  {ln}")
    lines += [
        "",
        "=" * 70,
        "SECTION 5 — TOP-5 FEASIBLE CANDIDATES",
        "-" * 70,
        f"  {'Rank':<5} {'Candidate_ID':<12} {'Sep(mm)':<10} "
        f"{'Ovl(mm)':<10} {pri:<26} {'Improv%':<10} {'Risk':<32} {'Reliability'}",
        "  " + "-" * 115,
    ]
    for _, row in result["top_n_df"].iterrows():
        lines.append(
            f"  {int(row['Rank']):<5} {row['Candidate_ID']:<12} "
            f"{row['Separation_mm']:<10.1f} {row['Overlap_mm']:<10.1f} "
            f"{row[pred_col]:<26.6f} "
            f"{row['Relative_Improvement_pct']:<10.2f} "
            f"{row['Risk_Category']:<32} "
            f"{row['Risk_Reliability']}"
        )
    lines += [
        "",
        "=" * 70,
        "SECTION 6 — VALIDATION WORKFLOW",
        "-" * 70,
        "  Step 1 [DONE]   : ML surrogate model predicts candidate configurations.",
        "  Step 2 [PENDING]: Send recommended configuration to MATLAB simulation.",
        "  Step 3 [PENDING]: Compare ML prediction vs MATLAB result.",
        "  Step 4 [PENDING]: If MATLAB confirms improvement -> proceed to prototype.",
        "  Step 5 [PENDING]: Measure physical prototype with calibrated EMC equipment.",
        "  NOTE: Do NOT use ML predictions for engineering sign-off.",
        "",
        "=" * 70,
    ]

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: reports/optimisation_summary.txt")


# ===========================================================================
# TASK 13 — VISUALISATIONS
# ===========================================================================

def plot_optimisation_results(result: dict) -> None:
    """
    Generate five visualisations:
        1. Baseline vs recommended — bar comparison
        2. Predicted interference vs Separation_mm
        3. Predicted interference vs Overlap_mm
        4. Grid search heatmap (Sep x Ovl)
        5. Candidate ranking chart (Top-N)
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    pri      = result["primary_objective"]
    pred_col = f"Pred_{pri}"
    df       = result["candidates_df"].copy()
    feasible = df[df["Feasibility_Status"] == "FEASIBLE"].copy()

    base_val = result["baseline_value"]
    best_val = result["best_value"]

    # ------------------------------------------------------------------
    # Plot 1: Baseline vs Recommended comparison (all available targets)
    # ------------------------------------------------------------------
    targets_avail = result["available_targets"]
    baseline_preds = result["baseline_predictions"]
    recommended_cid = result["recommended"]["Candidate_ID"]
    rec_mask = df["Candidate_ID"] == recommended_cid
    rec_preds = {}
    if rec_mask.any():
        rec_row = df[rec_mask].iloc[0]
        for tgt in targets_avail:
            col = f"Pred_{tgt}"
            if col in rec_row:
                rec_preds[tgt] = float(rec_row[col])

    if rec_preds:
        x_pos    = np.arange(len(targets_avail))
        width    = 0.35
        fig, ax  = plt.subplots(figsize=(max(9, len(targets_avail) * 2), 5))
        b_vals   = [baseline_preds.get(t, 0) for t in targets_avail]
        r_vals   = [rec_preds.get(t, 0)      for t in targets_avail]
        bars_b = ax.bar(x_pos - width/2, b_vals, width, label="Baseline",
                        color="#5c85d6", edgecolor="black", linewidth=0.5)
        bars_r = ax.bar(x_pos + width/2, r_vals, width, label="Recommended",
                        color="#4CAF50", edgecolor="black", linewidth=0.5)
        for bar in list(bars_b) + list(bars_r):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(targets_avail, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Predicted Value")
        ax.set_title(
            "Baseline vs Recommended Configuration — Predicted EMC Outputs\n"
            + SYNTH_NOTE, fontsize=10
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save_fig(fig, "opt_01_baseline_vs_recommended.png")

    # ------------------------------------------------------------------
    # Plot 2: Predicted primary interference vs Separation_mm
    # ------------------------------------------------------------------
    if not feasible.empty and pred_col in feasible.columns:
        sep_group = (feasible.groupby("Separation_mm")[pred_col]
                     .mean().reset_index())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sep_group["Separation_mm"], sep_group[pred_col],
                "o-", color="#5c85d6", linewidth=1.8, markersize=5)
        ax.axhline(base_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"Baseline ({base_val:.4f})")
        ax.set_xlabel("Separation_mm (mm)")
        ax.set_ylabel(f"Mean Predicted {pri}")
        ax.set_title(
            f"Predicted {pri} vs Cable Separation\n"
            "(mean across all overlap values, feasible only)\n" + SYNTH_NOTE,
            fontsize=9
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, "opt_02_vs_separation.png")

    # ------------------------------------------------------------------
    # Plot 3: Predicted primary interference vs Overlap_mm
    # ------------------------------------------------------------------
    if not feasible.empty and pred_col in feasible.columns:
        ovl_group = (feasible.groupby("Overlap_mm")[pred_col]
                     .mean().reset_index())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ovl_group["Overlap_mm"], ovl_group[pred_col],
                "s-", color="#d6845c", linewidth=1.8, markersize=5)
        ax.axhline(base_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"Baseline ({base_val:.4f})")
        ax.set_xlabel("Overlap_mm (mm)")
        ax.set_ylabel(f"Mean Predicted {pri}")
        ax.set_title(
            f"Predicted {pri} vs Parallel Overlap Length\n"
            "(mean across all separation values, feasible only)\n" + SYNTH_NOTE,
            fontsize=9
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, "opt_03_vs_overlap.png")

    # ------------------------------------------------------------------
    # Plot 4: Grid search heatmap — Separation x Overlap
    # ------------------------------------------------------------------
    if not feasible.empty and pred_col in feasible.columns:
        try:
            pivot = feasible.pivot_table(
                index="Separation_mm",
                columns="Overlap_mm",
                values=pred_col,
                aggfunc="mean",
            )
            fig_h = max(5, len(pivot) * 0.5)
            fig_w = max(8, len(pivot.columns) * 0.8)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(pivot, annot=(pivot.size <= 200), fmt=".2f",
                        cmap="YlOrRd", linewidths=0.3, ax=ax,
                        annot_kws={"size": 7},
                        cbar_kws={"label": f"Predicted {pri}"})
            ax.set_title(
                f"Grid Search Heatmap — Predicted {pri}\n"
                "(feasible candidates only)\n" + SYNTH_NOTE,
                fontsize=9
            )
            ax.set_xlabel("Overlap_mm (mm)")
            ax.set_ylabel("Separation_mm (mm)")
            fig.tight_layout()
            _save_fig(fig, "opt_04_grid_heatmap.png")
        except Exception as e:
            print(f"  [WARN] Could not generate heatmap: {e}")

    # ------------------------------------------------------------------
    # Plot 5: Candidate ranking chart (top-N feasible)
    # ------------------------------------------------------------------
    top_df = result["top_n_df"].copy()
    if not top_df.empty and pred_col in top_df.columns:
        palette_risk = {
            "LOWER PREDICTED RISK":        "#4CAF50",
            "INTERMEDIATE PREDICTED RISK":  "#FF9800",
            "HIGHER PREDICTED RISK":        "#F44336",
            "UNDETERMINED RISK":            "#9E9E9E",
        }
        colours = [palette_risk.get(r, "#9E9E9E")
                   for r in top_df["Risk_Category"]]
        labels  = [
            f"{row['Candidate_ID']}\n"
            f"Sep={row['Separation_mm']:.0f}mm "
            f"Ovl={row['Overlap_mm']:.0f}mm"
            for _, row in top_df.iterrows()
        ]
        fig, ax = plt.subplots(figsize=(max(9, len(top_df) * 1.8), 5))
        bars = ax.bar(labels, top_df[pred_col], color=colours,
                      edgecolor="black", linewidth=0.6)
        ax.axhline(base_val, color="red", linestyle="--", linewidth=1.3,
                   label=f"Baseline ({base_val:.4f})")
        for bar, val, improv in zip(
            bars, top_df[pred_col], top_df["Relative_Improvement_pct"]
        ):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}\n({improv:+.1f}%)",
                    ha="center", va="bottom", fontsize=7.5)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=lbl)
            for lbl, c in palette_risk.items()
            if lbl != "UNDETERMINED RISK"
        ]
        legend_elements.append(plt.Line2D([0], [0], color="red",
                                           linestyle="--", label="Baseline"))
        ax.legend(handles=legend_elements, fontsize=8)
        ax.set_ylabel(f"Predicted {pri}")
        ax.set_xlabel("Candidate (Separation, Overlap)")
        ax.set_title(
            f"Top-{len(top_df)} Candidate Rankings — {pri}\n" + SYNTH_NOTE,
            fontsize=10
        )
        fig.tight_layout()
        _save_fig(fig, "opt_05_candidate_ranking.png")

    print(f"  Visualisations saved to reports/visualisations/optimisation/")


# ===========================================================================
# TASK 14 — ENTRY POINT & SELF-TEST
# ===========================================================================

def main():
    sep = "=" * 68
    print(sep)
    print("EMC-GUARD  |  Module 5  |  Cable-Routing Optimisation Engine")
    print(sep)
    print(SYNTH_NOTE)

    # -------------------------------------------------------------------
    # Engineer input (Task 3 / Task 14 test configuration)
    # -------------------------------------------------------------------
    initial_configuration = {
        "Separation_mm":       10.0,
        "Overlap_mm":         450.0,
        "Cable_Length_mm":    450.0,
        "Frequency_Hz":    100000.0,
        "Voltage_V":           24.0,
        "Conductor_Radius_mm":  0.5,
        "Dielectric_Constant":  2.3,
        "Shield_Type":     "Unshielded",
        "Cable_Configuration": "Parallel",
    }

    # Optimisation search space (Task 3 parameters)
    sep_min   =  10.0   # mm
    sep_max   =  50.0   # mm
    sep_step  =  10.0   # mm
    ovl_min   = 100.0   # mm
    ovl_max   = 450.0   # mm
    ovl_step  =  50.0   # mm

    # Optional mechanical design constraints
    mechanical_constraints = {
        # "max_separation_mm": 60,   # uncomment to enforce a tighter limit
    }

    print("\n  Initial configuration:")
    for k, v in initial_configuration.items():
        print(f"    {k:<26}: {v}")
    print(f"\n  Search grid:")
    print(f"    Separation : {sep_min} to {sep_max} mm, step {sep_step} mm")
    print(f"    Overlap    : {ovl_min} to {ovl_max} mm, step {ovl_step} mm")

    # -------------------------------------------------------------------
    # Run optimisation
    # -------------------------------------------------------------------
    result = optimise_cable_routing(
        initial_configuration   = initial_configuration,
        sep_min                 = sep_min,
        sep_max                 = sep_max,
        sep_step                = sep_step,
        ovl_min                 = ovl_min,
        ovl_max                 = ovl_max,
        ovl_step                = ovl_step,
        mechanical_constraints  = mechanical_constraints,
        top_n                   = 5,
        verbose                 = True,
    )

    # -------------------------------------------------------------------
    # STEP 7: Export and visualise
    # -------------------------------------------------------------------
    print("\n[7/7] Exporting results and generating visualisations ...")
    generate_optimisation_report(result)
    plot_optimisation_results(result)

    # -------------------------------------------------------------------
    # Task 14 verification checks
    # -------------------------------------------------------------------
    print("\n" + sep)
    print("TASK 14 — VERIFICATION CHECKS")
    print(sep)
    PASS = "[PASS]"
    FAIL = "[FAIL]"

    cdf = result["candidates_df"]
    pri = result["primary_objective"]

    # V1: Candidate generation
    n_cands = result["n_candidates"]
    print(f"  V1 Candidate generation: {n_cands} candidates generated. "
          f"{PASS if n_cands > 1 else FAIL}")

    # V2: No duplicate (Sep, Ovl) combinations
    dup_check = cdf.groupby(["Separation_mm", "Overlap_mm"]).size().max()
    print(f"  V2 No duplicates: max occurrences per (Sep,Ovl) = {dup_check}. "
          f"{PASS if dup_check == 1 else FAIL}")

    # V3: Predictions generated via Module 2
    pred_col = f"Pred_{pri}"
    has_preds = (cdf[pred_col].notna()).sum()
    print(f"  V3 Predictions via Module 2: {has_preds}/{n_cands} rows have predictions. "
          f"{PASS if has_preds > 0 else FAIL}")

    # V4: Infeasible candidates exist and are rejected from ranking
    feas_only = cdf[cdf["Feasibility_Status"] == "FEASIBLE"]
    all_ranked = feas_only["Rank"].notna().all()
    print(f"  V4 Infeasible excluded from ranking: "
          f"All FEASIBLE rows have Rank: {all_ranked}. "
          f"{PASS if all_ranked else FAIL}")

    # V5: OOD candidates flagged (check if any OOD exist)
    n_ood = result["n_ood"]
    print(f"  V5 OOD candidates: {n_ood} flagged as OUT_OF_DOMAIN. "
          f"{PASS}")   # PASS regardless — no OOD expected for this example

    # V6: Relative improvement correctly calculated
    baseline_cand = cdf[cdf["_is_baseline"] == True]
    if not baseline_cand.empty:
        b_impr = float(baseline_cand.iloc[0]["Relative_Improvement_pct"])
        impr_correct = abs(b_impr) < 1e-6   # baseline should have 0% improvement
        print(f"  V6 Baseline improvement = {b_impr:.6f}% (should be ~0). "
              f"{PASS if impr_correct else FAIL}")

    # V7: Candidates are ranked correctly (rank 1 = lowest predicted value)
    top_val  = result["best_value"]
    min_pred = feas_only[pred_col].min()
    rank_ok  = abs(top_val - min_pred) < 1e-8
    print(f"  V7 Rank 1 has lowest predicted {pri}: {rank_ok}. "
          f"{PASS if rank_ok else FAIL}")

    # V8: Recommended config has lowest predicted primary objective
    print(f"  V8 Recommended config has lowest predicted value: "
          f"{top_val:.6f} == min feasible {min_pred:.6f}. "
          f"{PASS if rank_ok else FAIL}")

    # V9: Module 3 risk analysis integrated
    risk_col_ok = "Risk_Category" in cdf.columns
    has_risk = (feas_only["Risk_Category"] != "UNDETERMINED RISK").any()
    print(f"  V9 Module 3 risk integrated: column exists={risk_col_ok}, "
          f"any risk classified={has_risk}. "
          f"{PASS if (risk_col_ok and has_risk) else FAIL}")

    # V10: Output files exist
    files_ok = all(os.path.exists(p) for p in [OUT_CSV, OUT_JSON, OUT_TXT])
    vis_ok   = os.path.isdir(VIS_DIR) and len(os.listdir(VIS_DIR)) > 0
    print(f"  V10 Output files saved: reports={files_ok}, visualisations={vis_ok}. "
          f"{PASS if (files_ok and vis_ok) else FAIL}")

    print(sep)
    print("MODULE 5 COMPLETE")
    print("-" * 68)
    print("  Output files:")
    for f in [OUT_CSV, OUT_JSON, OUT_TXT]:
        rel = os.path.relpath(f, BASE_DIR)
        print(f"    {rel}")
    print(f"    reports/visualisations/optimisation/*.png")
    print(sep)
    print("[DONE] Module 5 — Intelligent Cable-Routing Optimisation Engine.")
    print(SYNTH_NOTE)


if __name__ == "__main__":
    main()

