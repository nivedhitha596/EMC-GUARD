"""
emc_guard.py
------------
EMC-GUARD Configuration Assessment Engine.

Generates feasible alternative configurations within the frozen model domain,
predicts coupling parameters for each, ranks candidates by engineering objectives,
and produces transparent recommendations.

IMPORTANT TERMINOLOGY:
  - "Lower predicted coupling" / "Higher predicted coupling"
  - "Preferred under selected objective"
  - "Requires further validation"

No EMC compliance, safety or certification claims are made.
"""

import itertools
from typing import List, Optional

import numpy as np
import pandas as pd

from models.prediction_engine import (
    SUPPORTED_CLEARANCES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_PITCHES,
    SUPPORTED_SEPARATIONS,
    predict,
)


# ---------------------------------------------------------------------------
# Feasible configuration enumeration
# ---------------------------------------------------------------------------
def enumerate_all_configurations() -> pd.DataFrame:
    """Return all 216 feasible configurations as a DataFrame."""
    records = []
    for sep, pitch, ov, gc in itertools.product(
        SUPPORTED_SEPARATIONS, SUPPORTED_PITCHES,
        SUPPORTED_OVERLAPS, SUPPORTED_CLEARANCES,
    ):
        records.append({
            "PairSeparation_mm":  sep,
            "PairPitch_mm":       pitch,
            "Overlap_mm":         ov,
            "GroundClearance_mm": gc,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Prediction batch
# ---------------------------------------------------------------------------
def predict_candidates(
    configs_df: pd.DataFrame,
    engine: str = "frozen_matlab",
) -> pd.DataFrame:
    """
    Predict capacitance and mutual inductance for a DataFrame of configurations.
    Adds predicted columns in-place and returns the enriched DataFrame.
    """
    caps, muts, cids = [], [], []
    for _, row in configs_df.iterrows():
        res = predict(
            row["PairSeparation_mm"], row["PairPitch_mm"],
            row["Overlap_mm"], row["GroundClearance_mm"],
            engine=engine,
        )
        if res["status"] == "OK":
            caps.append(res["predictions"]["TotalCapacitance_pF_per_m"]["value"])
            muts.append(res["predictions"]["AbsMutualInductance_nH"]["value"])
            cids.append(res["config_id"])
        else:
            caps.append(np.nan)
            muts.append(np.nan)
            cids.append("ERROR")

    df = configs_df.copy()
    df["Config_ID"]                 = cids
    df["TotalCapacitance_pF_per_m"] = caps
    df["AbsMutualInductance_nH"]    = muts
    return df.dropna(subset=["TotalCapacitance_pF_per_m", "AbsMutualInductance_nH"])


# ---------------------------------------------------------------------------
# Objective scoring
# ---------------------------------------------------------------------------
OBJECTIVE_LABELS = {
    "minimize_capacitance":     "Minimise Capacitive Coupling",
    "minimize_inductance":      "Minimise Inductive Coupling",
    "minimize_combined":        "Minimise Combined Coupling (Weighted)",
}


def score_configurations(
    df: pd.DataFrame,
    objective: str = "minimize_combined",
    w_cap: float = 0.5,
    w_mut: float = 0.5,
) -> pd.DataFrame:
    """
    Compute a ranking score for each candidate configuration.

    Scoring is based on normalised coupling values; lower score = preferred.

    Parameters
    ----------
    df        : DataFrame with TotalCapacitance_pF_per_m and AbsMutualInductance_nH
    objective : one of OBJECTIVE_LABELS keys
    w_cap     : weight for capacitance (used when objective="minimize_combined")
    w_mut     : weight for mutual inductance (same)
    """
    df = df.copy()

    cap_vals = df["TotalCapacitance_pF_per_m"].values
    mut_vals = df["AbsMutualInductance_nH"].values

    # Normalise to [0, 1] across candidates
    cap_norm = _safe_normalise(cap_vals)
    mut_norm = _safe_normalise(mut_vals)

    if objective == "minimize_capacitance":
        df["ObjectiveScore"] = cap_norm
        df["ObjectiveLabel"] = "Normalised capacitive coupling"
    elif objective == "minimize_inductance":
        df["ObjectiveScore"] = mut_norm
        df["ObjectiveLabel"] = "Normalised inductive coupling"
    elif objective == "minimize_combined":
        total_w = w_cap + w_mut
        df["ObjectiveScore"] = (w_cap * cap_norm + w_mut * mut_norm) / total_w
        df["ObjectiveLabel"] = f"Combined coupling (w_cap={w_cap:.2f}, w_mut={w_mut:.2f})"
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    df["Rank"] = df["ObjectiveScore"].rank(method="min").astype(int)
    df = df.sort_values("Rank").reset_index(drop=True)

    # Add coupling assessment label (no compliance claim)
    median_cap = np.median(cap_vals)
    median_mut = np.median(mut_vals)
    labels = []
    for _, row in df.iterrows():
        c_label = "lower" if row["TotalCapacitance_pF_per_m"] < median_cap else "higher"
        m_label = "lower" if row["AbsMutualInductance_nH"]    < median_mut else "higher"
        labels.append(f"{c_label} cap. / {m_label} ind. coupling (vs. median)")
    df["CouplingAssessment"] = labels

    return df


def _safe_normalise(arr: np.ndarray) -> np.ndarray:
    rng = arr.max() - arr.min()
    if rng < 1e-30:
        return np.zeros_like(arr)
    return (arr - arr.min()) / rng


# ---------------------------------------------------------------------------
# EMC-GUARD assessment workflow
# ---------------------------------------------------------------------------
def run_emc_guard_assessment(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
    objective: str = "minimize_combined",
    w_cap: float = 0.5,
    w_mut: float = 0.5,
    engine: str = "frozen_matlab",
    constrain_separation: Optional[List[float]] = None,
    constrain_overlap:    Optional[List[float]] = None,
    constrain_clearance:  Optional[List[float]] = None,
) -> dict:
    """
    Full EMC-GUARD configuration assessment.

    1. Predict for the reference configuration.
    2. Enumerate feasible alternatives (subject to optional constraints).
    3. Predict all candidates.
    4. Rank by the selected objective.
    5. Return structured results for display.

    Returns
    -------
    dict with keys:
        reference, candidates_df, ranking_df, recommendation, objective_info
    """
    # --- Step 1: Predict reference ---
    ref_result = predict(sep, pitch, overlap, ground, engine=engine)
    if ref_result["status"] != "OK":
        return {"error": ref_result.get("error", "Prediction failed for reference config.")}

    ref_cap = ref_result["predictions"]["TotalCapacitance_pF_per_m"]["value"]
    ref_mut = ref_result["predictions"]["AbsMutualInductance_nH"]["value"]

    # --- Step 2: Enumerate alternatives ---
    all_configs = enumerate_all_configurations()

    seps  = constrain_separation or SUPPORTED_SEPARATIONS
    ovs   = constrain_overlap    or SUPPORTED_OVERLAPS
    gcs   = constrain_clearance  or SUPPORTED_CLEARANCES

    mask = (
        all_configs["PairSeparation_mm"].isin(seps)
        & all_configs["PairPitch_mm"].isin(SUPPORTED_PITCHES)
        & all_configs["Overlap_mm"].isin(ovs)
        & all_configs["GroundClearance_mm"].isin(gcs)
    )
    candidates = all_configs[mask].copy()

    # --- Step 3: Predict candidates ---
    candidates = predict_candidates(candidates, engine=engine)

    # --- Step 4: Score and rank ---
    ranked = score_configurations(candidates, objective, w_cap, w_mut)

    # --- Step 5: Identify top candidate ---
    top = ranked.iloc[0]
    top_is_ref = (
        abs(top["PairSeparation_mm"]  - sep)    < 1e-6
        and abs(top["PairPitch_mm"]   - pitch)  < 1e-6
        and abs(top["Overlap_mm"]     - overlap) < 1e-6
        and abs(top["GroundClearance_mm"] - ground) < 1e-6
    )

    # Build recommendation text (no compliance language)
    if top_is_ref:
        recommendation = (
            f"The reference configuration ({ref_result['config_id']}) "
            f"is already preferred under the selected objective "
            f"({OBJECTIVE_LABELS.get(objective, objective)})."
        )
    else:
        cap_delta = top["TotalCapacitance_pF_per_m"] - ref_cap
        mut_delta = top["AbsMutualInductance_nH"]    - ref_mut
        cap_pct   = 100 * cap_delta / ref_cap if ref_cap != 0 else 0.0
        mut_pct   = 100 * mut_delta / ref_mut if ref_mut != 0 else 0.0

        def _fmt(v):
            return f"{'lower' if v <= 0 else 'higher'} by {abs(v):.1f}%"

        recommendation = (
            f"Configuration {top['Config_ID']} "
            f"(Sep={top['PairSeparation_mm']} mm, "
            f"Pitch={top['PairPitch_mm']}, "
            f"Overlap={top['Overlap_mm']} mm, "
            f"Ground={top['GroundClearance_mm']} mm) "
            f"is preferred under '{OBJECTIVE_LABELS.get(objective, objective)}'. "
            f"Predicted capacitance is {_fmt(cap_pct)} than the reference. "
            f"Predicted mutual inductance is {_fmt(mut_pct)} than the reference. "
            "Further validation is required before installation decisions."
        )

    return {
        "reference": {
            "config_id": ref_result["config_id"],
            "sep": sep, "pitch": pitch, "overlap": overlap, "ground": ground,
            "cap": ref_cap, "mut": ref_mut,
        },
        "ranked_df":     ranked,
        "recommendation": recommendation,
        "top_config":     top.to_dict(),
        "top_is_reference": top_is_ref,
        "objective_info": {
            "objective": objective,
            "label":     OBJECTIVE_LABELS.get(objective, objective),
            "w_cap":     w_cap,
            "w_mut":     w_mut,
        },
        "n_candidates":  len(ranked),
        "engine":        engine,
        "notice": (
            "EMC-GUARD assessments compare predicted coupling parameters only. "
            "No EMC compliance, safety or certification determination is made. "
            "Results require engineering review and experimental validation."
        ),
    }

