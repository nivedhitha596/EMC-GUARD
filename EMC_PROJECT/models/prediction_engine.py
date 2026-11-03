"""
prediction_engine.py
--------------------
Implements the EMC-GUARD prediction interface.

Prediction strategy:
  PRIMARY  : Stage 12A frozen MATLAB least-squares model
             (beta @ X_row for each config, machine-precision accuracy)
  SECONDARY: Stage 14A ExtraTrees surrogate (used when frozen model
             is unavailable, clearly flagged in output)

Both models are restricted to the original 216 configurations.
No interpolation or extrapolation is permitted.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from models.model_loader import (
    load_extratrees_models,
    load_frozen_matlab_model,
    load_physics_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported domain constants
# ---------------------------------------------------------------------------
SUPPORTED_SEPARATIONS   = [8, 12, 16, 20, 24, 28]       # mm
SUPPORTED_PITCHES       = [2, 3, 4]                      # mm
SUPPORTED_OVERLAPS      = [150, 300, 450]                # mm
SUPPORTED_CLEARANCES    = [10, 25, 50, 100]              # mm


def is_in_domain(sep: float, pitch: float, overlap: float, ground: float) -> bool:
    """Return True iff the configuration is in the supported 216-point design grid."""
    return (
        any(abs(sep    - s) < 1e-6 for s in SUPPORTED_SEPARATIONS)
        and any(abs(pitch  - p) < 1e-6 for p in SUPPORTED_PITCHES)
        and any(abs(overlap- o) < 1e-6 for o in SUPPORTED_OVERLAPS)
        and any(abs(ground - g) < 1e-6 for g in SUPPORTED_CLEARANCES)
    )


def _find_config_index(sep: float, pitch: float, overlap: float, ground: float) -> int:
    """
    Find the 0-based column index in the frozen model matrices for the given config.
    Returns -1 if not found.
    """
    df = load_physics_dataset()
    mask = (
        np.isclose(df["PairSeparation_mm"],   sep,    atol=1e-6)
        & np.isclose(df["PairPitch_mm"],       pitch,  atol=1e-6)
        & np.isclose(df["Overlap_mm"],         overlap, atol=1e-6)
        & np.isclose(df["GroundClearance_mm"], ground, atol=1e-6)
    )
    idx = np.where(mask)[0]
    if len(idx) != 1:
        return -1
    return int(idx[0])


# ---------------------------------------------------------------------------
# Primary: frozen MATLAB linear model prediction
# ---------------------------------------------------------------------------
def predict_frozen_matlab(sep: float, pitch: float, overlap: float, ground: float) -> dict:
    """
    Use the Stage 12A frozen MATLAB least-squares model to predict for one config.
    X columns are ordered identically to the 216-row physics dataset.

    Returns:
        capacitance_pF_per_m  (float)
        abs_mutual_inductance_nH (float)
        config_id (str)
        column_index (int)  – 0-based position in the 216-column design matrix
        model_engine (str)
    """
    model = load_frozen_matlab_model()
    df    = load_physics_dataset()

    col_idx = _find_config_index(sep, pitch, overlap, ground)
    if col_idx < 0:
        raise ValueError(
            f"Configuration (sep={sep}, pitch={pitch}, overlap={overlap}, "
            f"ground={ground}) is not in the 216-point design grid. "
            "No extrapolation is permitted."
        )

    betaCap = model["betaCap"]   # shape (94,)
    betaMut = model["betaMut"]   # shape (84,)
    XCap    = model["XCap"]      # shape (94, 216)
    XMut    = model["XMut"]      # shape (84, 216)

    cap_pred = float(betaCap @ XCap[:, col_idx])
    mut_pred = float(betaMut @ XMut[:, col_idx])

    row = df.iloc[col_idx]
    return {
        "capacitance_pF_per_m":       cap_pred,
        "abs_mutual_inductance_nH":   mut_pred,
        "config_id":                  str(row["Config_ID"]),
        "column_index":               col_idx,
        "model_engine":               "Stage12A_FrozenMATLAB",
        "capacitance_model_name":     model["model_names"]["capacitance"],
        "inductance_model_name":      model["model_names"]["mutual_inductance"],
        "reference_cap":              float(row["TotalCapacitance_pF_per_m"]),
        "reference_mut":              float(row["AbsMutualInductance_nH"]),
    }


# ---------------------------------------------------------------------------
# Secondary: ExtraTrees surrogate prediction
# ---------------------------------------------------------------------------
def predict_extratrees(sep: float, pitch: float, overlap: float, ground: float) -> dict:
    """
    Use the Stage 14A ExtraTrees surrogate to predict.
    Capacitance uses [Sep, Pitch, GC]; Inductance uses [Sep, Pitch, Overlap].
    """
    assets = load_extratrees_models()
    df     = load_physics_dataset()

    col_idx = _find_config_index(sep, pitch, overlap, ground)
    if col_idx < 0:
        raise ValueError(
            f"Configuration (sep={sep}, pitch={pitch}, overlap={overlap}, "
            f"ground={ground}) is not in the 216-point design grid."
        )

    cap_model = assets["cap_model"]
    mut_model = assets["mut_model"]
    meta      = assets["metadata"]

    cap_feats = meta["targets"]["TotalCapacitance_pF_per_m"]["features"]
    mut_feats = meta["targets"]["AbsMutualInductance_nH"]["features"]

    feat_map = {
        "PairSeparation_mm":   sep,
        "PairPitch_mm":        pitch,
        "Overlap_mm":          overlap,
        "GroundClearance_mm":  ground,
    }

    X_cap = pd.DataFrame([[feat_map[f] for f in cap_feats]], columns=cap_feats)
    X_mut = pd.DataFrame([[feat_map[f] for f in mut_feats]], columns=mut_feats)

    cap_pred = float(cap_model.predict(X_cap)[0])
    mut_pred = float(mut_model.predict(X_mut)[0])

    row = df.iloc[col_idx]
    return {
        "capacitance_pF_per_m":       cap_pred,
        "abs_mutual_inductance_nH":   mut_pred,
        "config_id":                  str(row["Config_ID"]),
        "column_index":               col_idx,
        "model_engine":               "Stage14A_ExtraTrees",
        "capacitance_model_name":     "ExtraTreesRegressor",
        "inductance_model_name":      "ExtraTreesRegressor",
        "reference_cap":              float(row["TotalCapacitance_pF_per_m"]),
        "reference_mut":              float(row["AbsMutualInductance_nH"]),
    }


# ---------------------------------------------------------------------------
# Main prediction interface
# ---------------------------------------------------------------------------
def predict(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
    engine: str = "frozen_matlab",
) -> dict:
    """
    Predict capacitance and absolute mutual inductance for one cable configuration.

    Parameters
    ----------
    sep    : PairSeparation_mm   – must be one of {8,12,16,20,24,28}
    pitch  : PairPitch_mm        – must be one of {2,3,4}
    overlap: Overlap_mm          – must be one of {150,300,450}
    ground : GroundClearance_mm  – must be one of {10,25,50,100}
    engine : "frozen_matlab" (default) or "extratrees"

    Returns
    -------
    dict with keys:
        status, config_id, geometry, predictions, model_info, timestamp,
        validation_notice
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Domain validation
    if not is_in_domain(sep, pitch, overlap, ground):
        return {
            "status": "ERROR",
            "error": (
                f"Configuration (sep={sep}, pitch={pitch}, overlap={overlap}, "
                f"ground={ground}) is outside the supported 216-point design grid. "
                "Extrapolation is NOT permitted."
            ),
            "timestamp": timestamp,
        }

    try:
        if engine == "frozen_matlab":
            raw = predict_frozen_matlab(sep, pitch, overlap, ground)
        elif engine == "extratrees":
            raw = predict_extratrees(sep, pitch, overlap, ground)
        else:
            raise ValueError(f"Unknown prediction engine: {engine!r}")

    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return {
            "status": "ERROR",
            "error":  str(exc),
            "timestamp": timestamp,
        }

    return {
        "status":    "OK",
        "config_id": raw["config_id"],
        "geometry":  {
            "PairSeparation_mm":   sep,
            "PairPitch_mm":        pitch,
            "Overlap_mm":          overlap,
            "GroundClearance_mm":  ground,
        },
        "predictions": {
            "TotalCapacitance_pF_per_m": {
                "value":      raw["capacitance_pF_per_m"],
                "unit":       "pF/m",
                "model":      raw["capacitance_model_name"],
                "reference":  raw["reference_cap"],
                "abs_diff":   abs(raw["capacitance_pF_per_m"] - raw["reference_cap"]),
            },
            "AbsMutualInductance_nH": {
                "value":      raw["abs_mutual_inductance_nH"],
                "unit":       "nH",
                "model":      raw["inductance_model_name"],
                "reference":  raw["reference_mut"],
                "abs_diff":   abs(raw["abs_mutual_inductance_nH"] - raw["reference_mut"]),
            },
        },
        "model_info": {
            "engine":              raw["model_engine"],
            "domain":              "Original 216-point MATLAB design grid",
            "extrapolation":       "Not permitted",
            "column_index":        raw["column_index"],
        },
        "timestamp": timestamp,
        "validation_notice": (
            "Predictions reproduce the MATLAB-generated deterministic dataset. "
            "Independent experimental validation has not yet been established."
        ),
    }


# ---------------------------------------------------------------------------
# Batch prediction over multiple configs
# ---------------------------------------------------------------------------
def predict_all_configs(engine: str = "frozen_matlab") -> pd.DataFrame:
    """
    Run prediction for all 216 configurations and return a DataFrame.
    """
    df = load_physics_dataset()
    rows = []
    for _, r in df.iterrows():
        res = predict(
            r["PairSeparation_mm"], r["PairPitch_mm"],
            r["Overlap_mm"], r["GroundClearance_mm"],
            engine=engine,
        )
        if res["status"] == "OK":
            rows.append({
                "Config_ID":                  res["config_id"],
                "PairSeparation_mm":          r["PairSeparation_mm"],
                "PairPitch_mm":               r["PairPitch_mm"],
                "Overlap_mm":                 r["Overlap_mm"],
                "GroundClearance_mm":         r["GroundClearance_mm"],
                "TotalCapacitance_pF_per_m":  res["predictions"]["TotalCapacitance_pF_per_m"]["value"],
                "AbsMutualInductance_nH":     res["predictions"]["AbsMutualInductance_nH"]["value"],
                "Ref_Cap_pF_per_m":           res["predictions"]["TotalCapacitance_pF_per_m"]["reference"],
                "Ref_Mut_nH":                 res["predictions"]["AbsMutualInductance_nH"]["reference"],
            })
    return pd.DataFrame(rows)

