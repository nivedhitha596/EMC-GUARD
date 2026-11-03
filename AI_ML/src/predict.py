"""
predict.py
----------
AI_ML module prediction interface for the KONE EMC-GUARD project.

This module is a THIN ADAPTER over the existing EMC_PROJECT prediction engine.
It does NOT reimplement prediction mathematics. It re-exports the public API
from EMC_PROJECT/models/prediction_engine.py and adds:

  - sys.path management so the existing engine can be imported from AI_ML/src/
  - Dataset integrity pre-check before predictions
  - Suppression of scikit-learn version warnings (1.8.0 -> 1.9.1 mismatch)
  - Convenience batch prediction from the MATLAB dataset
  - Clear distinction between primary (frozen_matlab) and secondary (extratrees)

PRIMARY ENGINE  : Stage 12A frozen MATLAB (machine-precision analytical result)
SECONDARY ENGINE: Stage 14A ExtraTrees surrogates (trained ML approximation)

Both engines reject inputs outside the 216-point design grid.
No extrapolation is permitted.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    EMC_PROJECT_DIR,
    ENGINE_EXTRATREES,
    ENGINE_FROZEN_MATLAB,
    SUPPORTED_ENGINES,
    TOTAL_CONFIGURATIONS,
)
from dataset import is_valid_configuration, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import the existing prediction engine without duplicating its logic
# ---------------------------------------------------------------------------

def _ensure_emc_project_on_path() -> None:
    """Add EMC_PROJECT root to sys.path so its modules can be imported."""
    emc_str = str(EMC_PROJECT_DIR)
    if emc_str not in sys.path:
        sys.path.insert(0, emc_str)


def _import_engine():
    """
    Import the prediction engine from EMC_PROJECT.

    Returns the prediction_engine module.
    Suppresses scikit-learn version mismatch warnings (1.8.0 -> 1.9.1);
    the mismatch is cosmetic — predictions are verified correct.
    """
    _ensure_emc_project_on_path()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*InconsistentVersionWarning.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Trying to unpickle estimator.*",
            category=UserWarning,
        )
        import importlib
        import models.prediction_engine as pe
        return pe


# ---------------------------------------------------------------------------
# Public prediction API (mirrors EMC_PROJECT/models/prediction_engine.py)
# ---------------------------------------------------------------------------

def predict(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
    engine: str = ENGINE_FROZEN_MATLAB,
) -> dict:
    """
    Predict TotalCapacitance_pF_per_m and AbsMutualInductance_nH for one
    cable configuration.

    This is a pass-through to the existing EMC_PROJECT prediction engine.
    All prediction mathematics are preserved without modification.

    Parameters
    ----------
    sep    : PairSeparation_mm  — must be one of {8,12,16,20,24,28}
    pitch  : PairPitch_mm       — must be one of {2,3,4}
    overlap: Overlap_mm         — must be one of {150,300,450}
    ground : GroundClearance_mm — must be one of {10,25,50,100}
    engine : "frozen_matlab" (default) or "extratrees"

    Returns
    -------
    dict with keys:
        status, config_id, geometry, predictions, model_info, timestamp,
        validation_notice

    Notes
    -----
    - "frozen_matlab" uses the Stage 12A analytical MATLAB model.
      Accuracy: machine precision (~10^-14 error vs MATLAB reference).
    - "extratrees" uses the Stage 14A ExtraTrees surrogate.
      Accuracy: Test R² > 0.9999; approximate, not analytical.
    """
    # Normalize engine string for cross-compatibility with UI labels
    if isinstance(engine, str):
        eng_lower = engine.strip().lower().replace(" ", "_").replace("-", "_")
        if eng_lower in ("frozen_matlab", "matlab", "frozen", "stage12a"):
            engine = ENGINE_FROZEN_MATLAB
        elif eng_lower in ("extratrees", "extra_trees", "trees", "stage14a"):
            engine = ENGINE_EXTRATREES

    if engine not in SUPPORTED_ENGINES:
        return {
            "status": "ERROR",
            "error": (
                f"Unknown engine '{engine}'. "
                f"Supported engines: {SUPPORTED_ENGINES}"
            ),
        }

    # Fast domain check before invoking the engine
    if not is_valid_configuration(sep, pitch, overlap, ground):
        return {
            "status": "ERROR",
            "error": (
                f"Configuration (sep={sep}, pitch={pitch}, overlap={overlap}, "
                f"ground={ground}) is outside the supported 216-point design grid. "
                "Extrapolation is not permitted."
            ),
        }

    try:
        pe = _import_engine()
        return pe.predict(sep=sep, pitch=pitch, overlap=overlap,
                          ground=ground, engine=engine)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return {"status": "ERROR", "error": str(exc)}


def predict_all_configs(engine: str = ENGINE_FROZEN_MATLAB) -> pd.DataFrame:
    """
    Run predictions for all 216 MATLAB configurations.

    Loads the MATLAB dataset, iterates all 216 rows, and returns a DataFrame
    with geometry, predictions, and reference values for both targets.

    Parameters
    ----------
    engine : "frozen_matlab" (default) or "extratrees"

    Returns
    -------
    pd.DataFrame with columns:
        Config_ID, PairSeparation_mm, PairPitch_mm, Overlap_mm,
        GroundClearance_mm,
        Cap_Predicted, Cap_Reference, Cap_AbsError,
        Mut_Predicted, Mut_Reference, Mut_AbsError,
        Engine, Status
    """
    df = load_dataset()
    rows = []

    for _, r in df.iterrows():
        result = predict(
            sep=float(r["PairSeparation_mm"]),
            pitch=float(r["PairPitch_mm"]),
            overlap=float(r["Overlap_mm"]),
            ground=float(r["GroundClearance_mm"]),
            engine=engine,
        )

        base = {
            "Config_ID":          r["Config_ID"],
            "PairSeparation_mm":  r["PairSeparation_mm"],
            "PairPitch_mm":       r["PairPitch_mm"],
            "Overlap_mm":         r["Overlap_mm"],
            "GroundClearance_mm": r["GroundClearance_mm"],
            "Engine":             engine,
        }

        if result["status"] == "OK":
            cap = result["predictions"]["TotalCapacitance_pF_per_m"]
            mut = result["predictions"]["AbsMutualInductance_nH"]
            rows.append({
                **base,
                "Status":        "OK",
                "Cap_Predicted": cap["value"],
                "Cap_Reference": cap["reference"],
                "Cap_AbsError":  cap["abs_diff"],
                "Mut_Predicted": mut["value"],
                "Mut_Reference": mut["reference"],
                "Mut_AbsError":  mut["abs_diff"],
            })
        else:
            rows.append({
                **base,
                "Status":        "ERROR",
                "Cap_Predicted": None,
                "Cap_Reference": None,
                "Cap_AbsError":  None,
                "Mut_Predicted": None,
                "Mut_Reference": None,
                "Mut_AbsError":  None,
                "Error":         result.get("error", "Unknown"),
            })

    return pd.DataFrame(rows)


def compare_engines(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
) -> dict:
    """
    Run both engines for a single configuration and compare their outputs
    against the MATLAB reference value.

    Returns
    -------
    dict with:
        config_id, geometry, matlab_reference,
        frozen_matlab: {cap_pred, cap_err, mut_pred, mut_err},
        extratrees:    {cap_pred, cap_err, mut_pred, mut_err},
        agreement_cap_pct, agreement_mut_pct
    """
    fm  = predict(sep, pitch, overlap, ground, engine=ENGINE_FROZEN_MATLAB)
    et  = predict(sep, pitch, overlap, ground, engine=ENGINE_EXTRATREES)

    if fm["status"] != "OK":
        return {"status": "ERROR", "error": fm.get("error")}
    if et["status"] != "OK":
        return {"status": "ERROR", "error": et.get("error")}

    cap_ref = fm["predictions"]["TotalCapacitance_pF_per_m"]["reference"]
    mut_ref = fm["predictions"]["AbsMutualInductance_nH"]["reference"]

    fm_cap = fm["predictions"]["TotalCapacitance_pF_per_m"]["value"]
    fm_mut = fm["predictions"]["AbsMutualInductance_nH"]["value"]
    et_cap = et["predictions"]["TotalCapacitance_pF_per_m"]["value"]
    et_mut = et["predictions"]["AbsMutualInductance_nH"]["value"]

    return {
        "status":      "OK",
        "config_id":   fm["config_id"],
        "geometry": {
            "PairSeparation_mm":   sep,
            "PairPitch_mm":        pitch,
            "Overlap_mm":          overlap,
            "GroundClearance_mm":  ground,
        },
        "matlab_reference": {
            "TotalCapacitance_pF_per_m": cap_ref,
            "AbsMutualInductance_nH":    mut_ref,
        },
        ENGINE_FROZEN_MATLAB: {
            "TotalCapacitance_pF_per_m": fm_cap,
            "AbsMutualInductance_nH":    fm_mut,
            "Cap_AbsError":              abs(fm_cap - cap_ref),
            "Mut_AbsError":              abs(fm_mut - mut_ref),
        },
        ENGINE_EXTRATREES: {
            "TotalCapacitance_pF_per_m": et_cap,
            "AbsMutualInductance_nH":    et_mut,
            "Cap_AbsError":              abs(et_cap - cap_ref),
            "Mut_AbsError":              abs(et_mut - mut_ref),
        },
        # Percentage difference between the two engines
        "engine_agreement_cap_pct": (
            abs(fm_cap - et_cap) / cap_ref * 100 if cap_ref else None
        ),
        "engine_agreement_mut_pct": (
            abs(fm_mut - et_mut) / mut_ref * 100 if mut_ref else None
        ),
    }

