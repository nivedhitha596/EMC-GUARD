"""
model_loader.py
---------------
Loads the frozen EMC-GUARD prediction assets:
  - Stage 12A frozen MATLAB linear-model coefficients (via h5py)
  - Stage 14A ExtraTrees surrogate models (via joblib)
  - Reference physics dataset (216 configurations)
  - AI model metadata

IMPORTANT: This module does NOT retrain, replace or modify the frozen models.
           All predictions are restricted to the original 216-point design grid.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to this file's location)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
DATA_DIR   = _PROJECT / "data"
MODELS_DIR = _PROJECT / "models"

FROZEN_MAT_PATH   = MODELS_DIR / "matlab_models" / "Stage12A_Selected_Models_FROZEN.mat"
CAP_JOBLIB_PATH   = MODELS_DIR / "EMC_GUARD_Capacitance_ExtraTrees.joblib"
MUT_JOBLIB_PATH   = MODELS_DIR / "EMC_GUARD_MutualInductance_ExtraTrees.joblib"
DATASET_PATH      = DATA_DIR / "EMC_GUARD_Physics_Dataset.csv"
AI_METADATA_PATH  = DATA_DIR / "EMC_GUARD_AI_Model_Metadata.json"


# ---------------------------------------------------------------------------
# Frozen MATLAB linear model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_frozen_matlab_model() -> dict:
    """
    Load the Stage 12A frozen MATLAB least-squares model from the HDF5 MAT file.

    Returns a dict containing:
        betaCap  : (94,) float64 – capacitance model coefficients
        betaMut  : (84,) float64 – mutual inductance model coefficients
        XCap     : (94, 216) float64 – capacitance design matrix
        XMut     : (84, 216) float64 – mutual inductance design matrix
        yCap     : (216,) float64 – target capacitance values [pF/m]
        yMut     : (216,) float64 – target mutual inductance values [nH]
    """
    if not FROZEN_MAT_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model file not found: {FROZEN_MAT_PATH}\n"
            "Cannot proceed without the Stage 12A model file."
        )

    with h5py.File(str(FROZEN_MAT_PATH), "r") as f:
        betaCap = f["betaCap"][()].flatten()
        betaMut = f["betaMut"][()].flatten()
        XCap    = f["Xcap"][()]    # shape (94, 216) – columns = configs
        XMut    = f["Xmut"][()]    # shape (84, 216)
        yCap    = f["yCap"][()].flatten()
        yMut    = f["yMut"][()].flatten()

    logger.info(
        "Frozen MATLAB model loaded: betaCap=%s, betaMut=%s, XCap=%s, XMut=%s",
        betaCap.shape, betaMut.shape, XCap.shape, XMut.shape,
    )

    return {
        "betaCap": betaCap,
        "betaMut": betaMut,
        "XCap": XCap,
        "XMut": XMut,
        "yCap": yCap,
        "yMut": yMut,
        "model_names": {
            "capacitance":       "B2_SP_Ground",
            "mutual_inductance": "B1_SP_Overlap",
        },
        "units": {
            "capacitance":       "pF/m",
            "mutual_inductance": "nH",
        },
        "num_terms": {
            "capacitance":       int(betaCap.shape[0]),
            "mutual_inductance": int(betaMut.shape[0]),
        },
    }


# ---------------------------------------------------------------------------
# ExtraTrees surrogate model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_extratrees_models() -> dict:
    """
    Load the Stage 14A ExtraTrees surrogate models (trained on the 216-point
    physics dataset, target-transformed with natural log in MATLAB, then fitted
    directly in scikit-learn on raw pF/m and nH values).

    Returns:
        cap_model : fitted ExtraTreesRegressor for TotalCapacitance_pF_per_m
        mut_model : fitted ExtraTreesRegressor for AbsMutualInductance_nH
        metadata  : AI model metadata dict
    """
    for p in (CAP_JOBLIB_PATH, MUT_JOBLIB_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"ExtraTrees model not found: {p}\n"
                "Cannot proceed without the trained surrogate model."
            )

    cap_model = joblib.load(str(CAP_JOBLIB_PATH))
    mut_model = joblib.load(str(MUT_JOBLIB_PATH))

    with open(str(AI_METADATA_PATH), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("ExtraTrees surrogate models loaded successfully.")
    return {
        "cap_model":  cap_model,
        "mut_model":  mut_model,
        "metadata":   metadata,
    }


# ---------------------------------------------------------------------------
# Physics dataset loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_physics_dataset() -> pd.DataFrame:
    """
    Load the authoritative 216-configuration physics dataset exported from MATLAB.

    Columns:
        Config_ID, PairSeparation_mm, PairPitch_mm, Overlap_mm,
        GroundClearance_mm, TotalCapacitance_pF_per_m, AbsMutualInductance_nH
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Physics dataset not found: {DATASET_PATH}")

    df = pd.read_csv(str(DATASET_PATH))
    expected_cols = [
        "Config_ID", "PairSeparation_mm", "PairPitch_mm",
        "Overlap_mm", "GroundClearance_mm",
        "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH",
    ]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Physics dataset is missing columns: {missing}")

    if len(df) != 216:
        raise ValueError(
            f"Physics dataset contains {len(df)} rows; expected 216."
        )

    logger.info("Physics dataset loaded: %d configurations.", len(df))
    return df


# ---------------------------------------------------------------------------
# Model status report (for UI)
# ---------------------------------------------------------------------------
def get_model_status() -> dict:
    """
    Attempt to load all model assets and return a status dict for the dashboard.
    """
    status = {
        "frozen_matlab":   {"ok": False, "error": None},
        "extratrees":      {"ok": False, "error": None},
        "physics_dataset": {"ok": False, "error": None, "n_configs": 0},
    }

    try:
        load_frozen_matlab_model()
        status["frozen_matlab"]["ok"] = True
    except Exception as exc:
        status["frozen_matlab"]["error"] = str(exc)

    try:
        load_extratrees_models()
        status["extratrees"]["ok"] = True
    except Exception as exc:
        status["extratrees"]["error"] = str(exc)

    try:
        df = load_physics_dataset()
        status["physics_dataset"]["ok"] = True
        status["physics_dataset"]["n_configs"] = len(df)
    except Exception as exc:
        status["physics_dataset"]["error"] = str(exc)

    return status

