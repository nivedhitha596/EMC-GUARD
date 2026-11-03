"""
dataset.py
----------
Dataset loader and validator for the KONE EMC-GUARD AI_ML module.

Loads the authoritative MATLAB physics dataset from the AI_ML integration
copy and validates its schema, row count, and domain coverage.

The dataset is never modified by this module. All reads are read-only.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    MATLAB_DATASET_AIML,
    MATLAB_DATASET_SOURCE,
    SUPPORTED_CLEARANCES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_PITCHES,
    SUPPORTED_SEPARATIONS,
    TARGET_COLUMNS,
    TOTAL_CONFIGURATIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the MATLAB physics dataset and validate its schema.

    Parameters
    ----------
    path : Path, optional
        Override the default dataset path. Defaults to the AI_ML integration
        copy at AI_ML/dataset/EMC_GUARD_Physics_Dataset_MATLAB.csv.

    Returns
    -------
    pd.DataFrame
        Validated 216-row DataFrame with all seven expected columns.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the expected path.
    ValueError
        If the dataset fails schema, row count, or domain validation.
    """
    dataset_path = Path(path) if path else MATLAB_DATASET_AIML

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"MATLAB dataset not found: {dataset_path}\n"
            "Expected the integration copy at AI_ML/dataset/. "
            "The original MATLAB source is at:\n"
            f"  {MATLAB_DATASET_SOURCE}"
        )

    df = pd.read_csv(str(dataset_path))
    _validate_schema(df, dataset_path)
    _validate_domain(df)

    logger.info(
        "Dataset loaded: %d configurations from %s",
        len(df),
        dataset_path.name,
    )
    return df


def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    """Verify that all expected columns are present and no NaNs exist."""
    missing = set(DATASET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing columns: {sorted(missing)}\n"
            f"Expected: {DATASET_COLUMNS}\n"
            f"Actual:   {df.columns.tolist()}"
        )

    if len(df) != TOTAL_CONFIGURATIONS:
        raise ValueError(
            f"Dataset has {len(df)} rows; expected {TOTAL_CONFIGURATIONS}. "
            f"File: {path}"
        )

    null_counts = df[DATASET_COLUMNS[1:]].isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Dataset contains missing values:\n{null_counts[null_counts > 0]}"
        )

    if df["Config_ID"].nunique() != TOTAL_CONFIGURATIONS:
        raise ValueError(
            f"Config_ID is not unique across all {TOTAL_CONFIGURATIONS} rows. "
            f"Unique count: {df['Config_ID'].nunique()}"
        )


def _validate_domain(df: pd.DataFrame) -> None:
    """Verify that discrete feature levels match the expected design grid."""
    checks = {
        "PairSeparation_mm": SUPPORTED_SEPARATIONS,
        "PairPitch_mm":      SUPPORTED_PITCHES,
        "Overlap_mm":        SUPPORTED_OVERLAPS,
        "GroundClearance_mm": SUPPORTED_CLEARANCES,
    }
    for col, expected in checks.items():
        actual = sorted(df[col].unique().tolist())
        if actual != expected:
            raise ValueError(
                f"Unexpected domain values for {col}.\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )

    for col in TARGET_COLUMNS:
        if (df[col] <= 0).any():
            raise ValueError(
                f"Target column '{col}' contains non-positive values. "
                "MATLAB physics targets must be strictly positive."
            )


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def get_config_row(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
) -> pd.Series:
    """
    Return the single dataset row matching the given configuration.

    Parameters
    ----------
    sep    : PairSeparation_mm
    pitch  : PairPitch_mm
    overlap: Overlap_mm
    ground : GroundClearance_mm

    Returns
    -------
    pd.Series with all seven dataset columns.

    Raises
    ------
    ValueError
        If the configuration is not in the 216-point design grid.
    """
    df = load_dataset()
    mask = (
        np.isclose(df["PairSeparation_mm"],   sep,     atol=1e-6)
        & np.isclose(df["PairPitch_mm"],       pitch,   atol=1e-6)
        & np.isclose(df["Overlap_mm"],         overlap, atol=1e-6)
        & np.isclose(df["GroundClearance_mm"], ground,  atol=1e-6)
    )
    matches = df[mask]
    if len(matches) != 1:
        raise ValueError(
            f"Configuration (sep={sep}, pitch={pitch}, overlap={overlap}, "
            f"ground={ground}) is not in the 216-point design grid. "
            "Only exact discrete grid values are accepted."
        )
    return matches.iloc[0]


def is_valid_configuration(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
) -> bool:
    """Return True iff all four values are in their respective supported sets."""
    return (
        any(abs(sep     - s) < 1e-6 for s in SUPPORTED_SEPARATIONS)
        and any(abs(pitch   - p) < 1e-6 for p in SUPPORTED_PITCHES)
        and any(abs(overlap - o) < 1e-6 for o in SUPPORTED_OVERLAPS)
        and any(abs(ground  - g) < 1e-6 for g in SUPPORTED_CLEARANCES)
    )


def get_dataset_summary() -> dict:
    """Return a summary dictionary describing the loaded dataset."""
    df = load_dataset()
    return {
        "n_configurations":    len(df),
        "n_columns":           len(df.columns),
        "columns":             df.columns.tolist(),
        "feature_columns":     FEATURE_COLUMNS,
        "target_columns":      TARGET_COLUMNS,
        "domain": {
            "PairSeparation_mm":  SUPPORTED_SEPARATIONS,
            "PairPitch_mm":       SUPPORTED_PITCHES,
            "Overlap_mm":         SUPPORTED_OVERLAPS,
            "GroundClearance_mm": SUPPORTED_CLEARANCES,
        },
        "target_stats": {
            col: {
                "min":  float(df[col].min()),
                "mean": float(df[col].mean()),
                "max":  float(df[col].max()),
                "unit": "pF/m" if "Capacitance" in col else "nH",
            }
            for col in TARGET_COLUMNS
        },
        "source_file": MATLAB_DATASET_AIML.name,
    }


def verify_dataset_integrity() -> dict:
    """
    Cross-check the AI_ML integration copy against the original MATLAB source.

    Returns a dict describing whether both files are byte-identical.
    Does NOT modify either file.
    """
    result = {
        "source_exists":      MATLAB_DATASET_SOURCE.exists(),
        "copy_exists":        MATLAB_DATASET_AIML.exists(),
        "byte_identical":     False,
        "source_rows":        None,
        "copy_rows":          None,
        "columns_match":      None,
        "data_identical":     None,
    }

    if not result["source_exists"]:
        result["error"] = f"Source not found: {MATLAB_DATASET_SOURCE}"
        return result

    if not result["copy_exists"]:
        result["error"] = f"Copy not found: {MATLAB_DATASET_AIML}"
        return result

    # MD5 checksum comparison
    def md5(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    result["byte_identical"] = md5(MATLAB_DATASET_SOURCE) == md5(MATLAB_DATASET_AIML)

    # DataFrame-level comparison
    src = pd.read_csv(str(MATLAB_DATASET_SOURCE))
    dst = pd.read_csv(str(MATLAB_DATASET_AIML))
    result["source_rows"]    = len(src)
    result["copy_rows"]      = len(dst)
    result["columns_match"]  = src.columns.tolist() == dst.columns.tolist()
    result["data_identical"] = bool(src.equals(dst))

    return result

