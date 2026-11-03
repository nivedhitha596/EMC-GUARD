"""
config.py
---------
Central path and domain configuration for the KONE EMC-GUARD AI_ML module.

All paths resolve relative to the repository root so the module works
regardless of the working directory it is invoked from.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (three levels up: AI_ML/src/config.py -> AI_ML/src -> AI_ML -> kone)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
# Authoritative MATLAB source (never modify this file)
MATLAB_DATASET_SOURCE = (
    ROOT / "01_MATLAB_Models" / "09_Stage13A_Export"
    / "EMC_GUARD_Physics_Dataset.csv"
)

# Integration copy inside AI_ML (used by this module as primary read source)
MATLAB_DATASET_AIML = (
    ROOT / "AI_ML" / "dataset" / "EMC_GUARD_Physics_Dataset_MATLAB.csv"
)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
# Stage 12A frozen MATLAB linear model (primary engine - machine precision)
FROZEN_MAT_PATH = (
    ROOT / "EMC_PROJECT" / "models" / "matlab_models"
    / "Stage12A_Selected_Models_FROZEN.mat"
)

# Stage 14A ExtraTrees surrogate models (secondary / fallback engines)
STAGE14A_DIR = (
    ROOT / "01_MATLAB_Models" / "10_Stage14A_AI_ML"
    / "models" / "stage14A_4_final"
)
CAP_JOBLIB_PATH  = STAGE14A_DIR / "EMC_GUARD_Capacitance_ExtraTrees.joblib"
MUT_JOBLIB_PATH  = STAGE14A_DIR / "EMC_GUARD_MutualInductance_ExtraTrees.joblib"
AI_METADATA_PATH = STAGE14A_DIR / "EMC_GUARD_AI_Model_Metadata.json"

# ---------------------------------------------------------------------------
# Existing prediction engine location (do not duplicate its logic)
# ---------------------------------------------------------------------------
EMC_PROJECT_DIR = ROOT / "EMC_PROJECT"

# ---------------------------------------------------------------------------
# AI_ML output directories
# ---------------------------------------------------------------------------
AIML_REPORTS_DIR = ROOT / "AI_ML" / "reports"
AIML_BACKUPS_DIR = ROOT / "AI_ML" / "backups"

# ---------------------------------------------------------------------------
# Domain constants - 216-point MATLAB design grid (DO NOT MODIFY)
# ---------------------------------------------------------------------------
SUPPORTED_SEPARATIONS = [8, 12, 16, 20, 24, 28]     # PairSeparation_mm
SUPPORTED_PITCHES     = [2, 3, 4]                     # PairPitch_mm
SUPPORTED_OVERLAPS    = [150, 300, 450]               # Overlap_mm
SUPPORTED_CLEARANCES  = [10, 25, 50, 100]             # GroundClearance_mm
TOTAL_CONFIGURATIONS  = 216                           # 6 x 3 x 3 x 4

# ---------------------------------------------------------------------------
# Feature selections (intentional - preserve)
# ---------------------------------------------------------------------------
# Capacitance is independent of Overlap_mm
CAP_FEATURES = ["PairSeparation_mm", "PairPitch_mm", "GroundClearance_mm"]
# Mutual inductance is independent of GroundClearance_mm
MUT_FEATURES = ["PairSeparation_mm", "PairPitch_mm", "Overlap_mm"]

# ---------------------------------------------------------------------------
# Dataset column names
# ---------------------------------------------------------------------------
DATASET_COLUMNS = [
    "Config_ID",
    "PairSeparation_mm",
    "PairPitch_mm",
    "Overlap_mm",
    "GroundClearance_mm",
    "TotalCapacitance_pF_per_m",
    "AbsMutualInductance_nH",
]
FEATURE_COLUMNS = DATASET_COLUMNS[1:5]
TARGET_COLUMNS  = DATASET_COLUMNS[5:]

# ---------------------------------------------------------------------------
# Prediction engine identifiers
# ---------------------------------------------------------------------------
ENGINE_FROZEN_MATLAB = "frozen_matlab"
ENGINE_EXTRATREES    = "extratrees"
SUPPORTED_ENGINES    = [ENGINE_FROZEN_MATLAB, ENGINE_EXTRATREES]

