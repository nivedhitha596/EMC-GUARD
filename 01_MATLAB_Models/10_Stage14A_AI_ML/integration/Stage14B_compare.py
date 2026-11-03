
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Project directories
INTEGRATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = INTEGRATION_DIR.parent

sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from emc_guard_ai_predict import predict_emc_guard

MATLAB_FILE = (
    INTEGRATION_DIR
    / "matlab_output"
    / "EMC_GUARD_MATLAB_Reference.csv"
)

OUTPUT_DIR = INTEGRATION_DIR / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "EMC_GUARD_Comparison.csv"

# Check MATLAB reference file
if not MATLAB_FILE.is_file():
    raise FileNotFoundError(
        f"MATLAB reference file not found: {MATLAB_FILE}"
    )

reference = pd.read_csv(MATLAB_FILE)

required = [
    "Config_ID",
    "PairSeparation_mm",
    "PairPitch_mm",
    "Overlap_mm",
    "GroundClearance_mm",
    "Capacitance_pF_per_m",
    "AbsMutualInductance_nH",
]

missing = set(required) - set(reference.columns)

if missing:
    raise ValueError(
        f"Missing MATLAB reference columns: {sorted(missing)}"
    )

results = []

for _, row in reference.iterrows():

    # Run AI prediction for the same geometry
    ai = predict_emc_guard(
        separation_mm=float(row["PairSeparation_mm"]),
        pitch_mm=float(row["PairPitch_mm"]),
        overlap_mm=float(row["Overlap_mm"]),
        ground_clearance_mm=float(row["GroundClearance_mm"]),
    )

    if ai["Configuration_ID"] != str(row["Config_ID"]):
        raise ValueError(
            f"Configuration mismatch: {row['Config_ID']}"
        )

    cap_ai = ai["Models"]["TotalCapacitance_pF_per_m"]["Prediction"]
    mutual_ai = ai["Models"]["AbsMutualInductance_nH"]["Prediction"]

    cap_matlab = float(row["Capacitance_pF_per_m"])
    mutual_matlab = float(row["AbsMutualInductance_nH"])

    cap_difference = abs(cap_ai - cap_matlab)
    mutual_difference = abs(mutual_ai - mutual_matlab)

    cap_percent = (
        100 * cap_difference / abs(cap_matlab)
        if cap_matlab != 0 else np.nan
    )

    mutual_percent = (
        100 * mutual_difference / abs(mutual_matlab)
        if mutual_matlab != 0 else np.nan
    )

    results.append({
        "Config_ID": str(row["Config_ID"]),
        "PairSeparation_mm": row["PairSeparation_mm"],
        "PairPitch_mm": row["PairPitch_mm"],
        "Overlap_mm": row["Overlap_mm"],
        "GroundClearance_mm": row["GroundClearance_mm"],

        "MATLAB_Capacitance_pF_per_m": cap_matlab,
        "AI_Capacitance_pF_per_m": cap_ai,
        "Capacitance_Absolute_Difference": cap_difference,
        "Capacitance_Difference_Percent": cap_percent,

        "MATLAB_AbsMutualInductance_nH": mutual_matlab,
        "AI_AbsMutualInductance_nH": mutual_ai,
        "MutualInductance_Absolute_Difference": mutual_difference,
        "MutualInductance_Difference_Percent": mutual_percent,

        "AI_Prediction_Status": ai["Prediction_Status"],
        "Comparison_Status": "Computed; tolerance not yet defined",
    })

comparison = pd.DataFrame(results)

comparison.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("EMC-GUARD | STAGE 14B-4")
print("MATLAB vs PYTHON AI COMPARISON")
print("=" * 65)

print("\nConfigurations compared:", len(comparison))

print("\nComparison results:")
print(comparison.to_string(index=False))

print("\nOutput saved:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print(
    "This comparison checks computational agreement between "
    "the MATLAB reference and AI surrogate."
)
print(
    "It does not establish independent physical validation "
    "or EMC compliance."
)
