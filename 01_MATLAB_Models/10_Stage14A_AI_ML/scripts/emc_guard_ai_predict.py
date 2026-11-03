
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

# Project root: 10_Stage14A_AI_ML
ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    ROOT / "models" / "stage14A_4_final"
)

DATA_FILE = (
    ROOT.parent / "09_Stage13A_Export"
    / "EMC_GUARD_Physics_Dataset.csv"
)

METADATA_FILE = (
    MODEL_DIR / "EMC_GUARD_AI_Model_Metadata.json"
)




with open(METADATA_FILE, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

DATA = pd.read_csv(DATA_FILE)

GEOMETRY_FIELDS = [
    "PairSeparation_mm",
    "PairPitch_mm",
    "Overlap_mm",
    "GroundClearance_mm",
]

MODELS = {}

for target, details in METADATA["targets"].items():
    model_path = MODEL_DIR / details["model_file"]
    MODELS[target] = joblib.load(model_path)


def predict_emc_guard(
    separation_mm,
    pitch_mm,
    overlap_mm,
    ground_clearance_mm,
):
    """
    Predict capacitance and absolute mutual inductance.

    Only exact configurations from the original 216-row
    dataset are accepted.

    Returns predictions and corresponding MATLAB reference
    values for computational comparison.
    """

    values = {
        "PairSeparation_mm": float(separation_mm),
        "PairPitch_mm": float(pitch_mm),
        "Overlap_mm": float(overlap_mm),
        "GroundClearance_mm": float(ground_clearance_mm),
    }

    # Exact supported configuration check.
    mask = np.ones(len(DATA), dtype=bool)

    for field, value in values.items():
        mask &= np.isclose(
            DATA[field].to_numpy(dtype=float),
            value,
            rtol=0,
            atol=1e-9,
        )

    matching_rows = DATA.loc[mask]

    if len(matching_rows) != 1:
        raise ValueError(
            "Unsupported geometry configuration. "
            "Only the original 216 MATLAB dataset "
            "configurations are currently accepted. "
            "No extrapolation or intermediate geometry "
            "prediction is permitted."
        )

    reference = matching_rows.iloc[0]

    results = {
        "Project": "EMC-GUARD",
        "Configuration_ID": str(reference["Config_ID"]),
        "Geometry": values,
        "Prediction_Status": "Accepted: exact dataset configuration",
        "Models": {},
        "Validation_Notice": (
            "AI models reproduce the MATLAB-generated dataset. "
            "This is not independent physical validation."
        ),
    }

    for target, details in METADATA["targets"].items():

        features = details["features"]

        X = pd.DataFrame(
            [[values[field] for field in features]],
            columns=features,
        )

        prediction = float(
            MODELS[target].predict(X)[0]
        )

        reference_value = float(reference[target])

        results["Models"][target] = {
            "Algorithm": details["algorithm"],
            "Prediction": prediction,
            "Unit": details["unit"],
            "MATLAB_Reference": reference_value,
            "Absolute_Difference": abs(
                prediction - reference_value
            ),
        }

    return results


if __name__ == "__main__":

    result = predict_emc_guard(
        separation_mm=8,
        pitch_mm=2,
        overlap_mm=150,
        ground_clearance_mm=10,
    )

    print("\nEMC-GUARD AI PREDICTION")
    print("=" * 55)

    print("Configuration:", result["Configuration_ID"])
    print("Status:", result["Prediction_Status"])

    for target, details in result["Models"].items():
        print("\n", target)
        print("AI Prediction:", details["Prediction"], details["Unit"])
        print("MATLAB Reference:", details["MATLAB_Reference"])
        print("Absolute Difference:", details["Absolute_Difference"])

    print("\nNotice:", result["Validation_Notice"])
