
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "09_Stage13A_Export"
    / "EMC_GUARD_Physics_Dataset.csv"
)

RESULTS_DIR = (
    ROOT / "10_Stage14A_AI_ML"
    / "results" / "stage14A_3_model_comparison"
)

LOLO_FILE = (
    RESULTS_DIR
    / "stage14A_3B_diagnostics"
    / "Axis_Wise_Performance.csv"
)
MODEL_DIR = (
    ROOT / "10_Stage14A_AI_ML"
    / "models" / "stage14A_4_final"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "TotalCapacitance_pF_per_m": {
        "features": [
            "PairSeparation_mm",
            "PairPitch_mm",
            "GroundClearance_mm",
        ],
        "filename": "EMC_GUARD_Capacitance_ExtraTrees.joblib",
        "unit": "pF/m",
    },
    "AbsMutualInductance_nH": {
        "features": [
            "PairSeparation_mm",
            "PairPitch_mm",
            "Overlap_mm",
        ],
        "filename": "EMC_GUARD_MutualInductance_ExtraTrees.joblib",
        "unit": "nH",
    },
}

print("=" * 68)
print("EMC-GUARD | STAGE 14A-4")
print("FINAL SURROGATE TRAINING AND MODEL FREEZE")
print("=" * 68)

# --------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

if len(df) != 216:
    raise ValueError(
        f"Expected 216 configurations, found {len(df)}."
    )

if df["Config_ID"].nunique() != 216:
    raise ValueError("Config_ID is not unique across all 216 rows.")

for target, cfg in CONFIG.items():
    required = cfg["features"] + [target]

    if df[required].isnull().any().any():
        raise ValueError(f"Missing values detected for {target}.")

    if (df[target] <= 0).any():
        raise ValueError(f"Non-positive target values for {target}.")

print("\nDataset validation passed.")
print("Rows:", len(df))
print("Unique configurations:", df["Config_ID"].nunique())

# --------------------------------------------------
# 2. READ LOLO VALIDATION EVIDENCE
# --------------------------------------------------

if not LOLO_FILE.exists():
    raise FileNotFoundError(
        f"LOLO validation report not found: {LOLO_FILE}"
    )

lolo = pd.read_csv(LOLO_FILE)

required_lolo_columns = [
    "Target",
    "Model",
    "ValidationAxis",
    "Pooled_RMSE",
    "Pooled_MAE",
    "Pooled_R2",
]

missing_lolo = set(required_lolo_columns) - set(lolo.columns)

if missing_lolo:
    raise ValueError(
        f"Missing validation columns: {sorted(missing_lolo)}"
    )

print("\nAxis-wise LOLO validation evidence loaded.")
print(lolo[required_lolo_columns].to_string(index=False))

# --------------------------------------------------
# 3. TRAIN FINAL MODELS ON ALL 216 ROWS
# --------------------------------------------------

model_records = {}

for target, cfg in CONFIG.items():

    features = cfg["features"]
    X = df[features]
    y = df[target]

    estimator = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model = TransformedTargetRegressor(
        regressor=estimator,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )

    model.fit(X, y)

    # In-sample reproduction diagnostics only.
    # These are NOT independent validation metrics.
    fitted = model.predict(X)

    train_rmse = float(
        np.sqrt(mean_squared_error(y, fitted))
    )
    train_mae = float(
        mean_absolute_error(y, fitted)
    )
    train_r2 = float(
        r2_score(y, fitted)
    )

    model_path = MODEL_DIR / cfg["filename"]
    joblib.dump(model, model_path)

    target_lolo = lolo[
        (lolo["Target"] == target)
        & (lolo["Model"] == "ExtraTrees")
    ]

    if target_lolo.empty:
        raise ValueError(
            f"ExtraTrees LOLO evidence missing for {target}."
        )

    print("\n" + "-" * 60)
    print("TARGET:", target)
    print("Features:", features)
    print("Saved:", model_path)
    print("Training reproduction RMSE:", train_rmse)
    print("Training reproduction MAE:", train_mae)
    print("Training reproduction R2:", train_r2)

    axis_evidence = {}

    for _, row in target_lolo.iterrows():
        axis_evidence[row["ValidationAxis"]] = {
            "Pooled_RMSE": float(row["Pooled_RMSE"]),
            "Pooled_MAE": float(row["Pooled_MAE"]),
            "Pooled_R2": float(row["Pooled_R2"]),
        }

    model_records[target] = {
        "model_file": cfg["filename"],
        "features": features,
        "unit": cfg["unit"],
        "algorithm": "ExtraTreesRegressor",
        "estimators": 400,
        "target_transform": "natural_log",
        "training_rows": int(len(df)),
        "training_reproduction": {
            "RMSE": train_rmse,
            "MAE": train_mae,
            "R2": train_r2,
            "interpretation": (
                "In-sample reproduction only; not independent validation."
            ),
        },
        "LOLO_validation": axis_evidence,
    }

# --------------------------------------------------
# 4. SAVE DOMAIN AND DATASET INFORMATION
# --------------------------------------------------

geometry_columns = [
    "PairSeparation_mm",
    "PairPitch_mm",
    "Overlap_mm",
    "GroundClearance_mm",
]

geometry_levels = {
    column: sorted(
        [float(x) for x in df[column].unique()]
    )
    for column in geometry_columns
}

valid_configurations = []

for _, row in df.iterrows():
    valid_configurations.append({
        column: float(row[column])
        for column in geometry_columns
    })

metadata = {
    "project": "EMC-GUARD",
    "stage": "14A-4",
    "status": "Final surrogate models trained",
    "dataset": DATA_FILE.name,
    "dataset_rows": int(len(df)),
    "unique_configurations": int(df["Config_ID"].nunique()),
    "model_directory": str(MODEL_DIR),
    "geometry_levels": geometry_levels,
    "supported_configurations": valid_configurations,
    "targets": model_records,
    "deployment_policy": {
        "accept_only_exact_dataset_configurations": True,
        "allow_extrapolation": False,
        "allow_unverified_intermediate_geometry": False,
    },
    "limitations": [
        "Models reproduce a deterministic MATLAB-generated dataset.",
        "LOLO validation is not physical validation.",
        "The model geometry is not a verified KONE cable cross-section.",
        "No real elevator EMC compliance claim is established.",
    ],
}

metadata_path = MODEL_DIR / "EMC_GUARD_AI_Model_Metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# --------------------------------------------------
# 5. SAVE TRAINING REPRODUCTION REPORT
# --------------------------------------------------

report_rows = []

for target, cfg in CONFIG.items():
    model = joblib.load(MODEL_DIR / cfg["filename"])
    predictions = model.predict(df[cfg["features"]])

    for index, prediction in enumerate(predictions):
        report_rows.append({
            "Config_ID": df.iloc[index]["Config_ID"],
            "Target": target,
            "Reference_MATLAB": float(df.iloc[index][target]),
            "AI_TrainingPrediction": float(prediction),
            "AbsoluteError": float(
                abs(df.iloc[index][target] - prediction)
            ),
        })

report = pd.DataFrame(report_rows)

report_path = MODEL_DIR / "FullDataset_Reproduction_Report.csv"
report.to_csv(report_path, index=False)

# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 68)
print("STAGE 14A-4 COMPLETE")
print("=" * 68)

print("Final model directory:")
print(MODEL_DIR)

print("\nSaved files:")
for file in sorted(MODEL_DIR.iterdir()):
    print(" -", file.name)

print("\nIMPORTANT:")
print(
    "The full-dataset training reproduction report is not "
    "independent validation. Retain the Stage 14A-3 LOLO "
    "results as the generalization evidence."
)

print("\nStage 14A-4 model training finished.")
