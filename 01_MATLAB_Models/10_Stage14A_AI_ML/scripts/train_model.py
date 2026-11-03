from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(
    r"D:\Downloads\kone\01_MATLAB_Models\10_Stage14A_AI_ML"
)

DATA_FILE = Path(
    r"D:\Downloads\kone\01_MATLAB_Models\09_Stage13A_Export"
    r"\EMC_GUARD_Physics_Dataset.csv"
)

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# DATASET DEFINITION
# --------------------------------------------------

FEATURES = [
    "PairSeparation_mm",
    "PairPitch_mm",
    "Overlap_mm",
    "GroundClearance_mm"
]

TARGETS = [
    "TotalCapacitance_pF_per_m",
    "AbsMutualInductance_nH"
]

df = pd.read_csv(DATA_FILE)

required = FEATURES + TARGETS

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if len(df) != 216:
    raise ValueError(f"Expected 216 rows, found {len(df)}")

if df[required].isnull().any().any():
    raise ValueError("Missing values found in model data.")

if not np.isfinite(df[required].to_numpy()).all():
    raise ValueError("Non-finite values found.")

if df.duplicated(subset=FEATURES).any():
    raise ValueError("Duplicate geometry configurations found.")

X = df[FEATURES]
config_ids = (
    df["Config_ID"].astype(str)
    if "Config_ID" in df.columns
    else pd.Series([f"EMC-{i:03d}" for i in range(1, 217)])
)

# --------------------------------------------------
# FIXED TRAIN / TEST SPLIT
# --------------------------------------------------

all_indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

split_manifest = pd.DataFrame({
    "Config_ID": config_ids,
    "Split": [
        "TRAIN" if i in set(train_idx) else "TEST"
        for i in all_indices
    ]
})

split_manifest.to_csv(
    RESULT_DIR / "data_split_manifest.csv",
    index=False
)

# --------------------------------------------------
# MODEL CANDIDATES
# --------------------------------------------------

def get_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_features=1.0,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            max_features=1.0,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=2,
            loss="squared_error",
            random_state=42
        )
    }

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

performance = []
feature_importance = []
predictions = []

# --------------------------------------------------
# TRAIN EACH TARGET INDEPENDENTLY
# --------------------------------------------------

for target in TARGETS:

    print(f"\n===== TRAINING TARGET: {target} =====")

    y = df[target]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    candidate_results = []

    for model_name, model in get_models().items():

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        mean_cv_rmse = -cv_scores.mean()

        candidate_results.append(
            (mean_cv_rmse, model_name, model)
        )

        print(
            f"{model_name}: "
            f"CV RMSE = {mean_cv_rmse:.10g}"
        )

    # Select using training-only cross-validation
    candidate_results.sort(key=lambda x: x[0])

    best_cv_rmse, best_name, best_model = candidate_results[0]

    # Fit selected model on complete training partition
    best_model.fit(X_train, y_train)

    # Evaluate once on locked test partition
    test_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    r2 = r2_score(y_test, test_pred)

    model_file = MODEL_DIR / f"{target}_{best_name}.joblib"

    joblib.dump(best_model, model_file)

    print(f"\nSelected model: {best_name}")
    print(f"Test MAE:  {mae:.12g}")
    print(f"Test RMSE: {rmse:.12g}")
    print(f"Test R2:   {r2:.8f}")

    performance.append({
        "Target": target,
        "SelectedModel": best_name,
        "CV_RMSE": best_cv_rmse,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
        "TrainingRows": len(train_idx),
        "TestRows": len(test_idx)
    })

    # Save feature importance for selected tree model
    if hasattr(best_model, "feature_importances_"):

        for feature, importance in zip(
            FEATURES,
            best_model.feature_importances_
        ):
            feature_importance.append({
                "Target": target,
                "Model": best_name,
                "Feature": feature,
                "Importance": importance
            })

    for config_id, actual, predicted in zip(
        config_ids.iloc[test_idx],
        y_test,
        test_pred
    ):
        predictions.append({
            "Config_ID": config_id,
            "Target": target,
            "Actual": actual,
            "Predicted": predicted,
            "AbsoluteError": abs(actual - predicted)
        })

# --------------------------------------------------
# EXPORT RESULTS
# --------------------------------------------------

performance_df = pd.DataFrame(performance)

performance_df.to_csv(
    RESULT_DIR / "model_performance.csv",
    index=False
)

pd.DataFrame(predictions).to_csv(
    RESULT_DIR / "heldout_test_predictions.csv",
    index=False
)

pd.DataFrame(feature_importance).to_csv(
    RESULT_DIR / "feature_importance.csv",
    index=False
)

metadata = {
    "project": "EMC-GUARD",
    "stage": "14A",
    "dataset_rows": len(df),
    "training_rows": len(train_idx),
    "test_rows": len(test_idx),
    "random_state": 42,
    "cross_validation": "5-fold KFold on training partition",
    "features": FEATURES,
    "targets": TARGETS,
    "model_selection": "Lowest mean cross-validation RMSE",
    "physical_validation": False
}

with open(
    RESULT_DIR / "training_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=4)

print("\n===== STAGE 14A COMPLETE =====")
print(performance_df.to_string(index=False))
print(f"\nResults saved to: {RESULT_DIR}")