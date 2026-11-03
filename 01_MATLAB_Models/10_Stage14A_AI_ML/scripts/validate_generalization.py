from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# EMC-GUARD - STAGE 14A-2
# LEAVE-ONE-GEOMETRY-LEVEL-OUT VALIDATION
# ==================================================

ROOT = Path(
    r"D:\Downloads\kone\01_MATLAB_Models\10_Stage14A_AI_ML"
)

DATA_FILE = Path(
    r"D:\Downloads\kone\01_MATLAB_Models"
    r"\09_Stage13A_Export\EMC_GUARD_Physics_Dataset.csv"
)

RESULT_DIR = ROOT / "results" / "generalization_validation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

# --------------------------------------------------
# LOAD AND CHECK DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

assert len(df) == 216, "Expected 216 configurations."

required = FEATURES + TARGETS

assert all(c in df.columns for c in required), \
    "Required dataset columns are missing."

assert not df[required].isnull().any().any(), \
    "Missing values detected."

assert np.isfinite(df[required].to_numpy()).all(), \
    "Non-finite values detected."

X = df[FEATURES]

# --------------------------------------------------
# MODEL FACTORY
# --------------------------------------------------

def create_extra_trees():

    return ExtraTreesRegressor(
        n_estimators=400,
        max_features=1.0,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )


def create_baseline():

    return DummyRegressor(strategy="mean")


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }


# --------------------------------------------------
# LEAVE-ONE-LEVEL-OUT EXPERIMENTS
# --------------------------------------------------

fold_results = []
prediction_results = []

total_folds = sum(
    df[feature].nunique() for feature in FEATURES
)

fold_number = 0

print("\n===== STAGE 14A-2 STARTED =====")
print(f"Total geometry-level experiments: {total_folds}")

for feature in FEATURES:

    levels = sorted(df[feature].unique())

    print(f"\n===== HOLDING OUT: {feature} =====")

    for level in levels:

        fold_number += 1

        train_mask = df[feature] != level
        test_mask = df[feature] == level

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        print(
            f"\nExperiment {fold_number}/{total_folds}: "
            f"{feature} = {level}"
        )

        for target in TARGETS:

            y_train = df.loc[train_mask, target]
            y_test = df.loc[test_mask, target]

            # Train Extra Trees
            model = create_extra_trees()
            model.fit(X_train, y_train)

            ai_prediction = model.predict(X_test)

            ai_metrics = calculate_metrics(
                y_test,
                ai_prediction
            )

            # Mean baseline
            baseline = create_baseline()
            baseline.fit(X_train, y_train)

            baseline_prediction = baseline.predict(X_test)

            baseline_metrics = calculate_metrics(
                y_test,
                baseline_prediction
            )

            # Store metrics
            fold_results.append({
                "HeldOutFeature": feature,
                "HeldOutLevel": level,
                "Target": target,
                "TestConfigurations": len(y_test),

                "AI_MAE": ai_metrics["MAE"],
                "AI_RMSE": ai_metrics["RMSE"],
                "AI_R2": ai_metrics["R2"],

                "Baseline_MAE": baseline_metrics["MAE"],
                "Baseline_RMSE": baseline_metrics["RMSE"],
                "Baseline_R2": baseline_metrics["R2"],

                "RMSE_Improvement_Percent": (
                    100 * (
                        1 - ai_metrics["RMSE"] /
                        baseline_metrics["RMSE"]
                    )
                    if baseline_metrics["RMSE"] > 0
                    else np.nan
                )
            })

            # Store individual predictions
            test_ids = (
                df.loc[test_mask, "Config_ID"].astype(str)
                if "Config_ID" in df.columns
                else df.index[test_mask].astype(str)
            )

            for config_id, actual, ai_pred, base_pred in zip(
                test_ids,
                y_test,
                ai_prediction,
                baseline_prediction
            ):

                prediction_results.append({
                    "Config_ID": config_id,
                    "HeldOutFeature": feature,
                    "HeldOutLevel": level,
                    "Target": target,
                    "Actual": actual,
                    "AI_Predicted": ai_pred,
                    "AI_AbsoluteError": abs(actual - ai_pred),
                    "Baseline_Predicted": base_pred,
                    "Baseline_AbsoluteError": abs(actual - base_pred)
                })

            print(
                f"{target}: "
                f"AI RMSE={ai_metrics['RMSE']:.8g}, "
                f"AI R2={ai_metrics['R2']:.6f}, "
                f"Baseline RMSE={baseline_metrics['RMSE']:.8g}"
            )

# --------------------------------------------------
# EXPORT FOLD RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

fold_file = RESULT_DIR / "LOLO_Fold_Results.csv"

fold_df.to_csv(fold_file, index=False)

# --------------------------------------------------
# EXPORT PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame(prediction_results)

prediction_file = (
    RESULT_DIR / "LOLO_All_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# --------------------------------------------------
# SUMMARY BY FEATURE AND TARGET
# --------------------------------------------------

summary_df = (
    fold_df.groupby(
        ["HeldOutFeature", "Target"],
        as_index=False
    )
    .agg(
        NumberOfLevels=("HeldOutLevel", "count"),
        Mean_AI_MAE=("AI_MAE", "mean"),
        Mean_AI_RMSE=("AI_RMSE", "mean"),
        Mean_AI_R2=("AI_R2", "mean"),
        Mean_Baseline_RMSE=("Baseline_RMSE", "mean"),
        Mean_RMSE_Improvement_Percent=(
            "RMSE_Improvement_Percent", "mean"
        )
    )
)

summary_file = RESULT_DIR / "LOLO_Summary.csv"

summary_df.to_csv(summary_file, index=False)

# --------------------------------------------------
# DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n===== STAGE 14A-2 SUMMARY =====")

print(summary_df.to_string(index=False))

print("\nTotal experiments:", len(fold_df))

print("Total individual predictions:", len(prediction_df))

print("\nSaved files:")
print(fold_file)
print(prediction_file)
print(summary_file)

print("\n===== STAGE 14A-2 COMPLETE =====")