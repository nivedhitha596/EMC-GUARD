
"""
EMC-GUARD | Stage 14A-3
Physics-Informed Surrogate Model Comparison

Targets:
    1. TotalCapacitance_pF_per_m
    2. AbsMutualInductance_nH

Validation:
    Leave-One-Feature-Level-Out (LOLO)

Models:
    - Extra Trees
    - Random Forest
    - Gradient Boosting
    - Polynomial Ridge Degree 2
    - Polynomial Ridge Degree 3
    - RBF Support Vector Regression

All metrics are calculated in the original target units.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. PATH CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "09_Stage13A_Export"
    / "EMC_GUARD_Physics_Dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "10_Stage14A_AI_ML"
    / "results"
    / "stage14A_3_model_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("EMC-GUARD | STAGE 14A-3")
print("PHYSICS-INFORMED SURROGATE MODEL COMPARISON")
print("=" * 65)

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded:")
print("File:", DATA_FILE)
print("Rows:", len(df))
print("Columns:", list(df.columns))

# ---------------------------------------------------------
# 3. DEFINE PHYSICS-INFORMED INPUTS
# ---------------------------------------------------------

TARGET_CONFIG = {
    "TotalCapacitance_pF_per_m": {
        "features": [
            "PairSeparation_mm",
            "PairPitch_mm",
            "GroundClearance_mm",
        ],
        "validation_axes": [
            "PairSeparation_mm",
            "PairPitch_mm",
            "GroundClearance_mm",
        ],
    },

    "AbsMutualInductance_nH": {
        "features": [
            "PairSeparation_mm",
            "PairPitch_mm",
            "Overlap_mm",
        ],
        "validation_axes": [
            "PairSeparation_mm",
            "PairPitch_mm",
            "Overlap_mm",
        ],
    },
}

# Verify required columns
required_columns = set()

for target, config in TARGET_CONFIG.items():
    required_columns.add(target)
    required_columns.update(config["features"])
    required_columns.update(config["validation_axes"])

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing dataset columns: {sorted(missing)}")

if df[list(required_columns)].isnull().any().any():
    raise ValueError("Missing values detected. Resolve before training.")

# Positive target values are required for logarithmic transformation.
for target in TARGET_CONFIG:
    if (df[target] <= 0).any():
        raise ValueError(f"{target} contains non-positive values.")

print("\nPhysics-informed feature selection:")
for target, config in TARGET_CONFIG.items():
    print(f"\n{target}")
    print("Inputs:", config["features"])
    print("Validation axes:", config["validation_axes"])

# ---------------------------------------------------------
# 4. MODEL FACTORY
# ---------------------------------------------------------

def build_models():

    base_models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=RANDOM_STATE,
        ),

        "PolynomialRidge_Degree2": Pipeline([
            ("poly", PolynomialFeatures(
                degree=2,
                include_bias=False
            )),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),

        "PolynomialRidge_Degree3": Pipeline([
            ("poly", PolynomialFeatures(
                degree=3,
                include_bias=False
            )),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),

        "RBF_SVR": Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(
                kernel="rbf",
                C=100.0,
                epsilon=0.01,
                gamma="scale",
            )),
        ]),
    }

    # Log-transform the target during training.
    # Predictions are automatically converted back to original units.
    models = {}

    for name, estimator in base_models.items():
        models[name] = TransformedTargetRegressor(
            regressor=clone(estimator),
            func=np.log,
            inverse_func=np.exp,
            check_inverse=False,
        )

    return models


# ---------------------------------------------------------
# 5. METRIC FUNCTION
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    if len(actual) >= 2 and np.var(actual) > 0:
        r2 = r2_score(actual, predicted)
    else:
        r2 = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ---------------------------------------------------------
# 6. LOLO VALIDATION
# ---------------------------------------------------------

all_predictions = []
fold_results = []
summary_results = []

for target, config in TARGET_CONFIG.items():

    features = config["features"]
    validation_axes = config["validation_axes"]

    X = df[features].copy()
    y = df[target].copy()

    models = build_models()

    print("\n" + "=" * 65)
    print("TARGET:", target)
    print("=" * 65)

    for axis in validation_axes:

        unique_levels = sorted(df[axis].unique())

        print(f"\nHolding out feature: {axis}")
        print("Levels:", unique_levels)

        for level in unique_levels:

            test_mask = df[axis] == level
            train_mask = ~test_mask

            X_train = X.loc[train_mask]
            X_test = X.loc[test_mask]

            y_train = y.loc[train_mask]
            y_test = y.loc[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # Mean baseline: calculated only from training data.
            baseline_value = float(y_train.mean())

            fold_id = f"{target}__{axis}__{level}"

            print(
                f"  Fold: {axis}={level} | "
                f"Train={len(X_train)} | Test={len(X_test)}"
            )

            # Baseline predictions
            baseline_pred = np.full(
                len(y_test),
                baseline_value,
                dtype=float
            )

            baseline_metrics = calculate_metrics(
                y_test,
                baseline_pred
            )

            fold_results.append({
                "Target": target,
                "Model": "MeanBaseline",
                "ValidationAxis": axis,
                "HeldOutLevel": level,
                "FoldID": fold_id,
                "TrainRows": len(X_train),
                "TestRows": len(X_test),
                **baseline_metrics,
            })

            for model_name, model in models.items():

                try:
                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                    # Protect against numerical underflow or invalid values.
                    if not np.isfinite(predictions).all():
                        raise ValueError("Non-finite predictions.")

                    if (predictions <= 0).any():
                        raise ValueError("Non-positive predictions.")

                    metrics = calculate_metrics(
                        y_test,
                        predictions
                    )

                    fold_results.append({
                        "Target": target,
                        "Model": model_name,
                        "ValidationAxis": axis,
                        "HeldOutLevel": level,
                        "FoldID": fold_id,
                        "TrainRows": len(X_train),
                        "TestRows": len(X_test),
                        **metrics,
                    })

                    for row_index, actual, predicted in zip(
                        df.index[test_mask],
                        y_test.to_numpy(),
                        predictions,
                    ):
                        prediction_row = {
                            "Target": target,
                            "Model": model_name,
                            "ValidationAxis": axis,
                            "HeldOutLevel": level,
                            "FoldID": fold_id,
                            "SourceRowIndex": int(row_index),
                            "Actual": float(actual),
                            "Predicted": float(predicted),
                            "AbsoluteError": float(
                                abs(actual - predicted)
                            ),
                        }

                        for feature in features:
                            prediction_row[feature] = float(
                                df.loc[row_index, feature]
                            )

                        all_predictions.append(prediction_row)

                except Exception as error:
                    print(
                        f"    ERROR: {model_name}: {error}"
                    )

                    fold_results.append({
                        "Target": target,
                        "Model": model_name,
                        "ValidationAxis": axis,
                        "HeldOutLevel": level,
                        "FoldID": fold_id,
                        "TrainRows": len(X_train),
                        "TestRows": len(X_test),
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                    })

# ---------------------------------------------------------
# 7. CREATE OUTPUT TABLES
# ---------------------------------------------------------

fold_df = pd.DataFrame(fold_results)
prediction_df = pd.DataFrame(all_predictions)

if prediction_df.empty:
    raise RuntimeError("No successful model predictions were generated.")

fold_file = OUTPUT_DIR / "LOLO_Fold_Results.csv"
prediction_file = OUTPUT_DIR / "LOLO_All_Predictions.csv"

fold_df.to_csv(fold_file, index=False)
prediction_df.to_csv(prediction_file, index=False)

# ---------------------------------------------------------
# 8. POOLED METRICS
# ---------------------------------------------------------

for target in TARGET_CONFIG:

    target_predictions = prediction_df[
        prediction_df["Target"] == target
    ]

    baseline_rows = fold_df[
        (fold_df["Target"] == target)
        & (fold_df["Model"] == "MeanBaseline")
    ]

    # Rebuild baseline predictions from the saved fold records.
    baseline_predictions = []

    for _, fold in baseline_rows.iterrows():

        mask = (
            (df[fold["ValidationAxis"]] == fold["HeldOutLevel"])
        )

        y_test = df.loc[mask, target].to_numpy()

        # The fold's training mean can be recovered from the
        # baseline RMSE only with more information, so baseline
        # predictions are rebuilt in a separate direct pass.
        train_values = df.loc[~mask, target]
        baseline_value = float(train_values.mean())

        for actual in y_test:
            baseline_predictions.append({
                "Actual": float(actual),
                "Predicted": baseline_value,
            })

    baseline_df = pd.DataFrame(baseline_predictions)

    baseline_metrics = calculate_metrics(
        baseline_df["Actual"],
        baseline_df["Predicted"]
    )

    for model_name in sorted(target_predictions["Model"].unique()):

        model_rows = target_predictions[
            target_predictions["Model"] == model_name
        ]

        metrics = calculate_metrics(
            model_rows["Actual"],
            model_rows["Predicted"]
        )

        improvement = (
            100.0
            * (baseline_metrics["RMSE"] - metrics["RMSE"])
            / baseline_metrics["RMSE"]
            if baseline_metrics["RMSE"] > 0
            else np.nan
        )

        summary_results.append({
            "Target": target,
            "Model": model_name,
            "Pooled_MAE": metrics["MAE"],
            "Pooled_RMSE": metrics["RMSE"],
            "Pooled_R2": metrics["R2"],
            "Baseline_MAE": baseline_metrics["MAE"],
            "Baseline_RMSE": baseline_metrics["RMSE"],
            "RMSE_Improvement_Percent": improvement,
            "PredictionCount": len(model_rows),
        })

summary_df = pd.DataFrame(summary_results)

summary_df = summary_df.sort_values(
    ["Target", "Pooled_RMSE"]
)

summary_file = OUTPUT_DIR / "LOLO_Pooled_Summary.csv"
summary_df.to_csv(summary_file, index=False)

# ---------------------------------------------------------
# 9. SELECT BEST CANDIDATE PER TARGET
# ---------------------------------------------------------

best_models = {}

for target in TARGET_CONFIG:

    candidates = summary_df[
        summary_df["Target"] == target
    ].dropna(subset=["Pooled_RMSE"])

    if candidates.empty:
        continue

    best = candidates.sort_values(
        ["Pooled_RMSE", "Pooled_MAE"]
    ).iloc[0]

    best_models[target] = {
        "SelectedModel": best["Model"],
        "Pooled_MAE": float(best["Pooled_MAE"]),
        "Pooled_RMSE": float(best["Pooled_RMSE"]),
        "Pooled_R2": float(best["Pooled_R2"]),
        "RMSE_Improvement_Percent": float(
            best["RMSE_Improvement_Percent"]
        ),
        "SelectionBasis": "Lowest pooled LOLO RMSE",
    }

# ---------------------------------------------------------
# 10. SAVE METADATA
# ---------------------------------------------------------

metadata = {
    "Stage": "14A-3",
    "Project": "EMC-GUARD",
    "Validation": "Leave-One-Feature-Level-Out",
    "TargetConfigurations": TARGET_CONFIG,
    "ModelCandidates": [
        "ExtraTrees",
        "RandomForest",
        "GradientBoosting",
        "PolynomialRidge_Degree2",
        "PolynomialRidge_Degree3",
        "RBF_SVR",
    ],
    "TargetTransformation": "Natural logarithm during fitting",
    "DatasetRows": int(len(df)),
    "PredictionRows": int(len(prediction_df)),
    "BestCandidates": best_models,
    "ImportantLimitation": (
        "Validation is against the deterministic MATLAB dataset. "
        "It does not establish independent physical validation, "
        "manufacturer cable accuracy, or performance outside the "
        "tested geometry domain."
    ),
}

metadata_file = OUTPUT_DIR / "Stage14A_3_Metadata.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

# ---------------------------------------------------------
# 11. FINAL REPORT
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("STAGE 14A-3 COMPLETE")
print("=" * 65)

print("\nPOOLED MODEL COMPARISON")
print(
    summary_df[
        [
            "Target",
            "Model",
            "Pooled_MAE",
            "Pooled_RMSE",
            "Pooled_R2",
            "RMSE_Improvement_Percent",
        ]
    ].to_string(index=False)
)

print("\nBEST CANDIDATES")
for target, result in best_models.items():
    print(f"\n{target}")
    for key, value in result.items():
        print(f"  {key}: {value}")

print("\nOUTPUT FILES:")
print(f"1. {fold_file}")
print(f"2. {prediction_file}")
print(f"3. {summary_file}")
print(f"4. {metadata_file}")

print("\nIMPORTANT:")
print(
    "These results validate surrogate reproduction of the MATLAB "
    "dataset only. Physical validation remains a separate stage."
)

print("\nStage 14A-3 finished.")
