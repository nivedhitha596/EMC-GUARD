
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "09_Stage13A_Export" / "EMC_GUARD_Physics_Dataset.csv"

RESULTS = (
    ROOT / "10_Stage14A_AI_ML" / "results"
    / "stage14A_3_model_comparison"
)

PREDICTIONS = RESULTS / "LOLO_All_Predictions.csv"

OUT = RESULTS / "stage14A_3B_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
pred = pd.read_csv(PREDICTIONS)

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) > 1 and np.var(actual) > 0
            else np.nan
        ),
    }

# --------------------------------------------------
# Axis-wise pooled performance and worst fold
# --------------------------------------------------

axis_rows = []

for (target, model, axis), group in pred.groupby(
    ["Target", "Model", "ValidationAxis"]
):
    score = metrics(group["Actual"], group["Predicted"])

    fold_scores = []

    for fold_id, fold in group.groupby("FoldID"):
        fold_score = metrics(fold["Actual"], fold["Predicted"])
        fold_scores.append({
            "FoldID": fold_id,
            **fold_score,
            "TestRows": len(fold),
        })

    fold_table = pd.DataFrame(fold_scores)
    worst = fold_table.loc[fold_table["RMSE"].idxmax()]

    # Reconstruct a mean baseline for this held-out axis.
    baseline_actual = []
    baseline_predicted = []

    for level in sorted(df[axis].unique()):
        test_mask = df[axis] == level
        train_values = df.loc[~test_mask, target]
        test_values = df.loc[test_mask, target]

        baseline_value = float(train_values.mean())

        baseline_actual.extend(test_values.tolist())
        baseline_predicted.extend(
            [baseline_value] * len(test_values)
        )

    baseline_score = metrics(
        baseline_actual,
        baseline_predicted
    )

    improvement = (
        100.0
        * (baseline_score["RMSE"] - score["RMSE"])
        / baseline_score["RMSE"]
        if baseline_score["RMSE"] > 0
        else np.nan
    )

    axis_rows.append({
        "Target": target,
        "Model": model,
        "ValidationAxis": axis,
        "Pooled_MAE": score["MAE"],
        "Pooled_RMSE": score["RMSE"],
        "Pooled_R2": score["R2"],
        "Baseline_RMSE": baseline_score["RMSE"],
        "RMSE_Improvement_Percent": improvement,
        "WorstFold": worst["FoldID"],
        "WorstFold_RMSE": worst["RMSE"],
        "WorstFold_R2": worst["R2"],
    })

axis_summary = pd.DataFrame(axis_rows)
axis_summary = axis_summary.sort_values(
    ["Target", "ValidationAxis", "Pooled_RMSE"]
)

axis_file = OUT / "Axis_Wise_Performance.csv"
axis_summary.to_csv(axis_file, index=False)

# --------------------------------------------------
# Detailed fold ranking
# --------------------------------------------------

fold_rows = []

for (target, model, axis, fold_id), group in pred.groupby(
    ["Target", "Model", "ValidationAxis", "FoldID"]
):
    score = metrics(group["Actual"], group["Predicted"])

    fold_rows.append({
        "Target": target,
        "Model": model,
        "ValidationAxis": axis,
        "FoldID": fold_id,
        "TestRows": len(group),
        **score,
    })

fold_summary = pd.DataFrame(fold_rows)

fold_summary = fold_summary.sort_values(
    ["Target", "RMSE"],
    ascending=[True, False]
)

fold_file = OUT / "Worst_Fold_Performance.csv"
fold_summary.to_csv(fold_file, index=False)

# --------------------------------------------------
# Print concise report
# --------------------------------------------------

print("\n" + "=" * 70)
print("EMC-GUARD | STAGE 14A-3B")
print("AXIS-WISE GENERALIZATION DIAGNOSTICS")
print("=" * 70)

for target in axis_summary["Target"].unique():
    print("\nTARGET:", target)

    target_table = axis_summary[
        axis_summary["Target"] == target
    ]

    print(
        target_table[
            [
                "Model",
                "ValidationAxis",
                "Pooled_MAE",
                "Pooled_RMSE",
                "Pooled_R2",
                "RMSE_Improvement_Percent",
                "WorstFold",
                "WorstFold_RMSE",
            ]
        ].to_string(index=False)
    )

print("\nWORST 15 FOLDS ACROSS ALL MODELS")
print(fold_summary.head(15).to_string(index=False))

print("\nFILES SAVED:")
print(axis_file)
print(fold_file)

print("\nStage 14A-3B complete.")
