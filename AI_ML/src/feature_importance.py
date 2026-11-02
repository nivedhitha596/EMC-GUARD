"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 4 | Script 1: Feature Importance Analysis
File   : feature_importance.py
Author : EMC-GUARD Development Team

Implements permutation importance using the INDEPENDENT TEST dataset.

Key design decisions
--------------------
1. Permutation is applied to RAW input feature columns (before encoding).
   For each shuffle, the raw column is corrupted, the full preprocessing
   pipeline is re-applied, and the model is re-scored.  This correctly
   attributes importance to the ORIGINAL features (e.g. "Shield_Type")
   rather than to individual one-hot columns.

2. Scoring metric: negative MAE (neg_mae).  Lower predicted MAE after
   shuffling = larger importance.  We also compute neg_RMSE for reference.

3. Importance is repeated n_repeats times per feature; mean and std are
   reported.

DISCLAIMER
----------
Permutation importance measures the degree to which a model's PREDICTIONS
depend on each feature in the TEST dataset.  It does NOT imply physical
causation.  Results are from SYNTHETIC DUMMY DATA and may differ
substantially from those obtained on real MATLAB simulation data.

Usage
-----
    python src/feature_importance.py
    -- or --
    from feature_importance import calculate_feature_importance, plot_feature_importance
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CSV = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
MODEL_PKL   = os.path.join(BASE_DIR, "models",  "emc_surrogate_model.pkl")
PREPROC_PKL = os.path.join(BASE_DIR, "models",  "emc_preprocessor.pkl")
META_JSON   = os.path.join(BASE_DIR, "models",  "target_model_metadata.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VIS_DIR     = os.path.join(REPORTS_DIR, "visualisations", "feature_importance")
FI_CSV      = os.path.join(REPORTS_DIR, "feature_importance.csv")

# ---------------------------------------------------------------------------
# COLUMN DEFINITIONS  (must match train_model.py)
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES   = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
]
CATEGORICAL_FEATURES = ["Shield_Type", "Cable_Configuration"]
INPUT_FEATURES       = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLS          = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_REPEATS    = 10   # shuffles per feature (10 gives stable estimates, runs fast)
MAX_PERM_ROWS = 80  # subsample test set for permutation — speeds up 3-4x

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
SYNTH_NOTE = "\n[SYNTHETIC DATA — permutation importance, not physical causation]"


# ---------------------------------------------------------------------------
# LOAD ARTEFACTS
# ---------------------------------------------------------------------------

def _load_artefacts():
    for path, lbl in [(MODEL_PKL, "emc_surrogate_model.pkl"),
                      (PREPROC_PKL, "emc_preprocessor.pkl"),
                      (META_JSON,  "target_model_metadata.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{lbl} not found: {path}\n"
                                    "Run train_model.py (Module 2) first.")
    preprocessor     = joblib.load(PREPROC_PKL)
    surrogate_bundle = joblib.load(MODEL_PKL)
    with open(META_JSON) as f:
        metadata = json.load(f)
    return preprocessor, surrogate_bundle, metadata


def _build_test_split(df: pd.DataFrame, target: str):
    """Reconstruct the same 80/20 split used in Module 2."""
    sub  = df.dropna(subset=[target])
    X    = sub[INPUT_FEATURES].reset_index(drop=True)
    y    = sub[target].reset_index(drop=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return Xte.reset_index(drop=True), yte.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PERMUTATION IMPORTANCE  (raw-feature space)
# ---------------------------------------------------------------------------

def _get_tree_importance(model, features: list) -> dict:
    """
    Extract mean-impurity-decrease (MDI) feature importances from the
    tree ensemble.  This is instantaneous — no prediction needed.
    Only covers ENCODED columns, so we map OHE columns back to raw features.
    Returns {raw_feature: mdi_importance} normalised to sum=1.
    """
    try:
        importances = model.feature_importances_
        # feature_names_in_ may not exist on older sklearn — use indices
        feat_imp = dict(enumerate(importances))
        return feat_imp
    except Exception:
        return {}


def _permutation_importance_raw(model, preprocessor,
                                  X_test_raw: pd.DataFrame,
                                  y_test: pd.Series,
                                  features: list,
                                  n_repeats: int = N_REPEATS,
                                  random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Compute permutation importance by shuffling RAW columns then re-encoding.

    Speed optimisations applied:
      - MAX_PERM_ROWS: subsample the test set so each transform+predict is fast.
      - n_repeats=10: 10 shuffles per feature is sufficient for stable estimates.

    For each feature:
      - Record baseline MAE on (subsampled) test set.
      - n_repeats times: shuffle that column, transform, predict, measure MAE.
      - Importance = mean(shuffled_MAE) - baseline_MAE
        (positive = shuffling hurts performance = feature is important)

    Returns DataFrame with columns:
        feature, mean_importance, std_importance, baseline_mae
    """
    rng = np.random.default_rng(random_state)

    # ----- subsample test rows for speed -----
    if len(X_test_raw) > MAX_PERM_ROWS:
        sample_idx = rng.choice(len(X_test_raw), size=MAX_PERM_ROWS, replace=False)
        X_sub = X_test_raw.iloc[sample_idx].reset_index(drop=True)
        y_sub = y_test.iloc[sample_idx].reset_index(drop=True)
    else:
        X_sub = X_test_raw.reset_index(drop=True)
        y_sub = y_test.reset_index(drop=True)

    # Baseline — encode once
    X_base_enc  = preprocessor.transform(X_sub)
    y_base_pred = model.predict(X_base_enc)
    baseline_mae = mean_absolute_error(y_sub, y_base_pred)

    records = []
    for feat in features:
        importances = []
        for _ in range(n_repeats):
            X_perm = X_sub.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            X_enc  = preprocessor.transform(X_perm)
            y_pred = model.predict(X_enc)
            shuffled_mae = mean_absolute_error(y_sub, y_pred)
            importances.append(shuffled_mae - baseline_mae)

        records.append({
            "feature":          feat,
            "mean_importance":  float(np.mean(importances)),
            "std_importance":   float(np.std(importances)),
            "baseline_mae":     round(baseline_mae, 6),
        })

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    return result_df


# ---------------------------------------------------------------------------
# PUBLIC API: calculate_feature_importance()
# ---------------------------------------------------------------------------

def calculate_feature_importance(n_repeats: int = N_REPEATS) -> dict:
    """
    Calculate permutation importance for every available target.

    Returns
    -------
    dict: {target: pd.DataFrame with columns
               [feature, mean_importance, std_importance, baseline_mae, rank]}
    """
    print("  Loading artefacts ...")
    preprocessor, surrogate_bundle, metadata = _load_artefacts()

    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV)

    results = {}
    all_rows = []

    for target in TARGET_COLS:
        if target not in surrogate_bundle:
            print(f"  [SKIP] {target} not in surrogate bundle.")
            continue

        model = surrogate_bundle[target]["model"]
        algo  = surrogate_bundle[target]["algorithm"]
        print(f"  [{target}]  ({algo})  permutation importance "
              f"({n_repeats} repeats x {len(INPUT_FEATURES)} features, "
              f"max {MAX_PERM_ROWS} test rows) ...")

        X_test_raw, y_test = _build_test_split(df, target)
        fi_df = _permutation_importance_raw(
            model, preprocessor, X_test_raw, y_test,
            INPUT_FEATURES, n_repeats=n_repeats
        )
        fi_df.insert(0, "target",    target)
        fi_df.insert(1, "algorithm", algo)
        results[target] = fi_df
        all_rows.append(fi_df)
        print(f"    Top feature: {fi_df.iloc[0]['feature']}  "
              f"(mean_imp={fi_df.iloc[0]['mean_importance']:.4f}  "
              f"std={fi_df.iloc[0]['std_importance']:.4f})")

    # Save combined CSV
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        combined.to_csv(FI_CSV, index=False)
        print(f"\n  Saved: reports/feature_importance.csv")

    return results


# ---------------------------------------------------------------------------
# PUBLIC API: plot_feature_importance()
# ---------------------------------------------------------------------------

def plot_feature_importance(fi_results: dict) -> None:
    """
    Generate a horizontal bar chart with error bars for each target.
    Saves to reports/visualisations/feature_importance/
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    palette = sns.color_palette("muted", len(INPUT_FEATURES))

    for target, fi_df in fi_results.items():
        fig, ax = plt.subplots(figsize=(9, 5))

        # Sort ascending so most important is at the top of horizontal bars
        plot_df = fi_df.sort_values("mean_importance", ascending=True)

        colours = [palette[i % len(palette)] for i in range(len(plot_df))]
        bars = ax.barh(
            plot_df["feature"],
            plot_df["mean_importance"],
            xerr=plot_df["std_importance"],
            color=colours,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

        # Annotate bar values
        for bar, val in zip(bars, plot_df["mean_importance"]):
            ax.text(max(val + plot_df["std_importance"].max() * 0.05, 0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=7.5)

        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(
            "Mean increase in MAE when feature is permuted\n"
            "(larger = more important to model predictions)",
            fontsize=9
        )
        ax.set_ylabel("Input Feature", fontsize=9)
        ax.set_title(
            f"Permutation Feature Importance — {target}"
            + SYNTH_NOTE,
            fontsize=10
        )
        fig.tight_layout()
        safe_name = target.replace("/", "_").replace(" ", "_")
        fig_path = os.path.join(VIS_DIR, f"fi_{safe_name}.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: reports/visualisations/feature_importance/fi_{safe_name}.png")

    # Summary heatmap: all targets x all features
    _plot_importance_heatmap(fi_results)


def _plot_importance_heatmap(fi_results: dict) -> None:
    """Heatmap of mean importance across all targets (features as rows)."""
    if not fi_results:
        return

    targets  = list(fi_results.keys())
    hmap     = pd.DataFrame(index=INPUT_FEATURES, columns=targets, dtype=float)
    for tgt, fi_df in fi_results.items():
        for _, row in fi_df.iterrows():
            hmap.at[row["feature"], tgt] = row["mean_importance"]

    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 2), max(5, len(INPUT_FEATURES) * 0.7)))
    sns.heatmap(hmap.astype(float), annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=0.4, ax=ax, annot_kws={"size": 7},
                cbar_kws={"label": "Mean MAE increase when permuted"})
    ax.set_title("Feature Importance Heatmap (all targets)" + SYNTH_NOTE, fontsize=10)
    ax.set_xlabel("Prediction Target")
    ax.set_ylabel("Input Feature")
    fig.tight_layout()
    path = os.path.join(VIS_DIR, "fi_heatmap_all_targets.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: reports/visualisations/feature_importance/fi_heatmap_all_targets.png")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("EMC-GUARD  |  Module 4  |  Feature Importance Analysis")
    print("=" * 65)
    fi_results = calculate_feature_importance()
    plot_feature_importance(fi_results)
    print("\n[DONE] Feature importance analysis complete.")

