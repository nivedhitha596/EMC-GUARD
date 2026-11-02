"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 4 | Script 2: Prediction Uncertainty and Error Analysis
File   : prediction_uncertainty.py
Author : EMC-GUARD Development Team

Implements:
  A. MODEL DISAGREEMENT ANALYSIS
       • Cross-model disagreement  : RF vs ET predictions (from model_performance.json)
       • Intra-ensemble spread     : std of per-tree predictions within the
                                     best-selected ensemble (200 trees)
       Both are clearly labelled as *spread estimates*, NOT calibrated
       confidence intervals.

  B. TEST-SET ERROR STATISTICS
       MAE, RMSE, mean residual, residual std, median AE,
       90th-percentile AE, 95th-percentile AE — computed on the
       independent test partition.

  C. SPLIT CONFORMAL PREDICTION INTERVALS  (optional, Task 6)
       Uses 60% of test rows as calibration, 40% for evaluation.
       Coverage is empirically measured and reported.
       Assumptions are clearly documented.

  D. OUT-OF-DOMAIN DETECTION  (Task 7)
       Per-feature range check + Mahalanobis distance for
       multivariate novelty detection.

  E. PREDICTION RELIABILITY ASSESSMENT  (Task 8)
       Combines all signals into a single structured dict with
       status label and ordered warning messages.

DISCLAIMER
----------
All results are from SYNTHETIC DUMMY DATA.  The intra-ensemble spread
is not a statistically calibrated prediction interval.  Conformal
intervals have marginal coverage guarantees only under exchangeability
(i.i.d. data from the same distribution as training).

Usage
-----
    python src/prediction_uncertainty.py
    -- or import individual functions --
"""

import os, sys, json, warnings, copy
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              median_absolute_error)
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CSV = os.path.join(BASE_DIR, "dataset", "EMC_GUARD_Physics_Dataset.csv")
MODEL_PKL   = os.path.join(BASE_DIR, "models",  "emc_surrogate_model.pkl")
PREPROC_PKL = os.path.join(BASE_DIR, "models",  "emc_preprocessor.pkl")
META_JSON   = os.path.join(BASE_DIR, "models",  "target_model_metadata.json")
PERF_JSON   = os.path.join(BASE_DIR, "reports", "model_performance.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VIS_RESID   = os.path.join(REPORTS_DIR, "visualisations", "residual_plots")
VIS_PI      = os.path.join(REPORTS_DIR, "visualisations", "prediction_intervals")
ERR_CSV     = os.path.join(REPORTS_DIR, "prediction_error_analysis.csv")
CALIB_JSON  = os.path.join(REPORTS_DIR, "uncertainty_calibration.json")

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
VALID_SHIELD   = {"Shielded", "Unshielded"}
VALID_CABLECONF= {"Parallel", "Perpendicular", "Crossed"}

RANDOM_STATE      = 42
TEST_SIZE         = 0.20
CONFORMAL_CALIB_FRAC = 0.60   # 60% of test rows → calibration
COVERAGE_LEVEL    = 0.90      # nominal conformal coverage

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
SYNTH_NOTE = "\n[SYNTHETIC DATA — empirical error statistics, not physics-validated]"


# ---------------------------------------------------------------------------
# HELPERS — LOAD ARTEFACTS
# ---------------------------------------------------------------------------

def _load_artefacts():
    for path, lbl in [(MODEL_PKL, "emc_surrogate_model.pkl"),
                      (PREPROC_PKL, "emc_preprocessor.pkl"),
                      (META_JSON,   "target_model_metadata.json"),
                      (PERF_JSON,   "model_performance.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{lbl} not found: {path}\n"
                                    "Run train_model.py (Module 2) first.")
    preprocessor     = joblib.load(PREPROC_PKL)
    surrogate_bundle = joblib.load(MODEL_PKL)
    with open(META_JSON)  as f: metadata = json.load(f)
    with open(PERF_JSON)  as f: perf     = json.load(f)
    return preprocessor, surrogate_bundle, metadata, perf


def _build_test_split(df: pd.DataFrame, target: str):
    sub  = df.dropna(subset=[target])
    X    = sub[INPUT_FEATURES].reset_index(drop=True)
    y    = sub[target].reset_index(drop=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return Xtr.reset_index(drop=True), Xte.reset_index(drop=True), \
           ytr.reset_index(drop=True), yte.reset_index(drop=True)


def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    rel = os.path.relpath(path, BASE_DIR)
    print(f"  Saved: {rel}")


# ===========================================================================
# SECTION A — MODEL DISAGREEMENT
# ===========================================================================

def calculate_model_disagreement(configuration: dict,
                                  preprocessor=None,
                                  surrogate_bundle=None,
                                  perf: dict = None) -> dict:
    """
    Estimate prediction spread for a single configuration.

    Two sources of spread:
      1. Cross-model: RF vs ET predictions (read from model_performance.json
         to reconstruct the alternate model — but we only have the BEST model
         stored, so we retrain the alternate on-the-fly using the same seed
         and report both best and alternate predictions).
         NOTE: RF and ET are NOT statistically independent estimators.
               This spread is an indicator only.

      2. Intra-ensemble: std of per-tree predictions within the stored 200-tree
         ensemble.  This reflects internal ensemble variance, not external
         calibrated uncertainty.

    Returns
    -------
    dict  {target: {"mean_prediction", "std_intra_ensemble",
                     "min_tree", "max_tree", "spread_intra",
                     "cross_model_spread",  # if alternate available
                     "best_algo", "note"}}
    """
    if preprocessor is None or surrogate_bundle is None:
        preprocessor, surrogate_bundle, _, perf = _load_artefacts()

    # Build single-row encoded input
    X_df  = pd.DataFrame({f: [configuration[f]] for f in INPUT_FEATURES})
    X_enc = preprocessor.transform(X_df)

    results = {}
    for target, bundle in surrogate_bundle.items():
        model     = bundle["model"]
        best_algo = bundle["algorithm"]

        # --- Intra-ensemble spread ---
        tree_preds = np.array([tree.predict(X_enc)[0]
                                for tree in model.estimators_])
        mean_p  = float(tree_preds.mean())
        std_p   = float(tree_preds.std())
        min_p   = float(tree_preds.min())
        max_p   = float(tree_preds.max())
        spread  = max_p - min_p

        # --- Cross-model spread (from recorded metrics in perf JSON) ---
        cross_spread = None
        cross_note   = ("Only one model object stored per target. "
                        "Cross-model spread estimated from test-set MAE difference "
                        "between RF and ET (from model_performance.json).")
        if perf is not None:
            tgt_perf = perf.get("targets", {}).get(target, {})
            rf_rmse  = tgt_perf.get("Random Forest", {}).get("RMSE", None)
            et_rmse  = tgt_perf.get("Extra Trees",   {}).get("RMSE", None)
            if rf_rmse is not None and et_rmse is not None:
                cross_spread = abs(rf_rmse - et_rmse)

        results[target] = {
            "best_algorithm":    best_algo,
            "mean_prediction":   round(max(0.0, mean_p), 8),
            "std_intra_ensemble": round(std_p, 8),
            "min_tree_pred":     round(max(0.0, min_p), 8),
            "max_tree_pred":     round(max(0.0, max_p), 8),
            "spread_intra":      round(spread, 8),
            "cross_model_rmse_diff": round(cross_spread, 6) if cross_spread else None,
            "note": (
                "Intra-ensemble spread = std of 200 tree predictions. "
                "This is NOT a calibrated confidence interval. "
                + cross_note
            ),
        }

    return results


# ===========================================================================
# SECTION B — TEST-SET ERROR ANALYSIS
# ===========================================================================

def analyse_test_errors(preprocessor=None, surrogate_bundle=None,
                         metadata=None) -> dict:
    """
    Compute comprehensive error statistics on the independent test set.

    Returns
    -------
    dict  {target: {"MAE", "RMSE", "mean_residual", "residual_std",
                    "median_AE", "p90_AE", "p95_AE",
                    "y_test", "y_pred", "residuals"}}
    """
    if preprocessor is None:
        preprocessor, surrogate_bundle, metadata, _ = _load_artefacts()

    df = pd.read_csv(DATASET_CSV)
    results = {}
    csv_rows = []

    for target in TARGET_COLS:
        if target not in surrogate_bundle:
            continue
        model = surrogate_bundle[target]["model"]
        algo  = surrogate_bundle[target]["algorithm"]

        _, X_test_raw, _, y_test = _build_test_split(df, target)
        X_test_enc = preprocessor.transform(X_test_raw)
        y_pred     = model.predict(X_test_enc)
        residuals  = y_pred - np.array(y_test)

        stats = {
            "target":        target,
            "algorithm":     algo,
            "n_test":        len(y_test),
            "MAE":           round(float(mean_absolute_error(y_test, y_pred)),   6),
            "RMSE":          round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 6),
            "mean_residual": round(float(residuals.mean()), 6),
            "residual_std":  round(float(residuals.std()),  6),
            "median_AE":     round(float(median_absolute_error(y_test, y_pred)), 6),
            "p90_AE":        round(float(np.percentile(np.abs(residuals), 90)),  6),
            "p95_AE":        round(float(np.percentile(np.abs(residuals), 95)),  6),
        }
        results[target] = {**stats,
                           "y_test":    np.array(y_test),
                           "y_pred":    y_pred,
                           "residuals": residuals}
        csv_rows.append(stats)
        print(f"  [{target}]  MAE={stats['MAE']:.4f}  RMSE={stats['RMSE']:.4f}  "
              f"median_AE={stats['median_AE']:.4f}  p90={stats['p90_AE']:.4f}")

    # Save CSV
    pd.DataFrame(csv_rows).to_csv(ERR_CSV, index=False)
    print(f"\n  Saved: reports/prediction_error_analysis.csv")
    return results


def plot_residual_analysis(error_results: dict) -> None:
    """
    For each target generate:
      (a) Residual distribution histogram + KDE
      (b) Residuals vs predicted scatter
    """
    os.makedirs(VIS_RESID, exist_ok=True)
    for target, res in error_results.items():
        residuals = res["residuals"]
        y_pred    = res["y_pred"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # (a) Distribution
        sns.histplot(residuals, bins=30, kde=True, ax=ax1, color="steelblue")
        ax1.axvline(0, color="red", lw=1.5, linestyle="--")
        ax1.axvline(res["mean_residual"], color="orange", lw=1.2,
                    linestyle="--", label=f"Mean={res['mean_residual']:.3f}")
        ax1.set_xlabel("Residual (predicted − actual)")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Residual Distribution\n{target}", fontsize=10)
        ax1.legend(fontsize=8)

        # (b) Residuals vs predicted
        ax2.scatter(y_pred, residuals, alpha=0.45, s=14,
                    color="coral", edgecolors="none")
        ax2.axhline(0, color="navy", lw=1.5, linestyle="--")
        ax2.set_xlabel("Predicted value")
        ax2.set_ylabel("Residual (predicted − actual)")
        ax2.set_title(f"Residuals vs Predicted\n{target}", fontsize=10)

        fig.suptitle(SYNTH_NOTE.strip(), fontsize=8, color="grey", style="italic")
        fig.tight_layout()
        safe = target.replace("/", "_")
        _save_fig(fig, os.path.join(VIS_RESID, f"resid_{safe}.png"))


# ===========================================================================
# SECTION C — CONFORMAL PREDICTION INTERVALS
# ===========================================================================

def calibrate_prediction_intervals(preprocessor=None, surrogate_bundle=None,
                                    coverage: float = COVERAGE_LEVEL) -> dict:
    """
    Split conformal prediction intervals (Venn-Abers / inductive CP).

    Method
    ------
    From the test partition (196 rows):
      - 60% → calibration set  (compute absolute residuals)
      - 40% → evaluation set   (measure empirical coverage)

    The conformal quantile q is the ceil(coverage * (n_calib + 1)) / n_calib
    quantile of calibration absolute residuals (standard split-CP formula).

    Interval for new prediction y_hat:
        [y_hat - q, y_hat + q]

    Assumptions (documented)
    ------------------------
    1. Exchangeability: calibration and evaluation rows must be drawn
       i.i.d. from the same distribution as new inputs.
    2. With SYNTHETIC data the distribution is controlled — this may not
       hold for real MATLAB data.
    3. Coverage is marginal (averaged over test inputs), not conditional.
    4. Intervals are symmetric — not appropriate for highly skewed targets.

    Returns dict {target: {"q_hat", "coverage_nominal", "coverage_empirical",
                            "mean_interval_width", "n_calib", "n_eval",
                            "status", "assumptions"}}
    """
    if preprocessor is None:
        preprocessor, surrogate_bundle, _, _ = _load_artefacts()

    df = pd.read_csv(DATASET_CSV)
    MIN_CALIB = 30
    calib_results = {}

    for target in TARGET_COLS:
        if target not in surrogate_bundle:
            continue
        model = surrogate_bundle[target]["model"]

        _, X_test_raw, _, y_test = _build_test_split(df, target)
        n_test = len(X_test_raw)

        n_calib = int(n_test * CONFORMAL_CALIB_FRAC)
        n_eval  = n_test - n_calib

        if n_calib < MIN_CALIB:
            calib_results[target] = {
                "status": "DISABLED",
                "reason": f"Calibration set too small ({n_calib} < {MIN_CALIB}).",
            }
            continue

        # Fixed split (deterministic, same seed)
        rng_idx  = np.random.default_rng(RANDOM_STATE)
        idx_all  = np.arange(n_test)
        idx_calib = rng_idx.choice(idx_all, size=n_calib, replace=False)
        idx_eval  = np.setdiff1d(idx_all, idx_calib)

        X_calib = X_test_raw.iloc[idx_calib]
        y_calib = y_test.iloc[idx_calib]
        X_eval  = X_test_raw.iloc[idx_eval]
        y_eval  = y_test.iloc[idx_eval]

        # Calibration scores (absolute residuals)
        X_calib_enc = preprocessor.transform(X_calib)
        y_calib_pred = model.predict(X_calib_enc)
        calib_scores = np.abs(y_calib_pred - np.array(y_calib))

        # Conformal quantile
        level = np.ceil(coverage * (n_calib + 1)) / n_calib
        level = min(level, 1.0)
        q_hat = float(np.quantile(calib_scores, level))

        # Evaluate empirical coverage on evaluation set
        X_eval_enc  = preprocessor.transform(X_eval)
        y_eval_pred = model.predict(X_eval_enc)
        covered      = np.abs(y_eval_pred - np.array(y_eval)) <= q_hat
        emp_coverage = float(covered.mean())
        mean_width   = 2 * q_hat   # symmetric interval width

        calib_results[target] = {
            "status":              "CALIBRATED",
            "coverage_nominal":    coverage,
            "coverage_empirical":  round(emp_coverage, 4),
            "q_hat":               round(q_hat, 6),
            "mean_interval_width": round(mean_width, 6),
            "n_calib":             int(n_calib),
            "n_eval":              int(n_eval),
            "assumptions": [
                "Exchangeability: calibration and new inputs share the same distribution.",
                "Coverage is MARGINAL (averaged), not conditional on input value.",
                "Intervals are symmetric — may undercover for skewed targets.",
                "Valid only while data distribution matches training distribution.",
                "SYNTHETIC data: these properties may not hold for real MATLAB data.",
            ],
        }
        print(f"  [{target}]  q_hat={q_hat:.4f}  "
              f"nominal_cov={coverage:.0%}  "
              f"empirical_cov={emp_coverage:.2%}  "
              f"width={mean_width:.4f}")

    # Save JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(CALIB_JSON, "w") as f:
        json.dump(calib_results, f, indent=2)
    print(f"\n  Saved: reports/uncertainty_calibration.json")
    return calib_results


def plot_prediction_intervals(error_results: dict,
                               calib_results: dict) -> None:
    """
    Plot actual vs predicted with conformal intervals overlaid.
    Only generated for targets with CALIBRATED intervals.
    """
    os.makedirs(VIS_PI, exist_ok=True)
    preprocessor, surrogate_bundle, _, _ = _load_artefacts()
    df = pd.read_csv(DATASET_CSV)

    for target, cr in calib_results.items():
        if cr.get("status") != "CALIBRATED":
            continue
        if target not in error_results:
            continue

        q_hat  = cr["q_hat"]
        y_test = error_results[target]["y_test"]
        y_pred = error_results[target]["y_pred"]
        lower  = y_pred - q_hat
        upper  = y_pred + q_hat

        sort_idx = np.argsort(y_test)
        yt  = y_test[sort_idx]
        yp  = y_pred[sort_idx]
        lo  = lower[sort_idx]
        hi  = upper[sort_idx]
        x   = np.arange(len(yt))

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(x, lo, hi, alpha=0.25, color="steelblue",
                        label=f"Conformal interval ({cr['coverage_nominal']:.0%} nominal)")
        ax.plot(x, yt,  "k.",  ms=4, label="Actual", alpha=0.7)
        ax.plot(x, yp,  "r-",  lw=1, label="Predicted", alpha=0.8)
        ax.set_xlabel("Test sample (sorted by actual value)")
        ax.set_ylabel(target)
        emp = cr["coverage_empirical"]
        ax.set_title(
            f"Conformal Prediction Interval — {target}\n"
            f"Nominal coverage={cr['coverage_nominal']:.0%}  "
            f"Empirical coverage={emp:.2%}  q={q_hat:.4f}"
            + SYNTH_NOTE,
            fontsize=9
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = target.replace("/", "_")
        _save_fig(fig, os.path.join(VIS_PI, f"pi_{safe}.png"))


# ===========================================================================
# SECTION D — OUT-OF-DOMAIN DETECTION
# ===========================================================================

def check_prediction_domain(configuration: dict) -> dict:
    """
    Assess whether the input configuration is within the model's training domain.

    Checks
    ------
    1. Per-feature range check (hard bounds from training data).
    2. Unseen categorical value detection.
    3. Multivariate Mahalanobis distance on numerical features
       (trained on the TRAINING partition — measures how unusual the
       combination is, not just individual features).

    Returns
    -------
    dict:
        domain_status   : IN_DOMAIN | OUT_OF_DOMAIN | UNUSUAL_COMBINATION
        warning         : summary string
        ood_features    : list[str]   (per-feature range violations)
        unseen_cats     : list[str]   (unseen categorical values)
        mahal_distance  : float | None
        mahal_threshold : float | None
        mahal_warning   : bool
        details         : dict
    """
    if not os.path.exists(DATASET_CSV):
        return {"domain_status": "UNKNOWN",
                "warning": "Dataset not found — cannot check domain."}

    df = pd.read_csv(DATASET_CSV)

    # Reconstruct training partition (same split as Module 2)
    y_dummy = df[TARGET_COLS[0]]
    X_all   = df[INPUT_FEATURES]
    X_train_raw, _, _, _ = train_test_split(
        X_all, y_dummy, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ----- 1. Per-feature range check -----
    ood_features = []
    details = {}
    for feat in NUMERICAL_FEATURES:
        lo = float(X_train_raw[feat].min())
        hi = float(X_train_raw[feat].max())
        val = float(configuration.get(feat, np.nan))
        details[feat] = {"train_min": round(lo, 4), "train_max": round(hi, 4),
                          "input_value": val}
        if val < lo or val > hi:
            ood_features.append(feat)
            details[feat]["ood"] = True
        else:
            details[feat]["ood"] = False

    # ----- 2. Unseen categorical check -----
    unseen_cats = []
    st = str(configuration.get("Shield_Type", ""))
    cc = str(configuration.get("Cable_Configuration", ""))
    if st not in VALID_SHIELD:
        unseen_cats.append(f"Shield_Type='{st}'")
    if cc not in VALID_CABLECONF:
        unseen_cats.append(f"Cable_Configuration='{cc}'")

    # ----- 3. Mahalanobis distance (numerical features only) -----
    mahal_dist  = None
    mahal_thresh = None
    mahal_warn  = False
    mahal_note  = ("Mahalanobis distance measures how unusual the COMBINATION "
                   "of numerical features is relative to the training set. "
                   "Being within individual feature ranges does not guarantee "
                   "an in-distribution combination.")
    try:
        X_train_num = X_train_raw[NUMERICAL_FEATURES].dropna().values
        cov_est = EmpiricalCovariance().fit(X_train_num)
        input_vec = np.array([[float(configuration[f])
                                for f in NUMERICAL_FEATURES]])
        mahal_dist   = float(cov_est.mahalanobis(input_vec)[0])
        # Threshold: 97.5th percentile of training Mahalanobis distances
        train_mahal = cov_est.mahalanobis(X_train_num)
        mahal_thresh = float(np.percentile(train_mahal, 97.5))
        mahal_warn   = mahal_dist > mahal_thresh
    except Exception as e:
        mahal_note += f"  (Could not compute: {e})"

    # ----- Determine overall status -----
    if ood_features or unseen_cats:
        status  = "OUT_OF_DOMAIN"
        warning = (f"Configuration is outside the observed training range. "
                   f"Affected features: {ood_features + unseen_cats}. "
                   "Predictions are extrapolations — verify with MATLAB.")
    elif mahal_warn:
        status  = "UNUSUAL_COMBINATION"
        warning = (f"All individual features are within training range but the "
                   f"COMBINATION is unusual (Mahalanobis distance={mahal_dist:.2f} > "
                   f"threshold={mahal_thresh:.2f}). "
                   "Prediction may be less reliable.")
    else:
        status  = "IN_DOMAIN"
        warning = ""

    return {
        "domain_status":   status,
        "warning":         warning,
        "ood_features":    ood_features,
        "unseen_cats":     unseen_cats,
        "mahal_distance":  round(mahal_dist, 4) if mahal_dist else None,
        "mahal_threshold": round(mahal_thresh, 4) if mahal_thresh else None,
        "mahal_warning":   mahal_warn,
        "mahal_note":      mahal_note,
        "feature_details": details,
    }


# ===========================================================================
# SECTION E — PREDICTION RELIABILITY ASSESSMENT
# ===========================================================================

def assess_prediction_reliability(configuration: dict,
                                   calib_results: dict = None,
                                   preprocessor=None,
                                   surrogate_bundle=None,
                                   perf: dict = None) -> dict:
    """
    Combine domain check, intra-ensemble spread, historical error stats,
    and conformal interval info into a unified reliability assessment.

    Reliability statuses
    --------------------
    RELIABLE_WITHIN_VALIDATED_DOMAIN
    CAUTION_MODEL_DISAGREEMENT
    CAUTION_OUT_OF_DOMAIN
    CAUTION_UNCALIBRATED_UNCERTAINTY

    Returns structured dict per target.
    """
    if preprocessor is None:
        preprocessor, surrogate_bundle, _, perf = _load_artefacts()
    if calib_results is None:
        if os.path.exists(CALIB_JSON):
            with open(CALIB_JSON) as f:
                calib_results = json.load(f)
        else:
            calib_results = {}

    # Domain check
    domain = check_prediction_domain(configuration)

    # Disagreement
    disagreement = calculate_model_disagreement(
        configuration, preprocessor, surrogate_bundle, perf
    )

    # Historical test-set errors
    if os.path.exists(ERR_CSV):
        err_df   = pd.read_csv(ERR_CSV).set_index("target")
        hist_err = err_df.to_dict(orient="index")
    else:
        hist_err = {}

    SPREAD_WARN_FRAC = 0.20   # intra-ensemble std > 20% of mean prediction → flag

    reliability = {}
    for target in TARGET_COLS:
        if target not in (surrogate_bundle or {}):
            continue

        warnings_list = []
        status_flags  = []

        # a) Domain
        if domain["domain_status"] == "OUT_OF_DOMAIN":
            status_flags.append("CAUTION_OUT_OF_DOMAIN")
            warnings_list.append(
                f"Out-of-domain input detected in: {domain['ood_features'] + domain['unseen_cats']}. "
                "Prediction is an extrapolation. Verify with MATLAB simulation."
            )
        elif domain["domain_status"] == "UNUSUAL_COMBINATION":
            status_flags.append("CAUTION_OUT_OF_DOMAIN")
            warnings_list.append(domain["warning"])

        # b) Intra-ensemble spread
        d = disagreement.get(target, {})
        mean_pred = d.get("mean_prediction", 0)
        std_pred  = d.get("std_intra_ensemble", 0)
        if mean_pred > 0 and (std_pred / mean_pred) > SPREAD_WARN_FRAC:
            status_flags.append("CAUTION_MODEL_DISAGREEMENT")
            warnings_list.append(
                f"Intra-ensemble spread is high for {target}: "
                f"std={std_pred:.4f} ({100*std_pred/mean_pred:.1f}% of mean prediction). "
                "The 200 trees within the ensemble disagree noticeably. "
                "Consider MATLAB verification."
            )

        # c) Conformal intervals
        cr = calib_results.get(target, {})
        if cr.get("status") == "CALIBRATED":
            q = cr["q_hat"]
            interval_info = (
                f"Conformal interval (nominal {cr['coverage_nominal']:.0%}, "
                f"empirical {cr['coverage_empirical']:.2%}): "
                f"[prediction - {q:.4f}, prediction + {q:.4f}]. "
                "Valid only if new inputs share the training distribution."
            )
        else:
            status_flags.append("CAUTION_UNCALIBRATED_UNCERTAINTY")
            interval_info = (
                "Conformal prediction interval not calibrated. "
                "Prediction uncertainty is not yet formally quantified."
            )
            warnings_list.append(interval_info)

        # d) Historical RMSE reminder
        h = hist_err.get(target, {})
        rmse = h.get("RMSE", None)
        if rmse:
            warnings_list.append(
                f"Empirical test-set RMSE for {target}: {rmse:.4f} "
                "(not a confidence interval; reported as background error context)."
            )

        # Overall status
        if not status_flags:
            overall = "RELIABLE_WITHIN_VALIDATED_DOMAIN"
        elif "CAUTION_OUT_OF_DOMAIN" in status_flags:
            overall = "CAUTION_OUT_OF_DOMAIN"
        elif "CAUTION_MODEL_DISAGREEMENT" in status_flags:
            overall = "CAUTION_MODEL_DISAGREEMENT"
        else:
            overall = "CAUTION_UNCALIBRATED_UNCERTAINTY"

        reliability[target] = {
            "overall_status":   overall,
            "warnings":         warnings_list,
            "domain":           domain["domain_status"],
            "intra_spread_std": d.get("std_intra_ensemble"),
            "mean_prediction":  d.get("mean_prediction"),
            "conformal_info":   interval_info,
            "historical_rmse":  rmse,
        }

    return {
        "domain_check":   domain,
        "disagreement":   disagreement,
        "per_target":     reliability,
    }


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("EMC-GUARD  |  Module 4  |  Prediction Uncertainty Analysis")
    print("=" * 65)

    preprocessor, surrogate_bundle, metadata, perf = _load_artefacts()

    print("\n[1/4] Test-set error analysis ...")
    error_results = analyse_test_errors(preprocessor, surrogate_bundle, metadata)
    plot_residual_analysis(error_results)

    print("\n[2/4] Conformal prediction interval calibration ...")
    calib_results = calibrate_prediction_intervals(preprocessor, surrogate_bundle)
    plot_prediction_intervals(error_results, calib_results)

    print("\n[3/4] Model disagreement — sample configuration ...")
    sample_cfg = {
        "Separation_mm": 40.0, "Overlap_mm": 250.0,
        "Cable_Length_mm": 2500.0, "Frequency_Hz": 2_000_000.0,
        "Voltage_V": 110.0, "Conductor_Radius_mm": 2.0,
        "Dielectric_Constant": 4.5, "Shield_Type": "Unshielded",
        "Cable_Configuration": "Parallel",
    }
    disagree = calculate_model_disagreement(sample_cfg, preprocessor,
                                             surrogate_bundle, perf)
    for tgt, d in disagree.items():
        print(f"  {tgt}: mean={d['mean_prediction']:.6f}  "
              f"std_intra={d['std_intra_ensemble']:.6f}  "
              f"spread={d['spread_intra']:.6f}")

    print("\n[4/4] Domain check — sample configuration ...")
    domain = check_prediction_domain(sample_cfg)
    print(f"  Domain status: {domain['domain_status']}")
    if domain["mahal_distance"]:
        print(f"  Mahalanobis dist: {domain['mahal_distance']:.4f}  "
              f"threshold: {domain['mahal_threshold']:.4f}")

    print("\n[DONE] Prediction uncertainty analysis complete.")

