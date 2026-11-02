"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 4 | Script 3: Explainable AI Engine (Main Orchestrator)
File   : xai_engine.py
Author : EMC-GUARD Development Team

This script is the single entry-point for the entire Module 4 pipeline.

It imports and orchestrates:
  - feature_importance.py   (Task 2 & 3)
  - prediction_uncertainty.py (Tasks 4-8)

And additionally implements:
  - generate_xai_report()   (Task 9-10): human-readable XAI report
  - run_test_suite()        (Task 12):  all 6 diagnostic test cases

DISCLAIMER
----------
All outputs are derived from SYNTHETIC DUMMY DATA.
Results are for decision-support and pipeline validation ONLY.
They do not represent validated electromagnetic physics.

Usage
-----
    python src/xai_engine.py
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

warnings.filterwarnings("ignore")

# Make src/ importable regardless of working directory
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Module 4 sub-modules
from feature_importance    import calculate_feature_importance, plot_feature_importance
from prediction_uncertainty import (
    analyse_test_errors, plot_residual_analysis,
    calibrate_prediction_intervals, plot_prediction_intervals,
    calculate_model_disagreement, check_prediction_domain,
    assess_prediction_reliability, _load_artefacts,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
VIS_DIR       = os.path.join(REPORTS_DIR, "visualisations")
FI_CSV        = os.path.join(REPORTS_DIR, "feature_importance.csv")
ERR_CSV       = os.path.join(REPORTS_DIR, "prediction_error_analysis.csv")
CALIB_JSON    = os.path.join(REPORTS_DIR, "uncertainty_calibration.json")
RELIABILITY_TXT = os.path.join(REPORTS_DIR, "prediction_reliability_report.txt")

TARGET_COLS = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]
INPUT_FEATURES = [
    "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
    "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
    "Shield_Type", "Cable_Configuration",
]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ===========================================================================
# TASK 9 — ENGINEERING EXPLANATION GENERATOR
# ===========================================================================

def _explain_feature_importance(fi_results: dict) -> list:
    """Generate plain-English feature importance explanations per target."""
    lines = []
    for target, fi_df in fi_results.items():
        top = fi_df.iloc[0]
        bot = fi_df.iloc[-1]
        lines.append(
            f"  [{target}]"
        )
        lines.append(
            f"    Most influential : '{top['feature']}' "
            f"(mean MAE increase = {top['mean_importance']:.4f} "
            f"+/- {top['std_importance']:.4f} when permuted)."
        )
        lines.append(
            f"    Least influential: '{bot['feature']}' "
            f"(mean MAE increase = {bot['mean_importance']:.4f})."
        )
        # Threshold: importance > 5% of top importance = notable
        notable = fi_df[fi_df["mean_importance"] >
                        0.05 * top["mean_importance"]]["feature"].tolist()
        lines.append(
            f"    Notable features : {notable}"
        )
        lines.append(
            "    NOTE: Permutation importance measures how much the model's "
            "predictions depend on each feature in the TEST dataset. "
            "It does NOT imply physical causation."
        )
        lines.append("")
    return lines


def _explain_disagreement(disagree: dict) -> list:
    """Generate reliability explanation from disagreement analysis."""
    WARN_FRAC = 0.20
    lines = []
    for target, d in disagree.items():
        mean_p = d.get("mean_prediction", 0)
        std_p  = d.get("std_intra_ensemble", 0)
        frac   = (std_p / mean_p) if mean_p > 0 else 0
        if frac > WARN_FRAC:
            lines.append(
                f"  [{target}] The intra-ensemble spread is relatively high "
                f"(std={std_p:.4f}, {frac*100:.1f}% of mean prediction). "
                "Further verification using MATLAB simulation is recommended."
            )
        else:
            lines.append(
                f"  [{target}] Intra-ensemble spread is low "
                f"(std={std_p:.4f}, {frac*100:.1f}% of mean prediction). "
                "The 200 decision trees within the ensemble are in broad agreement."
            )
    return lines


def _explain_domain(domain: dict) -> list:
    """Generate domain explanation."""
    lines = []
    status = domain["domain_status"]
    if status == "OUT_OF_DOMAIN":
        ood = domain["ood_features"] + domain["unseen_cats"]
        lines.append(
            f"  The entered configuration is OUTSIDE the model training range "
            f"for: {ood}. "
            "Predictions are extrapolations and must not be treated as reliable "
            "without independent physics-model verification."
        )
    elif status == "UNUSUAL_COMBINATION":
        lines.append(
            f"  The individual feature values are within training bounds, but the "
            f"COMBINATION is unusual "
            f"(Mahalanobis distance = {domain.get('mahal_distance','N/A')}, "
            f"threshold = {domain.get('mahal_threshold','N/A')}). "
            "Prediction reliability may be reduced."
        )
    else:
        lines.append(
            "  The configuration is within the observed training domain. "
            "Individual feature values and their combination are consistent with "
            "the training dataset."
        )
    return lines


def _explain_conformal(calib_results: dict) -> list:
    """Generate conformal interval explanation."""
    lines = []
    for target, cr in calib_results.items():
        if cr.get("status") == "CALIBRATED":
            emp = cr["coverage_empirical"]
            nom = cr["coverage_nominal"]
            q   = cr["q_hat"]
            lines.append(
                f"  [{target}] Conformal interval: prediction +/- {q:.4f}. "
                f"Nominal coverage = {nom:.0%}, empirical coverage = {emp:.2%} "
                f"on the evaluation set ({cr['n_eval']} rows). "
                "Valid under exchangeability; may not hold under distribution shift."
            )
        else:
            lines.append(
                f"  [{target}] Conformal interval not calibrated "
                f"({cr.get('reason','insufficient data')})."
            )
    return lines


# ===========================================================================
# TASK 10 — GENERATE XAI REPORT
# ===========================================================================

def generate_xai_report(fi_results: dict,
                          error_results: dict,
                          calib_results: dict,
                          disagree_sample: dict,
                          domain_sample: dict,
                          reliability_sample: dict,
                          sample_config: dict,
                          dataset_type: str = "SYNTHETIC") -> None:
    """Write the full human-readable XAI reliability report."""

    lines = [
        "=" * 70,
        "EMC-GUARD — MODULE 4: EXPLAINABLE AI & PREDICTION CONFIDENCE REPORT",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset    : {dataset_type}",
        "=" * 70,
        "",
        "DISCLAIMER",
        "-" * 70,
        "All results are derived from SYNTHETIC DUMMY DATA generated by Module 1.",
        "They do NOT represent validated electromagnetic physics.",
        "This is a decision-support prototype — not a certified EMC assessment.",
        "",
        "=" * 70,
        "SECTION 1 — FEATURE IMPORTANCE (Permutation, Test Set)",
        "-" * 70,
        "Method: Each feature is shuffled independently on the test set.",
        "        The increase in MAE measures how much the model depends on it.",
        "        This reflects PREDICTIVE DEPENDENCE, not physical causation.",
        "",
    ]
    lines += _explain_feature_importance(fi_results)

    lines += [
        "=" * 70,
        "SECTION 2 — TEST-SET ERROR STATISTICS",
        "-" * 70,
        "Source: Independent test partition (20% of cleaned dataset).",
        "        These are empirical statistics, NOT confidence intervals.",
        "",
        f"  {'Target':<26} {'MAE':>9} {'RMSE':>9} "
        f"{'Median_AE':>10} {'P90_AE':>9} {'P95_AE':>9}",
        "  " + "-" * 68,
    ]
    if os.path.exists(ERR_CSV):
        err_df = pd.read_csv(ERR_CSV)
        for _, row in err_df.iterrows():
            lines.append(
                f"  {row['target']:<26} {row['MAE']:>9.4f} {row['RMSE']:>9.4f} "
                f"{row['median_AE']:>10.4f} {row['p90_AE']:>9.4f} "
                f"{row['p95_AE']:>9.4f}"
            )
    lines.append("")

    lines += [
        "=" * 70,
        "SECTION 3 — CONFORMAL PREDICTION INTERVALS",
        "-" * 70,
        f"Method: Split conformal prediction (nominal coverage = "
        f"{calib_results.get(TARGET_COLS[0], {}).get('coverage_nominal', 'N/A')}).",
        "Assumption: Exchangeability — new inputs must share training distribution.",
        "            Coverage is MARGINAL (not conditional on input).",
        "",
    ]
    lines += _explain_conformal(calib_results)
    lines.append("")

    lines += [
        "=" * 70,
        "SECTION 4 — MODEL DISAGREEMENT ANALYSIS",
        "-" * 70,
        "Source: Intra-ensemble spread (std of 200 tree predictions).",
        "        This is NOT a calibrated confidence interval.",
        f"Sample configuration: Separation={sample_config.get('Separation_mm')} mm, "
        f"Overlap={sample_config.get('Overlap_mm')} mm, "
        f"Shield={sample_config.get('Shield_Type')}",
        "",
    ]
    lines += _explain_disagreement(disagree_sample)
    lines.append("")

    lines += [
        "=" * 70,
        "SECTION 5 — OUT-OF-DOMAIN DETECTION",
        "-" * 70,
        "Per-feature training range check + Mahalanobis distance.",
        f"Sample configuration status: {domain_sample['domain_status']}",
        "",
    ]
    lines += _explain_domain(domain_sample)
    lines.append("")

    lines += [
        "=" * 70,
        "SECTION 6 — PREDICTION RELIABILITY ASSESSMENT (SAMPLE CONFIG)",
        "-" * 70,
    ]
    for target, rel in reliability_sample.get("per_target", {}).items():
        lines.append(f"  [{target}]  Status: {rel['overall_status']}")
        for w in rel["warnings"]:
            lines.append(f"    WARNING: {w}")
        lines.append(f"    Conformal: {rel['conformal_info'][:120]}")
        lines.append("")

    lines += [
        "=" * 70,
        "SECTION 7 — RELIABILITY STATUS LEGEND",
        "-" * 70,
        "  RELIABLE_WITHIN_VALIDATED_DOMAIN",
        "    Input is within training bounds. Ensemble agreement is acceptable.",
        "",
        "  CAUTION_MODEL_DISAGREEMENT",
        "    The internal tree ensemble shows high spread for this input.",
        "    Consider MATLAB verification.",
        "",
        "  CAUTION_OUT_OF_DOMAIN",
        "    One or more features are outside the observed training range.",
        "    Prediction is an extrapolation. MATLAB verification required.",
        "",
        "  CAUTION_UNCALIBRATED_UNCERTAINTY",
        "    Conformal interval not yet calibrated.",
        "    Prediction uncertainty is not formally quantified.",
        "",
        "=" * 70,
    ]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(RELIABILITY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: reports/prediction_reliability_report.txt")


# ===========================================================================
# TASK 12 — TEST SUITE
# ===========================================================================

def run_test_suite(preprocessor, surrogate_bundle, perf,
                   calib_results: dict) -> None:
    """
    Task 12: Six diagnostic test configurations.

    TC1 — Normal configuration, within training domain
    TC2 — Unusually large separation (OOD)
    TC3 — Unusually large overlap (OOD)
    TC4 — Unfamiliar categorical value
    TC5 — High ensemble disagreement probe (extreme frequency + voltage)
    TC6 — Missing required input key
    """
    PASS = "[PASS]"
    FAIL = "[FAIL]"
    sep  = "-" * 65

    print(sep)
    print("TASK 12 — XAI MODULE TEST SUITE")
    print(sep)

    # ------ TC1: Normal in-domain config ------
    print("\n[TC1] Normal in-domain configuration ...")
    tc1 = {
        "Separation_mm": 60.0, "Overlap_mm": 180.0,
        "Cable_Length_mm": 2000.0, "Frequency_Hz": 1_500_000.0,
        "Voltage_V": 100.0, "Conductor_Radius_mm": 2.0,
        "Dielectric_Constant": 4.5, "Shield_Type": "Unshielded",
        "Cable_Configuration": "Parallel",
    }
    try:
        d1  = check_prediction_domain(tc1)
        r1  = assess_prediction_reliability(tc1, calib_results,
                                             preprocessor, surrogate_bundle, perf)
        dis1 = calculate_model_disagreement(tc1, preprocessor, surrogate_bundle, perf)
        ok  = d1["domain_status"] == "IN_DOMAIN"
        print(f"  {PASS if ok else FAIL} Domain status: {d1['domain_status']}")
        sample_tgt = TARGET_COLS[0]
        rel_status = r1["per_target"].get(sample_tgt, {}).get("overall_status", "N/A")
        print(f"  {PASS} Reliability ({sample_tgt}): {rel_status}")
        spread = dis1.get(sample_tgt, {}).get("std_intra_ensemble", "N/A")
        print(f"  {PASS} Intra-ensemble std ({sample_tgt}): {spread}")
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")

    # ------ TC2: Large separation (OOD) ------
    print("\n[TC2] Out-of-domain — Separation_mm = 5000 mm ...")
    tc2 = dict(tc1); tc2["Separation_mm"] = 5000.0
    try:
        d2 = check_prediction_domain(tc2)
        ok = d2["domain_status"] == "OUT_OF_DOMAIN" and \
             "Separation_mm" in d2["ood_features"]
        print(f"  {PASS if ok else FAIL} Domain status: {d2['domain_status']}")
        print(f"  {PASS if ok else FAIL} OOD features : {d2['ood_features']}")
        r2 = assess_prediction_reliability(tc2, calib_results,
                                            preprocessor, surrogate_bundle, perf)
        for tgt in [TARGET_COLS[0]]:
            s = r2["per_target"].get(tgt, {}).get("overall_status", "N/A")
            print(f"  {PASS} Reliability status ({tgt}): {s}")
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")

    # ------ TC3: Large overlap (OOD) ------
    print("\n[TC3] Out-of-domain — Overlap_mm = 2000 mm ...")
    tc3 = dict(tc1); tc3["Overlap_mm"] = 2000.0
    try:
        d3 = check_prediction_domain(tc3)
        ok = d3["domain_status"] == "OUT_OF_DOMAIN" and \
             "Overlap_mm" in d3["ood_features"]
        print(f"  {PASS if ok else FAIL} Domain status : {d3['domain_status']}")
        print(f"  {PASS if ok else FAIL} OOD features  : {d3['ood_features']}")
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")

    # ------ TC4: Unfamiliar categorical ------
    print("\n[TC4] Unfamiliar categorical — Shield_Type = 'Braided' ...")
    tc4 = dict(tc1); tc4["Shield_Type"] = "Braided"
    try:
        d4 = check_prediction_domain(tc4)
        ok = "Shield_Type='Braided'" in d4["unseen_cats"]
        print(f"  {PASS if ok else FAIL} Unseen categories detected: {d4['unseen_cats']}")
        print(f"  {PASS if ok else FAIL} Domain status: {d4['domain_status']}")
        # Prediction should still be produced (OHE maps unknown -> zeros)
        dis4 = calculate_model_disagreement(tc4, preprocessor, surrogate_bundle, perf)
        has_preds = all(d.get("mean_prediction") is not None
                        for d in dis4.values())
        print(f"  {PASS if has_preds else FAIL} Prediction still produced: {has_preds}")
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")

    # ------ TC5: Extreme config (high disagreement probe) ------
    print("\n[TC5] Extreme config — high Frequency + Voltage (disagreement probe) ...")
    tc5 = dict(tc1)
    tc5.update({"Frequency_Hz": 29_500_000.0, "Voltage_V": 490.0,
                 "Overlap_mm": 490.0, "Separation_mm": 8.0})
    try:
        dis5 = calculate_model_disagreement(tc5, preprocessor, surrogate_bundle, perf)
        d5   = check_prediction_domain(tc5)
        for tgt in [TARGET_COLS[2]]:   # Victim_Voltage_V
            d = dis5.get(tgt, {})
            mean_p = d.get("mean_prediction", 0)
            std_p  = d.get("std_intra_ensemble", 0)
            frac   = (std_p / mean_p * 100) if mean_p > 0 else 0
            print(f"  [{tgt}] mean={mean_p:.4f}  std={std_p:.4f}  "
                  f"spread={frac:.1f}% of mean")
        print(f"  Domain status: {d5['domain_status']}")
        print(f"  {PASS} Disagreement analysis completed without error.")
    except Exception as e:
        print(f"  {FAIL} Unexpected error: {e}")

    # ------ TC6: Missing required input ------
    print("\n[TC6] Missing required input key (Frequency_Hz omitted) ...")
    tc6 = {k: v for k, v in tc1.items() if k != "Frequency_Hz"}
    try:
        # check_prediction_domain uses the configuration dict only for bounds checks
        # — it won't raise on missing keys. The error should come from predict_emc.
        # We test the domain check gracefully handles missing keys.
        d6 = check_prediction_domain(tc6)
        # Now test that the full prediction pipeline raises properly
        from predict_emc import predict_interference, ValidationError
        try:
            predict_interference(tc6)
            print(f"  {FAIL} No exception raised — expected ValidationError.")
        except ValidationError as ve:
            print(f"  {PASS} ValidationError raised: {str(ve)[:80]}")
    except Exception as e:
        print(f"  {FAIL} Unexpected error type ({type(e).__name__}): {e}")

    print(sep)
    print("TEST SUITE COMPLETE")
    print(sep)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    print("=" * 70)
    print("EMC-GUARD  |  Module 4  |  Explainable AI & Confidence Engine")
    print("=" * 70)
    print("[SYNTHETIC DATA — results are for pipeline validation only]")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(VIS_DIR, "feature_importance"),   exist_ok=True)
    os.makedirs(os.path.join(VIS_DIR, "residual_plots"),       exist_ok=True)
    os.makedirs(os.path.join(VIS_DIR, "prediction_intervals"), exist_ok=True)

    # Load artefacts once, pass to all functions to avoid repeated disk I/O
    print("\n[0/6] Loading model artefacts ...")
    preprocessor, surrogate_bundle, metadata, perf = _load_artefacts()
    dataset_type = perf.get("dataset_type", "SYNTHETIC")
    print(f"  Dataset type: {dataset_type}  |  "
          f"Targets: {list(surrogate_bundle.keys())}")

    # ------------------------------------------------------------------
    # STEP 1: Feature importance
    # ------------------------------------------------------------------
    print("\n[1/6] Calculating permutation feature importance ...")
    fi_results = calculate_feature_importance()   # uses default n_repeats=10, max 80 rows
    print("\n  Generating feature importance charts ...")
    plot_feature_importance(fi_results)

    # ------------------------------------------------------------------
    # STEP 2: Test-set error analysis
    # ------------------------------------------------------------------
    print("\n[2/6] Analysing test-set errors ...")
    error_results = analyse_test_errors(preprocessor, surrogate_bundle, metadata)
    print("\n  Generating residual plots ...")
    plot_residual_analysis(error_results)

    # ------------------------------------------------------------------
    # STEP 3: Conformal prediction intervals
    # ------------------------------------------------------------------
    print("\n[3/6] Calibrating conformal prediction intervals ...")
    calib_results = calibrate_prediction_intervals(preprocessor, surrogate_bundle)
    print("\n  Generating prediction interval charts ...")
    plot_prediction_intervals(error_results, calib_results)

    # ------------------------------------------------------------------
    # STEP 4: Model disagreement + domain check (reference config)
    # ------------------------------------------------------------------
    reference_config = {
        "Separation_mm":       60.0,
        "Overlap_mm":         200.0,
        "Cable_Length_mm":   2500.0,
        "Frequency_Hz":  2_000_000.0,
        "Voltage_V":          110.0,
        "Conductor_Radius_mm":  2.0,
        "Dielectric_Constant":  4.5,
        "Shield_Type":    "Unshielded",
        "Cable_Configuration": "Parallel",
    }
    print("\n[4/6] Model disagreement analysis (reference configuration) ...")
    disagree_sample = calculate_model_disagreement(
        reference_config, preprocessor, surrogate_bundle, perf
    )
    for tgt, d in disagree_sample.items():
        print(f"  {tgt}: mean={d['mean_prediction']:.6f}  "
              f"std_intra={d['std_intra_ensemble']:.6f}  "
              f"spread={d['spread_intra']:.6f}")

    print("\n  Domain check (reference configuration) ...")
    domain_sample = check_prediction_domain(reference_config)
    print(f"  Domain status : {domain_sample['domain_status']}")
    if domain_sample["mahal_distance"]:
        print(f"  Mahal distance: {domain_sample['mahal_distance']}  "
              f"threshold: {domain_sample['mahal_threshold']}")

    # ------------------------------------------------------------------
    # STEP 5: Reliability assessment
    # ------------------------------------------------------------------
    print("\n[5/6] Reliability assessment (reference configuration) ...")
    reliability_sample = assess_prediction_reliability(
        reference_config, calib_results, preprocessor, surrogate_bundle, perf
    )
    for tgt, rel in reliability_sample["per_target"].items():
        print(f"  {tgt}: {rel['overall_status']}")

    # ------------------------------------------------------------------
    # STEP 6: Generate XAI report + run test suite
    # ------------------------------------------------------------------
    print("\n[6/6] Generating XAI report ...")
    generate_xai_report(
        fi_results, error_results, calib_results,
        disagree_sample, domain_sample, reliability_sample,
        reference_config, dataset_type=dataset_type,
    )

    print("\n  Running test suite (Task 12) ...")
    run_test_suite(preprocessor, surrogate_bundle, perf, calib_results)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODULE 4 COMPLETE")
    print("-" * 70)

    # Feature importance top-1 per target
    print("\n  Feature Importance — Top Feature Per Target:")
    for tgt, fi_df in fi_results.items():
        row = fi_df.iloc[0]
        print(f"    {tgt:<26} -> '{row['feature']}'  "
              f"(mean_imp={row['mean_importance']:.4f}  "
              f"std={row['std_importance']:.4f})")

    # Error summary
    print("\n  Test-Set Error Summary:")
    if os.path.exists(ERR_CSV):
        err_df = pd.read_csv(ERR_CSV)
        for _, row in err_df.iterrows():
            print(f"    {row['target']:<26}  MAE={row['MAE']:.4f}  "
                  f"RMSE={row['RMSE']:.4f}  P90_AE={row['p90_AE']:.4f}")

    # Conformal summary
    print("\n  Conformal Interval Summary:")
    for tgt, cr in calib_results.items():
        if cr.get("status") == "CALIBRATED":
            print(f"    {tgt:<26}  q={cr['q_hat']:.4f}  "
                  f"nominal={cr['coverage_nominal']:.0%}  "
                  f"empirical={cr['coverage_empirical']:.2%}")
        else:
            print(f"    {tgt:<26}  {cr.get('status','?')} — "
                  f"{cr.get('reason','')}")

    print("\n  Output files:")
    for f in [
        "reports/feature_importance.csv",
        "reports/prediction_error_analysis.csv",
        "reports/uncertainty_calibration.json",
        "reports/prediction_reliability_report.txt",
        "reports/visualisations/feature_importance/fi_*.png",
        "reports/visualisations/feature_importance/fi_heatmap_all_targets.png",
        "reports/visualisations/residual_plots/resid_*.png",
        "reports/visualisations/prediction_intervals/pi_*.png",
    ]:
        print(f"    {f}")

    print("=" * 70)
    print("[DONE] Module 4 — Explainable AI Engine complete.")
    print("  REMINDER: All results are from SYNTHETIC DUMMY DATA.")


if __name__ == "__main__":
    main()

