"""
test_suite.py
-------------
Automated test suite for EMC-PREDICT + EMC-GUARD.
Covers all 12 test categories from the master development prompt.

Run: python tests/test_suite.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def run_test(name, fn):
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  {PASS}  {name}")
    except Exception as exc:
        results.append((name, FAIL, str(exc)))
        print(f"  {FAIL}  {name}\n         Error: {exc}")


print("\n" + "="*65)
print("  EMC-PREDICT + EMC-GUARD  –  Automated Test Suite")
print("="*65 + "\n")


# ──────────────────────────────────────────────────────────────────────────
# TEST 1: Load all frozen MATLAB models
# ──────────────────────────────────────────────────────────────────────────
def t1_load_frozen_matlab():
    from models.model_loader import load_frozen_matlab_model
    m = load_frozen_matlab_model()
    assert m["betaCap"].shape == (94,), f"betaCap shape {m['betaCap'].shape}"
    assert m["betaMut"].shape == (84,), f"betaMut shape {m['betaMut'].shape}"
    assert m["XCap"].shape == (94, 216), f"XCap shape {m['XCap'].shape}"
    assert m["XMut"].shape == (84, 216), f"XMut shape {m['XMut'].shape}"

def t1_load_extratrees():
    from models.model_loader import load_extratrees_models
    m = load_extratrees_models()
    assert m["cap_model"] is not None
    assert m["mut_model"] is not None

def t1_load_dataset():
    from models.model_loader import load_physics_dataset
    df = load_physics_dataset()
    assert len(df) == 216, f"Expected 216 rows, got {len(df)}"

run_test("TEST 1a – Frozen MATLAB model loads correctly", t1_load_frozen_matlab)
run_test("TEST 1b – ExtraTrees surrogates load correctly", t1_load_extratrees)
run_test("TEST 1c – Physics dataset loads (216 rows)", t1_load_dataset)


# ──────────────────────────────────────────────────────────────────────────
# TEST 2: Predict a known configuration and compare Python vs MATLAB reference
# ──────────────────────────────────────────────────────────────────────────
def t2_known_config_frozen():
    from models.prediction_engine import predict
    from models.model_loader import load_physics_dataset
    df = load_physics_dataset()
    row = df[df["Config_ID"] == "EMC-001"].iloc[0]
    result = predict(8.0, 2.0, 150.0, 10.0, engine="frozen_matlab")
    assert result["status"] == "OK", result
    cap = result["predictions"]["TotalCapacitance_pF_per_m"]["value"]
    mut = result["predictions"]["AbsMutualInductance_nH"]["value"]
    ref_c = row["TotalCapacitance_pF_per_m"]
    ref_m = row["AbsMutualInductance_nH"]
    assert abs(cap - ref_c) < 1e-8, f"Cap error: {abs(cap-ref_c):.3e}"
    assert abs(mut - ref_m) < 1e-8, f"Mut error: {abs(mut-ref_m):.3e}"

def t2_known_config_extratrees():
    from models.prediction_engine import predict
    from models.model_loader import load_physics_dataset
    df = load_physics_dataset()
    row = df[df["Config_ID"] == "EMC-001"].iloc[0]
    result = predict(8.0, 2.0, 150.0, 10.0, engine="extratrees")
    assert result["status"] == "OK", result
    cap = result["predictions"]["TotalCapacitance_pF_per_m"]["value"]
    mut = result["predictions"]["AbsMutualInductance_nH"]["value"]
    # ExtraTrees: allow up to 0.1 pF/m and 0.01 nH
    assert abs(cap - row["TotalCapacitance_pF_per_m"]) < 0.1
    assert abs(mut - row["AbsMutualInductance_nH"]) < 0.01

run_test("TEST 2a – Known config (EMC-001) frozen model matches MATLAB reference", t2_known_config_frozen)
run_test("TEST 2b – Known config (EMC-001) ExtraTrees within tolerance", t2_known_config_extratrees)


# ──────────────────────────────────────────────────────────────────────────
# TEST 3: Test multiple configurations from the 216-configuration domain
# ──────────────────────────────────────────────────────────────────────────
def t3_multiple_configs():
    from models.prediction_engine import predict
    from models.model_loader import load_physics_dataset, load_frozen_matlab_model
    df = load_physics_dataset()
    m = load_frozen_matlab_model()
    # Test 10 configurations spread across the dataset
    indices = [0, 25, 50, 75, 100, 125, 150, 175, 200, 215]
    errors_c, errors_m = [], []
    for i in indices:
        row = df.iloc[i]
        result = predict(
            row["PairSeparation_mm"], row["PairPitch_mm"],
            row["Overlap_mm"], row["GroundClearance_mm"],
            engine="frozen_matlab",
        )
        assert result["status"] == "OK", f"Row {i}: {result}"
        dc = abs(result["predictions"]["TotalCapacitance_pF_per_m"]["abs_diff"])
        dm = abs(result["predictions"]["AbsMutualInductance_nH"]["abs_diff"])
        errors_c.append(dc)
        errors_m.append(dm)
    max_ec = max(errors_c)
    max_em = max(errors_m)
    assert max_ec < 1e-8, f"Max cap error across 10 configs: {max_ec:.3e}"
    assert max_em < 1e-8, f"Max mut error across 10 configs: {max_em:.3e}"

run_test("TEST 3 – Multiple configs from 216 grid all match MATLAB reference", t3_multiple_configs)


# ──────────────────────────────────────────────────────────────────────────
# TEST 4: Units and output precision
# ──────────────────────────────────────────────────────────────────────────
def t4_units_precision():
    from models.prediction_engine import predict
    result = predict(16.0, 3.0, 300.0, 25.0, engine="frozen_matlab")
    assert result["status"] == "OK"
    cap = result["predictions"]["TotalCapacitance_pF_per_m"]
    mut = result["predictions"]["AbsMutualInductance_nH"]
    assert cap["unit"] == "pF/m", f"Cap unit: {cap['unit']}"
    assert mut["unit"] == "nH",   f"Mut unit: {mut['unit']}"
    # Physics range checks (capacitance ~10–60 pF/m; inductance ~0.1–26 nH)
    assert 5 < cap["value"] < 100,  f"Cap out of physical range: {cap['value']}"
    assert 0.05 < mut["value"] < 30, f"Mut out of physical range: {mut['value']}"

run_test("TEST 4 – Output units and physical range checks", t4_units_precision)


# ──────────────────────────────────────────────────────────────────────────
# TEST 5: Invalid inputs rejected cleanly
# ──────────────────────────────────────────────────────────────────────────
def t5_invalid_inputs():
    from models.prediction_engine import predict
    # Out-of-domain value
    result = predict(15.0, 2.0, 150.0, 10.0)  # sep=15 is not in {8,12,16,20,24,28}
    assert result["status"] == "ERROR", "Expected ERROR for out-of-domain sep=15"
    assert "ERROR" in result["status"]

    # Invalid overlap
    result2 = predict(8.0, 2.0, 200.0, 10.0)
    assert result2["status"] == "ERROR", "Expected ERROR for overlap=200"

    # Invalid ground clearance
    result3 = predict(8.0, 2.0, 150.0, 30.0)
    assert result3["status"] == "ERROR", "Expected ERROR for GC=30"

run_test("TEST 5 – Invalid/out-of-domain inputs return ERROR status", t5_invalid_inputs)


# ──────────────────────────────────────────────────────────────────────────
# TEST 6: Configuration comparison
# ──────────────────────────────────────────────────────────────────────────
def t6_config_comparison():
    from models.prediction_engine import predict
    configs = [
        (8.0, 2.0, 150.0, 10.0),
        (28.0, 4.0, 450.0, 100.0),
        (16.0, 3.0, 300.0, 50.0),
    ]
    results = [predict(*c) for c in configs]
    for r in results:
        assert r["status"] == "OK", r
    caps = [r["predictions"]["TotalCapacitance_pF_per_m"]["value"] for r in results]
    muts = [r["predictions"]["AbsMutualInductance_nH"]["value"] for r in results]
    # All positive and not equal
    assert all(c > 0 for c in caps)
    assert all(m > 0 for m in muts)
    assert len(set(round(c, 4) for c in caps)) > 1, "Caps should not all be identical"

run_test("TEST 6 – Configuration comparison returns distinct valid values", t6_config_comparison)


# ──────────────────────────────────────────────────────────────────────────
# TEST 7: EMC-GUARD ranking logic
# ──────────────────────────────────────────────────────────────────────────
def t7_guard_ranking():
    from core.emc_guard import run_emc_guard_assessment
    result = run_emc_guard_assessment(
        16.0, 3.0, 300.0, 25.0,
        objective="minimize_combined",
        w_cap=0.5, w_mut=0.5,
    )
    assert "error" not in result, result.get("error")
    ranked = result["ranked_df"]
    assert len(ranked) == 216, f"Expected 216 candidates, got {len(ranked)}"
    # Ranks should be monotonically increasing
    assert all(ranked["Rank"].values[i] <= ranked["Rank"].values[i+1]
               for i in range(len(ranked)-1)), "Ranks not sorted"
    # Top rank should have lowest or equal objective score
    assert ranked.iloc[0]["ObjectiveScore"] <= ranked.iloc[-1]["ObjectiveScore"]
    # Recommendation must not contain compliance language
    rec = result["recommendation"].lower()
    assert "compliant" not in rec and "safe" not in rec and "certified" not in rec

run_test("TEST 7 – EMC-GUARD ranking logic and terminology check", t7_guard_ranking)


# ──────────────────────────────────────────────────────────────────────────
# TEST 8: 3D visualisation builds without error
# ──────────────────────────────────────────────────────────────────────────
def t8_3d_visualisation():
    from digital_twin.geometry import build_3d_figure, build_cross_section_figure
    fig3d = build_3d_figure(16.0, 3.0, 300.0, 25.0)
    assert fig3d is not None
    assert len(fig3d.data) >= 3  # At least cables + ground

    fig2d = build_cross_section_figure(16.0, 3.0, 25.0)
    assert fig2d is not None

    # Test with different parameters
    fig3d_2 = build_3d_figure(28.0, 4.0, 450.0, 100.0)
    assert fig3d_2 is not None

run_test("TEST 8 – 3D and cross-section geometry figures build correctly", t8_3d_visualisation)


# ──────────────────────────────────────────────────────────────────────────
# TEST 9: PDF report generates without error
# ──────────────────────────────────────────────────────────────────────────
def t9_pdf_report():
    try:
        from core.report_generator import generate_report
        pdf = generate_report(
            config_id="EMC-050",
            sep=16.0, pitch=3.0, overlap=300.0, ground=25.0,
            cap_value=28.5, mut_value=2.5,
            cap_model="B2_SP_Ground",
            mut_model="B1_SP_Overlap",
            engine="Stage12A_FrozenMATLAB",
            recommendation="Test recommendation for unit testing.",
        )
        assert isinstance(pdf, bytes) and len(pdf) > 1000, "PDF too small or not bytes"
    except ImportError:
        raise AssertionError("reportlab not available")

run_test("TEST 9 – PDF report generation", t9_pdf_report)


# ──────────────────────────────────────────────────────────────────────────
# TEST 10: Imports all succeed (clean environment check)
# ──────────────────────────────────────────────────────────────────────────
def t10_imports():
    import numpy, pandas, plotly, streamlit, joblib, h5py, scipy
    from models.model_loader import get_model_status
    status = get_model_status()
    assert status["frozen_matlab"]["ok"], f"Frozen MATLAB model status: {status['frozen_matlab']}"
    assert status["extratrees"]["ok"],    f"ExtraTrees status: {status['extratrees']}"
    assert status["physics_dataset"]["ok"], "Dataset not available"
    assert status["physics_dataset"]["n_configs"] == 216

run_test("TEST 10 – All imports and model_status pass", t10_imports)


# ──────────────────────────────────────────────────────────────────────────
# TEST 11: No fabricated/stale predictions
# ──────────────────────────────────────────────────────────────────────────
def t11_no_fabricated_data():
    from models.prediction_engine import predict
    from models.model_loader import load_physics_dataset
    df = load_physics_dataset()
    # Each frozen-model prediction must exactly equal the MATLAB reference (within fp tolerance)
    errors = []
    for i in [0, 10, 50, 100, 200, 215]:
        row = df.iloc[i]
        result = predict(
            row["PairSeparation_mm"], row["PairPitch_mm"],
            row["Overlap_mm"], row["GroundClearance_mm"],
            engine="frozen_matlab",
        )
        cap_diff = abs(result["predictions"]["TotalCapacitance_pF_per_m"]["abs_diff"])
        mut_diff = abs(result["predictions"]["AbsMutualInductance_nH"]["abs_diff"])
        if cap_diff > 1e-8 or mut_diff > 1e-8:
            errors.append(f"Config {i}: cap_diff={cap_diff:.3e}, mut_diff={mut_diff:.3e}")
    assert not errors, "\n".join(errors)

run_test("TEST 11 – Frozen model predictions match MATLAB references (no fabrication)", t11_no_fabricated_data)


# ──────────────────────────────────────────────────────────────────────────
# TEST 12: Missing model file error handling
# ──────────────────────────────────────────────────────────────────────────
def t12_missing_model_error():
    from pathlib import Path
    import models.model_loader as ml

    # Temporarily patch the path to something that doesn't exist
    orig_path = ml.FROZEN_MAT_PATH
    ml.FROZEN_MAT_PATH = Path("nonexistent_model.mat")
    ml.load_frozen_matlab_model.cache_clear()

    try:
        ml.load_frozen_matlab_model()
        ml.FROZEN_MAT_PATH = orig_path
        ml.load_frozen_matlab_model.cache_clear()
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError as exc:
        pass  # Expected
    finally:
        ml.FROZEN_MAT_PATH = orig_path
        ml.load_frozen_matlab_model.cache_clear()
        # Reload to restore cache
        ml.load_frozen_matlab_model()

run_test("TEST 12 – Missing model file raises clear FileNotFoundError", t12_missing_model_error)


# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TEST SUMMARY")
print("="*65)
passed = sum(1 for _, r, _ in results if r == PASS)
failed = sum(1 for _, r, _ in results if r == FAIL)
for name, r, err in results:
    mark = "[PASS]" if r == PASS else "[FAIL]"
    print(f"  {mark}  {name}")
    if err:
        print(f"      -> {err[:100]}")

print(f"\n  Passed: {passed} / {len(results)}")
print(f"  Failed: {failed} / {len(results)}")
print("="*65 + "\n")

if failed > 0:
    sys.exit(1)
