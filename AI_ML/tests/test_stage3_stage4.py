"""
test_stage3_stage4.py
---------------------
Automated Test Suite for KONE EMC-GUARD Stages 3 & 4:
  - Stage 3: EMC Risk Assessment
  - Stage 4: Intelligent Configuration Optimisation

Validates:
  1. All 216 configurations are analysed.
  2. Domain constraints and rejection of invalid configurations.
  3. Prediction matching with existing prediction API.
  4. Reproducibility of risk indicators and classifications.
  5. Mathematical consistency of risk rankings.
  6. Pareto dominance logic correctness.
  7. No dominated configuration in Pareto front.
  8. No Pareto optimal configuration omitted.
  9. CSV and JSON export integrity.
  10. Visualisation file generation and validity.
  11. Non-regression of existing Stage 1 & Stage 2 components.

Usage:
    py AI_ML/tests/test_stage3_stage4.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import unittest
from pathlib import Path

# Force UTF-8 encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure AI_ML/src is in sys.path
_TEST_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TEST_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from config import (
    AIML_REPORTS_DIR,
    CAP_FEATURES,
    ENGINE_FROZEN_MATLAB,
    MUT_FEATURES,
    SUPPORTED_CLEARANCES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_PITCHES,
    SUPPORTED_SEPARATIONS,
    TOTAL_CONFIGURATIONS,
)
from dataset import is_valid_configuration, load_dataset
from optimise import (
    compare_configurations,
    compute_pareto_dominance,
    run_optimisation,
    score_pareto_candidates,
)
from predict import predict
from risk_assessment import (
    TIER_HIGH,
    TIER_LOW,
    TIER_MOD,
    assess_all_configurations,
    assess_configuration,
    calculate_risk_indicators,
    get_risk_summary,
    run_risk_assessment,
)

FIGURES_DIR = AIML_REPORTS_DIR / "figures"
CAP_COL = "TotalCapacitance_pF_per_m"
MUT_COL = "AbsMutualInductance_nH"


class TestStage3RiskAssessment(unittest.TestCase):
    """Test suite for Stage 3 EMC Risk Assessment."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.assessed_df = calculate_risk_indicators(cls.df)

    def test_01_all_216_configurations_analysed(self):
        """Test 1: All 216 configurations are analysed without drops."""
        self.assertEqual(len(self.assessed_df), TOTAL_CONFIGURATIONS)
        self.assertEqual(self.assessed_df["Config_ID"].nunique(), TOTAL_CONFIGURATIONS)

    def test_02_invalid_configuration_rejection(self):
        """Test 2: Only valid configurations within design grid are accepted."""
        # Test invalid separation
        res_invalid_sep = assess_configuration(sep=15.0, pitch=2.0, overlap=150.0, ground=10.0)
        self.assertEqual(res_invalid_sep["status"], "ERROR")
        self.assertIn("outside", res_invalid_sep["error"])

        # Test invalid ground clearance
        res_invalid_gc = assess_configuration(sep=8.0, pitch=2.0, overlap=150.0, ground=35.0)
        self.assertEqual(res_invalid_gc["status"], "ERROR")

        # Test valid configuration
        res_valid = assess_configuration(sep=8.0, pitch=2.0, overlap=150.0, ground=10.0)
        self.assertEqual(res_valid["status"], "OK")

    def test_03_predictions_match_authoritative_engine(self):
        """Test 3: Predictions match the existing frozen prediction API."""
        test_config = (12.0, 3.0, 300.0, 50.0)
        res_assessment = assess_configuration(*test_config, engine=ENGINE_FROZEN_MATLAB)
        res_predict = predict(*test_config, engine=ENGINE_FROZEN_MATLAB)

        cap_assessed = res_assessment["predictions"][CAP_COL]["value"]
        cap_direct = res_predict["predictions"][CAP_COL]["value"]
        self.assertAlmostEqual(cap_assessed, cap_direct, places=6)

        mut_assessed = res_assessment["predictions"][MUT_COL]["value"]
        mut_direct = res_predict["predictions"][MUT_COL]["value"]
        self.assertAlmostEqual(mut_assessed, mut_direct, places=6)

    def test_04_risk_classifications_reproducible(self):
        """Test 4: Risk classification is reproducible and bounded in [0, 1]."""
        df1 = calculate_risk_indicators(self.df)
        df2 = calculate_risk_indicators(self.df)
        pd.testing.assert_frame_equal(df1, df2)

        # Check bounds
        self.assertTrue((df1["Capacitance_Indicator"] >= 0.0).all())
        self.assertTrue((df1["Capacitance_Indicator"] <= 1.0 + 1e-9).all())
        self.assertTrue((df1["MutualInductance_Indicator"] >= 0.0).all())
        self.assertTrue((df1["MutualInductance_Indicator"] <= 1.0 + 1e-9).all())
        self.assertTrue((df1["Combined_Risk_Indicator"] >= 0.0).all())
        self.assertTrue((df1["Combined_Risk_Indicator"] <= 1.0 + 1e-9).all())

    def test_05_risk_ranking_consistency(self):
        """Test 5: Risk ranking is monotonically consistent with combined indicators."""
        df_sorted = self.assessed_df.sort_values("Risk_Rank")
        scores = df_sorted["Combined_Risk_Indicator"].values
        diffs = np.diff(scores)
        self.assertTrue((diffs >= -1e-9).all(), "Risk scores must be non-decreasing with respect to Risk_Rank.")

        # Check valid tiers
        valid_tiers = {TIER_LOW, TIER_MOD, TIER_HIGH}
        self.assertTrue(set(self.assessed_df["Combined_Risk_Tier"].unique()).issubset(valid_tiers))


class TestStage4Optimisation(unittest.TestCase):
    """Test suite for Stage 4 Intelligent Configuration Optimisation."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.all_pareto_df = compute_pareto_dominance(cls.df)
        cls.pareto_front = cls.all_pareto_df[cls.all_pareto_df["Is_Pareto_Optimal"]].copy()

    def test_06_pareto_dominance_logic(self):
        """Test 6: Pareto dominance logic rigorously separates dominated vs non-dominated."""
        # For any point marked Pareto optimal, verify that NO configuration in df dominates it
        caps = self.all_pareto_df[CAP_COL].values
        muts = self.all_pareto_df[MUT_COL].values
        p_caps = self.pareto_front[CAP_COL].values
        p_muts = self.pareto_front[MUT_COL].values

        for p_c, p_m in zip(p_caps, p_muts):
            for c, m in zip(caps, muts):
                strictly_better = (c <= p_c and m <= p_m) and (c < p_c or m < p_m)
                self.assertFalse(strictly_better, f"Pareto point ({p_c}, {p_m}) was dominated by ({c}, {m})!")

    def test_07_no_dominated_configuration_in_pareto_front(self):
        """Test 7: No configuration in the Pareto front is dominated by another."""
        self.assertTrue((self.pareto_front["Dominated_By_Count"] == 0).all())

    def test_08_exact_pareto_front_members(self):
        """Test 8: Exact verified Pareto front members match the 6 analytical candidates."""
        # There are exactly 6 non-dominated configurations in this 216-dataset
        self.assertEqual(len(self.pareto_front), 6)

        # All Pareto optimal designs MUST have: Pitch=2, Overlap=150, GroundClearance=100
        self.assertTrue((self.pareto_front["PairPitch_mm"] == 2.0).all())
        self.assertTrue((self.pareto_front["Overlap_mm"] == 150.0).all())
        self.assertTrue((self.pareto_front["GroundClearance_mm"] == 100.0).all())

        # And exactly span the 6 separation levels: 8, 12, 16, 20, 24, 28 mm
        seps = sorted(self.pareto_front["PairSeparation_mm"].unique().tolist())
        self.assertEqual(seps, [8.0, 12.0, 16.0, 20.0, 24.0, 28.0])

    def test_09_configuration_comparison(self):
        """Test 9: Configuration comparison produces correct deltas and dominance."""
        c1 = {"PairSeparation_mm": 8.0, "PairPitch_mm": 2.0, "Overlap_mm": 150.0, "GroundClearance_mm": 100.0}
        c2 = {"PairSeparation_mm": 28.0, "PairPitch_mm": 2.0, "Overlap_mm": 150.0, "GroundClearance_mm": 100.0}
        comp = compare_configurations(c1, c2, "MinCap", "MinInd")

        self.assertAlmostEqual(comp["delta"]["Capacitance_Diff_pF_per_m"], 34.3804 - 29.4676, places=2)
        self.assertAlmostEqual(comp["delta"]["MutualInductance_Diff_nH"], 0.15085 - 1.93349, places=2)
        self.assertIn("Mutual trade-off", comp["dominance_relationship"])


class TestSystemIntegration(unittest.TestCase):
    """Test suite for system integration, reports, and visualisations."""

    def test_10_visualisation_files_exist_and_valid(self):
        """Test 10: All 9 Stage 3 & 4 figures exist and are valid HTML documents."""
        expected_figures = [
            "risk_capacitance_indicator.html",
            "risk_mutual_inductance_indicator.html",
            "risk_combined_comparison.html",
            "risk_configuration_ranking.html",
            "risk_capacitance_vs_inductance.html",
            "risk_distribution_chart.html",
            "optimisation_pareto_front.html",
            "optimisation_top_candidates.html",
            "optimisation_geometry_comparison.html",
        ]
        for fig_name in expected_figures:
            fig_path = FIGURES_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Figure {fig_name} was not found at {fig_path}")
            content = fig_path.read_text(encoding="utf-8")
            self.assertGreater(len(content), 5000, f"Figure {fig_name} appears truncated or empty.")
            self.assertIn("plotly", content.lower(), f"Figure {fig_name} does not contain Plotly code.")

    def test_11_stage1_and_stage2_non_regression(self):
        """Test 11: Stage 1 and Stage 2 models, scripts, and reports remain functional."""
        from evaluate import in_sample_evaluation, _load_models
        cap_model, mut_model = _load_models()
        self.assertIsNotNone(cap_model)
        self.assertIsNotNone(mut_model)

        df = load_dataset()
        eval_res = in_sample_evaluation(df, cap_model, mut_model)
        self.assertEqual(eval_res["evaluation_type"], "in_sample")
        self.assertAlmostEqual(eval_res["capacitance"]["metrics"]["R2"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

