import unittest
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\nived\OneDrive\Desktop\EMC_GAURD")
DATASET_PATH = REPO_ROOT / "EMC_PROJECT" / "data" / "EMC_GUARD_Physics_Dataset.csv"

class TestAnsysWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matlab_df = pd.read_csv(DATASET_PATH)

    def test_01_matlab_dataset_integrity(self):
        """Verify baseline analytical dataset has 216 rows and required columns."""
        self.assertEqual(len(self.matlab_df), 216)
        required_cols = ["Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm", "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH"]
        for c in required_cols:
            self.assertIn(c, self.matlab_df.columns)

    def test_02_q3d_matching_and_error_calculation(self):
        """Verify Q3D matching, absolute error and percentage error calculations."""
        # Create synthetic fixture
        sample = self.matlab_df.iloc[:10].copy()
        q3d_sim = pd.DataFrame({
            "Config_ID": sample["Config_ID"],
            "TotalCapacitance_pF_per_m": sample["TotalCapacitance_pF_per_m"] * 1.05  # +5% simulated deviation
        })
        
        merged = pd.merge(self.matlab_df, q3d_sim, on="Config_ID", suffixes=("_MATLAB", "_ANSYS"))
        self.assertEqual(len(merged), 10)
        
        abs_diff = (merged["TotalCapacitance_pF_per_m_ANSYS"] - merged["TotalCapacitance_pF_per_m_MATLAB"]).abs()
        pct_diff = (abs_diff / merged["TotalCapacitance_pF_per_m_MATLAB"]) * 100
        
        # In our synthetic fixture deviation is exactly 5%
        for val in pct_diff:
            self.assertAlmostEqual(val, 5.0, places=4)

    def test_03_maxwell_matching_and_error_calculation(self):
        """Verify Maxwell matching, absolute error and percentage error calculations."""
        sample = self.matlab_df.iloc[10:25].copy()
        maxwell_sim = pd.DataFrame({
            "Config_ID": sample["Config_ID"],
            "AbsMutualInductance_nH": sample["AbsMutualInductance_nH"] * 0.98  # -2% simulated deviation
        })
        
        merged = pd.merge(self.matlab_df, maxwell_sim, on="Config_ID", suffixes=("_MATLAB", "_ANSYS"))
        self.assertEqual(len(merged), 15)
        
        abs_diff = (merged["AbsMutualInductance_nH_ANSYS"] - merged["AbsMutualInductance_nH_MATLAB"]).abs()
        pct_diff = (abs_diff / merged["AbsMutualInductance_nH_MATLAB"]) * 100
        
        for val in pct_diff:
            self.assertAlmostEqual(val, 2.0, places=4)

    def test_04_geometric_matching_without_config_id(self):
        """Verify matching by geometric coordinates when Config_ID is absent."""
        sample = self.matlab_df.iloc[30:35].copy()
        q3d_geom = pd.DataFrame({
            "PairSeparation_mm": sample["PairSeparation_mm"],
            "PairPitch_mm": sample["PairPitch_mm"],
            "Overlap_mm": sample["Overlap_mm"],
            "GroundClearance_mm": sample["GroundClearance_mm"],
            "TotalCapacitance_pF_per_m": sample["TotalCapacitance_pF_per_m"] * 1.01
        })
        
        geom_keys = ["PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm"]
        merged = pd.merge(self.matlab_df, q3d_geom, on=geom_keys, suffixes=("_MATLAB", "_ANSYS"))
        self.assertEqual(len(merged), 5)
        self.assertEqual(set(merged["Config_ID"]), set(sample["Config_ID"]))

    def test_05_missing_columns_detection(self):
        """Verify that missing simulation target columns are detected."""
        invalid_df = pd.DataFrame({
            "Config_ID": ["EMC-001", "EMC-002"],
            "Unrelated_Column": [1, 2]
        })
        has_cap = any(c in invalid_df.columns for c in ["TotalCapacitance_pF_per_m", "C11_pF"])
        self.assertFalse(has_cap)

if __name__ == "__main__":
    unittest.main()

