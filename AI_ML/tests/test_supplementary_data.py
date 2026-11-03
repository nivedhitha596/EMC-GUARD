"""
test_supplementary_data.py
--------------------------
Unit tests verifying the safe loading, validation, and integrity checks
for the three supplementary simulation datasets.
"""

import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import supplementary_data as sd


class TestSupplementaryData(unittest.TestCase):

    def test_emc_simulated_loading(self):
        df, info = sd.load_emc_simulated_dataset()
        self.assertIsNotNone(df, "emc_predict_SIMULATED.csv should load successfully")
        self.assertEqual(info["rows"], 105, "Expected exactly 105 rows")
        self.assertEqual(info["cols"], 21, "Expected exactly 21 columns")
        self.assertEqual(info["status"], "VALIDATED")
        self.assertIn("victim_vpp_mv", df.columns)
        self.assertIn("victim_vrms_mv", df.columns)
        self.assertIn("fund_mag_dbv", df.columns)

    def test_rg_sweep_loading(self):
        df, info = sd.load_rg_sweep_dataset()
        self.assertIsNotNone(df, "rg_sweep_data_with_noise.csv should load successfully")
        self.assertEqual(info["rows"], 3, "Expected exactly 3 records")
        self.assertEqual(info["cols"], 18, "Expected exactly 18 columns")
        self.assertEqual(info["status"], "VALIDATED")
        self.assertListEqual(list(df["Rg_ohm"]), [10, 47, 100])
        # dv_dt slew rate decreases with Rg
        self.assertGreater(df.loc[df["Rg_ohm"] == 10, "dv_dt_V_per_ns"].values[0],
                           df.loc[df["Rg_ohm"] == 100, "dv_dt_V_per_ns"].values[0])
        # switching loss increases with Rg
        self.assertLess(df.loc[df["Rg_ohm"] == 10, "Turn_off_loss_uJ"].values[0],
                        df.loc[df["Rg_ohm"] == 100, "Turn_off_loss_uJ"].values[0])

    def test_rg_template_integrity(self):
        df, info = sd.load_rg_template_dataset()
        self.assertIsNotNone(df, "rg_sweep_template.csv should load successfully")
        self.assertEqual(info["rows"], 3, "Expected exactly 3 rows")
        self.assertEqual(info["cols"], 20, "Expected exactly 20 columns")
        self.assertEqual(info["status"], "AWAITING_SIMULATION_RESULTS")
        # Ensure outputs are completely unpopulated (not zero filled)
        self.assertEqual(info["total_outputs_missing"], 30, "Expected exactly 30 missing output values (10 cols x 3 rows)")
        self.assertEqual(info["percent_outputs_missing"], 100.0)
        for out_col in sd.TEMPLATE_OUTPUT_COLUMNS:
            self.assertTrue(df[out_col].isnull().all(), f"Column {out_col} should be all NaN, not populated")

    def test_nonexistent_file_handling(self):
        df, info = sd.load_emc_simulated_dataset(Path("non_existent_file.csv"))
        self.assertIsNone(df)
        self.assertEqual(info["status"], "FILE_NOT_FOUND")

    def test_load_all_supplementary(self):
        res = sd.load_all_supplementary_datasets()
        self.assertTrue(res["all_files_exist"], "All 3 supplementary files should exist")
        self.assertEqual(res["emc_simulated"]["info"]["rows"], 105)
        self.assertEqual(res["rg_sweep"]["info"]["rows"], 3)
        self.assertEqual(res["rg_template"]["info"]["rows"], 3)


if __name__ == "__main__":
    unittest.main()
