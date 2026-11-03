
from pathlib import Path
import json
from datetime import datetime

import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "comparison" / "EMC_GUARD_Comparison.csv"
OUTPUT_DIR = ROOT / "comparison" / "Stage14B_5_Validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# COMPUTATIONAL COMPARISON TOLERANCES
# These are NOT physical EMC tolerances.
# --------------------------------------------------
C_TOL = 1e-9       # pF/m
M_TOL = 1e-10      # nH

REQUIRED_COLUMNS = [
    "Config_ID",
    "AI_Prediction_Status",
    "Comparison_Status",
    "Capacitance_Absolute_Difference",
    "MutualInductance_Absolute_Difference",
]

# --------------------------------------------------
# LOAD AND CHECK
# --------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Comparison file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

if len(df) != 216:
    raise ValueError(f"Expected 216 rows, found {len(df)}")

if df["Config_ID"].nunique() != 216:
    raise ValueError("Config_ID values are not 216 unique configurations.")

if df["Config_ID"].isna().any():
    raise ValueError("Missing Config_ID values detected.")

if df[REQUIRED_COLUMNS].isna().any().any():
    raise ValueError("Missing values detected in required columns.")

# Convert numerical difference columns safely
for col in [
    "Capacitance_Absolute_Difference",
    "MutualInductance_Absolute_Difference",
]:
    df[col] = pd.to_numeric(df[col], errors="raise")

# --------------------------------------------------
# VALIDATION CHECKS
# --------------------------------------------------
exact_grid_count = (
    df["AI_Prediction_Status"]
    .astype(str)
    .str.startswith("Accepted: exact dataset configuration")
    .sum()
)

c_max = float(df["Capacitance_Absolute_Difference"].abs().max())
m_max = float(df["MutualInductance_Absolute_Difference"].abs().max())

c_pass = c_max <= C_TOL
m_pass = m_max <= M_TOL
grid_pass = exact_grid_count == 216
row_pass = len(df) == 216
unique_pass = df["Config_ID"].nunique() == 216
missing_pass = not df[REQUIRED_COLUMNS].isna().any().any()

overall_pass = all([
    c_pass,
    m_pass,
    grid_pass,
    row_pass,
    unique_pass,
    missing_pass,
])

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
summary = {
    "Stage": "Stage14B-5",
    "Project": "EMC-GUARD",
    "Generated_At": datetime.now().isoformat(timespec="seconds"),
    "Configurations": int(len(df)),
    "Unique_Configuration_IDs": int(df["Config_ID"].nunique()),
    "Exact_Grid_Accepted": int(exact_grid_count),
    "Missing_Required_Values": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
    "Capacitance_Max_Absolute_Difference_pF_per_m": c_max,
    "Capacitance_Absolute_Tolerance_pF_per_m": C_TOL,
    "Capacitance_Tolerance_Check": bool(c_pass),
    "MutualInductance_Max_Absolute_Difference_nH": m_max,
    "MutualInductance_Absolute_Tolerance_nH": M_TOL,
    "MutualInductance_Tolerance_Check": bool(m_pass),
    "Exact_Grid_Check": bool(grid_pass),
    "Row_Count_Check": bool(row_pass),
    "Unique_ID_Check": bool(unique_pass),
    "Missing_Value_Check": bool(missing_pass),
    "Overall_Computational_Integration_Result":
        "PASS" if overall_pass else "FAIL",
    "Validation_Scope":
        "Computational agreement between AI surrogate and MATLAB reference "
        "on the original 216-point dataset only.",
    "Limitations":
        "This does not establish independent physical validation, "
        "real elevator cable accuracy, or EMC standards compliance.",
}

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
json_file = OUTPUT_DIR / "Stage14B_5_Validation_Summary.json"
csv_file = OUTPUT_DIR / "Stage14B_5_Validation_Summary.csv"
txt_file = OUTPUT_DIR / "Stage14B_5_Validation_Report.txt"

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

pd.DataFrame([summary]).to_csv(csv_file, index=False)

report = f"""
EMC-GUARD
STAGE14B-5: MATLAB–AI INTEGRATION VALIDATION REPORT
===================================================

Generated: {summary['Generated_At']}

1. DATASET INTEGRITY
--------------------
Configurations processed: {len(df)}
Unique configuration IDs: {df['Config_ID'].nunique()}
Exact-grid accepted: {exact_grid_count}
Missing required values: {summary['Missing_Required_Values']}

2. NUMERICAL COMPARISON
-----------------------
Capacitance maximum absolute difference:
{c_max:.12e} pF/m

Capacitance computational tolerance:
{C_TOL:.12e} pF/m

Capacitance tolerance check: {"PASS" if c_pass else "FAIL"}

Mutual inductance maximum absolute difference:
{m_max:.12e} nH

Mutual inductance computational tolerance:
{M_TOL:.12e} nH

Mutual inductance tolerance check: {"PASS" if m_pass else "FAIL"}

3. INTEGRATION RESULT
---------------------
Overall computational integration result:
{"PASS" if overall_pass else "FAIL"}

4. VALIDATION SCOPE
-------------------
The AI surrogate was compared against the MATLAB reference
for the original 216-point geometry dataset.

The result demonstrates computational agreement between
the two implementations within the defined numerical
tolerances.

5. IMPORTANT LIMITATIONS
------------------------
- The AI surrogate reproduces MATLAB-generated outputs.
- The comparison is not independent physical validation.
- No real elevator cable accuracy is established.
- No EMC standards compliance is established.
- Predictions are restricted to the original 216-point grid.
- Physical experimental validation remains a separate task.

===================================================
END OF REPORT
"""

txt_file.write_text(report, encoding="utf-8")

print(report)
print("Saved validation files:")
print(json_file)
print(csv_file)
print(txt_file)

if not overall_pass:
    raise SystemExit("Stage14B-5 FAILED. Inspect the validation report.")
