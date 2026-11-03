
from pathlib import Path
import pandas as pd

integration_dir = Path(__file__).resolve().parent
project_dir = integration_dir.parent

source = (
    project_dir.parent
    / "09_Stage13A_Export"
    / "EMC_GUARD_Physics_Dataset.csv"
)

output = integration_dir / "input" / "EMC_GUARD_Input.csv"

df = pd.read_csv(source)

columns = [
    "Config_ID",
    "PairSeparation_mm",
    "PairPitch_mm",
    "Overlap_mm",
    "GroundClearance_mm",
]

if len(df) != 216:
    raise ValueError(f"Expected 216 rows, found {len(df)}")

if df["Config_ID"].nunique() != 216:
    raise ValueError("Duplicate configuration IDs detected")

batch = df[columns].copy()
batch.to_csv(output, index=False)

print("STAGE 14B-5 | BATCH INPUT READY")
print("Configurations:", len(batch))
print("Unique IDs:", batch["Config_ID"].nunique())
print("Output:", output)
print(batch.head())
