"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 1 | Script 1: Synthetic Dummy Dataset Generator
File   : generate_dummy_data.py
Author : EMC-GUARD Development Team

DISCLAIMER
----------
This script generates SYNTHETIC DUMMY DATA for development and pipeline testing
purposes ONLY. The data and the illustrative equations used to derive target
variables DO NOT represent actual electromagnetic physics, validated simulation
results, or real MATLAB/FEM outputs. They exist solely to exercise the
preprocessing pipeline until the MATLAB teammate provides the real simulation
CSV.

When the real MATLAB dataset is available, replace the input CSV path in the
configuration section of each script — NO other code changes are required.

Usage
-----
    python src/generate_dummy_data.py

Outputs
-------
    dataset/dummy_simulation_dataset.csv   — raw synthetic dataset (never modified)
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION — change only these paths when switching to the real dataset
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dummy_simulation_dataset.csv")

RANDOM_SEED = 42          # Fixed seed for full reproducibility
N_RECORDS   = 1000        # Number of synthetic records

# Fraction of records to corrupt for pipeline-stress testing
MISSING_FRAC   = 0.03     # 3 % of cells set to NaN
DUPLICATE_FRAC = 0.02     # 2 % extra duplicate rows appended
INVALID_FRAC   = 0.01     # 1 % of cells set to physically impossible values


def generate_clean_records(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """
    Generate N clean synthetic records with realistic engineering ranges.

    Feature definitions and units
    ------------------------------
    Separation_mm       : Separation between cables          [mm]   5 – 200
    Overlap_mm          : Longitudinal overlap length         [mm]   10 – 500
    Cable_Length_mm     : Total cable length                  [mm]   100 – 5000
    Frequency_Hz        : Driving signal frequency            [Hz]   1 000 – 30 000 000
    Voltage_V           : Source voltage amplitude            [V]    1 – 500
    Conductor_Radius_mm : Copper conductor radius             [mm]   0.5 – 5.0
    Dielectric_Constant : Relative permittivity of insulation [-]    2.0 – 10.0
    Shield_Type         : Shielded / Unshielded               [cat]
    Cable_Configuration : Parallel / Perpendicular / Crossed  [cat]

    Target definitions (illustrative synthetic relationships only)
    --------------------------------------------------------------
    MutualInductance_nH  : Mutual inductance                  [nH]
    MutualCapacitance_pF : Mutual capacitance                 [pF]
    Victim_Voltage_V     : Induced voltage on victim cable     [V]
    Victim_RMS_V         : RMS value of victim voltage         [V]
    Victim_PeakToPeak_V  : Peak-to-peak value of victim voltage[V]
    """

    config_ids = [f"CFG_{i+1:04d}" for i in range(n)]

    # --- Input features ---------------------------------------------------
    separation        = rng.uniform(5.0,       200.0,      n)   # mm
    overlap           = rng.uniform(10.0,      500.0,      n)   # mm
    cable_length      = rng.uniform(100.0,     5000.0,     n)   # mm
    frequency         = rng.uniform(1_000.0,   30_000_000, n)   # Hz
    voltage           = rng.uniform(1.0,       500.0,      n)   # V
    conductor_radius  = rng.uniform(0.5,       5.0,        n)   # mm
    dielectric        = rng.uniform(2.0,       10.0,       n)   # dimensionless
    shield_type       = rng.choice(["Shielded", "Unshielded"], size=n)
    cable_config      = rng.choice(["Parallel", "Perpendicular", "Crossed"], size=n)

    # --- Illustrative synthetic target equations ---------------------------
    # NOTE: These equations are invented for dataset-generation convenience.
    #       They are NOT derived from Maxwell's equations or validated models.

    # Shield attenuation factor (Shielded cables reduce coupling by ~20 dB)
    shield_factor = np.where(shield_type == "Shielded", 0.1, 1.0)

    # Configuration coupling factor (Perpendicular/Crossed reduce coupling)
    config_factor = np.select(
        [cable_config == "Parallel",
         cable_config == "Perpendicular",
         cable_config == "Crossed"],
        [1.0, 0.3, 0.5]
    )

    noise = lambda scale: rng.normal(0, scale, n)

    # MutualInductance_nH: increases with overlap and cable length,
    # decreases with separation; modulated by shield and config.
    mutual_inductance = (
        (overlap * cable_length / (separation + 1.0)) * 0.0005
        * shield_factor * config_factor
        + noise(0.5)
    )
    mutual_inductance = np.clip(mutual_inductance, 0.1, 1e5)   # physical bounds [nH]

    # MutualCapacitance_pF: increases with overlap, dielectric constant;
    # decreases with separation; modulated by shield.
    mutual_capacitance = (
        (overlap * dielectric * conductor_radius / (separation + 1.0)) * 0.08
        * shield_factor * config_factor
        + noise(0.2)
    )
    mutual_capacitance = np.clip(mutual_capacitance, 0.01, 1e4)  # [pF]

    # Victim_Voltage_V: inductive + capacitive coupling contribution
    omega = 2.0 * np.pi * frequency
    victim_voltage = (
        (omega * mutual_inductance * 1e-9 * voltage
         + omega * mutual_capacitance * 1e-12 * voltage * 50.0)  # 50 Ω load
        + noise(0.01)
    )
    victim_voltage = np.abs(victim_voltage)
    victim_voltage = np.clip(victim_voltage, 0.0, voltage)       # cannot exceed source

    # Victim_RMS_V: RMS of a sinusoidal victim signal
    victim_rms = victim_voltage / np.sqrt(2.0) + noise(0.002)
    victim_rms = np.abs(victim_rms)

    # Victim_PeakToPeak_V: peak-to-peak span
    victim_pp = 2.0 * victim_voltage + noise(0.005)
    victim_pp = np.abs(victim_pp)

    df = pd.DataFrame({
        "Configuration_ID"    : config_ids,
        "Separation_mm"       : np.round(separation,       3),
        "Overlap_mm"          : np.round(overlap,           3),
        "Cable_Length_mm"     : np.round(cable_length,      3),
        "Frequency_Hz"        : np.round(frequency,         2),
        "Voltage_V"           : np.round(voltage,           3),
        "Conductor_Radius_mm" : np.round(conductor_radius,  4),
        "Dielectric_Constant" : np.round(dielectric,        4),
        "Shield_Type"         : shield_type,
        "Cable_Configuration" : cable_config,
        "MutualInductance_nH" : np.round(mutual_inductance,  6),
        "MutualCapacitance_pF": np.round(mutual_capacitance, 6),
        "Victim_Voltage_V"    : np.round(victim_voltage,     8),
        "Victim_RMS_V"        : np.round(victim_rms,         8),
        "Victim_PeakToPeak_V" : np.round(victim_pp,          8),
    })

    return df


def inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Deliberately inject controlled data-quality issues so the cleaning
    pipeline has something to detect and handle.

    Issues introduced
    -----------------
    1. Missing values  — random cells in numerical columns set to NaN
    2. Duplicate rows  — a subset of rows appended with new Config IDs
    3. Invalid values  — physically impossible negatives or zeros
    4. Outlier values  — extreme but non-negative engineering values
    """
    df = df.copy()
    n = len(df)

    numerical_cols = [
        "Separation_mm", "Overlap_mm", "Cable_Length_mm", "Frequency_Hz",
        "Voltage_V", "Conductor_Radius_mm", "Dielectric_Constant",
        "MutualInductance_nH", "MutualCapacitance_pF",
        "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V"
    ]

    # 1. Missing values (3 % of cells in numerical columns)
    n_missing = int(n * len(numerical_cols) * MISSING_FRAC)
    for _ in range(n_missing):
        row = rng.integers(0, n)
        col = rng.choice(numerical_cols)
        df.at[row, col] = np.nan

    # 2. Duplicate configurations (2 % duplicate rows)
    n_dups = max(1, int(n * DUPLICATE_FRAC))
    dup_indices = rng.choice(n, size=n_dups, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    # Give duplicates new IDs so they look like separate configurations
    dup_rows["Configuration_ID"] = [f"CFG_DUP_{i+1:03d}" for i in range(n_dups)]
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3. Invalid numerical values (negative values in physically positive columns)
    n_invalid = max(1, int(n * INVALID_FRAC))
    for i in range(n_invalid):
        row = rng.integers(0, n)
        col = rng.choice(["Separation_mm", "Frequency_Hz",
                          "MutualInductance_nH", "MutualCapacitance_pF"])
        df.at[row, col] = rng.uniform(-500, -0.001)   # Negative — physically invalid

    # 4. Outlier values (extreme but positive)
    n_outliers = max(1, int(n * INVALID_FRAC))
    for i in range(n_outliers):
        row = rng.integers(0, n)
        col = rng.choice(["Voltage_V", "Cable_Length_mm", "MutualInductance_nH"])
        df.at[row, col] = rng.uniform(1e6, 1e7)       # Extreme but non-negative

    return df


def add_metadata_header(filepath: str) -> None:
    """Prepend a metadata block to the CSV file for documentation."""
    header = (
        "# ============================================================\n"
        "# EMC-GUARD — SYNTHETIC DUMMY DATASET\n"
        "# ============================================================\n"
        "# DISCLAIMER: This file contains SYNTHETIC DATA generated by\n"
        "# NumPy with a fixed random seed for pipeline development and\n"
        "# testing purposes ONLY.\n"
        "#\n"
        "# It does NOT represent actual MATLAB simulation results,\n"
        "# validated electromagnetic measurements, or real physics.\n"
        "#\n"
        "# The target-variable equations are illustrative conveniences,\n"
        "# NOT derived from Maxwell's equations or any validated model.\n"
        "#\n"
        "# When the MATLAB teammate provides the real simulation CSV,\n"
        "# update INPUT_CSV in each script — no other changes required.\n"
        "# ============================================================\n"
    )
    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + original)


def main():
    print("=" * 65)
    print("EMC-GUARD  |  Module 1  |  Synthetic Dummy Dataset Generator")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\n[1/3] Generating {N_RECORDS} clean synthetic records …")
    df_clean = generate_clean_records(rng, N_RECORDS)
    print(f"      Clean records generated: {len(df_clean)}")

    print("\n[2/3] Injecting controlled data-quality issues …")
    df_dirty = inject_data_quality_issues(df_clean, rng)
    print(f"      Total records after injection: {len(df_dirty)}")
    print(f"      Missing cells  : {df_dirty.isnull().sum().sum()}")
    print(f"      Duplicate rows : {df_dirty.duplicated(subset=df_dirty.columns[1:]).sum()}")

    print(f"\n[3/3] Saving raw dataset to:\n      {OUTPUT_FILE}")
    df_dirty.to_csv(OUTPUT_FILE, index=False)
    add_metadata_header(OUTPUT_FILE)
    print("      Done.")

    print("\n" + "=" * 65)
    print("Dataset Summary")
    print("-" * 65)
    print(f"  Rows    : {len(df_dirty)}")
    print(f"  Columns : {len(df_dirty.columns)}")
    print(f"  File    : {os.path.relpath(OUTPUT_FILE, BASE_DIR)}")
    print("=" * 65)
    print("\n[DONE] Dummy dataset generation complete.")
    print("  REMINDER: This is SYNTHETIC data for testing only.")


if __name__ == "__main__":
    main()
