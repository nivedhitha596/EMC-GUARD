# KONE EMC-GUARD — AI_ML Module

**Module version:** 4.0 (Stages 1 to 4: AI Verification, Evaluation, XAI, Risk Assessment & Pareto Optimisation)  
**Dataset:** MATLAB 216-configuration physics dataset (Stage 13A)  
**Primary engine:** Stage 12A Frozen MATLAB (machine-precision analytical)  
**Secondary engine:** Stage 14A ExtraTrees surrogates (ML approximation)

---

## Module Structure

```
AI_ML/
├── dataset/
│   └── EMC_GUARD_Physics_Dataset_MATLAB.csv   ← Integration copy of MATLAB dataset
├── reports/                                    ← Verification, Evaluation, XAI, Risk & Optimisation reports
│   ├── figures/                                ← 30 Interactive Plotly HTML visualisations (21 Stage 2 + 9 Stage 3/4)
│   ├── MODEL_EVALUATION_AND_XAI_REPORT.md      ← Stage 2 evaluation and XAI report
│   ├── EMC_RISK_ASSESSMENT_REPORT.md           ← Stage 3 relative risk assessment report
│   ├── CONFIGURATION_OPTIMISATION_REPORT.md    ← Stage 4 multi-objective optimisation report
│   ├── emc_risk_assessment_all_configs_*.csv   ← Full 216-configuration risk assessment data
│   ├── pareto_optimal_configurations_*.csv     ← 6 non-dominated Pareto front configurations
│   ├── all_configurations_pareto_scored_*.csv  ← All 216 configurations with Pareto dominance rankings
│   └── evaluation_summary_*.json / *.csv       ← Raw validation & importance outputs
├── backups/                                    ← Pre-modification file snapshots
├── tests/
│   └── test_stage3_stage4.py                  ← 11 automated verification & validation unit tests
└── src/
    ├── __init__.py
    ├── config.py             ← Paths, domain constants, feature selections
    ├── dataset.py            ← Dataset loading, validation, utilities
    ├── predict.py            ← Prediction API (thin adapter over EMC_PROJECT engine)
    ├── verify.py             ← End-to-end verification and spot-checking
    ├── evaluate.py           ← In-sample & Leave-One-Level-Out (LOLO) evaluation
    ├── feature_importance.py ← MDI and permutation importance rankings
    ├── explainability.py     ← XAI: Permutation, ICE, PDP, Sensitivity, Individual explanations
    ├── visualise.py          ← Plotly interactive visualisation generator (21 Stage 2 plots)
    ├── risk_assessment.py    ← Stage 3: Relative EMC risk assessment & classification
    └── optimise.py           ← Stage 4: Multi-objective Pareto-front optimisation
```

---

## Quick Start & Execution Guide

All commands run directly using Python from the repository root:

### 1. Run Automated Test Suite
Executes 11 unit tests covering all 216 configurations, Pareto non-dominance mathematical correctness, risk classification consistency, and Stage 1/2 non-regression:
```powershell
python AI_ML/tests/test_stage3_stage4.py
```

### 2. Stage 3: Relative EMC Risk Assessment (`risk_assessment.py`)
Computes normalized capacitance ($I_C$) and mutual inductance ($I_M$) indicators, weighted combined risk scores ($I_{\text{comb}}$), percentile ranks, and XAI-driven geometric attributions. Generates 6 interactive Plotly HTML figures:
```powershell
python AI_ML/src/risk_assessment.py
```
Outputs:
- Full CSV export: `AI_ML/reports/emc_risk_assessment_all_configs_*.csv`
- Summary JSON: `AI_ML/reports/emc_risk_assessment_summary_*.json`
- 6 Interactive Figures:
  - `risk_capacitance_indicator.html`
  - `risk_mutual_inductance_indicator.html`
  - `risk_combined_comparison.html`
  - `risk_configuration_ranking.html`
  - `risk_capacitance_vs_inductance.html`
  - `risk_distribution_chart.html`

### 3. Stage 4: Intelligent Configuration Optimisation (`optimise.py`)
Executes non-dominated sorting across all 216 configurations to isolate the 6 Pareto optimal designs. Evaluates trade-offs, utopia distances, and side-by-side configuration deltas. Generates 3 interactive Plotly HTML figures:
```powershell
python AI_ML/src/optimise.py
```
Outputs:
- Pareto front CSV: `AI_ML/reports/pareto_optimal_configurations_*.csv`
- Full ranked dataset CSV: `AI_ML/reports/all_configurations_pareto_scored_*.csv`
- Summary JSON: `AI_ML/reports/optimisation_summary_*.json`
- 3 Interactive Figures:
  - `optimisation_pareto_front.html`
  - `optimisation_top_candidates.html`
  - `optimisation_geometry_comparison.html`

### 4. Stage 2: Model Evaluation & XAI Suite
```powershell
# Model evaluation (In-sample & LOLO CV)
python AI_ML/src/evaluate.py

# Feature importance analysis (MDI & Permutation)
python AI_ML/src/feature_importance.py

# Explainability analysis (Global, ICE, PDP, Sensitivity, Local attributions)
python AI_ML/src/explainability.py

# Generate all 21 Stage 2 interactive plots
python AI_ML/src/visualise.py
```

---

## Stage 3 & 4 Key Results

### Relative EMC Risk Tier Summary
- **Low Relative Indicator:** **155 configurations (71.8%)** ($I_{\text{comb}} \le 0.333$)
- **Moderate Relative Indicator:** **60 configurations (27.8%)** ($0.333 < I_{\text{comb}} \le 0.667$)
- **High Relative Indicator:** **1 configuration (0.5%)** (`EMC-027`, $I_{\text{comb}} = 0.8231$)

### Non-Dominated Pareto Optimal Set (6 Configurations)
All Pareto optimal configurations share: `PairPitch_mm = 2.0mm`, `Overlap_mm = 150.0mm`, `GroundClearance_mm = 100.0mm`.

| Config ID | Separation | Pitch | Overlap | Ground Clearance | Capacitance | Mutual Inductance | Role / Recommendation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`EMC-010`** | $8\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | **$29.47\text{ pF/m}$** | $1.9335\text{ nH}$ | **Minimum Capacitance Champion** |
| **`EMC-046`** | $12\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $30.90\text{ pF/m}$ | $0.8425\text{ nH}$ | **Recommended Balanced Design (Primary)** |
| **`EMC-082`** | $16\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $32.00\text{ pF/m}$ | $0.4698\text{ nH}$ | **Balanced Design (Moderate Spacing)** |
| **`EMC-118`** | $20\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $32.91\text{ pF/m}$ | $0.2989\text{ nH}$ | **Inductance-Favouring Balanced Design** |
| **`EMC-154`** | $24\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $33.69\text{ pF/m}$ | $0.2064\text{ nH}$ | **Ultra-Low Inductance Candidate** |
| **`EMC-190`** | $28\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $34.38\text{ pF/m}$ | **$0.1509\text{ nH}$** | **Minimum Inductance Champion** |

---

## Scientific Limitations & Engineering Rules

1. **Analytical Dataset Origin:** All rankings reflect deterministic simulation calculations from MATLAB analytical formulations. No empirical elevator hoistway measurement data is included.
2. **Relative Assessment Only:** Classifications are relative comparisons across the 216 available configurations. Configurations are **not** labelled "safe" or "unsafe".
3. **No Safety Certification:** Predictions and candidate rankings do not constitute compliance with IEC, EN (e.g. EN 12015/12016) or KONE internal EMC standards.
4. **Discrete Grid Boundaries:** Only the supported discrete parameter levels are permitted; no continuous extrapolation is performed.
5. **Preservation of Core Assets:** Production models (`EMC_GUARD_Capacitance_ExtraTrees.joblib`, `EMC_GUARD_MutualInductance_ExtraTrees.joblib`, `Stage12A_Selected_Models_FROZEN.mat`) and the existing Streamlit dashboard remain completely untouched.

---

## Supplementary Simulation Datasets (Visualisation & Engineering Analysis Only)

The platform incorporates three supplementary simulation datasets stored under `AI_ML/dataset/supplementary/` for auxiliary engineering exploration and switching circuit characterisation:

| Dataset File | Dimensions | Data Source | Simulation Status | Purpose |
|---|---|---|---|---|
| `emc_predict_SIMULATED.csv` | 105 rows × 21 cols | Switching Circuit Simulation | Complete (105 runs) | Electrical switching interference, victim coupling ($V_{pp}, V_{rms}$), and harmonic spectral analysis across frequencies, spacings, and shielding configurations. |
| `rg_sweep_data_with_noise.csv` | 3 rows × 18 cols | LTspice Simulation (IRFH5302) | Complete (3 runs) | Exploratory gate resistance ($R_g \in \{10, 47, 100\}\,\Omega$) switching sweep comparing slew rate ($dV/dt$), rise time, overshoot, noise, and turn-off dissipation. |
| `rg_sweep_template.csv` | 3 rows × 20 cols | LTspice Simulation Template | Incomplete (10/10 outputs pending) | Unpopulated simulation template tracking 30 pending measurement outputs (represented strictly as `NaN`, zero fabrication). |

### Dashboard Location
- **Screen:** Screen 9 – Engineering Report Generation & Data Packages
- **Tab:** **Supplementary Simulation Analysis**
  - *Tab 1 – Electrical Interference Analysis (105 Runs)*: Interactive filters, spatial attenuation boxplots, harmonic scatter charts, and shield/ground comparison.
  - *Tab 2 – MOSFET Switching Sweep Analysis (Rg Sweep)*: Metric cards, dual-axis $dV/dt$ (EMI proxy) vs turn-off loss (thermal proxy) trade-off curves.
  - *Tab 3 – Simulation Template (Awaiting Runs)*: Input vs pending output schema breakdown and status monitoring.

### Scientific Safeguards
1. **Strict Model Decoupling:** Supplementary datasets are **strictly decoupled** from the primary 216-configuration MATLAB analytical engine and ExtraTrees surrogates. They are NOT used for training, prediction, or validation scoring.
2. **No Data Merging:** Supplementary datasets are kept isolated in `AI_ML/dataset/supplementary/` and are never combined with the core MATLAB dataset.
3. **No Artificial Imputation:** Unpopulated cells in `rg_sweep_template.csv` are preserved as `NaN`; never zero-filled or fabricated.
4. **Exploratory Status:** These datasets represent simulated numerical sweeps and preliminary exploratory runs; they do not establish certified regulatory compliance (IEC/EN) or real hoistway experimental validation.

