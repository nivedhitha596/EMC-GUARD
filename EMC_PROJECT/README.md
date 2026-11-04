# EMC-PREDICT + EMC-GUARD

**Physics-Informed Configuration-Aware EMC Digital Twin for Elevator Cable Systems**

---

## Quick Start

```bash
cd EMC_PROJECT
pip install -r requirements.txt
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## Architecture

```
EMC_PROJECT/
├── app.py                          ← Streamlit entry point (9 screens)
├── requirements.txt
│
├── data/                           ← Reference data (from MATLAB)
│   ├── EMC_GUARD_Physics_Dataset.csv   (216 configs + targets)
│   ├── MATLAB_216_Verification.csv
│   ├── Stage12A_Selected_Model_Summary.csv
│   ├── Stage12A_Selected_Model_Predictions.csv
│   ├── LOLO_Summary.csv
│   └── model_performance.csv
│
├── models/
│   ├── model_loader.py             ← Loads frozen models and dataset
│   ├── prediction_engine.py        ← Prediction interface
│   ├── matlab_models/
│   │   └── Stage12A_Selected_Models_FROZEN.mat   ← FROZEN – do not modify
│   ├── EMC_GUARD_Capacitance_ExtraTrees.joblib
│   └── EMC_GUARD_MutualInductance_ExtraTrees.joblib
│
├── core/
│   ├── emc_guard.py                ← Configuration assessment engine
│   └── report_generator.py         ← PDF report generation
│
├── digital_twin/
│   └── geometry.py                 ← 3D / 2D cable visualisation
│
└── reports/                        ← Generated PDF reports
```

---

## Prediction Models

| Response | Model | Stage | Terms | Cond. No. | CV RMSE |
|----------|-------|-------|-------|-----------|---------|
| Total Capacitance | B2_SP_Ground | 12A (Frozen MATLAB) | 94 | 135.64 | 5.46×10⁻¹⁴ pF/m |
| Abs. Mutual Inductance | B1_SP_Overlap | 12A (Frozen MATLAB) | 84 | 103.79 | 1.07×10⁻¹⁴ nH |

Secondary: Stage 14A ExtraTrees surrogates (400 estimators, natural-log target transform).

---

## Supported Design Domain

| Parameter | Levels |
|-----------|--------|
| Pair Separation | 8, 12, 16, 20, 24, 28 mm |
| Cable Pitch | 2, 3, 4 |
| Parallel Overlap | 150, 300, 450 mm |
| Ground Clearance | 10, 25, 50, 100 mm |
| **Total** | **216 configurations** |

---

## Important Limitations

- Predictions reproduce a deterministic MATLAB-generated analytical dataset.
- This platform does **not** predict victim voltage, conducted emission, radiated emission or immunity.
- No EMC compliance, safety certification or experimental validation has been established.
- Extrapolation outside the 216-point design grid is not permitted.
- Use terminology: "lower/higher predicted coupling" — never "compliant/non-compliant".

---

## Screens (Integrated Stage 1–5 Platform)

1. **Executive Overview** — System readiness cards, model provenance cards, and end-to-end architecture pipeline.
2. **Configuration Manager** — Parameter selection, domain validation, relative risk indicator preview, and dominant geometric feature insight.
3. **EMC-PREDICT Engine** — Primary Frozen MATLAB & secondary ExtraTrees predictions, relative risk indicator metrics, and XAI geometric attribution breakdown.
4. **Digital Twin Geometry** — Dark-themed interactive 3D conductor spatial routing and 2D cross-section visualisation.
5. **EMC-GUARD Assessment & Pareto Optimisation** — 
   - *Tab 1 (Assessment)*: Relative risk ranking, distribution summary, and automated trade-off classification.
   - *Tab 2 (Intelligent Pareto Optimisation)*: Interactive multi-objective Pareto front scatter plot, 6 non-dominated candidate cards, recommended champions (Balanced, Min Capacitance, Min Mutual Inductance), and Pareto dataset CSV export.
6. **Configuration Comparison** — Multi-configuration comparison with automated Pareto trade-off and dominance evaluation.
7. **Model Validation & Explainable AI** — Actual MATLAB validation evidence (P-vs-A, residual histograms, LOLO CV with physical boundary explanations) plus XAI Feature Importance (MDI & PFI) and OAT dynamic sensitivity ranges.
8. **Report Generation & Data Packages** — Downloadable PDF engineering summary, direct technical report downloads (Stage 2 XAI, Stage 3 Risk, Stage 4 Optimisation), and exported CSV dataset packages.

---

## Running the Application

From project root:
```bash
streamlit run EMC_PROJECT/app.py
```
Or directly using specific Python environment:
```powershell
& "D:\Python\python.exe" -m streamlit run EMC_PROJECT/app.py
```

---

*Platform Version: 2.0.0 (Stage 1–5 Integrated) · Model Stage: 12A (Frozen MATLAB) / 14A (ExtraTrees) · Dataset: 216 configurations*


