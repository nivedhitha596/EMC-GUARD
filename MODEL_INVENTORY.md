# MODEL INVENTORY: EMC-PREDICT + EMC-GUARD DIGITAL TWIN
**Project:** Physics-Informed Configuration-Aware Digital Twin for Predictive EMC Pre-Compliance of Elevator Systems  
**Root Path:** `D:\Downloads\kone\`  
**Date of Audit:** November 2026  
**Status:** Frozen Stage 12A Models & Verification Complete

---

## 1. Executive Summary

This document provides a comprehensive inventory of all engineering assets, frozen models, scripts, datasets, and visualization outputs developed for the KONE elevator cable EMC digital twin project.

The platform predicts:
- **Total Capacitance ($C_{tot}$)** in pF/m
- **Absolute Mutual Inductance ($|M_{12}|$ / $M_{abs}$)** in nH

over a deterministic analytical design grid of **216 valid configurations**.

---

## 2. Directory Hierarchy & Asset Inventory

```
D:\Downloads\kone\
├── 01_MATLAB_Models\
│   ├── 06_Stage11A_Results\
│   │   ├── 07_Stage12A_ModelFreeze\
│   │   │   ├── Stage12A_Selected_Models_FROZEN.mat          [Authoritative Frozen Models]
│   │   │   ├── Stage12A_Selected_Model_Summary.csv          [Model Performance Metadata]
│   │   │   ├── Stage12A_Selected_Model_Predictions.csv      [216 Predictions vs Reference]
│   │   │   └── Stage12A_Freeze_Audit_Report.txt            [Audit Verification Log]
│   │   ├── 08_Stage12B_Results\
│   │   │   ├── Stage12B7_AllConfigurations.csv             [216 Evaluated Configurations]
│   │   │   ├── Stage12B7_Corrected_Pareto.csv              [6-Point Non-Dominated Pareto Set]
│   │   │   ├── Stage12B7_ParetoConfigurations.csv          [9-Point Baseline Pareto Set]
│   │   │   ├── Stage12B6_GlobalSensitivitySummary.csv      [Global Sensitivity Indices]
│   │   │   └── Stage12B7_BalancedDesign.csv                [Recommended Trade-Off Design]
│   │   └── Stage11A_Analytical_Dataset.mat                 [Original Multi-Stage Data]
│   ├── 09_Stage13A_Export\
│   │   ├── EMC_GUARD_Physics_Dataset.csv                   [Canonical 216-Config Physics File]
│   │   ├── MATLAB_216_Verification.csv                     [Frozen Verification Values]
│   │   └── EMC_GUARD_Metadata.json                         [Domain Constraints & Units]
│   └── 10_Stage14A_AI_ML\
│       └── models\stage14A_4_final\
│           ├── EMC_GUARD_Capacitance_ExtraTrees.joblib     [Python Surrogate - Capacitance]
│           ├── EMC_GUARD_MutualInductance_ExtraTrees.joblib[Python Surrogate - Inductance]
│           ├── EMC_GUARD_Production_Metadata.json          [Surrogate Model Feature Meta]
│           └── Stage14A_Verification_Report.csv            [AI/ML Cross-Validation Metrics]
├── EMC_PROJECT\                                            [Streamlit Digital Twin App]
│   ├── app.py                                              [9-Screen Streamlit Dashboard]
│   ├── requirements.txt                                    [Python Dependencies]
│   ├── data\                                               [Embedded Reference Datasets]
│   ├── models\                                             [Loaders & Frozen Model Matrices]
│   ├── core\                                               [EMC-GUARD Scoring & PDF Reports]
│   ├── digital_twin\                                       [Plotly 3D Geometry Visualizer]
│   └── tests\test_suite.py                                 [Automated Regression Tests]
└── KONE_FINAL_RESULTS\                                     [Publication-Quality Visualisations]
    ├── KONE_Final_Engineering_Demo.m                       [Authoritative MATLAB Automation Script]
    ├── KONE_Final_Engineering_Demo.py                      [Python Matplotlib Reproduction Script]
    ├── A1_GlobalSensitivity_ConditionalRange.png           [Sensitivity Range Plots]
    ├── A2_SpearmanCorrelation.png                          [Spearman Correlation Heatmaps]
    ├── A3_SensitivitySweeps.png                            [1D Parameter Sensitivity Curves]
    ├── B1_Pareto_Front.png                                 [Objective Space Pareto Front]
    ├── B2_Pareto_BySeparation.png                          [Pareto Breakdown by Separation]
    ├── C1_ModelAgreement_PredVsActual.png                  [Predictions vs Analytical Truth]
    ├── D1_ResidualHistograms.png                           [Error Distribution Histograms]
    ├── D2_ResidualVsPredicted.png                          [Residual Homoscedasticity Plots]
    ├── E1_ResponseSurface_Sep_GC.png                       [Response Surface (Sep vs GC)]
    ├── E2_ResponseSurface_Sep_Pitch.png                    [Response Surface (Sep vs Pitch)]
    ├── F1_3D_Cable_Geometry.png                            [Annotated 3D Cable Assembly]
    ├── F2_CrossSection_MultipleConfigs.png                 [2D Cross-Section Profiles]
    └── *.csv                                               [Exported Pareto & Metrics Tables]
```

---

## 3. Detailed File Breakdown

### 3.1 Authoritative Frozen Models (`Stage12A`)
- **`Stage12A_Selected_Models_FROZEN.mat`**
  - **Format:** MATLAB v7.3 HDF5 binary.
  - **Contents:**
    - `betaCap`: $94 \times 1$ coefficient vector for model `B2_SP_Ground`.
    - `Xcap`: $94 \times 216$ basis matrix evaluated across all design points.
    - `yCap`: $216 \times 1$ analytical reference capacitance values.
    - `betaMut`: $84 \times 1$ coefficient vector for model `B1_SP_Overlap`.
    - `Xmut`: $84 \times 216$ basis matrix evaluated across all design points.
    - `yMut`: $216 \times 1$ analytical reference mutual inductance values.
  - **Verification:** Pooled CV RMSE = $5.46 \times 10^{-14}$ pF/m (Capacitance), $1.07 \times 10^{-14}$ nH (Inductance).

### 3.2 Reference Datasets (`Stage12B` & `Stage13A`)
- **`EMC_GUARD_Physics_Dataset.csv` / `Stage12B7_AllConfigurations.csv`**:
  - Contains all 216 combinations of:
    - `PairSeparation_mm`: $\{8, 12, 16, 20, 24, 28\}$
    - `PairPitch_mm`: $\{2, 3, 4\}$
    - `Overlap_mm`: $\{150, 300, 450\}$
    - `GroundClearance_mm`: $\{10, 25, 50, 100\}$
  - Response variables: `TotalCapacitance_pF_per_m` and `AbsMutualInductance_nH`.
- **`Stage12B7_Corrected_Pareto.csv`**:
  - The 6 non-dominated configurations minimizing both capacitance and mutual inductance.
  - Constant factors on the Pareto front: `PairPitch_mm = 2`, `Overlap_mm = 150`, `GroundClearance_mm = 100`.

### 3.3 Python AI/ML Surrogates (`Stage14A`)
- **`EMC_GUARD_Capacitance_ExtraTrees.joblib`**: ExtraTrees regressor trained on `[Separation, Pitch, GroundClearance]`.
- **`EMC_GUARD_MutualInductance_ExtraTrees.joblib`**: ExtraTrees regressor trained on `[Separation, Pitch, Overlap]`.
- **`EMC_GUARD_Production_Metadata.json`**: Feature sets, scalers, and training hyperparameters.

---

## 4. Verification and Audit Reports
- **`Stage12A_Freeze_Audit_Report.txt`**: Confirms model rank, conditioning ($135.64$ for Cap, $103.79$ for Mut), and non-singularity.
- **`Stage14A_Verification_Report.csv`**: Confirms high agreement between MATLAB linear surrogate and Python ExtraTrees.

---

## 5. Scope and Regulatory Boundaries

> [!IMPORTANT]
> - All models are calibrated on an internal analytical 216-point dataset.
> - The digital twin does **not** simulate victim circuit terminal voltages, transient surge waveforms, or conducted/radiated emissions according to EN 12015 / EN 12016.
> - Predictions indicate **relative coupling reduction** between design configurations, not certified compliance.
