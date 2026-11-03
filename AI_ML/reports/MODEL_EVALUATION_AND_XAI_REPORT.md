# KONE EMC-GUARD — Stage 2: AI Model Evaluation and Explainable AI (XAI) Report

**Document Reference:** KONE-EMC-GUARD-STAGE2-EVAL-XAI  
**Date:** 2026-11-03  
**Status:** Approved for Professional Engineering Review  
**Subject:** Rigorous Evaluation, Validation, and Explainability Analysis of Stage 14A Extra Trees Surrogate Models for Cable Crosstalk Prediction  

---

## 1. Executive Summary

As part of the **KONE EMC-GUARD** initiative, Stage 2 focuses on the comprehensive evaluation and interpretability of the machine learning surrogate models developed to approximate cable electromagnetic compatibility (EMC) parameters. 

The evaluation targets two distinct Extra Trees regressors serialised in Stage 14A:
1. **Total Capacitance Model** ($C_{\text{total}}$, in pF/m): Function of `PairSeparation_mm`, `PairPitch_mm`, and `GroundClearance_mm`.
2. **Absolute Mutual Inductance Model** ($|M|$, in nH): Function of `PairSeparation_mm`, `PairPitch_mm`, and `Overlap_mm`.

### Key Findings
- **In-Sample Reproduction:** The Extra Trees surrogates achieve near-perfect fidelity against the 216 deterministic analytical MATLAB training points ($R^2 = 1.000000$, $\text{MAE} < 10^{-12}$). This demonstrates that the decision-tree ensemble accurately acts as a surrogate for the analytical simulation engine.
- **Leave-One-Level-Out (LOLO) Generalisation:** Rigorous LOLO cross-validation reveals that the model generalises well across interior interpolation points (e.g. `PairSeparation_mm` with mean $R^2 = 0.9825$ for capacitance and $R^2 = 0.6228$ for mutual inductance). However, tree models exhibit significant degradation when withholding extreme outer boundary levels (notably `GroundClearance_mm = 10mm` and `100mm`, where step-function predictions cannot extrapolate steep non-linear asymptotic physical decays).
- **Dominant Features:**
  - For **Capacitance**, `GroundClearance_mm` overwhelmingly dominates the model with **92.8% Mean Decrease Impurity (MDI)** and a Permutation Importance $R^2$ drop of $1.8755$.
  - For **Mutual Inductance**, `PairSeparation_mm` is the primary factor (**58.4% MDI**, $R^2$ drop of $1.4633$), followed by `PairPitch_mm` (**25.8% MDI**) and `Overlap_mm` (**15.8% MDI**).
- **Physical Consistency:** Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) curves confirm that model responses strictly mirror qualitative electromagnetic field laws: increasing pair separation monotonically decreases mutual inductive coupling, increasing ground clearance decreases image capacitance to earth, and longer overlap monotonically increases mutual flux linkage.

---

## 2. Dataset & Evaluation Methodology

### 2.1 Authoritative Design Grid
The evaluation dataset consists of all **216 unique configurations** generated systematically in Stage 13A using MATLAB analytical field equations across a full factorial grid:
- **PairSeparation_mm** $\in \{8, 12, 16, 20, 24, 28\}$ (6 levels)
- **PairPitch_mm** $\in \{2, 3, 4\}$ (3 levels)
- **Overlap_mm** $\in \{150, 300, 450\}$ (3 levels)
- **GroundClearance_mm** $\in \{10, 25, 50, 100\}$ (4 levels)

### 2.2 In-Sample Reproduction vs Out-of-Sample Generalisation
- **In-Sample Evaluation:** Assesses surrogate fidelity. Because the MATLAB dataset contains zero measurement noise (pure analytical calculation), high in-sample $R^2$ indicates low surrogate bias.
- **Leave-One-Level-Out (LOLO) Cross-Validation:** For each discrete feature, all samples containing a given level (e.g., all 54 rows where `GroundClearance_mm = 10`) are held out. A temporary surrogate is trained on the remaining levels and evaluated exclusively on the unseen level. This tests true geometric generalisation and parameter interpolation/extrapolation.

---

## 3. Quantitative Evaluation Metrics

### 3.1 In-Sample Surrogate Fidelity (Full 216-Configuration Grid)

| Target Metric | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | Coefficient of Determination ($R^2$) |
| :--- | :--- | :--- | :--- | :--- |
| **TotalCapacitance_pF_per_m** | $9.2716 \times 10^{-13}\text{ pF/m}$ | $1.1904 \times 10^{-24}$ | $1.0910 \times 10^{-12}\text{ pF/m}$ | **1.00000000** |
| **AbsMutualInductance_nH** | $2.1125 \times 10^{-14}\text{ nH}$ | $2.1219 \times 10^{-27}$ | $4.6064 \times 10^{-14}\text{ nH}$ | **1.00000000** |

*Note: In-sample residuals are at numerical precision limits ($< 10^{-12}$), verifying that the Extra Trees estimators have completely learned the tabular grid.*

---

### 3.2 Leave-One-Level-Out (LOLO) Cross-Validation Results

#### Total Capacitance ($C_{\text{total}}$, pF/m)
| Withheld Feature | Levels Evaluated | Mean MAE (pF/m) | Mean RMSE (pF/m) | Baseline RMSE (pF/m) | Mean $R^2$ | RMSE Imprv % |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **PairSeparation_mm** | 6 | 0.818 | 0.926 | 13.531 | **+0.9825** | **92.9%** |
| **PairPitch_mm** | 3 | 1.125 | 1.288 | 13.460 | **+0.9865** | **90.3%** |
| **GroundClearance_mm** | 4 | 9.067 | 9.126 | 15.325 | **-6.3893** | **23.7%** |

*Detailed Ground Clearance LOLO Breakdown:*
- `GroundClearance_mm = 10mm`: $\text{RMSE} = 17.98\text{ pF/m}$, $R^2 = -8.67$ (Severe boundary extrapolation error)
- `GroundClearance_mm = 25mm`: $\text{RMSE} = 7.65\text{ pF/m}$, $R^2 = -2.99$
- `GroundClearance_mm = 50mm`: $\text{RMSE} = 4.05\text{ pF/m}$, $R^2 = -1.35$
- `GroundClearance_mm = 100mm`: $\text{RMSE} = 6.83\text{ pF/m}$, $R^2 = -12.54$

#### Absolute Mutual Inductance ($|M|$, nH)
| Withheld Feature | Levels Evaluated | Mean MAE (nH) | Mean RMSE (nH) | Baseline RMSE (nH) | Mean $R^2$ | RMSE Imprv % |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **PairSeparation_mm** | 6 | 1.466 | 1.811 | 4.063 | **+0.6228** | **67.2%** |
| **PairPitch_mm** | 3 | 1.642 | 2.537 | 4.671 | **-0.2418** | **45.7%** |
| **Overlap_mm** | 3 | 1.129 | 1.792 | 4.642 | **+0.3855** | **58.6%** |

*Detailed Mutual Inductance Separation Breakdown:*
- `PairSeparation_mm = 8mm`: $\text{RMSE} = 7.27\text{ nH}$, $R^2 = -0.01$ (Boundary extrapolation at tight separation)
- `PairSeparation_mm = 12mm`: $\text{RMSE} = 2.18\text{ nH}$, $R^2 = +0.45$
- `PairSeparation_mm = 16mm`: $\text{RMSE} = 0.67\text{ nH}$, $R^2 = +0.82$ (High interpolation fidelity)
- `PairSeparation_mm = 20mm`: $\text{RMSE} = 0.28\text{ nH}$, $R^2 = +0.92$ (High interpolation fidelity)
- `PairSeparation_mm = 24mm`: $\text{RMSE} = 0.13\text{ nH}$, $R^2 = +0.97$ (High interpolation fidelity)
- `PairSeparation_mm = 28mm`: $\text{RMSE} = 0.33\text{ nH}$, $R^2 = +0.58$ (Boundary level)

---

## 4. Feature Importance Analysis

Feature contributions were evaluated using two complementary techniques:
1. **Mean Decrease Impurity (MDI):** Standard tree-based gini/variance reduction.
2. **Permutation Feature Importance (PFI):** Evaluated over 30 independent shuffles using $R^2$ drop metrics.

### 4.1 Capacitance Feature Ranking
| Rank | Feature | MDI Importance (%) | PFI Mean $R^2$ Drop | PFI Std | Physical Interpretation |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `GroundClearance_mm` | **92.78%** | **1.8755** | 0.1490 | Primary distance to reference ground plane; determines image charge capacitance. |
| **2** | `PairSeparation_mm` | **6.28%** | **0.1386** | 0.0111 | Distance between active pair conductors; secondary influence. |
| **3** | `PairPitch_mm` | **0.94%** | **0.0240** | 0.0028 | Twist pitch variation introduces minor geometric displacement. |

### 4.2 Mutual Inductance Feature Ranking
| Rank | Feature | MDI Importance (%) | PFI Mean $R^2$ Drop | PFI Std | Physical Interpretation |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `PairSeparation_mm` | **58.38%** | **1.4633** | 0.1857 | Governs spatial magnetic flux decay rate ($B \propto 1/r$). |
| **2** | `PairPitch_mm` | **25.79%** | **0.7072** | 0.0855 | Determines periodic spatial field reversal rate and cancellation efficacy. |
| **3** | `Overlap_mm` | **15.83%** | **0.4509** | 0.0828 | Integrated parallel run length ($M \propto L_{\text{overlap}}$). |

---

## 5. Explainable AI (XAI) & Sensitivity Analysis

### 5.1 One-at-a-Time (OAT) Sensitivity at Median Reference
Baseline Reference Configuration: `PairSeparation_mm = 20mm`, `PairPitch_mm = 3mm`, `GroundClearance_mm = 50mm`, `Overlap_mm = 300mm`.
- **Reference Output:**
  - $C_{\text{total, ref}} = 40.91\text{ pF/m}$
  - $|M|_{\text{ref}} = 1.362\text{ nH}$

#### Dynamic Ranges (Sensitivity Spread)
- **Capacitance:**
  - `GroundClearance_mm` sweep ($10 \to 100\text{ mm}$): Range $= 36.17\text{ pF/m}$ (**88.4%** of reference value)
  - `PairSeparation_mm` sweep ($8 \to 28\text{ mm}$): Range $= 7.36\text{ pF/m}$ (**18.0%** of reference value)
  - `PairPitch_mm` sweep ($2 \to 4\text{ mm}$): Range $= 2.23\text{ pF/m}$ (**5.4%** of reference value)
- **Mutual Inductance:**
  - `PairSeparation_mm` sweep ($8 \to 28\text{ mm}$): Range $= 8.40\text{ nH}$ (**616.6%** of reference value)
  - `PairPitch_mm` sweep ($2 \to 4\text{ mm}$): Range $= 1.84\text{ nH}$ (**135.2%** of reference value)
  - `Overlap_mm` sweep ($150 \to 450\text{ mm}$): Range $= 1.37\text{ nH}$ (**100.5%** of reference value)

### 5.2 Local Prediction Attribution (Representative Configurations)
- **High Capacitance Configuration (`EMC-205`):**
  - Geometry: `PairSeparation = 28mm`, `PairPitch = 4mm`, `GroundClearance = 10mm`.
  - Predicted $C_{\text{total}} = 76.76\text{ pF/m}$ (vs Baseline $42.27\text{ pF/m}$).
  - Attribution: `GroundClearance = 10mm` contributed $+22.21\text{ pF/m}$, `PairPitch = 4mm` contributed $+2.59\text{ pF/m}$, `PairSeparation = 28mm` contributed $+2.16\text{ pF/m}$.
- **High Inductive Coupling Configuration (`EMC-027`):**
  - Geometry: `PairSeparation = 8mm`, `PairPitch = 4mm`, `Overlap = 450mm`.
  - Predicted $|M| = 25.89\text{ nH}$ (vs Baseline $3.59\text{ nH}$).
  - Attribution: `PairSeparation = 8mm` contributed $+4.81\text{ nH}$, `Overlap = 450mm` contributed $+1.95\text{ nH}$, `PairPitch = 4mm` contributed $+0.83\text{ nH}$.

---

## 6. Generated Visualisation Catalog

All figures are compiled as standalone, responsive, interactive HTML plots located in `AI_ML/reports/figures/`:

| Figure Identifier | File Name | Plot Type & Contents |
| :--- | :--- | :--- |
| **FIG-01** | `actual_vs_predicted_capacitance.html` | Scatter plot of in-sample actual vs predicted with identity line for $C_{\text{total}}$. |
| **FIG-02** | `actual_vs_predicted_mutual_inductance.html` | Scatter plot of in-sample actual vs predicted with identity line for $|M|$. |
| **FIG-03** | `residuals_capacitance.html` | Residual error vs predicted capacitance with $\pm 2\sigma$ error limits. |
| **FIG-04** | `residuals_mutual_inductance.html` | Residual error vs predicted mutual inductance with $\pm 2\sigma$ error limits. |
| **FIG-05** | `error_distribution_capacitance.html` | Histogram with fitted Gaussian probability density function for capacitance. |
| **FIG-06** | `error_distribution_mutual_inductance.html` | Histogram with fitted Gaussian probability density function for mutual inductance. |
| **FIG-07** | `avp_by_pairseparation_mm_capacitance.html` | Actual vs predicted capacitance segmented and coloured by pair separation level. |
| **FIG-08** | `avp_by_pairseparation_mm_mutual inductan.html` | Actual vs predicted mutual inductance segmented by separation level. |
| **FIG-09** | `lolo_r2_totalcapacitance_pf_per_m.html` | Grouped bar chart showing LOLO $R^2$ validation score across all held-out levels. |
| **FIG-10** | `lolo_r2_absmutualinductance_nh.html` | Grouped bar chart showing LOLO $R^2$ scores across mutual inductance features. |
| **FIG-11** | `lolo_rmse_improvement.html` | Mean percentage RMSE improvement over dummy baseline across all features. |
| **FIG-12** | `error_distribution_capacitance_lolo.html` | Cumulative residual error distribution for capacitance LOLO validation. |
| **FIG-13** | `error_distribution_mutual_inductance_lolo.html` | Cumulative residual error distribution for mutual inductance LOLO validation. |
| **FIG-14** | `fi_mdi_capacitance.html` | Horizontal bar chart of Mean Decrease Impurity for capacitance. |
| **FIG-15** | `fi_pfi_capacitance.html` | Horizontal bar chart of Permutation Importance ($R^2$ drop) for capacitance. |
| **FIG-16** | `fi_mdi_mutual_inductance.html` | Horizontal bar chart of Mean Decrease Impurity for mutual inductance. |
| **FIG-17** | `fi_pfi_mutual_inductance.html` | Horizontal bar chart of Permutation Importance ($R^2$ drop) for mutual inductance. |
| **FIG-18** | `xai_global_permutation_importance.html` | Combined multi-target global permutation feature importance comparison. |
| **FIG-19** | `sensitivity_totalcapacitance_pf_.html` | One-at-a-time sensitivity parameter sweeps for capacitance. |
| **FIG-20** | `sensitivity_absmutualinductance_.html` | One-at-a-time sensitivity parameter sweeps for mutual inductance. |
| **FIG-21** | `pdp_totalcapacitance_pf_.html` & `pdp_absmutualinductance_.html` | Partial Dependence Plots showing mean effect and $\pm 1\sigma$ ICE uncertainty bands. |

---

## 7. Scientific Limitations & Engineering Caveats

Engineers utilising this evaluation suite must observe the following constraints:
1. **Absence of Empirical Measurement Data:** All evaluations reflect fidelity against **analytical MATLAB calculations**. No physical cable harness measurements from actual KONE elevator hoistways are present in this dataset.
2. **Extrapolation Invalidation:** Tree-based models (such as Extra Trees) produce constant piecewise step functions outside their training ranges. The model **must never be evaluated outside the bounded 216-point grid**.
3. **Severe Boundary Degradation in LOLO:** As shown in Section 3.2, withholding boundary levels such as `GroundClearance_mm = 10mm` produces negative $R^2$ values because the non-linear asymptote of image capacitance cannot be approximated by flat trees without boundary points.
4. **Distinction Between Model Attribution and Causation:** Feature rankings reflect mathematical splitting thresholds within the random forest ensemble. While the directions align with Maxwell's equations, empirical causality must not be asserted from surrogate weights alone.
5. **No Regulatory Certification:** The surrogate predictions are conceptual design tools and do not constitute compliance certification for IEC/EN 12015 or EN 12016 EMC elevator directives.

---

## 8. Summary of Created & Modified Deliverables

| Deliverable Path | Status | Purpose |
| :--- | :--- | :--- |
| `AI_ML/src/evaluate.py` | Created | Comprehensive model evaluation script (In-sample + LOLO CV). |
| `AI_ML/src/feature_importance.py` | Created | MDI and Permutation Importance analysis script. |
| `AI_ML/src/explainability.py` | Created | XAI module (Global PFI, ICE, PDP, OAT Sensitivity, Local attributions). |
| `AI_ML/src/visualise.py` | Created | Plotly visualisation pipeline generating 21 interactive HTML figures. |
| `AI_ML/reports/figures/*.html` | Generated | 21 standalone interactive visualisations. |
| `AI_ML/reports/*.csv & *.json` | Generated | Full machine-readable numerical metric tables and summaries. |
| `AI_ML/README.md` | Updated | Comprehensive Stage 2 documentation and usage guide. |
| `AI_ML/backups/README_pre_stage2.md` | Created | Safe backup of pre-existing README prior to Stage 2 updates. |
| `AI_ML/reports/MODEL_EVALUATION_AND_XAI_REPORT.md` | Created | This formal technical engineering review report. |

