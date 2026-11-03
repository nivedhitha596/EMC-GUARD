# KONE EMC-GUARD — Stage 3: Relative EMC Risk Assessment Report

**Document Reference:** KONE-EMC-GUARD-STAGE3-RISK-ASSESSMENT  
**Date:** 2026-11-03  
**Status:** Approved for Professional Engineering Review  
**Subject:** Relative Electromagnetic Coupling Risk Assessment Across 216 Elevator Cable Configurations  

---

## 1. Executive Summary

Stage 3 implements a rigorous, transparent **EMC Risk Assessment Engine** within the `AI_ML` module. The assessment calculates normalised relative indicators for **Total Capacitance** ($C_{\text{total}}$, in pF/m) and **Absolute Mutual Inductance** ($|M|$, in nH), combining them into an overall relative coupling risk index.

### Key Classification Results
Across the authoritative 216-configuration MATLAB design grid:
- **Low Relative Indicator Tier:** **155 configurations (71.8%)**  
  Coupling values lie in the lower third of the relative simulated parameter space ($I_{\text{comb}} \le 0.333$).
- **Moderate Relative Indicator Tier:** **60 configurations (27.8%)**  
  Coupling values exhibit intermediate trade-offs ($0.333 < I_{\text{comb}} \le 0.667$).
- **High Relative Indicator Tier:** **1 configuration (0.5%)**  
  Only configuration `EMC-027` exhibits severe compounded coupling on both metrics ($I_{\text{comb}} = 0.8231 > 0.667$).

---

## 2. Mathematical Methodology & Classification Thresholds

### 2.1 Normalised Relative Indicators
Because the MATLAB dataset represents deterministic field calculations without empirical noise, relative indicators $I_C, I_M \in [0, 1]$ are formulated as linear normalisations over the analytical extrema:

$$I_C = \frac{C - C_{\min}}{C_{\max} - C_{\min}}, \quad I_M = \frac{|M| - |M|_{\min}}{|M|_{\max} - |M|_{\min}}$$

Where:
- $C_{\min} = 29.4676\text{ pF/m}$, $C_{\max} = 76.7644\text{ pF/m}$ ($\Delta C = 47.2968\text{ pF/m}$)
- $|M|_{\min} = 0.1509\text{ nH}$, $|M|_{\max} = 25.8878\text{ nH}$ ($\Delta |M| = 25.7369\text{ nH}$)

### 2.2 Combined Relative Risk Indicator
To balance capacitive (electric field) and inductive (magnetic field) coupling pathways, a weighted combined indicator is evaluated:

$$I_{\text{comb}} = w_C \cdot I_C + w_M \cdot I_M \quad (w_C = 0.5, w_M = 0.5 \text{ by default})$$

Additionally, the normalised Euclidean distance to the ideal zero-coupling utopia point is tracked:

$$I_{\text{dist}} = \frac{\sqrt{I_C^2 + I_M^2}}{\sqrt{2}}$$

### 2.3 Tier Definitions
- **Low Relative Indicator:** $I_{\text{comb}} \in [0.00, 0.333]$
- **Moderate Relative Indicator:** $I_{\text{comb}} \in (0.333, 0.667]$
- **High Relative Indicator:** $I_{\text{comb}} \in (0.667, 1.000]$

> **Engineering Note:** These thresholds are relative statistical divisions within the bounded 216-grid. The software architecture supports programmatic injection of certified customer/regulatory limits (e.g. maximum permissible capacitive line loading or inductive pickup microvolts) as physical requirements emerge.

---

## 3. Explainable AI (XAI) Attribution of Risk Tiers

Connecting Stage 2 XAI findings directly to risk classifications explains why specific configurations fall into their respective tiers:

### Capacitance Driving Mechanisms (92.8% Driven by `GroundClearance_mm`)
1. **Low Clearance ($10\text{ mm}$):** Strong electric field line concentration to the reference metal tray or elevator shaft wall significantly amplifies earth image capacitance. Configurations with $10\text{ mm}$ clearance average $62.4\text{ pF/m}$, universally elevating the capacitance indicator into Moderate or High.
2. **High Clearance ($100\text{ mm}$):** Dielectric spacing isolates active pairs from the ground plane, dropping average capacitance to $32.4\text{ pF/m}$ (a $\sim 48\%$ drop), placing these designs in the Low Capacitance tier.

### Mutual Inductance Driving Mechanisms (58.4% Separation, 25.8% Pitch, 15.8% Overlap)
1. **Compounded High Inductance:** Occurs when **Pair Separation is tight ($8\text{ mm}$)**, **Twist Pitch is loose ($4\text{ mm}$)**, and **Parallel Overlap is maximum ($450\text{ mm}$)**. This compounding generates the absolute maximum inductance of $25.89\text{ nH}$ in `EMC-027`.
2. **Inductance Suppression:** Expanding separation to $28\text{ mm}$ drops inductive coupling to $0.1509\text{ nH}$ (a $99.4\%$ reduction from worst-case). Tightening twist pitch to $2\text{ mm}$ provides rapid spatial magnetic cancellation.

---

## 4. Configuration Rankings

### Top 5 Most Favourable Configurations (Lowest Coupling Indicators)

| Rank | Config ID | Separation | Pitch | Overlap | Ground Clearance | Capacitance | Mutual Inductance | Combined Indicator |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `EMC-046` | $12\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $30.90\text{ pF/m}$ | $0.8425\text{ nH}$ | **0.0286** |
| **2** | `EMC-082` | $16\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $32.00\text{ pF/m}$ | $0.4698\text{ nH}$ | **0.0330** |
| **3** | `EMC-010` | $8\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $29.47\text{ pF/m}$ | $1.9335\text{ nH}$ | **0.0346** |
| **4** | `EMC-118` | $20\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $32.91\text{ pF/m}$ | $0.2989\text{ nH}$ | **0.0393** |
| **5** | `EMC-083` | $16\text{ mm}$ | $2\text{ mm}$ | $300\text{ mm}$ | $100\text{ mm}$ | $32.00\text{ pF/m}$ | $0.9436\text{ nH}$ | **0.0422** |

### Top 5 Unfavourable Configurations (Highest Coupling Indicators)

| Rank | Config ID | Separation | Pitch | Overlap | Ground Clearance | Capacitance | Mutual Inductance | Combined Indicator |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **216** | `EMC-027` | $8\text{ mm}$ | $4\text{ mm}$ | $450\text{ mm}$ | $10\text{ mm}$ | $60.02\text{ pF/m}$ | $25.8878\text{ nH}$ | **0.8231** |
| **215** | `EMC-030` | $8\text{ mm}$ | $4\text{ mm}$ | $450\text{ mm}$ | $25\text{ mm}$ | $44.46\text{ pF/m}$ | $25.8878\text{ nH}$ | **0.6585** |
| **214** | `EMC-026` | $8\text{ mm}$ | $4\text{ mm}$ | $300\text{ mm}$ | $10\text{ mm}$ | $60.02\text{ pF/m}$ | $17.2556\text{ nH}$ | **0.6554** |
| **213** | `EMC-063` | $12\text{ mm}$ | $4\text{ mm}$ | $450\text{ mm}$ | $10\text{ mm}$ | $65.60\text{ pF/m}$ | $10.5969\text{ nH}$ | **0.5850** |
| **212** | `EMC-033` | $8\text{ mm}$ | $4\text{ mm}$ | $450\text{ mm}$ | $50\text{ mm}$ | $36.43\text{ pF/m}$ | $25.8878\text{ nH}$ | **0.5736** |

---

## 5. Visualisations Generated for Stage 3

The following standalone interactive Plotly visualisations have been compiled in `AI_ML/reports/figures/`:
1. `risk_capacitance_indicator.html`: Sorted capacitance values with threshold cutoff lines.
2. `risk_mutual_inductance_indicator.html`: Sorted mutual inductance values with threshold cutoff lines.
3. `risk_combined_comparison.html`: Multi-parameter box plots illustrating ground clearance and separation interaction effects.
4. `risk_configuration_ranking.html`: Head-to-head bar comparison of top 15 most favourable vs top 15 highest coupling configurations.
5. `risk_capacitance_vs_inductance.html`: 2D scatter of capacitance vs mutual inductance with bubble sizing and color-coded risk tiers.
6. `risk_distribution_chart.html`: Dual-panel pie chart and bar distribution of the 216 configurations by risk tier.

---

## 6. Scientific Limitations & Boundaries

1. **Analytical Dataset Origin:** All rankings reflect deterministic simulation calculations. Real-world elevator installations will introduce variable cable shielding, ground bonding impedances, and structural reflections not captured here.
2. **Relative Assessment Only:** A "Low Relative Indicator" indicates lower coupling within the 216 evaluated permutations, but does not guarantee absence of EMC interference in a high-noise inverter environment.
3. **No Safety Certification:** This assessment does not substitute for physical compliance testing under EN 12015 (Emission) or EN 12016 (Immunity).

