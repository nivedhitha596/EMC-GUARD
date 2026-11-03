# KONE EMC-GUARD — Stage 4: Intelligent Configuration Optimisation Report

**Document Reference:** KONE-EMC-GUARD-STAGE4-OPTIMISATION  
**Date:** 2026-11-03  
**Status:** Approved for Professional Engineering Review  
**Subject:** Multi-Objective Pareto Optimisation of Cable Coupling Configurations  

---

## 1. Executive Summary

Stage 4 introduces a formal **Multi-Objective Pareto Optimisation Engine** within `AI_ML/src/optimise.py`. The engine evaluates trade-offs between two competing physical objectives:
1. **Minimising Total Capacitance** ($C_{\text{total}}$, pF/m) — Prevents capacitive charge loading, high-frequency common-mode leakage, and signal attenuation.
2. **Minimising Absolute Mutual Inductance** ($|M|$, nH) — Prevents magnetic induction and crosstalk from high-current hoist motor lines into sensitive sensor/encoder lines.

### Key Optimization Findings
- **Pareto Optimal Frontier:** Out of 216 discrete configurations, exactly **6 configurations (2.8%)** are non-dominated.
- **Universal Optimal Parameters:** All 6 Pareto-optimal designs share three identical parameters:
  - $\text{PairPitch\_mm} = 2.0\text{ mm}$ (Minimum twist pitch gives maximum field cancellation without increasing capacitance).
  - $\text{Overlap\_mm} = 150.0\text{ mm}$ (Shortest parallel run gives lowest exposed coupling length).
  - $\text{GroundClearance\_mm} = 100.0\text{ mm}$ (Maximum ground clearance minimises earth image capacitance).
- **The Core Trade-off Dimension:** The Pareto trade-off is governed exclusively by **Pair Separation** ($8\text{ mm} \to 28\text{ mm}$):
  - At $\text{Sep} = 8\text{ mm}$, capacitance is minimum ($29.47\text{ pF/m}$), but mutual inductance is higher ($1.9335\text{ nH}$).
  - Increasing separation to $\text{Sep} = 28\text{ mm}$ dramatically slashes mutual inductance by **$92.2\%$** (down to $0.1509\text{ nH}$) at the cost of a modest **$+16.7\%$** increase in capacitance ($34.38\text{ pF/m}$).

---

## 2. Non-Dominated Pareto Optimal Set

| Config ID | Separation | Pitch | Overlap | Ground Clearance | Capacitance ($C$) | Mutual Inductance ($|M|$) | Equal-Weight Score | Candidate Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`EMC-010`** | $8\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | **$29.47\text{ pF/m}$** | $1.9335\text{ nH}$ | 0.500 | **Minimum Capacitance Champion** |
| **`EMC-046`** | $12\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $30.90\text{ pF/m}$ | $0.8425\text{ nH}$ | **0.340** | **Recommended Balanced Design (Primary)** |
| **`EMC-082`** | $16\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $32.00\text{ pF/m}$ | $0.4698\text{ nH}$ | 0.348 | **Balanced Design (Moderate Spacing)** |
| **`EMC-118`** | $20\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $32.91\text{ pF/m}$ | $0.2989\text{ nH}$ | 0.392 | **Inductance-Favouring Balanced Design** |
| **`EMC-154`** | $24\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $33.69\text{ pF/m}$ | $0.2064\text{ nH}$ | 0.446 | **Ultra-Low Inductive Coupling Candidate** |
| **`EMC-190`** | $28\text{ mm}$ | $2\text{ mm}$ | $150\text{ mm}$ | $100\text{ mm}$ | $34.38\text{ pF/m}$ | **$0.1509\text{ nH}$** | 0.500 | **Minimum Inductance Champion** |

---

## 3. Engineering Analysis of Trade-Offs

### 3.1 Why No Single "Global" Optimum Exists
A design cannot simultaneously minimise capacitance and mutual inductance because pair separation exerts opposite physical pressures on the two coupling modes:
1. Narrower spacing ($8\text{ mm}$) decreases inter-wire capacitance because the projected field geometry between pairs has smaller dielectric cross-section.
2. Narrower spacing ($8\text{ mm}$) places conductors closer to the high magnetic flux field of the neighbouring pair ($B \propto 1/r$), increasing mutual inductance.

### 3.2 The Primary Recommendation: `EMC-046` ($12\text{ mm}$ Separation)
For general elevator hoistway signal trunking where neither mode is an extreme priority:
- Switching from the capacitance champion `EMC-010` ($8\text{ mm}$) to `EMC-046` ($12\text{ mm}$) incurs a minor capacitance penalty of only $+1.43\text{ pF/m}$ ($+4.9\%$).
- In exchange, it achieves an enormous **$56.4\%$ reduction in mutual inductance** (from $1.9335\text{ nH}$ down to $0.8425\text{ nH}$).
- Consequently, `EMC-046` achieves the lowest equal-weighted trade-off score ($0.340$) and represents the most cost-effective compromise.

### 3.3 The High-Sensitivity Recommendation: `EMC-190` ($28\text{ mm}$ Separation)
For high-sensitivity differential encoder, CAN bus, or weight sensor signals adjacent to variable-frequency motor drive cables:
- Inductive crosstalk ($V_{\text{induced}} = M \cdot \frac{di}{dt}$) is typically the dominant failure mechanism during motor acceleration.
- `EMC-190` reduces $M$ to $0.1509\text{ nH}$ (a $92.2\%$ reduction compared to $8\text{ mm}$), while total capacitance remains well below $35\text{ pF/m}$.

---

## 4. Visualisations Generated for Stage 4

The following standalone interactive Plotly visualisations have been compiled in `AI_ML/reports/figures/`:
1. `optimisation_pareto_front.html`: Interactive 2D scatter of all 216 configurations showing the 210 dominated points in light blue and the 6 non-dominated Pareto front configurations highlighted with step lines, diamond markers, and full hover annotations.
2. `optimisation_top_candidates.html`: Side-by-side bar charts of physical values alongside normalized trade-off curves for the 6 Pareto candidates.
3. `optimisation_geometry_comparison.html`: Multi-dimensional parallel coordinates plot illustrating how geometric choices (`Sep`, `Pitch`, `Overlap`, `GroundClearance`) funnel into Pareto optimality.

---

## 5. Scientific Limitations & Boundaries

1. **Discrete Grid Boundaries:** All candidates are selected strictly from the 216 pre-simulated grid points. The algorithm does not interpolate or extrapolate continuous intermediate parameters (e.g. $14.5\text{ mm}$ separation).
2. **Elevator Duct Space Constraints:** While `EMC-190` ($28\text{ mm}$ separation) is optimal for inductance, physical elevator travelling cables or narrow raceways may lack the physical cross-sectional space for $28\text{ mm}$ spacing.
3. **No Safety Certification:** Inclusion on the Pareto front signifies mathematical non-dominance in simulation, not certified immunity against electromagnetic transients in the field.

