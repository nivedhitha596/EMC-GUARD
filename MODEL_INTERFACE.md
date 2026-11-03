# MODEL INTERFACE SPECIFICATION: EMC-PREDICT + EMC-GUARD
**Document:** Exact Mathematical and Programmatic Interface for Frozen Surrogate Models  
**Reference Model File:** `Stage12A_Selected_Models_FROZEN.mat`  
**Host Application:** Python 3.10+ / MATLAB R2019b+  

---

## 1. Mathematical Formulation

### 1.1 Capacitance Prediction Model: `B2_SP_Ground`
- **Target Output:** Total Capacitance ($C_{tot}$) in **$\text{pF/m}$**.
- **Model Type:** Full-rank linear basis expansion.
- **Number of Basis Terms:** $P = 94$.
- **Condition Number:** $135.64$.
- **Rank:** $94$ (non-singular).
- **Equation:**
  $$\hat{y}_{\text{Cap}} = \boldsymbol{\beta}_{\text{Cap}}^T \mathbf{x}_{\text{Cap}}(s, p, o, h) = \sum_{j=1}^{94} \beta_{\text{Cap}, j} \, X_{\text{Cap}, j}$$
  where:
  - $\boldsymbol{\beta}_{\text{Cap}} \in \mathbb{R}^{94 \times 1}$ is the frozen coefficient vector.
  - $\mathbf{x}_{\text{Cap}} \in \mathbb{R}^{94 \times 1}$ is the pre-computed basis expansion vector for configuration $(s, p, o, h)$.
  - $s$: Cable Separation ($8, 12, 16, 20, 24, 28\text{ mm}$).
  - $p$: Cable Pitch ($2, 3, 4\text{ mm}$).
  - $o$: Cable Overlap ($150, 300, 450\text{ mm}$).
  - $h$: Ground Clearance ($10, 25, 50, 100\text{ mm}$).

### 1.2 Absolute Mutual Inductance Model: `B1_SP_Overlap`
- **Target Output:** Absolute Mutual Inductance ($|M_{12}|$) in **$\text{nH}$**.
- **Model Type:** Full-rank linear basis expansion.
- **Number of Basis Terms:** $P = 84$.
- **Condition Number:** $103.79$.
- **Rank:** $84$ (non-singular).
- **Equation:**
  $$\hat{y}_{\text{Mut}} = \boldsymbol{\beta}_{\text{Mut}}^T \mathbf{x}_{\text{Mut}}(s, p, o, h) = \sum_{j=1}^{84} \beta_{\text{Mut}, j} \, X_{\text{Mut}, j}$$
  where:
  - $\boldsymbol{\beta}_{\text{Mut}} \in \mathbb{R}^{84 \times 1}$ is the frozen coefficient vector.
  - $\mathbf{x}_{\text{Mut}} \in \mathbb{R}^{84 \times 1}$ is the pre-computed basis expansion vector for configuration $(s, p, o, h)$.

---

## 2. Frozen MAT File Structure (`Stage12A_Selected_Models_FROZEN.mat`)

The MAT file is formatted as **MATLAB v7.3 (HDF5 format)** and requires `h5py` in Python or native `load()` in MATLAB.

| Dataset / Key | Dimensions | Type | Engineering Meaning |
| :--- | :--- | :--- | :--- |
| `/betaCap` | $(94, 1)$ or $(94,)$ | `float64` | Frozen least-squares coefficients for Capacitance model |
| `/Xcap` | $(94, 216)$ | `float64` | Pre-evaluated basis design matrix for all 216 configurations |
| `/yCap` | $(216, 1)$ or $(216,)$ | `float64` | Target analytical reference capacitance values ($\text{pF/m}$) |
| `/betaMut` | $(84, 1)$ or $(84,)$ | `float64` | Frozen least-squares coefficients for Mutual Inductance model |
| `/Xmut` | $(84, 216)$ | `float64` | Pre-evaluated basis design matrix for all 216 configurations |
| `/yMut` | $(216, 1)$ or $(216,)$ | `float64` | Target analytical reference mutual inductance values ($\text{nH}$) |

---

## 3. Configuration Mapping & Index Alignment

The columns $k \in \{0, 1, \dots, 215\}$ of `/Xcap` and `/Xmut` are strictly indexed in exact correspondence with the 216 rows of `EMC_GUARD_Physics_Dataset.csv`:

$$\text{Column Index } k \iff \text{Row } k \text{ of Reference Physics Dataset}$$

### Configuration Ordering Rule:
1. `PairSeparation_mm` $\in \{8, 12, 16, 20, 24, 28\}$
2. `PairPitch_mm` $\in \{2, 3, 4\}$
3. `Overlap_mm` $\in \{150, 300, 450\}$
4. `GroundClearance_mm` $\in \{10, 25, 50, 100\}$

---

## 4. Python API Implementation

Located in `EMC_PROJECT/models/prediction_engine.py`:

```python
from models.prediction_engine import predict_frozen_matlab

result = predict_frozen_matlab(sep=12, pitch=2, overlap=150, ground=100)

print(result)
# {
#   "capacitance_pF_per_m": 1.7766,
#   "abs_mutual_inductance_nH": 6.8837,
#   "config_id": "CFG_016",
#   "column_index": 15,
#   "model_engine": "Stage12A_FrozenMATLAB",
#   "reference_cap": 1.7766,
#   "reference_mut": 6.8837
# }
```

---

## 5. Secondary AI/ML Fallback (Stage 14A ExtraTrees)

For runtime scenarios where HDF5 reading is bypassed or testing surrogate robustness:
- **Capacitance Surrogate:** `EMC_GUARD_Capacitance_ExtraTrees.joblib`
  - Inputs: `[PairSeparation_mm, PairPitch_mm, GroundClearance_mm]`
- **Mutual Inductance Surrogate:** `EMC_GUARD_MutualInductance_ExtraTrees.joblib`
  - Inputs: `[PairSeparation_mm, PairPitch_mm, Overlap_mm]`
- **Engine Flag:** When invoked, returns `"model_engine": "Stage14A_ExtraTrees"`.

---

## 6. Verification and Error Metrics

| Metric | Capacitance ($C_{tot}$) | Absolute Mutual Inductance ($M_{abs}$) |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | $1.350 \times 10^{-14}\text{ pF/m}$ | $4.613 \times 10^{-15}\text{ nH}$ |
| **Root Mean Squared Error (RMSE)** | $3.686 \times 10^{-14}\text{ pF/m}$ | $6.506 \times 10^{-15}\text{ nH}$ |
| **Maximum Absolute Error** | $2.665 \times 10^{-13}\text{ pF/m}$ | $3.553 \times 10^{-14}\text{ nH}$ |
| **$R^2$ Score** | $1.000000$ | $1.000000$ |

*Note: The machine-precision error values verify exact mathematical reproduction of the analytical training set.*

---

## 7. Strict Operational Constraints

1. **No Extrapolation:** Queries outside the discrete sets $\{8,12,16,20,24,28\}\text{ mm}$, $\{2,3,4\}\text{ mm}$, $\{150,300,450\}\text{ mm}$, and $\{10,25,50,100\}\text{ mm}$ are rejected with an explicit error.
2. **No Coefficient Mutation:** Model coefficients are read-only and immutable.
3. **No Compliance Assertions:** Outputs represent relative electromagnetic coupling estimates, not EMC certification.
