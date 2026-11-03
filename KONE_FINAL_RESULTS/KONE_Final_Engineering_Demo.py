"""
KONE_Final_Engineering_Demo.py
==============================
Python equivalent of KONE_Final_Engineering_Demo.m

Generates all 6 visualisation sections from the actual Stage 12A/12B CSV data
and saves figures to KONE_FINAL_RESULTS\.

Usage:
    python KONE_Final_Engineering_Demo.py

Requirements:
    pip install pandas numpy matplotlib scipy plotly kaleido
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")

# ─── PATHS ─────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path("D:/Downloads/kone/01_MATLAB_Models")
RESULTS_ROOT  = Path("D:/Downloads/kone/KONE_FINAL_RESULTS")
PATH_12B      = PROJECT_ROOT / "06_Stage11A_Results" / "08_Stage12B_Results"
PATH_12A      = PROJECT_ROOT / "06_Stage11A_Results" / "07_Stage12A_ModelFreeze"
PATH_11A      = PROJECT_ROOT / "06_Stage11A_Results"

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("  EMC-PREDICT + EMC-GUARD  –  Final Engineering Visualisation (Python)")
print("="*70 + "\n")

# ─── STYLE ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F9FA",
    "axes.edgecolor":     "#CCCCCC",
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         130,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
})

C_BLUE    = "#2566DA"
C_ORANGE  = "#E87722"
C_GREEN   = "#2CA02C"
C_RED     = "#D62728"
C_GREY    = "#7F7F7F"
C_GOLD    = "#BCBD22"
C_PURPLE  = "#9467BD"

def save(name: str):
    p = RESULTS_ROOT / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"   Saved {name}")

def notice_text(ax, text: str, y_frac: float = -0.12):
    ax.text(0.5, y_frac, text,
            transform=ax.transAxes, fontsize=8,
            color="#666666", ha="center", va="top",
            style="italic", wrap=True)

# ─── LOAD DATA ─────────────────────────────────────────────────────────────

print("[1/6] Loading data files...")

all_cfg     = pd.read_csv(PATH_12B / "Stage12B7_AllConfigurations.csv")
pareto_corr = pd.read_csv(PATH_12B / "Stage12B7_Corrected_Pareto.csv")
pareto_all  = pd.read_csv(PATH_12B / "Stage12B7_ParetoConfigurations.csv")
global_sens = pd.read_csv(PATH_12B / "Stage12B6_GlobalSensitivitySummary.csv")
factor_lvl  = pd.read_csv(PATH_12B / "Stage12B6_FactorLevelEffects.csv")
sep_sens    = pd.read_csv(PATH_12B / "Stage12B5_Separation_Sensitivity.csv")
pitch_sens  = pd.read_csv(PATH_12B / "Stage12B5_Pitch_Sensitivity.csv")
over_sens   = pd.read_csv(PATH_12B / "Stage12B5_Overlap_Sensitivity.csv")
gnd_sens    = pd.read_csv(PATH_12B / "Stage12B5_GroundClearance_Sensitivity.csv")
bal_rank    = pd.read_csv(PATH_12B / "Stage12B8_Balanced_Design_Ranking.csv")
model_pred  = pd.read_csv(PATH_12A / "Stage12A_Selected_Model_Predictions.csv")
model_summ  = pd.read_csv(PATH_12A / "Stage12A_Selected_Model_Summary.csv")
spearman    = pd.read_csv(PATH_11A / "Stage11A_Spearman_Results.csv")
sens_idx    = pd.read_csv(PATH_11A / "Stage11A_Sensitivity_Indices.csv")

print(f"   All configurations : {len(all_cfg)}")
print(f"   Pareto (corrected) : {len(pareto_corr)}")


def compute_metrics(actual, predicted):
    err  = predicted - actual
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    maxe = np.max(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((actual - actual.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return mae, rmse, maxe, r2


# ═══════════════════════════════════════════════════════════════════════════
# A. GLOBAL SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[A] Global sensitivity analysis...")

# ── A1: Average conditional range bar chart ─────────────────────────────────

cap_s = global_sens[global_sens["Response"] == "Capacitance"].copy()
mut_s = global_sens[global_sens["Response"] == "MutualInductance"].copy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("A1. Global Sensitivity – Average Conditional Range", fontsize=14, fontweight="bold")

for ax, data, col, ylabel, title_str in [
    (axes[0], cap_s, C_BLUE,   "Avg. Conditional Range (pF/m)", "Capacitance  (B2_SP_Ground)"),
    (axes[1], mut_s, C_ORANGE, "Avg. Conditional Range (nH)",   "Abs. Mutual Inductance  (B1_SP_Overlap)"),
]:
    data_sorted = data.sort_values("AverageConditionalRange", ascending=False)
    bars = ax.bar(data_sorted["Factor"], data_sorted["AverageConditionalRange"],
                  color=col, edgecolor="white", linewidth=0.8)
    if len(bars) > 0:
        bars[0].set_color(C_RED)
    for bar, val in zip(bars, data_sorted["AverageConditionalRange"].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * data_sorted["AverageConditionalRange"].max(),
                f"{val:.3g}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title_str, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Factor")

plt.figtext(0.5, 0.01,
    "Source: Stage12B6_GlobalSensitivitySummary.csv | Metric: avg conditional output range | Analytical dataset only",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
save("A1_GlobalSensitivity_ConditionalRange.png")

# ── A2: Spearman rank correlation ─────────────────────────────────────────

factors    = spearman["Row"].tolist()
cap_corr   = spearman["Capacitance"].values
mut_corr   = spearman["AbsMutualInductance"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("A2. Spearman Rank Correlation – Parameter Influence", fontsize=14, fontweight="bold")

for ax, data, col, title_str in [
    (axes[0], cap_corr, C_BLUE,   "Capacitance  (B2_SP_Ground)"),
    (axes[1], mut_corr, None,     "Abs. Mutual Inductance  (B1_SP_Overlap)"),
]:
    if col is None:
        colours = [C_RED if v < 0 else C_ORANGE for v in data]
    else:
        colours = [col] * len(data)
    bars = ax.barh(factors, data, color=colours, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_title(title_str, fontweight="bold")
    ax.set_xlabel("Spearman Rank Correlation (ρ)")
    for bar, v in zip(bars, data):
        ax.text(v + 0.02 * np.sign(v), bar.get_y() + bar.get_height()/2,
                f"{v:+.3f}", va="center", fontsize=9)

plt.figtext(0.5, 0.01,
    "Source: Stage11A_Spearman_Results.csv | Stage 11A sensitivity | Analytical dataset",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
save("A2_SpearmanCorrelation.png")

# ── A3: Individual sensitivity sweeps ────────────────────────────────────

sweep_data   = [sep_sens,   pitch_sens,   over_sens,    gnd_sens]
sweep_factor = ["Separation_mm", "Pitch_mm", "Overlap_mm", "GroundClearance_mm"]
sweep_xlabel = ["Separation (mm)", "Pitch", "Overlap (mm)", "Ground Clearance (mm)"]
sweep_title  = ["Separation Sweep", "Pitch Sweep", "Overlap Sweep", "GC Sweep"]
sweep_col    = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("A3. Individual Factor Sensitivity Sweeps (all others fixed)", fontsize=14, fontweight="bold")

for k, (sd, fx, xl, tt, col) in enumerate(zip(sweep_data, sweep_factor,
                                                sweep_xlabel, sweep_title, sweep_col)):
    fixed_fields = [f for f in sweep_factor if f != fx]
    fixed_str = ", ".join(f"{f}={sd[f].iloc[0]}" for f in fixed_fields)

    ax_cap = axes[0, k]
    ax_mut = axes[1, k]

    ax_cap.plot(sd[fx], sd["Capacitance_pF_per_m"], "o-", color=col, lw=2, ms=7, mfc=col)
    ax_cap.set_title(tt, fontweight="bold")
    ax_cap.set_xlabel(xl)
    ax_cap.set_ylabel("Capacitance (pF/m)")
    ax_cap.text(0.03, 0.97, fixed_str, transform=ax_cap.transAxes, fontsize=7.5,
                color="#555555", va="top")

    ax_mut.plot(sd[fx], sd["AbsMutualInductance_nH"], "s-", color=col, lw=2, ms=7, mfc=col)
    ax_mut.set_xlabel(xl)
    ax_mut.set_ylabel("Inductance (nH)")
    ax_mut.text(0.03, 0.97, fixed_str, transform=ax_mut.transAxes, fontsize=7.5,
                color="#555555", va="top")

plt.figtext(0.5, 0.005,
    "Source: Stage12B5 sensitivity CSVs | Frozen analytical model | Fixed factors shown per panel",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.025, 1, 0.96])
save("A3_SensitivitySweeps.png")


# ═══════════════════════════════════════════════════════════════════════════
# B. MULTI-OBJECTIVE PARETO OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n[B] Pareto optimisation plots...")

# ── B1: All configs + Pareto front ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("B1. Multi-Objective Pareto Front: Capacitance vs. Abs. Mutual Inductance",
             fontsize=14, fontweight="bold")

non_pareto = all_cfg[all_cfg["IsParetoOptimal"] == 0]
pareto_pts = all_cfg[all_cfg["IsParetoOptimal"] == 1]

ax.scatter(non_pareto["Capacitance_pF_per_m"], non_pareto["AbsMutualInductance_nH"],
           s=40, color=C_GREY, alpha=0.45, zorder=2,
           label=f"All configurations (n={len(non_pareto)})")

ax.scatter(pareto_corr["Capacitance_pF_per_m"], pareto_corr["AbsMutualInductance_nH"],
           s=140, color=C_ORANGE, marker="^", zorder=5, edgecolors="k", linewidths=0.8,
           label=f"Pareto-optimal – corrected (n={len(pareto_corr)})")

# Connect Pareto front
pc_sorted = pareto_corr.sort_values("Capacitance_pF_per_m")
ax.plot(pc_sorted["Capacitance_pF_per_m"], pc_sorted["AbsMutualInductance_nH"],
        "--", color=C_ORANGE, lw=1.5, zorder=4)

# Label Pareto points
for _, row in pareto_corr.iterrows():
    ax.annotate(f"Sep={int(row['Separation_mm'])}, GC={int(row['GroundClearance_mm'])}",
                xy=(row["Capacitance_pF_per_m"], row["AbsMutualInductance_nH"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8, color="#8B4513")

# Best balanced design
best = bal_rank.iloc[0]
ax.scatter(best["Capacitance_pF_per_m"], best["AbsMutualInductance_nH"],
           s=220, color=C_RED, marker="*", zorder=6, edgecolors="k", linewidths=0.8,
           label=f"Best balanced design (Sep={int(best['Separation_mm'])})")

ax.set_xlabel("Total Capacitance (pF/m)", fontsize=12)
ax.set_ylabel("Absolute Mutual Inductance (nH)", fontsize=12)
ax.legend(loc="upper right", framealpha=0.9)

plt.figtext(0.5, 0.003,
    "Source: Stage12B7_AllConfigurations.csv, Stage12B7_Corrected_Pareto.csv  |  "
    "Objective: minimise capacitance AND inductance  |  No EMC compliance claim implied",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 1])
save("B1_Pareto_Front.png")

# ── B2: Configuration space coloured by separation ─────────────────────────

fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle("B2. Configuration Space Coloured by Cable Separation",
             fontsize=14, fontweight="bold")

sep_vals = sorted(all_cfg["Separation_mm"].unique())
cmap = plt.cm.get_cmap("tab10", len(sep_vals))

for k, s in enumerate(sep_vals):
    grp = all_cfg[all_cfg["Separation_mm"] == s]
    ax.scatter(grp["Capacitance_pF_per_m"], grp["AbsMutualInductance_nH"],
               s=35, color=cmap(k), alpha=0.55, label=f"Sep = {int(s)} mm", zorder=2)

ax.scatter(pareto_corr["Capacitance_pF_per_m"], pareto_corr["AbsMutualInductance_nH"],
           s=130, color="black", marker="^", zorder=5, edgecolors="k",
           label="Pareto-optimal (corrected)")
pc_sorted = pareto_corr.sort_values("Capacitance_pF_per_m")
ax.plot(pc_sorted["Capacitance_pF_per_m"], pc_sorted["AbsMutualInductance_nH"],
        "k--", lw=1.5, zorder=4)

ax.set_xlabel("Total Capacitance (pF/m)", fontsize=12)
ax.set_ylabel("Absolute Mutual Inductance (nH)", fontsize=12)
ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=9)
plt.tight_layout()
save("B2_Pareto_BySeparation.png")

# Export Pareto table
pareto_corr.to_csv(RESULTS_ROOT / "B_Pareto_Optimal_Configurations.csv", index=False)
print("   Exported B_Pareto_Optimal_Configurations.csv")


# ═══════════════════════════════════════════════════════════════════════════
# C. MODEL AGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[C] Model agreement analysis...")

act_cap  = model_pred["ActualCapacitance_pF_m"].values
pred_cap = model_pred["PredictedCapacitance_pF_m"].values
act_mut  = model_pred["ActualAbsMutualInductance_nH"].values
pred_mut = model_pred["PredictedAbsMutualInductance_nH"].values

mae_c, rmse_c, maxe_c, r2_c = compute_metrics(act_cap, pred_cap)
mae_m, rmse_m, maxe_m, r2_m = compute_metrics(act_mut, pred_mut)

print(f"\n   Capacitance:          MAE={mae_c:.3e} pF/m  RMSE={rmse_c:.3e} pF/m  R²={r2_c:.8f}")
print(f"   Abs. Mut. Inductance: MAE={mae_m:.3e} nH   RMSE={rmse_m:.3e} nH   R²={r2_m:.8f}")
print("   (These metrics reflect dataset reproduction, not physical validation)")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("C1. Model Agreement: Predicted vs. Analytical Reference (Stage 12A)",
             fontsize=14, fontweight="bold")

for ax, actual, predicted, col, ylabel, title_str, mae, rmse, maxe, r2, unit in [
    (axes[0], act_cap, pred_cap, C_BLUE,   "Predicted (pF/m)", "Capacitance (B2_SP_Ground)",
     mae_c, rmse_c, maxe_c, r2_c, "pF/m"),
    (axes[1], act_mut, pred_mut, C_ORANGE, "Predicted (nH)",   "Abs. Mutual Inductance (B1_SP_Overlap)",
     mae_m, rmse_m, maxe_m, r2_m, "nH"),
]:
    ax.scatter(actual, predicted, s=45, color=col, alpha=0.6, zorder=3, label="Configurations")
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    margin = (hi - lo) * 0.04
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", lw=2, label="y = x (ideal)")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_title(title_str, fontweight="bold")
    ax.set_xlabel(f"Actual ({unit})")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    stats_str = (f"MAE   = {mae:.3e} {unit}\n"
                 f"RMSE  = {rmse:.3e} {unit}\n"
                 f"Max|e|= {maxe:.3e} {unit}\n"
                 f"R²    = {r2:.8f}")
    ax.text(0.04, 0.97, stats_str, transform=ax.transAxes, fontsize=9,
            va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#CCAA00", alpha=0.9))

plt.figtext(0.5, 0.003,
    "NOTICE: R² ≈ 1 and sub-femto residuals reflect EXACT least-squares reproduction of the analytical dataset, "
    "NOT independent experimental validation.  Source: Stage12A_Selected_Model_Predictions.csv",
    ha="center", fontsize=8, color="#9B5700",
    bbox=dict(facecolor="#FFF7E6", edgecolor="#CCAA00", alpha=0.8, pad=3))
plt.tight_layout(rect=[0, 0.06, 1, 1])
save("C1_ModelAgreement_PredVsActual.png")


# ═══════════════════════════════════════════════════════════════════════════
# D. RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[D] Residual analysis...")

res_c = model_pred["CapacitanceResidual_pF_m"].values
res_m = model_pred["AbsMutualInductanceResidual_nH"].values

# ── D1: Histograms ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("D1. Residual Distributions (Stage 12A Model)", fontsize=14, fontweight="bold")

for ax, res, col, title_str, unit in [
    (axes[0], res_c, C_BLUE,   "Capacitance Residuals",         "pF/m"),
    (axes[1], res_m, C_ORANGE, "Abs. Mutual Inductance Residuals", "nH"),
]:
    ax.hist(res, bins=20, color=col, edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero")
    ax.set_title(title_str, fontweight="bold")
    ax.set_xlabel(f"Residual ({unit})")
    ax.set_ylabel("Count")
    ax.legend()
    stats_str = (f"Mean  = {res.mean():.3e}\n"
                 f"Std   = {res.std():.3e}\n"
                 f"Max|e|= {np.abs(res).max():.3e}")
    ax.text(0.97, 0.97, stats_str, transform=ax.transAxes, fontsize=9,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4FF",
                      edgecolor="#AABBDD", alpha=0.9))

plt.figtext(0.5, 0.003,
    "Residuals at floating-point machine precision (~10⁻¹⁴ to 10⁻¹³). "
    "Reflects exact algebraic reconstruction, not measurement uncertainty.",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
save("D1_ResidualHistograms.png")

# ── D2: Residual vs. Predicted ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("D2. Residuals vs. Predicted Values", fontsize=14, fontweight="bold")

for ax, predicted, residuals, col, title_str, xunit, yunit in [
    (axes[0], pred_cap, res_c, C_BLUE,   "Capacitance",         "pF/m", "pF/m"),
    (axes[1], pred_mut, res_m, C_ORANGE, "Abs. Mutual Inductance", "nH",  "nH"),
]:
    ax.scatter(predicted, residuals, s=45, color=col, alpha=0.55, zorder=3)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(title_str, fontweight="bold")
    ax.set_xlabel(f"Predicted ({xunit})")
    ax.set_ylabel(f"Residual ({yunit})")

plt.tight_layout()
save("D2_ResidualVsPredicted.png")


# ═══════════════════════════════════════════════════════════════════════════
# E. ENGINEERING RESPONSE SURFACES
# ═══════════════════════════════════════════════════════════════════════════

print("\n[E] Engineering response surfaces...")

sep_lvls = sorted(all_cfg["Separation_mm"].unique())
gc_lvls  = sorted(all_cfg["GroundClearance_mm"].unique())
pitch_lvls = sorted(all_cfg["Pitch_mm"].unique())

# ── E1: Separation × Ground Clearance (Pitch=2, Overlap=300) ──────────────

sub = all_cfg[(all_cfg["Pitch_mm"] == 2) & (all_cfg["Overlap_mm"] == 300)].copy()

nS = len(sep_lvls)
nG = len(gc_lvls)
CapMat = np.zeros((nG, nS))
MutMat = np.zeros((nG, nS))
for i, s in enumerate(sep_lvls):
    for j, g in enumerate(gc_lvls):
        row = sub[(sub["Separation_mm"] == s) & (sub["GroundClearance_mm"] == g)]
        if not row.empty:
            CapMat[j, i] = row["Capacitance_pF_per_m"].values[0]
            MutMat[j, i] = row["AbsMutualInductance_nH"].values[0]

SEP, GC = np.meshgrid(sep_lvls, gc_lvls)

fig = plt.figure(figsize=(14, 5))
fig.suptitle("E1. Response Surface: Separation × Ground Clearance  (Pitch=2, Overlap=300 mm)",
             fontsize=13, fontweight="bold")

ax1 = fig.add_subplot(121, projection="3d")
surf1 = ax1.plot_surface(SEP, GC, CapMat, cmap="turbo", edgecolor="grey",
                          linewidth=0.3, alpha=0.9)
fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1, label="pF/m")
ax1.set_title("Capacitance (pF/m)", fontweight="bold")
ax1.set_xlabel("Separation (mm)")
ax1.set_ylabel("Ground Clearance (mm)")
ax1.set_zlabel("Capacitance (pF/m)")
ax1.view_init(elev=28, azim=-50)

ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(SEP, GC, MutMat, cmap="turbo", edgecolor="grey",
                          linewidth=0.3, alpha=0.9)
fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1, label="nH")
ax2.set_title("Abs. Mutual Inductance (nH)", fontweight="bold")
ax2.set_xlabel("Separation (mm)")
ax2.set_ylabel("Ground Clearance (mm)")
ax2.set_zlabel("Inductance (nH)")
ax2.view_init(elev=28, azim=-50)

plt.figtext(0.5, 0.003,
    "Fixed: Pitch=2, Overlap=300 mm  |  Discrete grid values only – no interpolation  |  "
    "Source: Stage12B7_AllConfigurations.csv",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
save("E1_ResponseSurface_Sep_GC.png")

# ── E2: Separation × Pitch (Overlap=300, GC=50) ──────────────────────────

sub2 = all_cfg[(all_cfg["Overlap_mm"] == 300) & (all_cfg["GroundClearance_mm"] == 50)].copy()

nP = len(pitch_lvls)
CapMat2 = np.zeros((nP, nS))
MutMat2 = np.zeros((nP, nS))
for i, s in enumerate(sep_lvls):
    for j, p in enumerate(pitch_lvls):
        row = sub2[(sub2["Separation_mm"] == s) & (sub2["Pitch_mm"] == p)]
        if not row.empty:
            CapMat2[j, i] = row["Capacitance_pF_per_m"].values[0]
            MutMat2[j, i] = row["AbsMutualInductance_nH"].values[0]

SEP2, PITCH = np.meshgrid(sep_lvls, pitch_lvls)

fig = plt.figure(figsize=(14, 5))
fig.suptitle("E2. Response Surface: Separation × Pitch  (Overlap=300 mm, GC=50 mm)",
             fontsize=13, fontweight="bold")

ax1 = fig.add_subplot(121, projection="3d")
surf1 = ax1.plot_surface(SEP2, PITCH, CapMat2, cmap="cool", edgecolor="grey",
                          linewidth=0.3, alpha=0.9)
fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1, label="pF/m")
ax1.set_title("Capacitance (pF/m)", fontweight="bold")
ax1.set_xlabel("Separation (mm)"); ax1.set_ylabel("Pitch")
ax1.set_zlabel("Capacitance (pF/m)"); ax1.view_init(elev=28, azim=-50)

ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(SEP2, PITCH, MutMat2, cmap="cool", edgecolor="grey",
                          linewidth=0.3, alpha=0.9)
fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1, label="nH")
ax2.set_title("Abs. Mutual Inductance (nH)", fontweight="bold")
ax2.set_xlabel("Separation (mm)"); ax2.set_ylabel("Pitch")
ax2.set_zlabel("Inductance (nH)"); ax2.view_init(elev=28, azim=-50)

plt.figtext(0.5, 0.003,
    "Fixed: Overlap=300 mm, GC=50 mm  |  Discrete grid values only",
    ha="center", fontsize=8, color="#666666", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
save("E2_ResponseSurface_Sep_Pitch.png")


# ═══════════════════════════════════════════════════════════════════════════
# F. 3D CABLE GEOMETRY VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n[F] 3D cable geometry visualisation...")

# Representative configuration: best balanced design
cfg = bal_rank.iloc[0]
cfg_sep   = int(cfg["Separation_mm"])
cfg_pitch = int(cfg["Pitch_mm"])
cfg_ol    = int(cfg["Overlap_mm"])
cfg_gc    = int(cfg["GroundClearance_mm"])

CABLE_R   = 3.0    # mm, representative only
N_THETA   = 48


def make_cylinder(cx, cy, length, r=CABLE_R, n=N_THETA):
    """Generate (X,Y,Z) arrays for a cylinder along Z."""
    theta = np.linspace(0, 2 * np.pi, n + 1)
    z     = np.array([-length / 2, length / 2])
    TH, Z = np.meshgrid(theta, z)
    X = cx + r * np.cos(TH)
    Y = cy + r * np.sin(TH)
    return X, Y, Z


fig = plt.figure(figsize=(13, 8), facecolor="#0F172A")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0F172A")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#334155")
ax.yaxis.pane.set_edgecolor("#334155")
ax.zaxis.pane.set_edgecolor("#334155")
ax.grid(True, alpha=0.2, color="#475569")

src_x = 0.0;      src_y = float(cfg_gc)
vic_x = cfg_sep;  vic_y = cfg_gc + cfg_pitch
src_len = cfg_ol + 60
vic_len = cfg_ol

# Source cable
Xs, Ys, Zs = make_cylinder(src_x, src_y, src_len)
ax.plot_surface(Xs, Ys, Zs, color="#2196F3", alpha=0.88, linewidth=0,
                antialiased=True)

# Victim cable
Xv, Yv, Zv = make_cylinder(vic_x, vic_y, vic_len)
ax.plot_surface(Xv, Yv, Zv, color="#FF9800", alpha=0.88, linewidth=0,
                antialiased=True)

# Ground plane
xg = np.array([[-CABLE_R - 12, vic_x + CABLE_R + 12],
                [-CABLE_R - 12, vic_x + CABLE_R + 12]])
zg = np.array([[-src_len/2 - 25, -src_len/2 - 25],
                [ src_len/2 + 25,  src_len/2 + 25]])
yg = np.zeros_like(xg)
ax.plot_surface(xg, yg, zg, color="#2D4A35", alpha=0.30, linewidth=0)

# Overlap boundaries
for zb in [-vic_len/2, vic_len/2]:
    ax.plot([src_x - CABLE_R - 3, vic_x + CABLE_R + 3],
            [src_y, vic_y], [zb, zb],
            "--", color="#EF4444", lw=1.5)

# Separation arrow
dim_z = src_len / 2 + 20
ax.plot([src_x, vic_x], [src_y, vic_y], [dim_z, dim_z],
        "-", color="#4ADE80", lw=2.5)
ax.text((src_x + vic_x)/2, (src_y + vic_y)/2, dim_z + 8,
        f"Sep = {cfg_sep} mm", color="#4ADE80", fontsize=10, fontweight="bold",
        ha="center")

# Ground clearance
gc_x = src_x - CABLE_R - 10
gc_z = -src_len / 2 + 25
ax.plot([gc_x, gc_x], [0, src_y], [gc_z, gc_z], "-", color="#CBD5E1", lw=2)
ax.text(gc_x - 2, src_y / 2, gc_z,
        f"GC = {cfg_gc} mm", color="#CBD5E1", fontsize=9, ha="right")

# Overlap label
ov_x = vic_x + CABLE_R + 12
ax.plot([ov_x, ov_x], [vic_y, vic_y], [-vic_len/2, vic_len/2],
        "-", color="#FCD34D", lw=2)
ax.text(ov_x + 4, vic_y, 0,
        f"OL = {cfg_ol} mm", color="#FCD34D", fontsize=9)

# Pitch
pt_z = vic_len / 2 + 15
ax.plot([vic_x, vic_x], [src_y, vic_y], [pt_z, pt_z],
        "-", color="#C084FC", lw=2)
ax.text(vic_x + 3, (src_y + vic_y)/2, pt_z,
        f"Pitch = {cfg_pitch}", color="#C084FC", fontsize=9)

# Labels
ax.set_xlabel("X – cross-cable (mm)", color="#94A3B8", labelpad=8)
ax.set_ylabel("Y – height (mm)",      color="#94A3B8", labelpad=8)
ax.set_zlabel("Z – cable run (mm)",   color="#94A3B8", labelpad=8)
ax.tick_params(colors="#64748B")
for l in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    l.set_color("#64748B")

ax.set_title(
    f"F1. Cable Geometry — Sep={cfg_sep} mm  |  Pitch={cfg_pitch}  |  "
    f"OL={cfg_ol} mm  |  GC={cfg_gc} mm",
    fontsize=12, fontweight="bold", color="#E2E8F0", pad=12,
)

src_patch   = mpatches.Patch(color="#2196F3", label="Source cable (blue)")
vic_patch   = mpatches.Patch(color="#FF9800", label="Victim cable (orange)")
gnd_patch   = mpatches.Patch(color="#2D4A35", label="Ground plane", alpha=0.5)
bnd_line    = plt.Line2D([0], [0], color="#EF4444", linestyle="--", lw=1.5,
                          label="Overlap boundary")
ax.legend(handles=[src_patch, vic_patch, gnd_patch, bnd_line],
          loc="upper right", fontsize=9, framealpha=0.25,
          labelcolor="white", facecolor="#1E293B", edgecolor="#334155")

ax.view_init(elev=22, azim=-42)

fig.text(0.5, 0.02,
    "ILLUSTRATIVE ENGINEERING REPRESENTATION: geometry parameterised by separation, pitch, "
    "overlap and ground clearance. NOT a verified KONE cross-section or EM field simulation.",
    ha="center", fontsize=8, color="#64748B",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", edgecolor="#334155", alpha=0.8))

plt.tight_layout()
save("F1_3D_Cable_Geometry.png")

# ── F2: Cross-section grid for representative configurations ─────────────

sample_configs = [
    (8,  2, 150,  10),
    (12, 2, 150, 100),
    (20, 3, 300,  50),
    (28, 4, 450, 100),
    (16, 3, 300,  25),
    (24, 2, 150,  50),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#0F172A")
fig.suptitle("F2. Cable Cross-Sections for Selected Configurations (Y–X plane)",
             fontsize=13, fontweight="bold", color="#E2E8F0")

theta = np.linspace(0, 2 * np.pi, 128)
col_src = "#2196F3"
col_vic = "#FF9800"
col_gnd = "#2D4A35"

for (ax, (s, p, ol, gc)) in zip(axes.flat, sample_configs):
    ax.set_facecolor("#0F172A")
    ax.set_aspect("equal")

    # Ground line
    x_span = [-CABLE_R - 8, s + CABLE_R + 8]
    ax.plot(x_span, [0, 0], "-", color=col_gnd, lw=2.5)
    ax.fill_between(x_span, [-2, -2], [0, 0], color=col_gnd, alpha=0.3)

    # Source cable
    ax.fill(0 + CABLE_R * np.cos(theta), gc + CABLE_R * np.sin(theta),
            color=col_src, alpha=0.85, edgecolor="#4FC3F7", lw=1.2)
    ax.text(0, gc, "SRC", color="white", fontsize=7, ha="center", va="center",
            fontweight="bold")

    # Victim cable
    ax.fill(s + CABLE_R * np.cos(theta), (gc + p) + CABLE_R * np.sin(theta),
            color=col_vic, alpha=0.85, edgecolor="#FFB74D", lw=1.2)
    ax.text(s, gc + p, "VIC", color="white", fontsize=7, ha="center", va="center",
            fontweight="bold")

    # Separation dimension
    ax.annotate("", xy=(s, gc), xytext=(0, gc),
                arrowprops=dict(arrowstyle="<->", color="#4ADE80", lw=1.5))
    ax.text(s/2, gc - 3, f"Sep={s} mm", color="#4ADE80", fontsize=7.5, ha="center")

    # GC
    ax.text(-CABLE_R - 5, gc/2, f"GC={gc}", color="#CBD5E1", fontsize=7, ha="right")

    ax.set_title(f"Sep={s}  P={p}  OL={ol}  GC={gc}",
                 fontsize=8.5, color="#E2E8F0", fontweight="bold")
    ax.set_xlim(-12, s + 12)
    ax.set_ylim(-5, max(gc + p + CABLE_R + 6, 25))
    ax.set_xlabel("X (mm)", fontsize=8, color="#64748B")
    ax.set_ylabel("Y (mm)", fontsize=8, color="#64748B")
    ax.tick_params(colors="#475569")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    ax.grid(True, alpha=0.15, color="#334155")

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("F2_CrossSection_MultipleConfigs.png")


# ─── EXPORT TABLES ─────────────────────────────────────────────────────────

print("\n[7] Exporting result tables...")
all_cfg.to_csv(RESULTS_ROOT / "All_216_Configurations_With_Pareto.csv", index=False)
bal_rank.to_csv(RESULTS_ROOT / "Balanced_Design_Ranking.csv", index=False)
model_summ.to_csv(RESULTS_ROOT / "Stage12A_Model_Summary.csv", index=False)
global_sens.to_csv(RESULTS_ROOT / "GlobalSensitivity_Summary.csv", index=False)
print("   Exported 4 result tables.")


# ─── FINAL SUMMARY ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  KONE Final Engineering Demo COMPLETE (Python)")
print("="*70)
print(f"  Output: {RESULTS_ROOT}")
print("\n  Figures:")
for f in sorted(RESULTS_ROOT.glob("*.png")):
    print(f"    {f.name}")
print("\n  IMPORTANT NOTICES:")
print("    * All results are from the deterministic analytical MATLAB dataset.")
print("    * R²≈1 reflects dataset reproduction, NOT independent validation.")
print("    * No physical EMC measurements or compliance determination is shown.")
print("    * 3D geometry is an illustrative parameterised representation.")
print("="*70 + "\n")
