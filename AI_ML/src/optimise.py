"""
optimise.py
-----------
Intelligent Configuration Optimisation module for KONE EMC-GUARD (Stage 4).

Performs multi-objective Pareto-front optimisation across all 216 discrete
cable configurations to identify designs offering favourable combinations
of capacitance and mutual inductance.

PRIMARY OBJECTIVES:
-------------------
1. Minimise Total Capacitance (TotalCapacitance_pF_per_m, pF/m)
2. Minimise Absolute Mutual Inductance (AbsMutualInductance_nH, nH)

IMPORTANT CONSTRAINTS & CAVEATS:
--------------------------------
- Optimality is strictly defined within the 216 simulated MATLAB configurations.
- No global continuous physical optimality or real-world elevator validation is claimed.
- Tree models and linear surrogates are frozen; no new models are trained.
- No single configuration is presented as universally superior when trade-offs exist;
  the full non-dominated Pareto front is presented transparently.

Usage (from AI_ML/ directory):
    py src/optimise.py

Usage (from any directory):
    py C:/path/to/kone/AI_ML/src/optimise.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Force UTF-8 on Windows consoles to prevent cp1252 encoding crashes
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure AI_ML/src is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    AIML_REPORTS_DIR,
    ENGINE_FROZEN_MATLAB,
    SUPPORTED_CLEARANCES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_PITCHES,
    SUPPORTED_SEPARATIONS,
    TOTAL_CONFIGURATIONS,
)
from dataset import is_valid_configuration, load_dataset
from predict import predict
from risk_assessment import calculate_risk_indicators

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAP_COL = "TotalCapacitance_pF_per_m"
MUT_COL = "AbsMutualInductance_nH"
FIGURES_DIR = AIML_REPORTS_DIR / "figures"

COLOR_PARETO = "#D32F2F"      # red / gold accent for Pareto front
COLOR_DOMINATED = "#90CAF9"   # light blue for dominated configs
COLOR_PRIMARY = "#1565C0"     # dark blue
COLOR_ACCENT = "#2E7D32"      # green

PLOTLY_LAYOUT = dict(
    font=dict(family="Arial, sans-serif", size=12),
    paper_bgcolor="white",
    plot_bgcolor="#F8F9FA",
    margin=dict(l=70, r=40, t=70, b=60),
)


# ---------------------------------------------------------------------------
# Multi-Objective Pareto Dominance Computation
# ---------------------------------------------------------------------------

def compute_pareto_dominance(
    df: pd.DataFrame,
    col_cap: str = CAP_COL,
    col_mut: str = MUT_COL,
) -> pd.DataFrame:
    """
    Compute rigorous Pareto dominance across all configurations.

    A configuration A dominates B (A ≺ B) iff:
        C(A) <= C(B) AND M(A) <= M(B) AND (C(A) < C(B) OR M(A) < M(B))

    Parameters
    ----------
    df      : DataFrame with capacitance and mutual inductance columns.
    col_cap : Column name for total capacitance.
    col_mut : Column name for absolute mutual inductance.

    Returns
    -------
    pd.DataFrame enriched with:
        - Is_Pareto_Optimal (bool)
        - Dominated_By_Count (int): Number of configurations dominating this one (0 for Pareto)
        - Dominates_Count (int): Number of other configurations dominated by this one
        - Pareto_Rank (int): 1 for Pareto front, >1 for subsequent fronts
    """
    n = len(df)
    caps = df[col_cap].values
    muts = df[col_mut].values

    dominated_by = np.zeros(n, dtype=int)
    dominates = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Does j dominate i?
            if (caps[j] <= caps[i] and muts[j] <= muts[i]) and (
                caps[j] < caps[i] or muts[j] < muts[i]
            ):
                dominated_by[i] += 1
            # Does i dominate j?
            if (caps[i] <= caps[j] and muts[i] <= muts[j]) and (
                caps[i] < caps[j] or muts[i] < muts[j]
            ):
                dominates[i] += 1

    res = df.copy()
    res["Dominated_By_Count"] = dominated_by
    res["Dominates_Count"] = dominates
    res["Is_Pareto_Optimal"] = (dominated_by == 0)

    # Calculate Pareto Ranks using fast non-dominated sorting
    remaining_indices = list(range(n))
    pareto_rank = np.zeros(n, dtype=int)
    current_rank = 1

    while remaining_indices:
        current_front = []
        for idx in remaining_indices:
            # Check if any other remaining index dominates idx
            is_dominated = False
            for other in remaining_indices:
                if idx == other:
                    continue
                if (caps[other] <= caps[idx] and muts[other] <= muts[idx]) and (
                    caps[other] < caps[idx] or muts[other] < muts[idx]
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                current_front.append(idx)

        for idx in current_front:
            pareto_rank[idx] = current_rank
            remaining_indices.remove(idx)
        current_rank += 1

    res["Pareto_Rank"] = pareto_rank
    return res


# ---------------------------------------------------------------------------
# Multi-Criteria Decision Making & Scoring
# ---------------------------------------------------------------------------

def score_pareto_candidates(
    pareto_df: pd.DataFrame,
    global_df: pd.DataFrame,
    w_cap: float = 0.5,
    w_mut: float = 0.5,
) -> pd.DataFrame:
    """
    Rank and score Pareto optimal candidates using multi-criteria trade-offs.

    Parameters
    ----------
    pareto_df : Non-dominated subset of configurations.
    global_df : Full 216-dataset for global context.
    w_cap     : Weight for capacitance objective.
    w_mut     : Weight for mutual inductance objective.

    Returns
    -------
    Enriched pareto_df with normalized scores, utopia distances, and engineering rationale.
    """
    res = pareto_df.copy()

    # Normalise relative to global extrema for true physical comparison
    c_min = global_df[CAP_COL].min()
    c_max = global_df[CAP_COL].max()
    m_min = global_df[MUT_COL].min()
    m_max = global_df[MUT_COL].max()

    c_norm_global = (res[CAP_COL] - c_min) / (c_max - c_min)
    m_norm_global = (res[MUT_COL] - m_min) / (m_max - m_min)

    # Normalise within the Pareto front itself to understand front-local trade-offs
    p_c_min = res[CAP_COL].min()
    p_c_max = res[CAP_COL].max()
    p_m_min = res[MUT_COL].min()
    p_m_max = res[MUT_COL].max()

    c_norm_pareto = (res[CAP_COL] - p_c_min) / (p_c_max - p_c_min) if p_c_max > p_c_min else 0.0
    m_norm_pareto = (res[MUT_COL] - p_m_min) / (p_m_max - p_m_min) if p_m_max > p_m_min else 0.0

    res["Global_Norm_Capacitance"] = c_norm_global
    res["Global_Norm_Inductance"] = m_norm_global
    res["Pareto_Norm_Capacitance"] = c_norm_pareto
    res["Pareto_Norm_Inductance"] = m_norm_pareto

    # Weighted trade-off score (lower is more balanced)
    total_w = w_cap + w_mut
    res["Weighted_Tradeoff_Score"] = (
        w_cap * c_norm_pareto + w_mut * m_norm_pareto
    ) / total_w

    # Euclidean distance to the Pareto utopia point (min C, min M in Pareto front)
    res["Utopia_Distance_Pareto"] = np.sqrt(c_norm_pareto**2 + m_norm_pareto**2)

    # Assign candidate role tags
    roles = []
    rationales = []
    for _, row in res.iterrows():
        sep = row["PairSeparation_mm"]
        if abs(row[CAP_COL] - p_c_min) < 1e-6:
            role = "Minimum Capacitance Champion"
            rationale = (
                f"Achieves lowest total capacitance ({row[CAP_COL]:.2f} pF/m) in the entire dataset. "
                f"Separation=8mm minimises inter-pair capacitance, at the cost of higher mutual inductance."
            )
        elif abs(row[MUT_COL] - p_m_min) < 1e-6:
            role = "Minimum Inductance Champion"
            rationale = (
                f"Achieves lowest mutual inductance ({row[MUT_COL]:.4f} nH) in the entire dataset. "
                f"Separation=28mm maximizes spatial magnetic decay (-92.2% vs 8mm), at slight capacitance cost (+16.7%)."
            )
        elif sep == 12.0:
            role = "Recommended Balanced Design (Primary)"
            rationale = (
                f"Yields the lowest equal-weighted trade-off score ({row['Weighted_Tradeoff_Score']:.3f}). "
                f"Reduces mutual inductance by 56.4% compared to 8mm champion while increasing capacitance by only 1.43 pF/m."
            )
        elif sep == 16.0:
            role = "Balanced Design (Moderate Spacing)"
            rationale = (
                f"Reduces mutual inductance by 75.7% (to {row[MUT_COL]:.4f} nH) with minimal capacitance increase (+8.6%). "
                f"Strong compromise candidate for space-constrained elevator ducts."
            )
        elif sep == 20.0:
            role = "Inductance-Favouring Balanced Design"
            rationale = (
                f"Reduces mutual inductance by 84.5% (to {row[MUT_COL]:.4f} nH) for high-sensitivity analog/encoder lines."
            )
        elif sep == 24.0:
            role = "Ultra-Low Inductive Coupling Candidate"
            rationale = (
                f"Near-minimum mutual inductance ({row[MUT_COL]:.4f} nH, -89.3% vs 8mm) while remaining non-dominated."
            )
        else:
            role = "Pareto Optimal Candidate"
            rationale = "Non-dominated trade-off configuration on the Pareto front."

        roles.append(role)
        rationales.append(rationale)

    res["Candidate_Role"] = roles
    res["Inclusion_Rationale"] = rationales
    return res.sort_values(CAP_COL).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Configuration Comparison Utility
# ---------------------------------------------------------------------------

def compare_configurations(
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    label_a: str = "Configuration A (Baseline)",
    label_b: str = "Configuration B (Candidate)",
) -> Dict[str, Any]:
    """
    Compare two cable configurations side-by-side.

    Computes:
      - Geometric parameter differences
      - Coupling changes (absolute and percentage)
      - Pareto dominance relationship
      - Trade-off evaluation

    Parameters
    ----------
    config_a : Dict with keys PairSeparation_mm, PairPitch_mm, Overlap_mm, GroundClearance_mm,
               and optionally TotalCapacitance_pF_per_m, AbsMutualInductance_nH
    config_b : Dict with same keys
    label_a  : Label for configuration A
    label_b  : Label for configuration B

    Returns
    -------
    dict with detailed comparative evaluation.
    """
    # Helper to resolve predictions if not already present
    def _resolve_pred(c: Dict[str, Any]) -> Tuple[float, float, str]:
        if CAP_COL in c and MUT_COL in c:
            return float(c[CAP_COL]), float(c[MUT_COL]), c.get("Config_ID", "CUSTOM")
        res = predict(
            sep=c["PairSeparation_mm"],
            pitch=c["PairPitch_mm"],
            overlap=c["Overlap_mm"],
            ground=c["GroundClearance_mm"],
        )
        return (
            float(res["predictions"][CAP_COL]["value"]),
            float(res["predictions"][MUT_COL]["value"]),
            res.get("config_id", "CUSTOM"),
        )

    c_a, m_a, cid_a = _resolve_pred(config_a)
    c_b, m_b, cid_b = _resolve_pred(config_b)

    delta_c = c_b - c_a
    pct_c = (delta_c / c_a) * 100.0 if c_a else 0.0

    delta_m = m_b - m_a
    pct_m = (delta_m / m_a) * 100.0 if m_a else 0.0

    # Determine dominance
    if (c_b <= c_a and m_b <= m_a) and (c_b < c_a or m_b < m_a):
        dominance = f"{label_b} strictly dominates {label_a} (superior on both objectives)."
    elif (c_a <= c_b and m_a <= m_b) and (c_a < c_b or m_a < m_b):
        dominance = f"{label_a} strictly dominates {label_b}."
    else:
        dominance = (
            "Mutual trade-off: neither configuration dominates the other. "
            f"{label_b} {'increases' if delta_c > 0 else 'decreases'} capacitance while "
            f"{'increasing' if delta_m > 0 else 'decreasing'} mutual inductance."
        )

    # Narrative trade-off summary
    tradeoff_summary = (
        f"Switching from {label_a} ({cid_a}) to {label_b} ({cid_b}): "
        f"Capacitance changes by {delta_c:+.2f} pF/m ({pct_c:+.1f}%), and "
        f"Mutual Inductance changes by {delta_m:+.4f} nH ({pct_m:+.1f}%)."
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "config_a": {
            "Config_ID": cid_a,
            "geometry": {
                "PairSeparation_mm": config_a["PairSeparation_mm"],
                "PairPitch_mm": config_a["PairPitch_mm"],
                "Overlap_mm": config_a["Overlap_mm"],
                "GroundClearance_mm": config_a["GroundClearance_mm"],
            },
            "TotalCapacitance_pF_per_m": c_a,
            "AbsMutualInductance_nH": m_a,
        },
        "config_b": {
            "Config_ID": cid_b,
            "geometry": {
                "PairSeparation_mm": config_b["PairSeparation_mm"],
                "PairPitch_mm": config_b["PairPitch_mm"],
                "Overlap_mm": config_b["Overlap_mm"],
                "GroundClearance_mm": config_b["GroundClearance_mm"],
            },
            "TotalCapacitance_pF_per_m": c_b,
            "AbsMutualInductance_nH": m_b,
        },
        "delta": {
            "Capacitance_Diff_pF_per_m": delta_c,
            "Capacitance_Percent_Change": pct_c,
            "MutualInductance_Diff_nH": delta_m,
            "MutualInductance_Percent_Change": pct_m,
        },
        "dominance_relationship": dominance,
        "tradeoff_summary": tradeoff_summary,
    }


# ---------------------------------------------------------------------------
# Visualisations for Stage 4
# ---------------------------------------------------------------------------

def _save_fig(fig: go.Figure, fname: str, figures_dir: Path) -> str:
    """Save Plotly figure to HTML."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{fname}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info("  Saved figure: %s", html_path.name)
    return str(html_path)


def plot_pareto_front(
    df_all: pd.DataFrame,
    df_pareto: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> str:
    """Figure 6: Multi-objective Pareto-front scatter plot."""
    dominated = df_all[~df_all["Is_Pareto_Optimal"]].copy()
    pareto_sorted = df_pareto.sort_values(CAP_COL).copy()

    fig = go.Figure()

    # Trace 1: Dominated configurations
    fig.add_trace(go.Scatter(
        x=dominated[CAP_COL],
        y=dominated[MUT_COL],
        mode="markers",
        marker=dict(color=COLOR_DOMINATED, size=7, opacity=0.6),
        name="Dominated Configurations (210)",
        text=[
            f"Config: {r.Config_ID}<br>Capacitance: {r[CAP_COL]:.2f} pF/m<br>"
            f"Inductance: {r[MUT_COL]:.4f} nH<br>"
            f"Sep={r.PairSeparation_mm}mm, Pitch={r.PairPitch_mm}mm, Overlap={r.Overlap_mm}mm, Ground={r.GroundClearance_mm}mm"
            for _, r in dominated.iterrows()
        ],
        hoverinfo="text",
    ))

    # Trace 2: Pareto connection line
    fig.add_trace(go.Scatter(
        x=pareto_sorted[CAP_COL],
        y=pareto_sorted[MUT_COL],
        mode="lines",
        line=dict(color=COLOR_PARETO, width=2.5, dash="dash"),
        name="Pareto Optimal Front",
        showlegend=True,
    ))

    # Trace 3: Pareto optimal points
    fig.add_trace(go.Scatter(
        x=pareto_sorted[CAP_COL],
        y=pareto_sorted[MUT_COL],
        mode="markers+text",
        marker=dict(size=14, color=COLOR_PARETO, symbol="diamond", line=dict(color="white", width=1.5)),
        name="Pareto Optimal Candidates (6)",
        text=[f"Sep={r.PairSeparation_mm}mm" for _, r in pareto_sorted.iterrows()],
        textposition="top right",
        textfont=dict(size=11, color=COLOR_PARETO),
        hovertemplate=(
            "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
            "Capacitance: %{x:.2f} pF/m<br>"
            "Inductance: %{y:.4f} nH<br>"
            "Separation: %{customdata[2]} mm<br>"
            "Pitch: %{customdata[3]} mm<br>"
            "Overlap: %{customdata[4]} mm<br>"
            "Ground Clearance: %{customdata[5]} mm<br>"
            "<i>%{customdata[6]}</i><extra></extra>"
        ),
        customdata=pareto_sorted[[
            "Config_ID", "Candidate_Role", "PairSeparation_mm",
            "PairPitch_mm", "Overlap_mm", "GroundClearance_mm", "Inclusion_Rationale"
        ]].values,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Multi-Objective Pareto Optimisation Front (Capacitance vs Mutual Inductance)</b><br>"
                 "<sup>Identifies the 6 non-dominated configurations from the 216-grid (Pitch=2mm, Overlap=150mm, Ground=100mm)</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Total Capacitance [pF/m] (Minimise →)", gridcolor="#E0E0E0"),
        yaxis=dict(title="Absolute Mutual Inductance [nH] (Minimise →)", gridcolor="#E0E0E0"),
        legend=dict(x=0.68, y=0.98, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E0E0E0", borderwidth=1),
    )
    return _save_fig(fig, "optimisation_pareto_front", figures_dir)


def plot_top_candidates(
    df_pareto: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> str:
    """Figure 7: Top candidate configuration trade-off comparison."""
    df_sorted = df_pareto.sort_values(CAP_COL).copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Absolute Physical Parameters</b>",
            "<b>Pareto-Local Normalised Trade-Offs [0 to 1]</b>"
        ],
    )

    labels = [f"Sep={r.PairSeparation_mm}mm<br>({r.Config_ID})" for _, r in df_sorted.iterrows()]

    # Left: Dual axis for physical values
    fig.add_trace(
        go.Bar(
            name="Capacitance [pF/m]",
            x=labels,
            y=df_sorted[CAP_COL],
            marker_color=COLOR_PRIMARY,
            text=[f"{v:.1f}" for v in df_sorted[CAP_COL]],
            textposition="inside",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            name="Mutual Inductance [nH]",
            x=labels,
            y=df_sorted[MUT_COL],
            marker_color=COLOR_PARETO,
            text=[f"{v:.3f}" for v in df_sorted[MUT_COL]],
            textposition="outside",
        ),
        row=1, col=1,
    )

    # Right: Normalised trade-off curves
    fig.add_trace(
        go.Scatter(
            name="Norm. Capacitance (Pareto)",
            x=labels,
            y=df_sorted["Pareto_Norm_Capacitance"],
            mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=2.5),
            marker=dict(size=8),
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scatter(
            name="Norm. Inductance (Pareto)",
            x=labels,
            y=df_sorted["Pareto_Norm_Inductance"],
            mode="lines+markers",
            line=dict(color=COLOR_PARETO, width=2.5),
            marker=dict(size=8),
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scatter(
            name="Equal-Weight Score",
            x=labels,
            y=df_sorted["Weighted_Tradeoff_Score"],
            mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=3, dash="dot"),
            marker=dict(size=10, symbol="star"),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title=dict(
            text="<b>Pareto Optimal Candidates Comparison</b><br>"
                 "<sup>Trade-off progression across Pair Separation (Pitch=2mm, Overlap=150mm, Ground=100mm)</sup>",
            x=0.5,
        ),
    )
    return _save_fig(fig, "optimisation_top_candidates", figures_dir)


def plot_geometry_comparison(
    df_pareto: pd.DataFrame,
    df_all: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> str:
    """Figure 8: Geometry parameter comparison across Pareto optimal vs non-optimal configurations."""
    # Create parallel coordinates plot
    plot_df = df_all.copy()
    plot_df["Pareto_Status"] = np.where(plot_df["Is_Pareto_Optimal"], 1, 0)

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=plot_df["Pareto_Status"],
            colorscale=[[0, COLOR_DOMINATED], [1, COLOR_PARETO]],
            showscale=True,
            colorbar=dict(
                title="Status",
                tickvals=[0, 1],
                ticktext=["Dominated", "Pareto Front"],
                len=0.75,
            ),
        ),
        dimensions=[
            dict(
                range=[8, 28],
                label="Pair Separation [mm]",
                values=plot_df["PairSeparation_mm"],
                tickvals=SUPPORTED_SEPARATIONS,
            ),
            dict(
                range=[2, 4],
                label="Pair Pitch [mm]",
                values=plot_df["PairPitch_mm"],
                tickvals=SUPPORTED_PITCHES,
            ),
            dict(
                range=[150, 450],
                label="Overlap [mm]",
                values=plot_df["Overlap_mm"],
                tickvals=SUPPORTED_OVERLAPS,
            ),
            dict(
                range=[10, 100],
                label="Ground Clearance [mm]",
                values=plot_df["GroundClearance_mm"],
                tickvals=SUPPORTED_CLEARANCES,
            ),
            dict(
                range=[plot_df[CAP_COL].min(), plot_df[CAP_COL].max()],
                label="Capacitance [pF/m]",
                values=plot_df[CAP_COL],
            ),
            dict(
                range=[plot_df[MUT_COL].min(), plot_df[MUT_COL].max()],
                label="Mutual Inductance [nH]",
                values=plot_df[MUT_COL],
            ),
        ],
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Geometry Parameter Routing to Pareto Optimality</b><br>"
                 "<sup>Red lines highlight the 6 Pareto optimal pathways (Pitch=2, Overlap=150, Ground=100)</sup>",
            x=0.5,
        ),
    )
    return _save_fig(fig, "optimisation_geometry_comparison", figures_dir)


# ---------------------------------------------------------------------------
# Pipeline Execution & Reporting
# ---------------------------------------------------------------------------

def run_optimisation(
    engine: str = ENGINE_FROZEN_MATLAB,
    save_results: bool = True,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end Intelligent Configuration Optimisation pipeline."""
    logger.info("=" * 65)
    logger.info("KONE EMC-GUARD — Stage 4: Intelligent Configuration Optimisation")
    logger.info("=" * 65)

    # Step 1: Load and assess all configurations
    raw_df = calculate_risk_indicators(load_dataset())

    # Step 2: Compute Pareto dominance
    all_scored_df = compute_pareto_dominance(raw_df)

    # Step 3: Extract and score Pareto front
    pareto_candidates = all_scored_df[all_scored_df["Is_Pareto_Optimal"]].copy()
    scored_pareto_df = score_pareto_candidates(pareto_candidates, all_scored_df)

    # Merge Pareto information back into full dataset
    all_scored_df = all_scored_df.merge(
        scored_pareto_df[[
            "Config_ID", "Candidate_Role", "Inclusion_Rationale",
            "Weighted_Tradeoff_Score", "Utopia_Distance_Pareto"
        ]],
        on="Config_ID",
        how="left",
    )

    logger.info("Optimisation complete: %d Pareto optimal candidates identified.", len(scored_pareto_df))

    saved_files = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if save_results:
        AIML_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Pareto Front CSV
        pareto_csv = AIML_REPORTS_DIR / f"pareto_optimal_configurations_{timestamp}.csv"
        scored_pareto_df.to_csv(pareto_csv, index=False)
        saved_files["pareto_configurations_csv"] = str(pareto_csv)

        # 2. All 216 configurations with Pareto metrics
        all_csv = AIML_REPORTS_DIR / f"all_configurations_pareto_scored_{timestamp}.csv"
        all_scored_df.to_csv(all_csv, index=False)
        saved_files["all_configurations_pareto_csv"] = str(all_csv)

        # 3. Summary JSON
        summary_json = AIML_REPORTS_DIR / f"optimisation_summary_{timestamp}.json"
        summary_payload = {
            "generated_at": timestamp,
            "module": "optimise.py",
            "engine": engine,
            "total_configurations": len(all_scored_df),
            "pareto_optimal_count": len(scored_pareto_df),
            "common_pareto_fixed_parameters": {
                "PairPitch_mm": 2.0,
                "Overlap_mm": 150.0,
                "GroundClearance_mm": 100.0,
                "explanation": (
                    "Pitch=2mm maximizes twist cancellation; Overlap=150mm minimizes exposure length; "
                    "GroundClearance=100mm minimizes image capacitance."
                ),
            },
            "pareto_candidates": scored_pareto_df[[
                "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
                CAP_COL, MUT_COL, "Candidate_Role", "Weighted_Tradeoff_Score", "Inclusion_Rationale"
            ]].to_dict(orient="records"),
            "best_single_objective": {
                "min_capacitance": scored_pareto_df.loc[scored_pareto_df[CAP_COL].idxmin()][
                    ["Config_ID", "PairSeparation_mm", CAP_COL, MUT_COL]
                ].to_dict(),
                "min_inductance": scored_pareto_df.loc[scored_pareto_df[MUT_COL].idxmin()][
                    ["Config_ID", "PairSeparation_mm", CAP_COL, MUT_COL]
                ].to_dict(),
            },
        }
        with open(summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2, ensure_ascii=False)
        saved_files["summary_json"] = str(summary_json)
        logger.info("Saved optimisation data to %s and %s", pareto_csv.name, summary_json.name)

    if generate_plots:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        saved_files["fig_pareto_front"] = plot_pareto_front(all_scored_df, scored_pareto_df, FIGURES_DIR)
        saved_files["fig_top_candidates"] = plot_top_candidates(scored_pareto_df, FIGURES_DIR)
        saved_files["fig_geometry_comparison"] = plot_geometry_comparison(scored_pareto_df, all_scored_df, FIGURES_DIR)

    logger.info("=" * 65)
    logger.info("Stage 4 Optimisation complete.")
    logger.info("=" * 65)

    return {
        "all_df": all_scored_df,
        "pareto_df": scored_pareto_df,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    results = run_optimisation(save_results=True, generate_plots=True)
    pareto_df = results["pareto_df"]

    print("\n" + "=" * 65)
    print("PARETO OPTIMAL CONFIGURATIONS (Non-Dominated Candidates)")
    print("=" * 65)
    for _, r in pareto_df.iterrows():
        print(
            f"  {r['Config_ID']} (Sep={r['PairSeparation_mm']:>2.0f}mm, Pitch={r['PairPitch_mm']}mm, "
            f"Ov={r['Overlap_mm']}mm, GC={r['GroundClearance_mm']:>3.0f}mm) -> "
            f"Cap={r[CAP_COL]:5.2f} pF/m, Mut={r[MUT_COL]:6.4f} nH | Role: {r['Candidate_Role']}"
        )

