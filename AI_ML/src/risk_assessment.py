"""
risk_assessment.py
------------------
EMC Risk Assessment module for the KONE EMC-GUARD AI_ML system (Stage 3).

Analyses predicted cable total capacitance (pF/m) and absolute mutual
inductance (nH) across cable configurations to determine relative electromagnetic
coupling risk indicators.

CRITICAL SCIENTIFIC PRINCIPLES:
-------------------------------
1. Relative Indicators Only:
   All classifications ("Low", "Moderate", "High") represent RELATIVE coupling
   levels within the bounded 216-configuration MATLAB simulation dataset.
2. No Regulatory / Compliance Claims:
   Configurations are NEVER classified as "safe" or "unsafe", and NO claim
   of compliance with IEC, EN (e.g. EN 12015/12016) or KONE internal standards
   is made without officially approved, verified physical thresholds.
3. Extensible Framework:
   Designed to seamlessly incorporate engineering thresholds and customer
   standards when validated empirical limits are established.
4. Explainable Attribution:
   Every risk classification incorporates Stage 2 Explainable AI (XAI) insights
   to identify the geometric parameters driving each indicator.

Usage (from AI_ML/ directory):
    py src/risk_assessment.py

Usage (from any directory):
    py C:/path/to/kone/AI_ML/src/risk_assessment.py
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
    CAP_FEATURES,
    ENGINE_FROZEN_MATLAB,
    MUT_FEATURES,
    SUPPORTED_CLEARANCES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_PITCHES,
    SUPPORTED_SEPARATIONS,
    TOTAL_CONFIGURATIONS,
)
from dataset import is_valid_configuration, load_dataset
from predict import predict

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
# Constants & Boundaries
# ---------------------------------------------------------------------------
CAP_COL = "TotalCapacitance_pF_per_m"
MUT_COL = "AbsMutualInductance_nH"
FIGURES_DIR = AIML_REPORTS_DIR / "figures"

# Relative indicator category labels
TIER_LOW = "Low Relative Indicator"
TIER_MOD = "Moderate Relative Indicator"
TIER_HIGH = "High Relative Indicator"

# Standard cutoff boundaries for normalised [0, 1] indicators
# [0.00, 0.33] -> Low, (0.33, 0.66] -> Moderate, (0.66, 1.00] -> High
DEFAULT_CUTOFF_LOW = 0.3333
DEFAULT_CUTOFF_HIGH = 0.6667

# Palette for Plotly figures
COLOR_LOW = "#2E7D32"       # green
COLOR_MOD = "#F57F17"       # amber / orange
COLOR_HIGH = "#C62828"      # red
COLOR_PRIMARY = "#1565C0"   # blue
COLOR_ACCENT = "#6A1B9A"    # purple

PLOTLY_LAYOUT = dict(
    font=dict(family="Arial, sans-serif", size=12),
    paper_bgcolor="white",
    plot_bgcolor="#F8F9FA",
    margin=dict(l=70, r=40, t=70, b=60),
)


# ---------------------------------------------------------------------------
# Core Risk Calculation & Indicator Normalisation
# ---------------------------------------------------------------------------

def calculate_risk_indicators(
    df: pd.DataFrame,
    w_cap: float = 0.5,
    w_mut: float = 0.5,
    cutoff_low: float = DEFAULT_CUTOFF_LOW,
    cutoff_high: float = DEFAULT_CUTOFF_HIGH,
) -> pd.DataFrame:
    """
    Calculate relative EMC coupling risk indicators across a dataset.

    Normalises capacitance and mutual inductance to [0, 1] relative to the
    dataset extrema, then calculates individual and combined risk indicators.

    Parameters
    ----------
    df          : DataFrame containing TotalCapacitance_pF_per_m and AbsMutualInductance_nH
    w_cap       : Weight for capacitive coupling (default 0.5)
    w_mut       : Weight for inductive coupling (default 0.5)
    cutoff_low  : Upper boundary for 'Low Relative Indicator' (default 0.333)
    cutoff_high : Lower boundary for 'High Relative Indicator' (default 0.667)

    Returns
    -------
    Enriched pd.DataFrame with normalized indicators, tier categories, and ranks.
    """
    if CAP_COL not in df.columns or MUT_COL not in df.columns:
        raise ValueError(f"Input DataFrame must contain '{CAP_COL}' and '{MUT_COL}'.")

    result = df.copy()

    c_vals = result[CAP_COL].values
    m_vals = result[MUT_COL].values

    # Dataset extrema (deterministic bounds of 216 configurations)
    c_min, c_max = float(np.min(c_vals)), float(np.max(c_vals))
    m_min, m_max = float(np.min(m_vals)), float(np.max(m_vals))

    c_range = c_max - c_min if c_max > c_min else 1.0
    m_range = m_max - m_min if m_max > m_min else 1.0

    # Normalised indicators [0, 1]
    result["Capacitance_Indicator"] = (c_vals - c_min) / c_range
    result["MutualInductance_Indicator"] = (m_vals - m_min) / m_range

    # Weighted combined risk indicator
    total_w = w_cap + w_mut
    norm_w_cap = w_cap / total_w
    norm_w_mut = w_mut / total_w

    result["Combined_Risk_Indicator"] = (
        norm_w_cap * result["Capacitance_Indicator"]
        + norm_w_mut * result["MutualInductance_Indicator"]
    )

    # Euclidean distance from utopia point (0, 0) normalized to [0, 1]
    result["Euclidean_Risk_Distance"] = np.sqrt(
        result["Capacitance_Indicator"] ** 2 + result["MutualInductance_Indicator"] ** 2
    ) / np.sqrt(2.0)

    # Percentile ranks (0 to 100%)
    result["Capacitance_Percentile"] = result[CAP_COL].rank(pct=True) * 100.0
    result["MutualInductance_Percentile"] = result[MUT_COL].rank(pct=True) * 100.0
    result["Combined_Percentile"] = result["Combined_Risk_Indicator"].rank(pct=True) * 100.0

    # Categorisation helper
    def _classify_tier(val: float) -> str:
        if val <= cutoff_low:
            return TIER_LOW
        elif val <= cutoff_high:
            return TIER_MOD
        return TIER_HIGH

    result["Capacitance_Risk_Tier"] = result["Capacitance_Indicator"].apply(_classify_tier)
    result["MutualInductance_Risk_Tier"] = result["MutualInductance_Indicator"].apply(_classify_tier)
    result["Combined_Risk_Tier"] = result["Combined_Risk_Indicator"].apply(_classify_tier)

    # Risk ranking (1 = lowest coupling / preferred, 216 = highest coupling)
    result["Risk_Rank"] = result["Combined_Risk_Indicator"].rank(method="min").astype(int)

    # Sort primarily by risk rank
    result = result.sort_values("Risk_Rank").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Individual Configuration Risk Assessment & Attribution
# ---------------------------------------------------------------------------

def assess_configuration(
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
    engine: str = ENGINE_FROZEN_MATLAB,
    w_cap: float = 0.5,
    w_mut: float = 0.5,
) -> Dict[str, Any]:
    """
    Perform complete EMC risk assessment for a single configuration.

    Parameters
    ----------
    sep     : PairSeparation_mm  (8, 12, 16, 20, 24, 28)
    pitch   : PairPitch_mm       (2, 3, 4)
    overlap : Overlap_mm         (150, 300, 450)
    ground  : GroundClearance_mm (10, 25, 50, 100)
    engine  : Prediction engine ("frozen_matlab" or "extratrees")
    w_cap   : Capacitance weighting factor (default 0.5)
    w_mut   : Mutual inductance weighting factor (default 0.5)

    Returns
    -------
    dict with configuration geometry, predictions, relative indicators,
    tier classifications, XAI-based parameter attribution, and scientific caveats.
    """
    if not is_valid_configuration(sep, pitch, overlap, ground):
        return {
            "status": "ERROR",
            "error": (
                f"Configuration ({sep}, {pitch}, {overlap}, {ground}) is outside "
                "the supported 216-point design grid. Extrapolation is strictly prohibited."
            ),
        }

    # Obtain authoritative prediction
    pred_res = predict(sep=sep, pitch=pitch, overlap=overlap, ground=ground, engine=engine)
    if pred_res.get("status") != "OK":
        return {"status": "ERROR", "error": pred_res.get("error", "Prediction failed")}

    c_val = float(pred_res["predictions"][CAP_COL]["value"])
    m_val = float(pred_res["predictions"][MUT_COL]["value"])
    config_id = pred_res.get("config_id", "UNKNOWN")

    # Load full dataset for global relative context
    ds_df = load_dataset()
    c_min, c_max = float(ds_df[CAP_COL].min()), float(ds_df[CAP_COL].max())
    m_min, m_max = float(ds_df[MUT_COL].min()), float(ds_df[MUT_COL].max())

    # Indicators
    c_ind = (c_val - c_min) / (c_max - c_min)
    m_ind = (m_val - m_min) / (m_max - m_min)
    total_w = w_cap + w_mut
    norm_w_cap = w_cap / total_w if total_w > 0 else 0.5
    norm_w_mut = w_mut / total_w if total_w > 0 else 0.5
    comb_ind = norm_w_cap * c_ind + norm_w_mut * m_ind

    # Classifications
    def _classify(val: float) -> str:
        if val <= DEFAULT_CUTOFF_LOW:
            return TIER_LOW
        elif val <= DEFAULT_CUTOFF_HIGH:
            return TIER_MOD
        return TIER_HIGH

    c_tier = _classify(c_ind)
    m_tier = _classify(m_ind)
    comb_tier = _classify(comb_ind)

    # Percentiles
    c_pct = float((ds_df[CAP_COL] < c_val).mean() * 100.0)
    m_pct = float((ds_df[MUT_COL] < m_val).mean() * 100.0)

    # XAI-based Geometric Parameter Attributions & Narrative
    # Capacitance is 92.8% driven by GroundClearance, 6.3% by PairSeparation
    cap_factors = []
    if ground <= 10.0:
        cap_factors.append(
            "Close proximity to ground plane (GroundClearance=10mm) strongly intensifies "
            "earth image capacitance (dominant 92.8% MDI factor), driving capacitance indicator high."
        )
    elif ground >= 100.0:
        cap_factors.append(
            "High ground clearance (GroundClearance=100mm) significantly weakens earth electric "
            "flux coupling, maintaining capacitance indicator at minimum levels."
        )
    else:
        cap_factors.append(
            f"Intermediate ground clearance ({ground}mm) provides moderate electrostatic isolation from ground."
        )

    if sep <= 8.0:
        cap_factors.append(
            "Narrow pair separation (8mm) yields lower inter-pair capacitance compared to wider spacing."
        )
    elif sep >= 28.0:
        cap_factors.append(
            "Wide pair separation (28mm) slightly increases inter-wire capacitance (+16.7% over 8mm baseline)."
        )

    # Mutual Inductance is 58.4% driven by PairSeparation, 25.8% by Pitch, 15.8% by Overlap
    mut_factors = []
    if sep <= 8.0:
        mut_factors.append(
            "Tight pair separation (8mm) places conductors in intense local magnetic field (B ∝ 1/r), "
            "sharply elevating inductive coupling indicator (dominant 58.4% MDI factor)."
        )
    elif sep >= 28.0:
        mut_factors.append(
            "Wide pair separation (28mm) places conductors in decayed far-field, reducing mutual "
            "inductance by up to 92.2% compared to 8mm."
        )

    if pitch <= 2.0:
        mut_factors.append(
            "Tight twist pitch (2mm) provides frequent spatial field reversal, maximizing cancellation of "
            "external magnetic flux (25.8% MDI contribution)."
        )
    elif pitch >= 4.0:
        mut_factors.append(
            "Loose twist pitch (4mm) provides weaker spatial field reversal, resulting in elevated inductive pickup."
        )

    if overlap >= 450.0:
        mut_factors.append(
            "Maximum parallel run (Overlap=450mm) linearly accumulates magnetic flux linkage."
        )
    elif overlap <= 150.0:
        mut_factors.append(
            "Short parallel run (Overlap=150mm) minimizes exposed coupling length."
        )

    # Synthesize holistic engineering rationale
    rationale = (
        f"Configuration {config_id} is classified as '{comb_tier}' with a combined indicator of "
        f"{comb_ind:.4f} (Capacitance indicator={c_ind:.4f}, Mutual Inductance indicator={m_ind:.4f}). "
        f"Capacitance is predominantly governed by GroundClearance ({ground}mm). "
        f"Mutual inductance is primarily governed by PairSeparation ({sep}mm) and PairPitch ({pitch}mm)."
    )

    return {
        "status": "OK",
        "config_id": config_id,
        "geometry": {
            "PairSeparation_mm": sep,
            "PairPitch_mm": pitch,
            "Overlap_mm": overlap,
            "GroundClearance_mm": ground,
        },
        "engine": engine,
        "predictions": {
            "TotalCapacitance_pF_per_m": {
                "value": c_val,
                "unit": "pF/m",
                "relative_indicator": c_ind,
                "percentile_in_dataset": c_pct,
                "risk_tier": c_tier,
            },
            "AbsMutualInductance_nH": {
                "value": m_val,
                "unit": "nH",
                "relative_indicator": m_ind,
                "percentile_in_dataset": m_pct,
                "risk_tier": m_tier,
            },
        },
        "combined_assessment": {
            "weights": {"w_cap": norm_w_cap, "w_mut": norm_w_mut},
            "combined_risk_indicator": comb_ind,
            "combined_risk_tier": comb_tier,
            "euclidean_risk_distance": float(np.sqrt(c_ind**2 + m_ind**2) / np.sqrt(2.0)),
        },
        "explainable_attribution": {
            "capacitance_drivers": cap_factors,
            "mutual_inductance_drivers": mut_factors,
            "engineering_rationale": rationale,
        },
        "scientific_disclaimer": (
            "This assessment represents a RELATIVE indicator comparison within the 216 deterministic "
            "MATLAB analytical configurations. It does NOT assert compliance with IEC/EN standards, "
            "guarantee field accuracy in physical elevator hoistways, or imply an absolute safety certification."
        ),
    }


# ---------------------------------------------------------------------------
# Batch Assessment of All 216 Configurations
# ---------------------------------------------------------------------------

def assess_all_configurations(
    engine: str = ENGINE_FROZEN_MATLAB,
    w_cap: float = 0.5,
    w_mut: float = 0.5,
) -> pd.DataFrame:
    """
    Execute relative EMC risk assessment across all 216 dataset configurations.

    Returns
    -------
    pd.DataFrame containing all geometry parameters, predictions, indicators,
    risk tiers, and ranks.
    """
    logger.info("Executing batch risk assessment across 216 configurations using engine '%s' …", engine)
    df = load_dataset()

    records = []
    for _, row in df.iterrows():
        sep = float(row["PairSeparation_mm"])
        pitch = float(row["PairPitch_mm"])
        ov = float(row["Overlap_mm"])
        gc = float(row["GroundClearance_mm"])

        pred_res = predict(sep=sep, pitch=pitch, overlap=ov, ground=gc, engine=engine)
        if pred_res.get("status") == "OK":
            c_val = pred_res["predictions"][CAP_COL]["value"]
            m_val = pred_res["predictions"][MUT_COL]["value"]
            cid = pred_res.get("config_id", row.get("Config_ID", "UNKNOWN"))
        else:
            # Fall back to dataset values if prediction failed
            c_val = float(row[CAP_COL])
            m_val = float(row[MUT_COL])
            cid = str(row.get("Config_ID", "UNKNOWN"))

        records.append({
            "Config_ID": cid,
            "PairSeparation_mm": sep,
            "PairPitch_mm": pitch,
            "Overlap_mm": ov,
            "GroundClearance_mm": gc,
            CAP_COL: c_val,
            MUT_COL: m_val,
        })

    raw_df = pd.DataFrame(records)
    assessed_df = calculate_risk_indicators(raw_df, w_cap=w_cap, w_mut=w_mut)
    logger.info("Risk assessment complete for %d configurations.", len(assessed_df))
    return assessed_df


def get_risk_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compile comprehensive risk distribution statistics from assessed DataFrame."""
    total_n = len(df)
    tier_counts = df["Combined_Risk_Tier"].value_counts().to_dict()
    cap_tiers = df["Capacitance_Risk_Tier"].value_counts().to_dict()
    mut_tiers = df["MutualInductance_Risk_Tier"].value_counts().to_dict()

    top_lowest = df.head(10)[[
        "Risk_Rank", "Config_ID", "PairSeparation_mm", "PairPitch_mm",
        "Overlap_mm", "GroundClearance_mm", CAP_COL, MUT_COL,
        "Combined_Risk_Indicator", "Combined_Risk_Tier"
    ]].to_dict(orient="records")

    top_highest = df.tail(10).sort_values("Risk_Rank", ascending=False)[[
        "Risk_Rank", "Config_ID", "PairSeparation_mm", "PairPitch_mm",
        "Overlap_mm", "GroundClearance_mm", CAP_COL, MUT_COL,
        "Combined_Risk_Indicator", "Combined_Risk_Tier"
    ]].to_dict(orient="records")

    return {
        "total_configurations_assessed": total_n,
        "tier_distribution": {
            "combined": {
                tier: {
                    "count": int(tier_counts.get(tier, 0)),
                    "percentage": float(tier_counts.get(tier, 0) / total_n * 100.0),
                }
                for tier in [TIER_LOW, TIER_MOD, TIER_HIGH]
            },
            "capacitance": {
                tier: int(cap_tiers.get(tier, 0)) for tier in [TIER_LOW, TIER_MOD, TIER_HIGH]
            },
            "mutual_inductance": {
                tier: int(mut_tiers.get(tier, 0)) for tier in [TIER_LOW, TIER_MOD, TIER_HIGH]
            },
        },
        "parameter_extrema": {
            "capacitance": {
                "min_pF_per_m": float(df[CAP_COL].min()),
                "max_pF_per_m": float(df[CAP_COL].max()),
                "mean_pF_per_m": float(df[CAP_COL].mean()),
            },
            "mutual_inductance": {
                "min_nH": float(df[MUT_COL].min()),
                "max_nH": float(df[MUT_COL].max()),
                "mean_nH": float(df[MUT_COL].mean()),
            },
        },
        "top_favourable_configurations": top_lowest,
        "top_unfavourable_configurations": top_highest,
    }


# ---------------------------------------------------------------------------
# Visualisations for Stage 3
# ---------------------------------------------------------------------------

def _save_fig(fig: go.Figure, fname: str, figures_dir: Path) -> str:
    """Save Plotly figure to HTML."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{fname}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info("  Saved figure: %s", html_path.name)
    return str(html_path)


def plot_capacitance_indicator(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> str:
    """Figure 1: Capacitance relative indicator sorted distribution and bands."""
    df_sorted = df.sort_values("Capacitance_Indicator").reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df_sorted) + 1)),
        y=df_sorted[CAP_COL],
        mode="lines+markers",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        marker=dict(
            size=6,
            color=df_sorted["Capacitance_Indicator"],
            colorscale="Tealgrn",
            showscale=True,
            colorbar=dict(title="Relative<br>Indicator"),
        ),
        text=[
            f"Config: {r.Config_ID}<br>Capacitance: {r[CAP_COL]:.2f} pF/m<br>"
            f"Ground Clearance: {r.GroundClearance_mm} mm<br>Tier: {r.Capacitance_Risk_Tier}"
            for _, r in df_sorted.iterrows()
        ],
        hoverinfo="text",
        name="Capacitance",
    ))

    # Add reference band thresholds
    c_min = df[CAP_COL].min()
    c_max = df[CAP_COL].max()
    t1 = c_min + DEFAULT_CUTOFF_LOW * (c_max - c_min)
    t2 = c_min + DEFAULT_CUTOFF_HIGH * (c_max - c_min)

    fig.add_hline(y=t1, line=dict(color=COLOR_LOW, dash="dash", width=1.5),
                  annotation_text=f"Low/Mod Cutoff ({t1:.1f} pF/m)", annotation_position="top left")
    fig.add_hline(y=t2, line=dict(color=COLOR_HIGH, dash="dash", width=1.5),
                  annotation_text=f"Mod/High Cutoff ({t2:.1f} pF/m)", annotation_position="top left")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Capacitance Relative Indicator Distribution</b><br>"
                 "<sup>Sorted across 216 configurations with relative indicator thresholds</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Configuration Index (Sorted by Capacitance)", gridcolor="#E0E0E0"),
        yaxis=dict(title="Total Capacitance [pF/m]", gridcolor="#E0E0E0"),
    )
    return _save_fig(fig, "risk_capacitance_indicator", figures_dir)


def plot_mutual_inductance_indicator(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> str:
    """Figure 2: Mutual inductance relative indicator sorted distribution and bands."""
    df_sorted = df.sort_values("MutualInductance_Indicator").reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df_sorted) + 1)),
        y=df_sorted[MUT_COL],
        mode="lines+markers",
        line=dict(color=COLOR_HIGH, width=2.5),
        marker=dict(
            size=6,
            color=df_sorted["MutualInductance_Indicator"],
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Relative<br>Indicator"),
        ),
        text=[
            f"Config: {r.Config_ID}<br>Inductance: {r[MUT_COL]:.4f} nH<br>"
            f"Separation: {r.PairSeparation_mm} mm<br>Pitch: {r.PairPitch_mm} mm<br>Tier: {r.MutualInductance_Risk_Tier}"
            for _, r in df_sorted.iterrows()
        ],
        hoverinfo="text",
        name="Mutual Inductance",
    ))

    m_min = df[MUT_COL].min()
    m_max = df[MUT_COL].max()
    t1 = m_min + DEFAULT_CUTOFF_LOW * (m_max - m_min)
    t2 = m_min + DEFAULT_CUTOFF_HIGH * (m_max - m_min)

    fig.add_hline(y=t1, line=dict(color=COLOR_LOW, dash="dash", width=1.5),
                  annotation_text=f"Low/Mod Cutoff ({t1:.2f} nH)", annotation_position="top left")
    fig.add_hline(y=t2, line=dict(color=COLOR_HIGH, dash="dash", width=1.5),
                  annotation_text=f"Mod/High Cutoff ({t2:.2f} nH)", annotation_position="top left")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Mutual Inductance Relative Indicator Distribution</b><br>"
                 "<sup>Sorted across 216 configurations with relative indicator thresholds</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Configuration Index (Sorted by Inductance)", gridcolor="#E0E0E0"),
        yaxis=dict(title="Absolute Mutual Inductance [nH]", gridcolor="#E0E0E0"),
    )
    return _save_fig(fig, "risk_mutual_inductance_indicator", figures_dir)


def plot_combined_risk_comparison(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> str:
    """Figure 3: Combined Risk Comparison grouped by Ground Clearance and Separation."""
    fig = px.box(
        df,
        x="GroundClearance_mm",
        y="Combined_Risk_Indicator",
        color="PairSeparation_mm",
        title="<b>Combined Risk Indicator Comparison by Geometry</b><br>"
              "<sup>Separation [mm] and Ground Clearance [mm] interaction effects</sup>",
        labels={
            "GroundClearance_mm": "Ground Clearance [mm]",
            "Combined_Risk_Indicator": "Combined Relative Risk Indicator [0 to 1]",
            "PairSeparation_mm": "Pair Separation [mm]",
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.add_hline(y=DEFAULT_CUTOFF_LOW, line=dict(color=COLOR_LOW, dash="dash"),
                  annotation_text="Low Threshold (0.33)")
    fig.add_hline(y=DEFAULT_CUTOFF_HIGH, line=dict(color=COLOR_HIGH, dash="dash"),
                  annotation_text="High Threshold (0.67)")
    return _save_fig(fig, "risk_combined_comparison", figures_dir)


def plot_configuration_risk_ranking(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> str:
    """Figure 4: Configuration risk ranking (top 15 lowest vs top 15 highest indicator)."""
    top15_low = df.head(15).copy()
    top15_high = df.tail(15).copy()
    comp = pd.concat([top15_low, top15_high]).sort_values("Risk_Rank")

    colors = [
        COLOR_LOW if t == TIER_LOW else (COLOR_MOD if t == TIER_MOD else COLOR_HIGH)
        for t in comp["Combined_Risk_Tier"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp["Config_ID"],
        y=comp["Combined_Risk_Indicator"],
        marker_color=colors,
        text=[f"{v:.3f}" for v in comp["Combined_Risk_Indicator"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>Combined Indicator: %{y:.4f}<br>"
            "Capacitance: %{customdata[0]:.2f} pF/m<br>"
            "Inductance: %{customdata[1]:.4f} nH<br>"
            "Sep=%{customdata[2]}mm, Pitch=%{customdata[3]}mm, Overlap=%{customdata[4]}mm, Ground=%{customdata[5]}mm<extra></extra>"
        ),
        customdata=comp[[
            CAP_COL, MUT_COL, "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm"
        ]].values,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>EMC Risk Ranking — Top 15 Favourable vs Top 15 Unfavourable Configurations</b><br>"
                 "<sup>Green: Low Relative Indicator | Orange: Moderate | Red: High</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Configuration ID", tickangle=-45, gridcolor="#E0E0E0"),
        yaxis=dict(title="Combined Risk Indicator [0 to 1]", gridcolor="#E0E0E0", range=[0, 1.1]),
    )
    return _save_fig(fig, "risk_configuration_ranking", figures_dir)


def plot_capacitance_vs_inductance(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> str:
    """Figure 5: Capacitance versus Mutual Inductance scatter plot coloured by Combined Risk."""
    fig = px.scatter(
        df,
        x=CAP_COL,
        y=MUT_COL,
        color="Combined_Risk_Tier",
        size="Combined_Risk_Indicator",
        hover_name="Config_ID",
        hover_data={
            "PairSeparation_mm": True,
            "PairPitch_mm": True,
            "Overlap_mm": True,
            "GroundClearance_mm": True,
            "Combined_Risk_Indicator": ":.4f",
            CAP_COL: ":.2f",
            MUT_COL: ":.4f",
        },
        color_discrete_map={
            TIER_LOW: COLOR_LOW,
            TIER_MOD: COLOR_MOD,
            TIER_HIGH: COLOR_HIGH,
        },
        title="<b>Capacitance vs Mutual Inductance Coupling Trade-Off</b><br>"
              "<sup>All 216 configurations classified by Combined Relative Risk Tier</sup>",
        labels={
            CAP_COL: "Total Capacitance [pF/m]",
            MUT_COL: "Absolute Mutual Inductance [nH]",
            "Combined_Risk_Tier": "Risk Tier",
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return _save_fig(fig, "risk_capacitance_vs_inductance", figures_dir)


def plot_risk_distribution(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> str:
    """Figure 9 (from Part C): Risk distribution pie & bar chart."""
    counts = df["Combined_Risk_Tier"].value_counts().reindex([TIER_LOW, TIER_MOD, TIER_HIGH]).fillna(0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["<b>Proportion by Tier</b>", "<b>Configuration Counts</b>"],
    )

    colors = [COLOR_LOW, COLOR_MOD, COLOR_HIGH]

    fig.add_trace(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker=dict(colors=colors),
            hole=0.4,
            textinfo="label+percent",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=colors,
            text=counts.values.tolist(),
            textposition="outside",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>EMC Risk Tier Distribution Across 216 Configurations</b><br>"
                 "<sup>Classification based on equal-weighted relative indicators</sup>",
            x=0.5,
        ),
        showlegend=False,
    )
    return _save_fig(fig, "risk_distribution_chart", figures_dir)


# ---------------------------------------------------------------------------
# Pipeline Execution & Reporting
# ---------------------------------------------------------------------------

def run_risk_assessment(
    engine: str = ENGINE_FROZEN_MATLAB,
    save_results: bool = True,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end EMC Risk Assessment pipeline."""
    logger.info("=" * 65)
    logger.info("KONE EMC-GUARD — Stage 3: EMC Risk Assessment Pipeline")
    logger.info("=" * 65)

    df_assessed = assess_all_configurations(engine=engine)
    summary = get_risk_summary(df_assessed)

    saved_files = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if save_results:
        AIML_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Full CSV table
        csv_path = AIML_REPORTS_DIR / f"emc_risk_assessment_all_configs_{timestamp}.csv"
        df_assessed.to_csv(csv_path, index=False)
        saved_files["all_configurations_csv"] = str(csv_path)

        # Summary JSON
        json_path = AIML_REPORTS_DIR / f"emc_risk_assessment_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({
                "generated_at": timestamp,
                "module": "risk_assessment.py",
                "engine": engine,
                "summary": summary,
            }, fh, indent=2, ensure_ascii=False)
        saved_files["summary_json"] = str(json_path)

        logger.info("Saved risk assessment data to %s and %s", csv_path.name, json_path.name)

    if generate_plots:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        saved_files["fig_capacitance"] = plot_capacitance_indicator(df_assessed, FIGURES_DIR)
        saved_files["fig_mutual_inductance"] = plot_mutual_inductance_indicator(df_assessed, FIGURES_DIR)
        saved_files["fig_combined"] = plot_combined_risk_comparison(df_assessed, FIGURES_DIR)
        saved_files["fig_ranking"] = plot_configuration_risk_ranking(df_assessed, FIGURES_DIR)
        saved_files["fig_scatter"] = plot_capacitance_vs_inductance(df_assessed, FIGURES_DIR)
        saved_files["fig_distribution"] = plot_risk_distribution(df_assessed, FIGURES_DIR)

    logger.info("=" * 65)
    logger.info("Stage 3 Risk Assessment complete.")
    logger.info("=" * 65)

    return {
        "assessed_df": df_assessed,
        "summary": summary,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    results = run_risk_assessment(save_results=True, generate_plots=True)
    summary = results["summary"]

    print("\n" + "=" * 65)
    print("EMC RISK ASSESSMENT SUMMARY (216 Configurations)")
    print("=" * 65)
    tier_dist = summary["tier_distribution"]["combined"]
    for tier, data in tier_dist.items():
        print(f"  {tier:<28}: {data['count']:>3} configs ({data['percentage']:5.1f}%)")

    print("\nTop 5 Most Favourable Configurations (Lowest Coupling):")
    for r in summary["top_favourable_configurations"][:5]:
        print(
            f"  Rank {r['Risk_Rank']:>2}: {r['Config_ID']} | Sep={r['PairSeparation_mm']}mm, "
            f"Pitch={r['PairPitch_mm']}mm, Overlap={r['Overlap_mm']}mm, Ground={r['GroundClearance_mm']}mm | "
            f"C={r[CAP_COL]:.2f} pF/m, M={r[MUT_COL]:.4f} nH | Score={r['Combined_Risk_Indicator']:.4f}"
        )

    print("\nTop 5 Highest Coupling Indicator Configurations:")
    for r in summary["top_unfavourable_configurations"][:5]:
        print(
            f"  Rank {r['Risk_Rank']:>2}: {r['Config_ID']} | Sep={r['PairSeparation_mm']}mm, "
            f"Pitch={r['PairPitch_mm']}mm, Overlap={r['Overlap_mm']}mm, Ground={r['GroundClearance_mm']}mm | "
            f"C={r[CAP_COL]:.2f} pF/m, M={r[MUT_COL]:.4f} nH | Score={r['Combined_Risk_Indicator']:.4f}"
        )
