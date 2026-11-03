"""
app.py
------
EMC-PREDICT + EMC-GUARD  –  Main Streamlit Application Entry Point

Physics-Informed Configuration-Aware EMC Digital Twin for Elevator Cable Systems.
Integrated with Stages 1–4 AI/ML Verification, Evaluation, XAI, Risk Assessment,
and Intelligent Configuration Optimisation.

Run:  streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Ensure project root and AI_ML/src are on sys.path ───────────────────────
_CURRENT_FILE = Path(__file__).resolve()
_CURRENT_DIR = _CURRENT_FILE.parent

# Detect repo root robustly whether file is in Dashboard/, EMC_PROJECT/, or root
if (_CURRENT_DIR / "EMC_PROJECT").exists():
    _REPO_ROOT = _CURRENT_DIR
elif (_CURRENT_DIR.parent / "EMC_PROJECT").exists():
    _REPO_ROOT = _CURRENT_DIR.parent
else:
    _REPO_ROOT = _CURRENT_FILE.parents[1]

_EMC_PROJECT = _REPO_ROOT / "EMC_PROJECT"
_AIML_SRC = _REPO_ROOT / "AI_ML" / "src"
_AIML_REPORTS = _REPO_ROOT / "AI_ML" / "reports"
_AIML_FIGURES = _AIML_REPORTS / "figures"
_DATA_DIR = _EMC_PROJECT / "data"

for p in [str(_CURRENT_DIR), str(_EMC_PROJECT), str(_AIML_SRC), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Project imports (EMC_PROJECT) ───────────────────────────────────────────
from models.model_loader import (
    get_model_status,
    load_extratrees_models,
    load_frozen_matlab_model,
    load_physics_dataset,
)
from models.prediction_engine import (
    SUPPORTED_CLEARANCES,
    SUPPORTED_OVERLAPS,
    SUPPORTED_PITCHES,
    SUPPORTED_SEPARATIONS,
    predict,
    predict_all_configs,
)
from core.emc_guard import (
    OBJECTIVE_LABELS,
    enumerate_all_configurations,
    predict_candidates,
    run_emc_guard_assessment,
    score_configurations,
)
from digital_twin.geometry import build_3d_figure, build_cross_section_figure

try:
    from core.report_generator import REPORTLAB_AVAILABLE, generate_report
except Exception:
    REPORTLAB_AVAILABLE = False

# ── AI_ML Stage 3 & 4 module imports ────────────────────────────────────────
try:
    import risk_assessment as ra
    import optimise as opt
    AIML_MODULES_AVAILABLE = True
except Exception as _e:
    AIML_MODULES_AVAILABLE = False
    logging.warning("AI_ML modules could not be imported: %s", _e)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# SAFE HTML RENDERING HELPER
# ---------------------------------------------------------------------------
def render_html(html_str: str, sidebar: bool = False) -> None:
    """
    Safely render custom HTML/CSS in Streamlit.
    Bypasses CommonMark markdown parsing (which mistakenly converts 4-space indentation
    into preformatted <pre><code> blocks and breaks at blank lines).
    Uses st.html() (Streamlit >= 1.34) or dedented st.markdown(..., unsafe_allow_html=True).
    """
    target = st.sidebar if sidebar else st
    clean_html = textwrap.dedent(html_str).strip()
    if hasattr(target, "html"):
        target.html(clean_html)
    else:
        target.markdown(clean_html, unsafe_allow_html=True)


# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EMC-PREDICT + EMC-GUARD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS (dark industrial theme)
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.main { background: #0F172A; }
[data-testid="stSidebar"] { background: #1E293B; border-right: 1px solid #334155; }
[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

/* ── Header banner ── */
.banner {
    background: linear-gradient(135deg, #1E3A5F 0%, #0F172A 100%);
    border: 1px solid #2563EB;
    border-radius: 8px;
    padding: 18px 24px 14px;
    margin-bottom: 20px;
}
.banner h1 { color: #E2E8F0; font-size: 1.6rem; margin: 0; font-weight: 700; }
.banner p  { color: #94A3B8; font-size: 0.9rem; margin: 4px 0 0; }

/* ── Metric cards ── */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.metric-card .label { color: #64748B; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { color: #E2E8F0; font-size: 1.4rem; font-weight: 700; margin: 4px 0; }
.metric-card .unit  { color: #94A3B8; font-size: 0.8rem; }

/* ── Status badges ── */
.badge-ok   { display:inline-block; background:#166534; color:#86EFAC; border-radius:4px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-err  { display:inline-block; background:#7F1D1D; color:#FCA5A5; border-radius:4px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-pend { display:inline-block; background:#78350F; color:#FCD34D; border-radius:4px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-info { display:inline-block; background:#1E3A8A; color:#93C5FD; border-radius:4px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* ── Notice boxes ── */
.notice {
    background: #1C2B4B;
    border-left: 4px solid #3B82F6;
    padding: 10px 14px;
    border-radius: 4px;
    color: #93C5FD;
    font-size: 0.85rem;
    margin: 8px 0;
}
.warning-box {
    background: #2D1B00;
    border-left: 4px solid #F59E0B;
    padding: 10px 14px;
    border-radius: 4px;
    color: #FCD34D;
    font-size: 0.85rem;
    margin: 8px 0;
}
.danger-box {
    background: #3B0D0D;
    border-left: 4px solid #EF4444;
    padding: 10px 14px;
    border-radius: 4px;
    color: #FCA5A5;
    font-size: 0.85rem;
    margin: 8px 0;
}

/* ── Section divider ── */
.section-divider { border-top: 1px solid #334155; margin: 20px 0; }

/* ── Streamlit overrides ── */
.stSelectbox label, .stSlider label, .stNumberInput label { color: #94A3B8 !important; }
.stButton > button {
    background: #2563EB; color: white; border: none;
    border-radius: 6px; padding: 8px 20px; font-weight: 600;
}
.stButton > button:hover { background: #1D4ED8; }
div[data-testid="metric-container"] {
    background: #1E293B; border-radius: 8px; padding: 12px;
    border: 1px solid #334155;
}
</style>
""")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION (9 SCREENS PRESERVED)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_html("""
    <div style='text-align:center; padding:8px 0 16px;'>
        <div style='color:#2563EB; font-size:1.8rem;'>⚡</div>
        <div style='color:#E2E8F0; font-weight:700; font-size:1.05rem;'>EMC-PREDICT</div>
        <div style='color:#94A3B8; font-size:0.8rem;'>+ EMC-GUARD AI/ML</div>
    </div>
    <hr style='border-color:#334155; margin:0 0 14px;'>
    """)

    SCREENS = {
        "1 · Executive Overview":          "overview",
        "2 · Configuration Manager":       "config",
        "3 · EMC-PREDICT Engine":          "predict",
        "4 · Digital Twin Geometry":       "geometry",
        "5 · EMC-GUARD Assessment":        "guard",
        "6 · Configuration Comparison":    "compare",
        "7 · Model Validation":            "validation",
        "8 · ANSYS Integration":           "ansys",
        "9 · Report Generation":           "report",
    }
    selected = st.radio("Navigation", list(SCREENS.keys()), label_visibility="collapsed")
    screen = SCREENS[selected]

    render_html("<hr style='border-color:#334155;'>", sidebar=True)
    render_html("""
    <div style='font-size:0.74rem; color:#475569; padding:0 4px;'>
        <b style='color:#64748B;'>Platform</b> KONE EMC-GUARD v2.0<br>
        <b style='color:#64748B;'>Primary Engine</b> Stage 12A MATLAB (Frozen)<br>
        <b style='color:#64748B;'>Surrogate</b> Stage 14A ExtraTrees<br>
        <b style='color:#64748B;'>Dataset</b> 216 Analytical Configurations<br>
        <b style='color:#64748B;'>Pareto Set</b> 6 Non-Dominated Candidates
    </div>
    """, sidebar=True)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _get_dataset() -> pd.DataFrame:
    return load_physics_dataset()

@st.cache_resource
def _get_model_status() -> dict:
    return get_model_status()

@st.cache_data(ttl=300)
def _get_risk_assessed_df() -> pd.DataFrame:
    if AIML_MODULES_AVAILABLE:
        return ra.calculate_risk_indicators(_get_dataset())
    return _get_dataset()

@st.cache_data(ttl=300)
def _get_pareto_front_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if AIML_MODULES_AVAILABLE:
        full_df = ra.calculate_risk_indicators(_get_dataset())
        all_pareto = opt.compute_pareto_dominance(full_df)
        pareto_front = all_pareto[all_pareto["Is_Pareto_Optimal"]].copy()
        scored_pareto = opt.score_pareto_candidates(pareto_front, all_pareto)
        return all_pareto, scored_pareto
    return pd.DataFrame(), pd.DataFrame()


def get_config_id(sep: float, pitch: float, overlap: float, ground: float) -> str:
    """Helper to lookup the Config_ID from geometry parameters."""
    ds = _get_dataset()
    m = ds[
        (ds["PairSeparation_mm"] == sep) &
        (ds["PairPitch_mm"] == pitch) &
        (ds["Overlap_mm"] == overlap) &
        (ds["GroundClearance_mm"] == ground)
    ]
    if len(m) > 0:
        return str(m.iloc[0]["Config_ID"])
    return "-"

def config_inputs(
    key_prefix: str = "main",
    default_indices: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[float, float, float, float]:
    """Render cable configuration selectors. Returns (sep, pitch, overlap, ground)."""
    idx_s, idx_p, idx_o, idx_g = default_indices if default_indices else (0, 0, 0, 0)
    # Clamp indices to valid range
    idx_s = max(0, min(idx_s, len(SUPPORTED_SEPARATIONS) - 1))
    idx_p = max(0, min(idx_p, len(SUPPORTED_PITCHES) - 1))
    idx_o = max(0, min(idx_o, len(SUPPORTED_OVERLAPS) - 1))
    idx_g = max(0, min(idx_g, len(SUPPORTED_CLEARANCES) - 1))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sep = st.selectbox(
            "Pair Separation (mm)", SUPPORTED_SEPARATIONS,
            index=idx_s, key=f"{key_prefix}_sep",
            help="Centre-to-centre distance between source and victim cable pairs."
        )
    with col2:
        pitch = st.selectbox(
            "Cable Pitch", SUPPORTED_PITCHES,
            index=idx_p, key=f"{key_prefix}_pitch",
            help="Structural pitch / vertical twist stagger parameter."
        )
    with col3:
        overlap = st.selectbox(
            "Parallel Overlap (mm)", SUPPORTED_OVERLAPS,
            index=idx_o, key=f"{key_prefix}_overlap",
            help="Length of parallel cable run region."
        )
    with col4:
        ground = st.selectbox(
            "Ground Clearance (mm)", SUPPORTED_CLEARANCES,
            index=idx_g, key=f"{key_prefix}_ground",
            help="Height of cable centre above reference ground plane."
        )
    return float(sep), float(pitch), float(overlap), float(ground)


def render_banner(subtitle: str = ""):
    render_html(f"""
    <div class="banner">
        <h1>⚡ EMC-PREDICT + EMC-GUARD</h1>
        <p>Physics-Informed Configuration-Aware EMC Digital Twin for Elevator Cable Systems
        {'&nbsp;·&nbsp;' + subtitle if subtitle else ''}</p>
    </div>
    """)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 1  –  EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if screen == "overview":
    render_banner("Executive Overview")

    status = _get_model_status()

    def _badge(ok, err=None):
        if ok:   return '<span class="badge-ok">✓ Available</span>'
        if err:  return '<span class="badge-err">✗ Error</span>'
        return '<span class="badge-pend">⋯ Pending</span>'

    # ── Primary Status Indicators ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_html(f"""<div class="metric-card">
            <div class="label">Frozen MATLAB Engine</div>
            <div style='margin:8px 0'>{_badge(status['frozen_matlab']['ok'], status['frozen_matlab']['error'])}</div>
            <div class="unit">Stage 12A · Primary (Machine Precision)</div>
        </div>""")
    with c2:
        render_html(f"""<div class="metric-card">
            <div class="label">ExtraTrees Surrogate</div>
            <div style='margin:8px 0'>{_badge(status['extratrees']['ok'], status['extratrees']['error'])}</div>
            <div class="unit">Stage 14A · R²=1.000 (In-Sample)</div>
        </div>""")
    with c3:
        n = status['physics_dataset']['n_configs']
        render_html(f"""<div class="metric-card">
            <div class="label">Physics Dataset</div>
            <div class="value">{n}</div>
            <div class="unit">MATLAB Analytical Grid</div>
        </div>""")
    with c4:
        render_html("""<div class="metric-card">
            <div class="label">Physical Validation</div>
            <div style='margin:8px 0'><span class="badge-pend">⋯ Simulation Only</span></div>
            <div class="unit">No empirical field data</div>
        </div>""")

    # ── AI/ML Stage 2–4 Status Badges ──────────────────────────────────────
    render_html("<div style='margin-top:14px'></div>")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_html("""<div class="metric-card">
            <div class="label">LOLO Validation</div>
            <div style='margin:6px 0'><span class="badge-ok">✓ Complete</span></div>
            <div class="unit">Evaluated across all levels</div>
        </div>""")
    with s2:
        render_html("""<div class="metric-card">
            <div class="label">XAI & Feature Importance</div>
            <div style='margin:6px 0'><span class="badge-ok">✓ Active</span></div>
            <div class="unit">MDI, PFI, ICE & PDP</div>
        </div>""")
    with s3:
        render_html("""<div class="metric-card">
            <div class="label">Relative EMC Risk</div>
            <div style='margin:6px 0'><span class="badge-ok">✓ Operational</span></div>
            <div class="unit">Low / Mod / High Tiers</div>
        </div>""")
    with s4:
        render_html("""<div class="metric-card">
            <div class="label">Pareto Optimisation</div>
            <div style='margin:6px 0'><span class="badge-ok">✓ 6 Candidates</span></div>
            <div class="unit">Non-dominated Frontier</div>
        </div>""")

    render_html("<div class='section-divider'></div>")

    # ── System Architecture ────────────────────────────────────────────────
    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.subheader("System Architecture")
        render_html("""
        ```
        ┌────────────────────────────────────────────────────────┐
        │            KONE EMC-GUARD PLATFORM v2.0                │
        ├──────────────┬─────────────────────────────────────────┤
        │  EMC-PREDICT │  Physics-Informed Prediction Engine     │
        │              │  ├─ Stage 12A Frozen MATLAB Model       │
        │              │  │   ├─ B2_SP_Ground  (Capacitance)    │
        │              │  │   └─ B1_SP_Overlap (Inductance)     │
        │              │  └─ Stage 14A ExtraTrees Surrogates     │
        ├──────────────┼─────────────────────────────────────────┤
        │  AI/ML XAI   │  Explainable AI & Evaluation (Stage 2) │
        │              │  ├─ Leave-One-Level-Out (LOLO) CV       │
        │              │  ├─ MDI & Permutation Feature Imp.     │
        │              │  └─ ICE, PDP & OAT Sensitivity Sweeps   │
        ├──────────────┼─────────────────────────────────────────┤
        │  RISK ENGINE │  EMC Risk Assessment (Stage 3)          │
        │              │  ├─ Relative Capacitive/Inductive Tiers │
        │              │  └─ Explainable Geometric Attribution   │
        ├──────────────┼─────────────────────────────────────────┤
        │  OPTIMISER   │  Intelligent Optimisation (Stage 4)     │
        │              │  ├─ Multi-objective Pareto Frontier     │
        │              │  └─ Non-dominated candidate ranking     │
        ├──────────────┼─────────────────────────────────────────┤
        │  DIGITAL     │  3D Digital Twin Geometry               │
        │  TWIN        │  (Configurable interactive spatial view)│
        ├──────────────┼─────────────────────────────────────────┤
        │  ANSYS I/F   │  (Ready for future Maxwell/Q3D data)    │
        └──────────────┴─────────────────────────────────────────┘
        ```
        """)

    with col_b:
        st.subheader("Prediction Parameters")
        param_data = {
            "Parameter":   ["Total Capacitance", "Abs. Mutual Inductance"],
            "Unit":        ["pF/m", "nH"],
            "Primary Model": ["Stage 12A Frozen MATLAB", "Stage 12A Frozen MATLAB"],
            "Surrogate":   ["ExtraTrees Regressor", "ExtraTrees Regressor"],
            "In-Sample R²": ["1.000000", "1.000000"],
            "LOLO Gen.":   ["High except GC=10mm", "High on interior Sep"],
        }
        st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

        st.subheader("Supported Design Domain")
        domain_data = {
            "Parameter":  ["Pair Separation", "Twist Pitch", "Parallel Overlap", "Ground Clearance"],
            "Levels":     [
                "8, 12, 16, 20, 24, 28 mm",
                "2, 3, 4",
                "150, 300, 450 mm",
                "10, 25, 50, 100 mm"
            ],
            "Count":      [6, 3, 3, 4],
        }
        st.dataframe(pd.DataFrame(domain_data), use_container_width=True, hide_index=True)

    # ── Scientific Limitations Notice ──────────────────────────────────────
    st.markdown("""<div class="warning-box">
    <b>⚠ Scientific Limitations & Boundary Statement</b><br>
    This platform predicts selected electromagnetic coupling parameters (total capacitance and absolute
    mutual inductance) across a discrete 216-configuration dataset derived from deterministic MATLAB
    analytical field calculations. It does <b>not</b> predict victim terminal voltage, conducted emissions,
    or radiated emission compliance. <b>No physical elevator field measurements, regulatory certifications
    (e.g., EN 12015/12016), or safety guarantees have been established.</b>
    </div>""")


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 2  –  CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "config":
    render_banner("Configuration Manager")

    st.subheader("Select Cable Configuration")
    sep, pitch, overlap, ground = config_inputs("cfg")

    df = _get_dataset()
    mask = (
        np.isclose(df["PairSeparation_mm"],   sep,     atol=1e-6)
        & np.isclose(df["PairPitch_mm"],       pitch,   atol=1e-6)
        & np.isclose(df["Overlap_mm"],         overlap, atol=1e-6)
        & np.isclose(df["GroundClearance_mm"], ground,  atol=1e-6)
    )
    matched = df[mask]
    if len(matched) == 1:
        row = matched.iloc[0]
        config_id = row["Config_ID"]
        render_html(f"""<div class="notice">
            ✓ <b>Valid configuration</b> &nbsp;·&nbsp; ID: <b>{config_id}</b>
            &nbsp;·&nbsp; Configuration index: {matched.index[0] + 1} of 216
        </div>""")
    else:
        st.error("Configuration not found in the 216-point design grid.")
        config_id = "—"

    render_html("<div class='section-divider'></div>")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Configuration Summary")
        summary = pd.DataFrame({
            "Parameter":  [
                "Pair Separation", "Cable Pitch", "Parallel Overlap",
                "Ground Clearance", "Configuration ID"
            ],
            "Value":      [
                f"{sep} mm", f"{pitch}", f"{overlap} mm", f"{ground} mm", config_id
            ],
            "Description": [
                "Horizontal distance between cable pair centres",
                "Twist pitch / vertical stagger parameter",
                "Length of parallel cable run",
                "Height of cable centre above ground plane",
                "Reference ID in 216-configuration dataset",
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Dataset Reference Values")
        if len(matched) == 1:
            ref_vals = pd.DataFrame({
                "Parameter": ["Total Capacitance", "Abs. Mutual Inductance"],
                "Reference Value": [
                    f"{row['TotalCapacitance_pF_per_m']:.6g}",
                    f"{row['AbsMutualInductance_nH']:.6g}",
                ],
                "Unit": ["pF/m", "nH"],
                "Source": ["MATLAB analytical model", "MATLAB analytical model"],
            })
            st.dataframe(ref_vals, use_container_width=True, hide_index=True)

            # Stage 3 Risk Indicator Quick Preview
            if AIML_MODULES_AVAILABLE:
                risk_res = ra.assess_configuration(sep, pitch, overlap, ground)
                if risk_res.get("status") == "OK":
                    tier = risk_res["combined_assessment"]["combined_risk_tier"]
                    badge_class = "badge-ok" if "Low" in tier else ("badge-pend" if "Mod" in tier else "badge-err")
                    render_html(f"""
                    <div style='margin-top:14px; padding:10px 14px; background:#1E293B; border-radius:6px; border:1px solid #334155;'>
                        <b style='color:#94A3B8; font-size:0.8rem; text-transform:uppercase;'>Relative EMC Risk Tier</b><br>
                        <span class="{badge_class}">{tier}</span>
                        <span style='color:#CBD5E1; font-size:0.85rem; margin-left:8px;'>
                            (Score: {risk_res['combined_assessment']['combined_risk_indicator']:.4f})
                        </span>
                    </div>
                    """)
        else:
            st.info("Select a valid configuration to see reference values.")

    render_html("""<div class="notice">
    <b>ℹ Domain policy:</b> Only the 216 discrete configurations from the MATLAB analytical dataset
    are supported. Interpolation and extrapolation outside this grid are strictly rejected.
    </div>""")


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 3  –  EMC-PREDICT ENGINE
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "predict":
    render_banner("EMC-PREDICT Engine")

    st.subheader("Prediction Configuration")
    sep, pitch, overlap, ground = config_inputs("pred")

    col_eng, col_btn = st.columns([3, 1])
    with col_eng:
        engine_choice = st.radio(
            "Prediction Engine",
            ["frozen_matlab", "extratrees"],
            format_func=lambda x: {
                "frozen_matlab": "Stage 12A – Frozen MATLAB Linear Model (Primary / Analytical Precision)",
                "extratrees":    "Stage 14A – ExtraTrees Surrogate (Secondary / ML Approximation)",
            }[x],
            horizontal=True,
        )
    with col_btn:
        render_html("<div style='height:26px'></div>")
        run_clicked = st.button("▶  Run Prediction", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Executing prediction…"):
            result = predict(sep, pitch, overlap, ground, engine=engine_choice)

        if result["status"] != "OK":
            st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
        else:
            render_html("<div class='section-divider'></div>")

            config_id = result["config_id"]
            cap = result["predictions"]["TotalCapacitance_pF_per_m"]
            mut = result["predictions"]["AbsMutualInductance_nH"]
            ts  = result["timestamp"]

            st.subheader("Predicted Outputs")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                render_html(f"""<div class="metric-card">
                    <div class="label">Total Capacitance</div>
                    <div class="value">{cap['value']:.5g}</div>
                    <div class="unit">pF/m</div>
                </div>""")
            with mc2:
                render_html(f"""<div class="metric-card">
                    <div class="label">Abs. Mutual Inductance</div>
                    <div class="value">{mut['value']:.5g}</div>
                    <div class="unit">nH</div>
                </div>""")
            with mc3:
                render_html(f"""<div class="metric-card">
                    <div class="label">Configuration ID</div>
                    <div class="value" style='font-size:1.2rem'>{config_id}</div>
                    <div class="unit">{engine_choice}</div>
                </div>""")

            # ── Detailed table ────────────────────────────────────────────
            st.subheader("Prediction Detail vs Analytical Reference")
            detail = pd.DataFrame({
                "Parameter": ["Total Capacitance", "Abs. Mutual Inductance"],
                "Predicted": [f"{cap['value']:.8g}", f"{mut['value']:.8g}"],
                "MATLAB Reference": [f"{cap['reference']:.8g}", f"{mut['reference']:.8g}"],
                "Absolute Difference": [f"{cap['abs_diff']:.3e}", f"{mut['abs_diff']:.3e}"],
                "Unit":  [cap['unit'], mut['unit']],
                "Model": [cap['model'], mut['model']],
            })
            st.dataframe(detail, use_container_width=True, hide_index=True)

            # ── Integrated Stage 3 Risk & XAI Attribution ─────────────────
            if AIML_MODULES_AVAILABLE:
                st.subheader("Relative EMC Risk Assessment & Geometric Attribution")
                risk_eval = ra.assess_configuration(sep, pitch, overlap, ground, engine=engine_choice)
                if risk_eval.get("status") == "OK":
                    comb = risk_eval["combined_assessment"]
                    tier = comb["combined_risk_tier"]
                    tier_badge = "badge-ok" if "Low" in tier else ("badge-pend" if "Mod" in tier else "badge-err")

                    rk1, rk2, rk3 = st.columns(3)
                    with rk1:
                        c_ind = risk_eval["predictions"]["TotalCapacitance_pF_per_m"]
                        render_html(f"""<div class="metric-card">
                            <div class="label">Capacitance Relative Indicator</div>
                            <div class="value">{c_ind['relative_indicator']:.3f}</div>
                            <div class="unit">Percentile: {c_ind['percentile_in_dataset']:.1f}% · {c_ind['risk_tier']}</div>
                        </div>""")
                    with rk2:
                        m_ind = risk_eval["predictions"]["AbsMutualInductance_nH"]
                        render_html(f"""<div class="metric-card">
                            <div class="label">Inductance Relative Indicator</div>
                            <div class="value">{m_ind['relative_indicator']:.3f}</div>
                            <div class="unit">Percentile: {m_ind['percentile_in_dataset']:.1f}% · {m_ind['risk_tier']}</div>
                        </div>""")
                    with rk3:
                        render_html(f"""<div class="metric-card">
                            <div class="label">Combined Relative Tier</div>
                            <div style='margin:6px 0'><span class="{tier_badge}">{tier}</span></div>
                            <div class="unit">Combined Score: {comb['combined_risk_indicator']:.4f}</div>
                        </div>""")

                    with st.expander("🔍 Explainable AI (XAI) Parameter Rationale", expanded=True):
                        render_html(f"**Engineering Rationale:** {risk_eval['explainable_attribution']['engineering_rationale']}")
                        st.markdown("**Capacitive Driving Mechanisms:**")
                        for d in risk_eval["explainable_attribution"]["capacitance_drivers"]:
                            st.markdown(f"- {d}")
                        st.markdown("**Inductive Driving Mechanisms:**")
                        for d in risk_eval["explainable_attribution"]["mutual_inductance_drivers"]:
                            st.markdown(f"- {d}")

            # ── Model Info & Disclaimer ───────────────────────────────────
            info = result["model_info"]
            st.markdown(f"""<div class="notice">
            <b>Model Metadata:</b> Engine: <b>{info['engine']}</b> · Domain: {info['domain']} ·
            Column Index: {info['column_index']} · Timestamp: {ts}
            </div>""")

            render_html(f"""<div class="warning-box">
            ⚠ {result['validation_notice']}
            </div>""")

            # Store in session state
            st.session_state["last_prediction"] = result
            st.session_state["last_config"] = (sep, pitch, overlap, ground)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 4  –  DIGITAL TWIN GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "geometry":
    render_banner("Digital Twin Geometry Visualisation")

    render_html("""<div class="notice">
    Interactive geometric visual representation of the cable bundle configuration updated
    in real time as parameters change. It is <b>not</b> a numerical electromagnetic field solver.
    </div>""")

    sep, pitch, overlap, ground = config_inputs("geom")

    tab3d, tab2d = st.tabs(["3D View", "Cross-Section (XY)"])

    with tab3d:
        with st.spinner("Building 3D geometric model…"):
            fig3d = build_3d_figure(sep, pitch, overlap, ground)
            fig3d.update_layout(paper_bgcolor="#0F172A", font=dict(color="#CBD5E1"))
        st.plotly_chart(fig3d, use_container_width=True)

        render_html(f"""
        | Geometric Parameter | Configuration Value |
        |---|---|
        | **Source Cable** | Blue · Centre at (0, {ground} mm) |
        | **Victim Cable** | Orange · Centre at ({sep} mm, {ground + pitch} mm) |
        | **Parallel Run (Overlap)** | {overlap} mm |
        | **Ground Clearance** | {ground} mm |
        | **Conductor Diameter** | ~6 mm (representative bundle) |
        """)

    with tab2d:
        fig2d = build_cross_section_figure(sep, pitch, ground)
        fig2d.update_layout(paper_bgcolor="#0F172A", font=dict(color="#CBD5E1"))
        st.plotly_chart(fig2d, use_container_width=True)

    st.markdown("""<div class="warning-box">
    ⚠ The cable cross-section geometry is a representative engineering rendering and has not been
    experimentally validated as an exact manufactured KONE cable harness.
    </div>""")


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 5  –  EMC-GUARD ASSESSMENT & PARETO OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "guard":
    render_banner("EMC-GUARD Configuration Assessment & Optimisation")

    tab_assess, tab_pareto = st.tabs([
        "Multi-Objective Assessment & Search",
        "Intelligent Pareto Optimisation (Stage 4)"
    ])

    with tab_assess:
        render_html("""<div class="notice">
        EMC-GUARD ranks alternative cable configurations based on predicted electromagnetic coupling.
        <b>No compliance, safety, or certification determinations are made.</b>
        </div>""")

        st.subheader("Reference Configuration")
        sep, pitch, overlap, ground = config_inputs("guard")

        st.subheader("Assessment Objective")
        col_obj, col_wc, col_wm = st.columns([2, 1, 1])
        with col_obj:
            objective = st.selectbox(
                "Ranking Objective", list(OBJECTIVE_LABELS.keys()),
                format_func=lambda k: OBJECTIVE_LABELS[k]
            )
        with col_wc:
            w_cap = st.slider("Capacitance weight", 0.0, 1.0, 0.5, 0.05,
                              disabled=(objective != "minimize_combined"))
        with col_wm:
            w_mut = st.slider("Inductance weight", 0.0, 1.0, 0.5, 0.05,
                              disabled=(objective != "minimize_combined"))

        st.subheader("Installation Constraints (optional)")
        col_cs, col_co, col_cg = st.columns(3)
        with col_cs:
            allowed_sep = st.multiselect("Allowed separations (mm)", SUPPORTED_SEPARATIONS, default=SUPPORTED_SEPARATIONS)
        with col_co:
            allowed_ov  = st.multiselect("Allowed overlaps (mm)", SUPPORTED_OVERLAPS, default=SUPPORTED_OVERLAPS)
        with col_cg:
            allowed_gc  = st.multiselect("Allowed clearances (mm)", SUPPORTED_CLEARANCES, default=SUPPORTED_CLEARANCES)

        engine_g = st.radio(
            "Prediction engine", ["frozen_matlab", "extratrees"],
            format_func=lambda x: {"frozen_matlab": "Frozen MATLAB (primary)", "extratrees": "ExtraTrees (secondary)"}[x],
            horizontal=True, key="guard_engine"
        )

        if st.button("▶  Run EMC-GUARD Assessment", type="primary"):
            with st.spinner("Evaluating candidate configurations…"):
                result = run_emc_guard_assessment(
                    sep, pitch, overlap, ground,
                    objective=objective, w_cap=w_cap, w_mut=w_mut,
                    engine=engine_g,
                    constrain_separation=allowed_sep or None,
                    constrain_overlap=allowed_ov or None,
                    constrain_clearance=allowed_gc or None,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                render_html("<div class='section-divider'></div>")
                st.subheader("Recommendation")
                render_html(f"""<div class="notice">💡 {result['recommendation']}</div>""")

                ref = result["reference"]
                top = result["top_config"]

                c1, c2, c3, c4 = st.columns(4)
                def _delta_label(new, old):
                    d = new - old
                    pct = 100 * d / old if old != 0 else 0
                    sign = "▲" if d > 0 else "▼"
                    return f"{sign} {abs(pct):.1f}%"

                with c1:
                    st.metric("Candidates evaluated", result["n_candidates"])
                with c2:
                    st.metric("Ref. Capacitance", f"{ref['cap']:.4g} pF/m")
                with c3:
                    st.metric("Top Cap. Predicted", f"{top['TotalCapacitance_pF_per_m']:.4g} pF/m",
                              delta=_delta_label(top['TotalCapacitance_pF_per_m'], ref['cap']))
                with c4:
                    st.metric("Top Mut. Predicted", f"{top['AbsMutualInductance_nH']:.4g} nH",
                              delta=_delta_label(top['AbsMutualInductance_nH'], ref['mut']))

                ranked = result["ranked_df"]
                st.subheader("Ranked Configurations (Top 20)")
                display_cols = [
                    "Rank", "Config_ID", "PairSeparation_mm", "PairPitch_mm",
                    "Overlap_mm", "GroundClearance_mm",
                    "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH",
                    "ObjectiveScore", "CouplingAssessment",
                ]
                disp = ranked[[c for c in display_cols if c in ranked.columns]].head(20)
                st.dataframe(disp.style.format({
                    "TotalCapacitance_pF_per_m": "{:.4g}",
                    "AbsMutualInductance_nH":    "{:.4g}",
                    "ObjectiveScore":             "{:.4f}",
                }), use_container_width=True, hide_index=True)

                st.session_state["guard_result"] = result

    # ── TAB 2: INTELLIGENT PARETO OPTIMISATION (STAGE 4) ────────────────────
    with tab_pareto:
        st.subheader("Multi-Objective Pareto Frontier Analysis")
        render_html("""<div class="notice">
        <b>Pareto Optimality Principle:</b> A configuration is Pareto-optimal if no other configuration
        in the 216-dataset can reduce mutual inductance without simultaneously increasing capacitance.
        </div>""")

        if AIML_MODULES_AVAILABLE:
            all_pareto_df, pareto_front_df = _get_pareto_front_data()

            if not pareto_front_df.empty:
                # Pareto Metric Highlights
                p1, p2, p3 = st.columns(3)
                min_c = pareto_front_df.loc[pareto_front_df["TotalCapacitance_pF_per_m"].idxmin()]
                min_m = pareto_front_df.loc[pareto_front_df["AbsMutualInductance_nH"].idxmin()]
                best_bal = pareto_front_df.loc[pareto_front_df["Weighted_Tradeoff_Score"].idxmin()]

                with p1:
                    render_html(f"""<div class="metric-card">
                        <div class="label">Min. Capacitance Champion</div>
                        <div class="value">{min_c['TotalCapacitance_pF_per_m']:.2f} <span style='font-size:0.9rem'>pF/m</span></div>
                        <div class="unit">{min_c['Config_ID']} · Sep={min_c['PairSeparation_mm']:.0f}mm · Mut={min_c['AbsMutualInductance_nH']:.3f}nH</div>
                    </div>""")
                with p2:
                    render_html(f"""<div class="metric-card">
                        <div class="label">Min. Inductance Champion</div>
                        <div class="value">{min_m['AbsMutualInductance_nH']:.4f} <span style='font-size:0.9rem'>nH</span></div>
                        <div class="unit">{min_m['Config_ID']} · Sep={min_m['PairSeparation_mm']:.0f}mm · Cap={min_m['TotalCapacitance_pF_per_m']:.2f}pF/m</div>
                    </div>""")
                with p3:
                    render_html(f"""<div class="metric-card">
                        <div class="label">Recommended Balanced Design</div>
                        <div class="value">{best_bal['Config_ID']}</div>
                        <div class="unit">Sep={best_bal['PairSeparation_mm']:.0f}mm · Cap={best_bal['TotalCapacitance_pF_per_m']:.2f} · Mut={best_bal['AbsMutualInductance_nH']:.4f}</div>
                    </div>""")

                render_html("<div style='margin-top:16px'></div>")

                # Interactive Plotly Pareto Front Chart
                fig_pf = go.Figure()
                dominated = all_pareto_df[~all_pareto_df["Is_Pareto_Optimal"]]
                fig_pf.add_trace(go.Scatter(
                    x=dominated["TotalCapacitance_pF_per_m"],
                    y=dominated["AbsMutualInductance_nH"],
                    mode="markers",
                    marker=dict(color="#475569", size=6, opacity=0.6),
                    name="Dominated (210)",
                    text=[f"{r.Config_ID}: C={r.TotalCapacitance_pF_per_m:.2f}, M={r.AbsMutualInductance_nH:.3f}" for _, r in dominated.iterrows()],
                    hoverinfo="text",
                ))
                p_sorted = pareto_front_df.sort_values("TotalCapacitance_pF_per_m")
                fig_pf.add_trace(go.Scatter(
                    x=p_sorted["TotalCapacitance_pF_per_m"],
                    y=p_sorted["AbsMutualInductance_nH"],
                    mode="lines+markers+text",
                    line=dict(color="#EF4444", width=2.5, dash="dash"),
                    marker=dict(size=12, color="#EF4444", symbol="diamond"),
                    text=[f"{r.Config_ID} ({r.PairSeparation_mm:.0f}mm)" for _, r in p_sorted.iterrows()],
                    textposition="top right",
                    textfont=dict(color="#F87171", size=11),
                    name="Pareto Optimal Front (6)",
                ))
                fig_pf.update_layout(
                    title="<b>Multi-Objective Pareto Optimisation Frontier</b>",
                    xaxis=dict(title="Total Capacitance [pF/m] (Minimise →)", gridcolor="#334155"),
                    yaxis=dict(title="Absolute Mutual Inductance [nH] (Minimise →)", gridcolor="#334155"),
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), height=450,
                )
                st.plotly_chart(fig_pf, use_container_width=True)

                # Pareto Candidates Table
                st.subheader("The 6 Non-Dominated Pareto Optimal Configurations")
                p_disp = pareto_front_df[[
                    "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
                    "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH", "Candidate_Role", "Inclusion_Rationale"
                ]]
                st.dataframe(p_disp, use_container_width=True, hide_index=True)

                # Download Pareto CSV
                csv_bytes = pareto_front_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇  Download Pareto Optimal Configurations CSV",
                    data=csv_bytes,
                    file_name="pareto_optimal_configurations.csv",
                    mime="text/csv",
                )
        else:
            st.info("Stage 4 optimisation module is loading or unavailable.")


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 6  –  CONFIGURATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "compare":
    render_banner("Configuration Comparison")

    st.subheader("Select Configurations to Compare")
    n_configs = st.slider("Number of configurations", 2, 6, 3)

    # Distinct preset defaults so Config 1, Config 2, Config 3 start with distinct configurations
    # Config 1: EMC-001 (Baseline: 8mm, pitch 2, 150mm overlap, 10mm GC)
    # Config 2: EMC-086 (Moderate: 16mm, pitch 3, 300mm overlap, 25mm GC)
    # Config 3: EMC-190 (Isolated: 28mm, pitch 4, 150mm overlap, 100mm GC)
    DEFAULT_COMP_INDICES = [
        (0, 0, 0, 0),  # EMC-001
        (2, 1, 1, 1),  # EMC-086
        (5, 2, 0, 3),  # EMC-190
        (1, 0, 2, 2),  # EMC-043
        (3, 2, 1, 0),  # EMC-122
        (4, 1, 0, 1),  # EMC-146
    ]

    configs = []
    for i in range(n_configs):
        def_idx = DEFAULT_COMP_INDICES[i % len(DEFAULT_COMP_INDICES)]
        with st.expander(f"Configuration {i+1}", expanded=(i < 3)):
            sc = config_inputs(f"cmp{i}", default_indices=def_idx)
            cfg_id = get_config_id(*sc)
            st.caption(f"Selected: **{cfg_id}** &nbsp;•&nbsp; Sep={sc[0]}mm, Pitch={sc[1]}, Overlap={sc[2]}mm, GC={sc[3]}mm")
            configs.append(sc)

    ref_idx = st.selectbox(
        "Reference for % difference", list(range(1, n_configs + 1)),
        format_func=lambda x: f"Configuration {x}"
    ) - 1

    if st.button("▶  Compare Configurations", type="primary"):
        with st.spinner("Executing predictions for all selected configurations…"):
            results = []
            for i, (s, p, o, g) in enumerate(configs):
                res = predict(s, p, o, g, engine="frozen_matlab")
                c_val = res["predictions"]["TotalCapacitance_pF_per_m"]["value"] if res["status"] == "OK" else None
                m_val = res["predictions"]["AbsMutualInductance_nH"]["value"] if res["status"] == "OK" else None
                results.append({
                    "Config #":     f"Config {i+1}",
                    "Config_ID":    res.get("config_id", "—"),
                    "Sep (mm)":     s,
                    "Pitch":        p,
                    "Overlap (mm)": o,
                    "GC (mm)":      g,
                    "Cap (pF/m)":   c_val,
                    "Mut (nH)":     m_val,
                    "Status":       res["status"],
                })
            df_cmp = pd.DataFrame(results)

        valid = df_cmp[df_cmp["Status"] == "OK"]
        if valid.empty:
            st.error("No valid predictions obtained.")
        else:
            ref_cap = valid.iloc[ref_idx]["Cap (pF/m)"] if ref_idx < len(valid) else valid.iloc[0]["Cap (pF/m)"]
            ref_mut = valid.iloc[ref_idx]["Mut (nH)"]   if ref_idx < len(valid) else valid.iloc[0]["Mut (nH)"]
            df_cmp["ΔCap %"] = df_cmp["Cap (pF/m)"].apply(
                lambda v: f"{100*(v-ref_cap)/ref_cap:+.2f}%" if pd.notna(v) and ref_cap != 0 else "—"
            )
            df_cmp["ΔMut %"] = df_cmp["Mut (nH)"].apply(
                lambda v: f"{100*(v-ref_mut)/ref_mut:+.2f}%" if pd.notna(v) and ref_mut != 0 else "—"
            )

            st.subheader("Comparison Table")
            st.dataframe(df_cmp.style.format({
                "Cap (pF/m)": "{:.4g}",
                "Mut (nH)":   "{:.4g}",
            }), use_container_width=True, hide_index=True)

            # Stage 4 Automated Trade-Off & Dominance Evaluation
            if AIML_MODULES_AVAILABLE and len(valid) >= 2:
                st.subheader("Pareto Dominance & Trade-Off Evaluation")
                ref_row = valid.iloc[ref_idx]
                for i, r in valid.iterrows():
                    if i == ref_idx:
                        continue
                    c_ref = {
                        "PairSeparation_mm": ref_row["Sep (mm)"], "PairPitch_mm": ref_row["Pitch"],
                        "Overlap_mm": ref_row["Overlap (mm)"], "GroundClearance_mm": ref_row["GC (mm)"],
                        "TotalCapacitance_pF_per_m": ref_row["Cap (pF/m)"], "AbsMutualInductance_nH": ref_row["Mut (nH)"],
                        "Config_ID": ref_row["Config_ID"],
                    }
                    c_cand = {
                        "PairSeparation_mm": r["Sep (mm)"], "PairPitch_mm": r["Pitch"],
                        "Overlap_mm": r["Overlap (mm)"], "GroundClearance_mm": r["GC (mm)"],
                        "TotalCapacitance_pF_per_m": r["Cap (pF/m)"], "AbsMutualInductance_nH": r["Mut (nH)"],
                        "Config_ID": r["Config_ID"],
                    }
                    comp_eval = opt.compare_configurations(c_ref, c_cand, f"Config {ref_idx+1}", r["Config #"])
                    render_html(f"**{r['Config #']} vs Config {ref_idx+1} ({ref_row['Config_ID']}):** {comp_eval['tradeoff_summary']}")
                    st.caption(comp_eval["dominance_relationship"])

            st.markdown("<div class='section-divider'></div>")

            c_bar, m_bar = st.columns(2)
            with c_bar:
                fig_cap = px.bar(
                    valid, x="Config #", y="Cap (pF/m)",
                    color="Config #", title="Total Capacitance Comparison",
                    labels={"Cap (pF/m)": "Capacitance (pF/m)"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_cap.update_layout(
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), showlegend=False,
                    yaxis=dict(gridcolor="#334155"), height=350
                )
                st.plotly_chart(fig_cap, use_container_width=True)

            with m_bar:
                fig_mut = px.bar(
                    valid, x="Config #", y="Mut (nH)",
                    color="Config #", title="Abs. Mutual Inductance Comparison",
                    labels={"Mut (nH)": "Inductance (nH)"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_mut.update_layout(
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), showlegend=False,
                    yaxis=dict(gridcolor="#334155"), height=350
                )
                st.plotly_chart(fig_mut, use_container_width=True)

            st.session_state["comparison_df"] = df_cmp


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 7  –  MODEL VALIDATION & EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "validation":
    render_banner("Model Validation & Explainable AI (XAI)")

    render_html("""<div class="notice">
    Validation evidence from Stage 12A MATLAB analytical models, Stage 14A ExtraTrees surrogate models,
    and Stage 2 Explainable AI (XAI) feature attribution.
    </div>""")

    try:
        pred_df = pd.read_csv(_DATA_DIR / "Stage12A_Selected_Model_Predictions.csv")
        summary_df = pd.read_csv(_DATA_DIR / "Stage12A_Selected_Model_Summary.csv")
        lolo_df = pd.read_csv(_DATA_DIR / "LOLO_Summary.csv")
        perf_df = pd.read_csv(_DATA_DIR / "model_performance.csv")
        data_ok = True
    except Exception as exc:
        st.error(f"Could not load validation data: {exc}")
        data_ok = False

    if data_ok:
        tab_sum, tab_pva, tab_res, tab_cv, tab_xai = st.tabs([
            "Model Summary", "Predicted vs Actual", "Residuals",
            "LOLO Cross-Validation", "Explainable AI (Stage 2)"
        ])

        with tab_sum:
            st.subheader("Stage 12A Frozen Model Summary")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.subheader("Stage 14A ExtraTrees Surrogate Performance (In-Sample)")
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            st.caption("Notice: In-sample R²=1.0 reflects reproduction of deterministic analytical equations, not empirical test generalisation.")

        with tab_pva:
            st.subheader("Predicted vs. Analytical Reference — Stage 12A")
            pv_c1, pv_c2 = st.columns(2)
            with pv_c1:
                fig_pvc = go.Figure()
                fig_pvc.add_trace(go.Scatter(
                    x=pred_df["ActualCapacitance_pF_m"], y=pred_df["PredictedCapacitance_pF_m"],
                    mode="markers", marker=dict(color="#2196F3", size=5, opacity=0.7),
                    name="Configurations",
                ))
                mn = pred_df["ActualCapacitance_pF_m"].min()
                mx = pred_df["ActualCapacitance_pF_m"].max()
                fig_pvc.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="#EF4444", dash="dash"), name="Ideal (1:1)"
                ))
                fig_pvc.update_layout(
                    title="Capacitance: Predicted vs. Actual",
                    xaxis_title="Actual (pF/m)", yaxis_title="Predicted (pF/m)",
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), height=380,
                    xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
                )
                st.plotly_chart(fig_pvc, use_container_width=True)

            with pv_c2:
                fig_pvm = go.Figure()
                fig_pvm.add_trace(go.Scatter(
                    x=pred_df["ActualAbsMutualInductance_nH"], y=pred_df["PredictedAbsMutualInductance_nH"],
                    mode="markers", marker=dict(color="#FF9800", size=5, opacity=0.7),
                    name="Configurations",
                ))
                mn = pred_df["ActualAbsMutualInductance_nH"].min()
                mx = pred_df["ActualAbsMutualInductance_nH"].max()
                fig_pvm.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="#EF4444", dash="dash"), name="Ideal (1:1)"
                ))
                fig_pvm.update_layout(
                    title="Mutual Inductance: Predicted vs. Actual",
                    xaxis_title="Actual (nH)", yaxis_title="Predicted (nH)",
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), height=380,
                    xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
                )
                st.plotly_chart(fig_pvm, use_container_width=True)

        with tab_res:
            st.subheader("Residual Error Distributions — Stage 12A")
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                res_c = pred_df["CapacitanceResidual_pF_m"]
                fig_rc = px.histogram(
                    res_c, nbins=30, title="Capacitance Residuals (pF/m)",
                    labels={"value": "Residual (pF/m)"},
                    color_discrete_sequence=["#2196F3"],
                )
                fig_rc.update_layout(
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), height=350,
                    xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
                )
                st.plotly_chart(fig_rc, use_container_width=True)
                render_html(f"RMSE: **{res_c.pow(2).mean()**0.5:.3e}** pF/m · MAE: **{res_c.abs().mean():.3e}** pF/m")

            with r_c2:
                res_m = pred_df["AbsMutualInductanceResidual_nH"]
                fig_rm = px.histogram(
                    res_m, nbins=30, title="Mutual Inductance Residuals (nH)",
                    labels={"value": "Residual (nH)"},
                    color_discrete_sequence=["#FF9800"],
                )
                fig_rm.update_layout(
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), height=350,
                    xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
                )
                st.plotly_chart(fig_rm, use_container_width=True)
                st.markdown(f"RMSE: **{res_m.pow(2).mean()**0.5:.3e}** nH · MAE: **{res_m.abs().mean():.3e}** nH")

        with tab_cv:
            st.subheader("Leave-One-Level-Out (LOLO) Generalisation Assessment")
            st.markdown("""<div class="notice">
            LOLO withholds all configurations at a specific factor level and evaluates predictions
            using a model trained exclusively on remaining data. This reveals interpolation versus
            boundary extrapolation limitations.
            </div>""")

            for target in lolo_df["Target"].unique():
                sub = lolo_df[lolo_df["Target"] == target]
                render_html(f"**{target}**")
                st.dataframe(sub[[
                    "HeldOutFeature", "NumberOfLevels",
                    "Mean_AI_RMSE", "Mean_AI_R2", "Mean_RMSE_Improvement_Percent"
                ]].rename(columns={
                    "HeldOutFeature": "Held-out factor",
                    "NumberOfLevels": "Levels",
                    "Mean_AI_RMSE": "Pooled RMSE",
                    "Mean_AI_R2": "Pooled R²",
                    "Mean_RMSE_Improvement_Percent": "RMSE improvement %",
                }), use_container_width=True, hide_index=True)

            st.markdown("""<div class="warning-box">
            ⚠ <b>Scientific Observation on Negative R² Values:</b><br>
            Notice negative R² when withholding <code>GroundClearance_mm=10mm</code> (-8.67) and <code>PairPitch_mm=2</code> (-2.25).
            Decision trees predict flat piecewise step functions and cannot extrapolate steep asymptotic field increases
            beyond observed training bounds. This highlights why predictions are restricted strictly to the 216-grid.
            </div>""")

        with tab_xai:
            st.subheader("Feature Importance & Physical Attribution (Stage 2)")
            fx1, fx2 = st.columns(2)
            with fx1:
                render_html("**Total Capacitance Feature Importance**")
                cap_imp = pd.DataFrame({
                    "Feature": ["GroundClearance_mm", "PairSeparation_mm", "PairPitch_mm"],
                    "MDI Importance": ["92.8%", "6.3%", "0.9%"],
                    "PFI R² Drop": ["+1.8755", "+0.1386", "+0.0240"],
                    "Physical Driver": ["Earth image charge coupling", "Inter-pair spacing", "Twist geometry"],
                })
                st.dataframe(cap_imp, use_container_width=True, hide_index=True)

            with fx2:
                st.markdown("**Mutual Inductance Feature Importance**")
                mut_imp = pd.DataFrame({
                    "Feature": ["PairSeparation_mm", "PairPitch_mm", "Overlap_mm"],
                    "MDI Importance": ["58.4%", "25.8%", "15.8%"],
                    "PFI R² Drop": ["+1.4633", "+0.7072", "+0.4509"],
                    "Physical Driver": ["Magnetic spatial field decay (1/r)", "Periodic twist cancellation", "Coupled parallel run length"],
                })
                st.dataframe(mut_imp, use_container_width=True, hide_index=True)

            st.markdown(r"""
            **One-at-a-Time (OAT) Dynamic Sensitivity Ranges (at Median Config):**
            - **Ground Clearance ($10 \to 100$ mm):** Drives a **$36.17$ pF/m ($88.4\%$)** change in capacitance.
            - **Pair Separation ($8 \to 28$ mm):** Drives an **$8.40$ nH ($616.6\%$)** change in mutual inductance.
            - **Pair Pitch ($2 \to 4$ mm):** Drives a **$1.84$ nH ($135.2\%$)** change in mutual inductance.
            """)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 8  –  ANSYS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "ansys":
    render_banner("ANSYS Electronics Desktop Integration")

    st.markdown("""<div class="notice">
    This module is prepared for future ANSYS Electronics Desktop numerical verification.
    Comparison functions are implemented and ready to activate once Maxwell / Q3D simulation data is provided.
    </div>""")

    render_html("""<div class="warning-box">
    ⚠ <b>ANSYS integration: Pending simulation data.</b><br>
    No ANSYS Q3D, Maxwell or HFSS results are currently available in the project repository.
    The comparison interface below will automatically populate when simulation data is supplied.
    </div>""")

    st.subheader("Expected Data Format")
    tab_q3d, tab_max, tab_comp = st.tabs(["Q3D Capacitance", "Maxwell Inductance", "Comparison Interface"])

    with tab_q3d:
        render_html("**Expected Q3D capacitance matrix columns:**")
        st.code("""
Config_ID, PairSeparation_mm, PairPitch_mm, Overlap_mm, GroundClearance_mm,
C11_pF, C12_pF, C22_pF, TotalCapacitance_pF_per_m, FrequencyHz, SimulationCase
        """)
        st.markdown("""Once supplied, the comparison will compute:
- Point-by-point difference between MATLAB and ANSYS predictions
- Percentage deviation across the 216-configuration grid
- Identification of configurations with highest discrepancy
        """)

    with tab_max:
        st.markdown("**Expected Maxwell / Q3D inductance matrix columns:**")
        st.code("""
Config_ID, PairSeparation_mm, PairPitch_mm, Overlap_mm, GroundClearance_mm,
L11_nH, L12_nH, L22_nH, AbsMutualInductance_nH, SimulationCase
        """)

    with tab_comp:
        st.subheader("MATLAB vs ANSYS Comparison Interface")
        st.info("Upload an ANSYS result CSV to activate the comparison engine.")
        uploaded = st.file_uploader("Upload ANSYS result CSV", type=["csv"])
        if uploaded:
            try:
                ansys_df = pd.read_csv(uploaded)
                matlab_df = _get_dataset()
                st.success(f"ANSYS file loaded: {len(ansys_df)} rows.")
                if "TotalCapacitance_pF_per_m" in ansys_df.columns:
                    merged = matlab_df.merge(ansys_df, on="Config_ID", suffixes=("_MATLAB", "_ANSYS"))
                    merged["ΔCap (%)"] = 100 * (
                        merged["TotalCapacitance_pF_per_m_ANSYS"] -
                        merged["TotalCapacitance_pF_per_m_MATLAB"]
                    ) / merged["TotalCapacitance_pF_per_m_MATLAB"]
                    st.dataframe(merged[[
                        "Config_ID", "TotalCapacitance_pF_per_m_MATLAB",
                        "TotalCapacitance_pF_per_m_ANSYS", "ΔCap (%)"
                    ]], use_container_width=True)
                else:
                    st.dataframe(ansys_df.head(20), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not parse ANSYS file: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 9  –  REPORT GENERATION & DATA PACKAGE DOWNLOADS
# ═══════════════════════════════════════════════════════════════════════════
elif screen == "report":
    render_banner("Engineering Report Generation & Data Packages")

    tab_pdf, tab_packages = st.tabs([
        "PDF Configuration Report",
        "Stage 2–4 Technical Reports & Data Packages"
    ])

    with tab_pdf:
        st.subheader("Generate Cable Assessment PDF")
        sep, pitch, overlap, ground = config_inputs("rep")

        last = st.session_state.get("last_prediction")
        guard_result = st.session_state.get("guard_result")
        comp_df = st.session_state.get("comparison_df")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            include_comparison = st.checkbox("Include configuration comparison table", value=(comp_df is not None))
        with col_opt2:
            include_recommendation = st.checkbox("Include EMC-GUARD recommendation", value=(guard_result is not None))

        if not REPORTLAB_AVAILABLE:
            st.warning("`reportlab` is not installed in the active environment. PDF compilation requires reportlab. You can download complete Markdown reports and CSV data packages in the next tab.")
        else:
            if st.button("📄  Generate PDF Report", type="primary"):
                with st.spinner("Compiling engineering assessment report…"):
                    res = predict(sep, pitch, overlap, ground, engine="frozen_matlab")
                    if res["status"] != "OK":
                        st.error(f"Prediction error: {res.get('error')}")
                        st.stop()

                    cap = res["predictions"]["TotalCapacitance_pF_per_m"]
                    mut = res["predictions"]["AbsMutualInductance_nH"]
                    recommendation = guard_result.get("recommendation") if (include_recommendation and guard_result) else None
                    df_for_report = comp_df if (include_comparison and comp_df is not None) else None

                    ts = datetime.now(timezone.utc)
                    fname = f"EMC_GUARD_Report_{res['config_id']}_{ts.strftime('%Y%m%d_%H%M%S')}.pdf"

                    pdf_bytes = generate_report(
                        config_id=res["config_id"],
                        sep=sep, pitch=pitch, overlap=overlap, ground=ground,
                        cap_value=cap["value"], mut_value=mut["value"],
                        cap_model=cap["model"], mut_model=mut["model"],
                        engine="Stage12A_FrozenMATLAB",
                        recommendation=recommendation,
                        comparison_df=df_for_report,
                        filename=fname,
                    )

                st.success(f"Report generated: {fname}")
                st.download_button(
                    label="⬇  Download PDF Report",
                    data=pdf_bytes, file_name=fname,
                    mime="application/pdf", type="primary",
                )

    with tab_packages:
        st.subheader("Technical Engineering Reports & Data Exports")
        st.markdown("""
        All official technical reports and dataset exports generated in Stages 2, 3, and 4
        can be downloaded directly for review, auditing, or offline analysis:
        """)

        # ── Engineering Markdown Reports ──────────────────────────────────
        st.markdown("#### Formal Engineering Reports (Markdown)")
        rep_cols = st.columns(3)

        eval_rep_path = _AIML_REPORTS / "MODEL_EVALUATION_AND_XAI_REPORT.md"
        risk_rep_path = _AIML_REPORTS / "EMC_RISK_ASSESSMENT_REPORT.md"
        opt_rep_path  = _AIML_REPORTS / "CONFIGURATION_OPTIMISATION_REPORT.md"

        with rep_cols[0]:
            if eval_rep_path.exists():
                st.download_button(
                    label="⬇  Model Evaluation & XAI Report",
                    data=eval_rep_path.read_text(encoding="utf-8"),
                    file_name="MODEL_EVALUATION_AND_XAI_REPORT.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with rep_cols[1]:
            if risk_rep_path.exists():
                st.download_button(
                    label="⬇  EMC Risk Assessment Report",
                    data=risk_rep_path.read_text(encoding="utf-8"),
                    file_name="EMC_RISK_ASSESSMENT_REPORT.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with rep_cols[2]:
            if opt_rep_path.exists():
                st.download_button(
                    label="⬇  Pareto Optimisation Report",
                    data=opt_rep_path.read_text(encoding="utf-8"),
                    file_name="CONFIGURATION_OPTIMISATION_REPORT.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        st.markdown("<div style='margin-top:10px'></div>")

        # ── CSV Datasets & Results ─────────────────────────────────────────
        st.markdown("#### Processed Datasets & Metric CSVs")
        data_cols = st.columns(3)

        with data_cols[0]:
            # Pareto optimal configurations CSV
            if AIML_MODULES_AVAILABLE:
                _, p_df = _get_pareto_front_data()
                if not p_df.empty:
                    st.download_button(
                        label="⬇  Pareto Front Configurations (CSV)",
                        data=p_df.to_csv(index=False).encode("utf-8"),
                        file_name="pareto_optimal_configurations.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with data_cols[1]:
            # Full 216 risk assessment CSV
            if AIML_MODULES_AVAILABLE:
                risk_df = _get_risk_assessed_df()
                if not risk_df.empty:
                    st.download_button(
                        label="⬇  All 216 Configs Risk Assessment (CSV)",
                        data=risk_df.to_csv(index=False).encode("utf-8"),
                        file_name="all_configurations_risk_assessment.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with data_cols[2]:
            # Raw MATLAB 216 Physics Dataset
            ds = _get_dataset()
            st.download_button(
                label="⬇  Original MATLAB Physics Dataset (CSV)",
                data=ds.to_csv(index=False).encode("utf-8"),
                file_name="EMC_GUARD_Physics_Dataset_MATLAB.csv",
                mime="text/csv",
                use_container_width=True,
            )
