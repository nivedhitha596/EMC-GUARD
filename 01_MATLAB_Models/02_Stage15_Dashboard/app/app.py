from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================================
# EMC-GUARD | Stage 15C Premium Engineering Workstation
# Reads frozen Stage 14B exports. It does not run MATLAB or live AI inference.
# ============================================================================

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
REPORTS_DIR = PROJECT / "reports"
PHYSICS_FILE = DATA_DIR / "EMC_GUARD_Physics_Dataset.csv"
MATLAB_FILE = DATA_DIR / "EMC_GUARD_MATLAB_Reference.csv"
COMPARISON_FILE = DATA_DIR / "EMC_GUARD_Comparison.csv"

APP_TITLE = "EMC-GUARD"
APP_SUBTITLE = (
    "Physics-Informed Configuration-Aware Digital Twin "
    "for Predictive EMC Pre-Compliance of Elevator Systems"
)

# Computational cross-tool comparison tolerances from Stage 14B-5.
# These are NOT physical measurement tolerances or EMC acceptance limits.
CAP_TOL = 1.0e-9       # pF/m
MUTUAL_TOL = 1.0e-10   # nH

# Geometry assumptions used by the current computational model.
# They must remain visibly labelled as model assumptions, not a verified KONE cable.
CABLE_WIDTH_MM = 35.45
CABLE_HEIGHT_MM = 5.80
COPPER_AREA_MM2 = 0.50
EQUIV_COPPER_RADIUS_MM = float(np.sqrt(COPPER_AREA_MM2 / np.pi))
EQUIV_COPPER_DIAMETER_MM = 2.0 * EQUIV_COPPER_RADIUS_MM

NAV_ITEMS = [
    "Command Center",
    "Geometry Explorer",
    "Design Advisor",
    "Model Agreement",
    "Pareto & Sensitivity",
    "Validation Status",
    "Data & Reports",
]

PALETTE = {
    "bg": "#0A101B",
    "panel": "#111B2A",
    "panel_2": "#142236",
    "stroke": "#28384D",
    "text": "#E7EEF8",
    "muted": "#91A2B9",
    "cyan": "#43D6C5",
    "blue": "#76A9FA",
    "amber": "#F1B65C",
    "red": "#F07878",
    "violet": "#B49AFF",
    "green": "#6BD6A5",
}

st.set_page_config(
    page_title="EMC-GUARD | Engineering Workstation",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------- Styling ----------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {{
      --eg-bg: {PALETTE['bg']};
      --eg-panel: {PALETTE['panel']};
      --eg-panel2: {PALETTE['panel_2']};
      --eg-stroke: {PALETTE['stroke']};
      --eg-text: {PALETTE['text']};
      --eg-muted: {PALETTE['muted']};
      --eg-cyan: {PALETTE['cyan']};
    }}

    html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}
    [data-testid="stAppViewContainer"] {{
      background:
        radial-gradient(ellipse at 12% 0%, rgba(48, 104, 142, .16), transparent 34%),
        radial-gradient(ellipse at 92% 8%, rgba(67, 214, 197, .075), transparent 26%),
        var(--eg-bg);
      color: var(--eg-text);
    }}
    [data-testid="stHeader"] {{ background: rgba(10,16,27,.78); }}
    [data-testid="stToolbar"] {{ right: 1rem; }}
    .block-container {{ padding-top: 4rem; padding-bottom: 2.5rem; max-width: 1600px; }}

    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #0D1726 0%, #0B1421 100%);
      border-right: 1px solid #223247;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: #9AABC0; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: .32rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
      border: 1px solid transparent;
      border-radius: 10px;
      padding: .50rem .65rem;
      background: rgba(255,255,255,.018);
      transition: .15s ease;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
      background: rgba(67,214,197,.07);
      border-color: rgba(67,214,197,.22);
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
      background: rgba(67,214,197,.105);
      border: 1px solid rgba(67,214,197,.37);
    }}

    h1, h2, h3, h4 {{ color: var(--eg-text) !important; letter-spacing: -.025em; }}
    h1 {{ font-weight: 800 !important; }}
    h2, h3 {{ font-weight: 750 !important; }}
    p, li {{ color: #C3CFDE; }}
    [data-testid="stCaptionContainer"] p {{ color: var(--eg-muted); }}
    hr {{ border-color: #26364A !important; }}

    [data-testid="stMetric"] {{
      background: linear-gradient(145deg, rgba(22,36,55,.96), rgba(15,26,42,.96));
      border: 1px solid #2A3B51;
      border-radius: 13px;
      padding: 15px 17px;
      min-height: 105px;
      box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }}
    [data-testid="stMetricLabel"] p {{ color: #91A4BC !important; font-size: .78rem; font-weight: 650; }}
    [data-testid="stMetricValue"] {{ color: #F0F5FC !important; font-size: 1.55rem; font-weight: 800; }}
    [data-testid="stMetricDelta"] {{ font-family: 'DM Mono', monospace; }}

    [data-testid="stSelectbox"], [data-testid="stSlider"], [data-testid="stFileUploader"] {{
      background: transparent;
    }}
    [data-baseweb="select"] > div {{
      background: #111E30; border-color: #34465E; border-radius: 9px;
    }}
    [data-testid="stDataFrame"] {{ border: 1px solid #2A3B51; border-radius: 10px; overflow: hidden; }}
    .stButton > button, .stDownloadButton > button {{
      border-radius: 9px; border: 1px solid #3B536D;
      background: linear-gradient(120deg, #18334A, #17283D);
      color: #EAF4FF; font-weight: 700;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
      border-color: #43D6C5; color: white;
      background: linear-gradient(120deg, #1B4656, #1C354E);
    }}
    [data-testid="stAlert"] {{ border-radius: 10px; }}

    .brand-lockup {{ display:flex; align-items:center; gap:11px; margin: 2px 0 22px 2px; }}
    .brand-mark {{
      width:38px; height:38px; border-radius:11px;
      display:flex; align-items:center; justify-content:center;
      color:#07131D; font-size:19px; font-weight:900;
      background:linear-gradient(145deg,#67E5D4,#3C9FC0);
      box-shadow:0 5px 22px rgba(67,214,197,.18);
    }}
    .brand-name {{ color:#F1F6FC; font-size:1.08rem; font-weight:850; letter-spacing:.11em; }}
    .brand-sub {{ color:#8094AD; font-size:.60rem; letter-spacing:.17em; margin-top:1px; }}

    .hero {{
      border: 1px solid #2B4058; border-radius: 17px; padding: 25px 28px;
      background:
        radial-gradient(circle at 88% 8%, rgba(67,214,197,.12), transparent 27%),
        linear-gradient(115deg, rgba(22,40,61,.98), rgba(14,26,43,.98));
      box-shadow: 0 14px 38px rgba(0,0,0,.16);
      margin-bottom: 17px;
    }}
    .hero-kicker {{ color:#66DCCF; font-family:'DM Mono',monospace; font-size:.70rem; letter-spacing:.15em; text-transform:uppercase; margin-bottom:9px; }}
    .hero-title {{ color:#F1F6FC; font-size:2.0rem; line-height:1.16; font-weight:850; letter-spacing:-.055em; }}
    .hero-copy {{ color:#AAB9CB; font-size:.92rem; max-width:900px; line-height:1.7; margin-top:10px; }}
    .hero-meta {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }}
    .pill {{ display:inline-flex; align-items:center; gap:6px; border:1px solid #354B63; border-radius:99px; padding:5px 10px; color:#C2D1E2; background:rgba(7,17,29,.36); font-size:.69rem; font-weight:700; }}
    .pill-cyan {{ color:#78E8DB; border-color:rgba(67,214,197,.35); background:rgba(67,214,197,.08); }}
    .pill-amber {{ color:#F3C779; border-color:rgba(241,182,92,.35); background:rgba(241,182,92,.08); }}

    .section-top {{ display:flex; align-items:center; gap:9px; margin: 19px 0 11px; }}
    .section-line {{ width:4px; height:19px; border-radius:5px; background:#43D6C5; }}
    .section-name {{ color:#E7EEF8; font-size:1.0rem; font-weight:800; letter-spacing:-.02em; }}
    .section-note {{ color:#8295AC; font-size:.75rem; margin-left:auto; }}

    .panel {{
      border:1px solid #293B51; border-radius:13px; padding:17px 18px;
      background:linear-gradient(150deg, rgba(20,34,53,.92), rgba(14,25,41,.92));
      height:100%;
    }}
    .panel-label {{ color:#8297B0; text-transform:uppercase; letter-spacing:.12em; font-size:.65rem; font-weight:800; }}
    .panel-value {{ color:#EDF4FC; font-size:1.35rem; font-weight:850; margin-top:5px; }}
    .panel-note {{ color:#8FA1B7; font-size:.75rem; margin-top:4px; line-height:1.5; }}
    .stat-card {{
      border:1px solid #2A3B51; border-radius:13px; padding:15px 17px; min-height:105px;
      background:linear-gradient(145deg, rgba(22,36,55,.96), rgba(15,26,42,.96));
      box-shadow:0 8px 24px rgba(0,0,0,.12);
    }}
    .stat-label {{ color:#91A4BC; font-size:.78rem; font-weight:650; }}
    .stat-value {{ color:#F0F5FC; font-size:1.55rem; font-weight:800; margin-top:4px; }}
    .stat-note {{ font-family:'DM Mono',monospace; font-size:.72rem; margin-top:6px; }}
    .stat-ok {{ color:#7DE0B3; }}
    .stat-pending {{ color:#F1C275; }}
    .stat-neutral {{ color:#8FA1B7; }}
    .status-row {{ display:flex; justify-content:space-between; gap:10px; padding:10px 0; border-bottom:1px solid rgba(65,84,108,.38); }}
    .status-row:last-child {{ border-bottom:0; }}
    .status-name {{ color:#CAD5E3; font-size:.82rem; }}
    .status-state {{ color:#7DE0B3; font-family:'DM Mono',monospace; font-size:.69rem; white-space:nowrap; }}
    .status-pending {{ color:#F1C275; }}
    .mono {{ font-family:'DM Mono',monospace; }}
    .tiny-muted {{ color:#8295AC; font-size:.72rem; line-height:1.55; }}
    .footer {{ border-top:1px solid #26364A; margin-top:30px; padding-top:13px; color:#6F829A; font-size:.68rem; letter-spacing:.04em; }}

    @keyframes egFadeUp {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:none; }} }}
    @keyframes egShimmer {{ 0% {{ background-position:0% 50%; }} 100% {{ background-position:200% 50%; }} }}
    [data-testid="stAppViewContainer"]::before {{
      content:""; position:fixed; top:0; left:0; right:0; height:3px; z-index:9999;
      background:linear-gradient(90deg,#43D6C5,#76A9FA,#B49AFF,#43D6C5); background-size:200% 100%;
      animation:egShimmer 6s linear infinite;
    }}
    .hero, .stat-card, .panel, .flow-step, .rec-card {{ animation: egFadeUp .5s ease both; }}
    .hero {{ position:relative; overflow:hidden; }}
    .hero::after {{
      content:""; position:absolute; inset:0; pointer-events:none;
      background-image:linear-gradient(rgba(120,160,200,.055) 1px, transparent 1px), linear-gradient(90deg, rgba(120,160,200,.055) 1px, transparent 1px);
      background-size:34px 34px;
      -webkit-mask-image:linear-gradient(120deg, transparent 35%, #000); mask-image:linear-gradient(120deg, transparent 35%, #000);
    }}
    .hero > * {{ position:relative; z-index:1; }}
    .stat-card, .panel {{ transition: transform .18s ease, border-color .18s ease; }}
    .stat-card:hover, .panel:hover {{ transform: translateY(-2px); border-color:#3F6A85; }}
    .flow {{ display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 4px; }}
    .flow-step {{
      flex:1 1 160px; border:1px solid #2F4560; border-radius:12px; padding:12px 14px;
      background:linear-gradient(150deg, rgba(23,40,60,.95), rgba(14,26,42,.95));
    }}
    .flow-step .n {{ font-family:'DM Mono',monospace; font-size:.62rem; letter-spacing:.14em; color:#66DCCF; }}
    .flow-step .t {{ color:#EDF4FC; font-weight:800; margin-top:4px; }}
    .flow-step .d {{ color:#8FA1B7; font-size:.72rem; margin-top:3px; line-height:1.45; }}
    .flow-step.pending {{ border-color:rgba(241,182,92,.38); }}
    .flow-step.pending .n {{ color:#F1C477; }}
    .insight {{
      border-left:3px solid #43D6C5; background:rgba(67,214,197,.06); border-radius:0 10px 10px 0;
      padding:11px 14px; margin:8px 0; color:#CFE0F0; font-size:.84rem; line-height:1.6;
    }}
    .rec-card {{
      border:1px solid rgba(67,214,197,.45); border-radius:15px; padding:20px 22px;
      background:radial-gradient(circle at 90% 0%, rgba(67,214,197,.14), transparent 40%),
                 linear-gradient(140deg, rgba(20,40,58,.98), rgba(13,26,42,.98));
    }}
    .rec-id {{ color:#F1F6FC; font-size:1.9rem; font-weight:850; letter-spacing:-.04em; }}
    @media (max-width: 850px) {{
      .hero-title {{ font-size:1.55rem; }}
      .hero {{ padding:20px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------- Helpers ----------------------------------

def section_header(title: str, note: str | None = None) -> None:
    note_html = f'<span class="section-note">{note}</span>' if note else ""
    st.markdown(
        f'<div class="section-top"><span class="section-line"></span>'
        f'<span class="section-name">{title}</span>{note_html}</div>',
        unsafe_allow_html=True,
    )


def plotly_style(fig: go.Figure, height: int = 390, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,19,31,.72)",
        font=dict(family="Manrope, sans-serif", color="#C8D5E5", size=11),
        height=height,
        margin=dict(l=18, r=18, t=55, b=18),
        legend=dict(
            bgcolor="rgba(12,21,34,.65)",
            bordercolor="rgba(75,96,122,.45)",
            borderwidth=1,
            font=dict(size=10),
        ) if legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor="#16263A", bordercolor="#3D5973",
            font=dict(color="#F0F5FC", family="Manrope"),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(105,129,157,.15)", zerolinecolor="rgba(105,129,157,.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(105,129,157,.15)", zerolinecolor="rgba(105,129,157,.25)")
    return fig


def add_hero(kicker: str, title: str, copy: str, tags: list[tuple[str, str]]) -> None:
    tag_html = "".join(
        f'<span class="pill {kind}">{label}</span>' for label, kind in tags
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-copy">{copy}</div>
          <div class="hero-meta">{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_panel(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="panel"><div class="panel-label">{label}</div>'
        f'<div class="panel-value">{value}</div>'
        f'<div class="panel-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, note: str, tone: str = "neutral") -> None:
    """Metric-style card without the up/down delta arrow. tone: ok | pending | neutral."""
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-note stat-{tone}">{note}</div></div>',
        unsafe_allow_html=True,
    )


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_files = [p.name for p in (PHYSICS_FILE, MATLAB_FILE, COMPARISON_FILE) if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing data file(s): " + ", ".join(missing_files)
            + f". Expected them inside: {DATA_DIR}"
        )

    physics = pd.read_csv(PHYSICS_FILE)
    matlab = pd.read_csv(MATLAB_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)

    for frame in (physics, matlab, comparison):
        if "Config_ID" in frame.columns:
            frame["Config_ID"] = frame["Config_ID"].astype(str)

    return physics, matlab, comparison


def validate_data(
    physics: pd.DataFrame,
    matlab: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[str]:
    issues: list[str] = []
    required = {
        "physics": [
            "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm",
            "GroundClearance_mm", "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH",
        ],
        "matlab": [
            "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm",
            "GroundClearance_mm", "Capacitance_pF_per_m", "AbsMutualInductance_nH",
        ],
        "comparison": [
            "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm",
            "GroundClearance_mm", "MATLAB_Capacitance_pF_per_m", "AI_Capacitance_pF_per_m",
            "MATLAB_AbsMutualInductance_nH", "AI_AbsMutualInductance_nH",
            "Capacitance_Absolute_Difference", "MutualInductance_Absolute_Difference",
        ],
    }
    frames = {"physics": physics, "matlab": matlab, "comparison": comparison}
    for name, columns in required.items():
        missing = [col for col in columns if col not in frames[name].columns]
        if missing:
            issues.append(f"{name}: missing columns {missing}")
        if len(frames[name]) != 216:
            issues.append(f"{name}: expected 216 rows, found {len(frames[name])}")
        if "Config_ID" in frames[name].columns and frames[name]["Config_ID"].nunique() != 216:
            issues.append(f"{name}: Config_ID is not unique across 216 rows")

    if not issues:
        id_sets = [set(x["Config_ID"]) for x in (physics, matlab, comparison)]
        if not (id_sets[0] == id_sets[1] == id_sets[2]):
            issues.append("The three datasets do not contain the same Config_ID set")

        geom = ["PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm"]
        for name, frame in frames.items():
            if frame[geom].isna().any().any():
                issues.append(f"{name}: missing geometry values detected")
        if not issues:
            unique_grid = comparison[geom].drop_duplicates().shape[0]
            if unique_grid != 216:
                issues.append(f"comparison: expected 216 unique geometry combinations, found {unique_grid}")

    return issues


def get_geometry_positions(separation: float, pitch: float) -> list[float]:
    # Same four-conductor centre-position definition used by the current model.
    return [
        -separation / 2.0 - pitch / 2.0,
        -separation / 2.0 + pitch / 2.0,
        separation / 2.0 - pitch / 2.0,
        separation / 2.0 + pitch / 2.0,
    ]


def geometry_cross_section(separation: float, pitch: float, clearance: float) -> go.Figure:
    xs = get_geometry_positions(separation, pitch)
    zc = clearance + CABLE_HEIGHT_MM / 2.0
    half_w = CABLE_WIDTH_MM / 2.0
    ztop = clearance + CABLE_HEIGHT_MM

    fig = go.Figure()
    # Reference ground plane.
    fig.add_shape(
        type="line", x0=-half_w - 7, x1=half_w + 7, y0=0, y1=0,
        line=dict(color="#E2AE5D", width=2, dash="dash"),
    )
    # Assumed cable envelope; not a shield representation.
    fig.add_shape(
        type="rect", x0=-half_w, x1=half_w, y0=clearance, y1=ztop,
        line=dict(color="#7A91AC", width=1.5, dash="dot"),
        fillcolor="rgba(91,126,160,.12)",
    )

    pair_specs = [
        (xs[0:2], "Pair A", PALETTE["cyan"]),
        (xs[2:4], "Pair B", PALETTE["blue"]),
    ]
    for pair_x, name, color in pair_specs:
        fig.add_trace(go.Scatter(
            x=pair_x, y=[zc, zc], mode="markers+text",
            text=["A1", "A2"] if name == "Pair A" else ["B1", "B2"],
            textposition="top center", textfont=dict(size=10, color=color),
            marker=dict(size=15, color=color, line=dict(color="#EAF4FC", width=1.2), symbol="circle"),
            name=name,
            hovertemplate=f"{name}<br>x = %{{x:.2f}} mm<br>z = %{{y:.2f}} mm<extra></extra>",
        ))

    fig.add_annotation(
        x=0, y=zc + 1.12, text="Conductor centres lie in one plane in this model",
        showarrow=False, font=dict(size=9, color="#9DB0C6"),
    )
    fig.add_annotation(
        x=half_w, y=clearance + CABLE_HEIGHT_MM / 2,
        text=" assumed envelope", showarrow=False, xanchor="left",
        font=dict(size=8, color="#8498B1"),
    )
    fig.add_annotation(
        x=-half_w - 6, y=0, text="REFERENCE GROUND",
        showarrow=False, xanchor="right", yshift=-13,
        font=dict(size=8, color="#E2AE5D"),
    )
    fig.update_layout(
        title="Four-conductor cross-section · schematic",
        xaxis_title="Horizontal position x (mm)",
        yaxis_title="Height above reference plane z (mm)",
        yaxis=dict(range=[-2, max(ztop + 4, 17)]),
        xaxis=dict(range=[-half_w - 10, half_w + 16], scaleanchor="y", scaleratio=1),
        showlegend=True,
    )
    return plotly_style(fig, height=420)


def geometry_3d(separation: float, pitch: float, overlap: float, clearance: float) -> go.Figure:
    xs = get_geometry_positions(separation, pitch)
    zc = clearance + CABLE_HEIGHT_MM / 2.0
    half_w = CABLE_WIDTH_MM / 2.0
    ztop = clearance + CABLE_HEIGHT_MM
    y0, y1 = 0.0, float(overlap)

    fig = go.Figure()
    # Ground reference plane.
    gx = np.array([[-half_w - 7, half_w + 7], [-half_w - 7, half_w + 7]])
    gy = np.array([[y0, y0], [y1, y1]])
    gz = np.zeros_like(gx)
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=gz, showscale=False, opacity=.13,
        colorscale=[[0, "#C99A52"], [1, "#C99A52"]],
        name="Reference plane", hoverinfo="skip",
    ))

    # Cable-envelope edges along the shared run (outline only).
    for xx in (-half_w, half_w):
        for zz in (clearance, ztop):
            fig.add_trace(go.Scatter3d(
                x=[xx, xx], y=[y0, y1], z=[zz, zz], mode="lines",
                line=dict(color="#8298B1", width=3, dash="dot"),
                name="Assumed cable envelope", showlegend=(xx == -half_w and zz == clearance),
                hoverinfo="skip",
            ))

    colors = [PALETTE["cyan"], PALETTE["cyan"], PALETTE["blue"], PALETTE["blue"]]
    labels = ["A1", "A2", "B1", "B2"]
    for xx, color, label in zip(xs, colors, labels):
        fig.add_trace(go.Scatter3d(
            x=[xx, xx], y=[y0, y1], z=[zc, zc], mode="lines+markers",
            line=dict(color=color, width=7),
            marker=dict(size=3, color=color),
            name=f"Conductor {label}",
            hovertemplate=f"{label}<br>x = {xx:.2f} mm<br>shared run = {overlap:.0f} mm<br>z = {zc:.2f} mm<extra></extra>",
        ))

    fig.update_layout(
        title=f"Parallel run · {overlap:.0f} mm shared overlap",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="Run length y (mm)", zaxis_title="z (mm)",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(105,129,157,.18)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(105,129,157,.18)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(gridcolor="rgba(105,129,157,.18)", backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.8, z=.55),
            camera=dict(eye=dict(x=1.65, y=-1.7, z=1.15)),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        height=470,
    )
    return fig


def pareto_mask(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    values_x = frame[x_col].to_numpy(dtype=float)
    values_y = frame[y_col].to_numpy(dtype=float)
    # True for a point if no other point is <= in both objectives and < in one.
    dominates = (
        (values_x[:, None] <= values_x[None, :])
        & (values_y[:, None] <= values_y[None, :])
        & ((values_x[:, None] < values_x[None, :]) | (values_y[:, None] < values_y[None, :]))
    )
    return ~np.any(dominates, axis=0)


def balanced_candidate(frame: pd.DataFrame, cap_weight: float, mutual_weight: float) -> pd.Series:
    cap = frame["TotalCapacitance_pF_per_m"].to_numpy(dtype=float)
    mutual = frame["AbsMutualInductance_nH"].to_numpy(dtype=float)
    cap_range = max(float(cap.max() - cap.min()), 1e-12)
    mutual_range = max(float(mutual.max() - mutual.min()), 1e-12)
    cap_norm = (cap - cap.min()) / cap_range
    mutual_norm = (mutual - mutual.min()) / mutual_range
    total = cap_weight + mutual_weight
    if total <= 0:
        cap_weight = mutual_weight = 0.5
        total = 1.0
    wc, wm = cap_weight / total, mutual_weight / total
    distance = np.sqrt(wc * cap_norm**2 + wm * mutual_norm**2)
    return frame.iloc[int(np.argmin(distance))]


PARAM_LABELS = {
    "PairSeparation_mm": "pair separation",
    "PairPitch_mm": "pair pitch",
    "Overlap_mm": "parallel overlap",
    "GroundClearance_mm": "ground clearance",
}
C_COL = "TotalCapacitance_pF_per_m"
M_COL = "AbsMutualInductance_nH"


def lookup_row(cfg: dict) -> pd.Series:
    """Exact-grid lookup in the frozen physics dataset (no interpolation)."""
    mask = np.ones(len(physics), dtype=bool)
    for col in PARAM_LABELS:
        mask &= np.isclose(physics[col].to_numpy(dtype=float), float(cfg[col]))
    return physics[mask].iloc[0]


def design_landscape(color_col: str, color_label: str) -> go.Figure:
    fig = go.Figure(go.Scatter3d(
        x=physics["PairSeparation_mm"], y=physics["Overlap_mm"], z=physics["GroundClearance_mm"],
        mode="markers",
        marker=dict(
            size=6, color=physics[color_col], colorscale="Viridis", opacity=.9,
            colorbar=dict(title=color_label, thickness=12, len=.7),
            line=dict(width=.4, color="rgba(230,240,250,.5)"),
        ),
        customdata=np.stack([physics["Config_ID"], physics["PairPitch_mm"], physics[color_col]], axis=1),
        hovertemplate="%{customdata[0]}<br>D=%{x} mm · L=%{y} mm · h=%{z} mm · p=%{customdata[1]} mm"
                      f"<br>{color_label}: " + "%{customdata[2]:.5g}<extra></extra>",
    ))
    fig.update_layout(
        title="Design-space map · 216 supported configurations",
        scene=dict(
            xaxis_title="Pair separation (mm)", yaxis_title="Overlap (mm)", zaxis_title="Ground clearance (mm)",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(105,129,157,.2)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(105,129,157,.2)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(gridcolor="rgba(105,129,157,.2)", backgroundcolor="rgba(0,0,0,0)"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#C8D5E5", size=11),
        margin=dict(l=0, r=0, t=50, b=0), height=500,
    )
    return fig


def response_heatmap(out_col: str, out_label: str, x_key: str, y_key: str, fixed: dict) -> go.Figure:
    d = physics.copy()
    for key, val in fixed.items():
        d = d[np.isclose(d[key].to_numpy(dtype=float), float(val))]
    pv = d.pivot_table(index=y_key, columns=x_key, values=out_col, aggfunc="mean").sort_index()
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=[f"{v:g}" for v in pv.columns], y=[f"{v:g}" for v in pv.index],
        colorscale="Tealgrn", texttemplate="%{z:.4g}", textfont=dict(size=11),
        colorbar=dict(title=out_label, thickness=12),
        hovertemplate=f"{PARAM_LABELS[x_key]}: %{{x}} mm<br>{PARAM_LABELS[y_key]}: %{{y}} mm<br>{out_label}: %{{z:.6g}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{out_label} · {PARAM_LABELS[y_key]} against {PARAM_LABELS[x_key]}",
        xaxis_title=f"{PARAM_LABELS[x_key].capitalize()} (mm)", yaxis_title=f"{PARAM_LABELS[y_key].capitalize()} (mm)",
    )
    return plotly_style(fig, 420, legend=False)


def grid_insights() -> list[str]:
    out = []
    for col, label in ((C_COL, "capacitance"), (M_COL, "mutual inductance")):
        best = None
        for key in PARAM_LABELS:
            g = physics.groupby(key)[col].mean()
            span = (g.max() - g.min()) / physics[col].mean() * 100.0
            if best is None or span > best[1]:
                best = (key, span, g.idxmin(), g.idxmax())
        key, span, lo, hi = best
        out.append(
            f"Largest descriptive influence on {label}: <b>{PARAM_LABELS[key]}</b> "
            f"(level-mean span ≈ {span:.0f}% of the grid mean; lowest mean at {lo:g} mm, highest at {hi:g} mm)."
        )
    return out


def single_change_table(base_cfg: dict, cand_cfg: dict) -> pd.DataFrame:
    base = lookup_row(base_cfg)
    rows = []
    for key in PARAM_LABELS:
        if not np.isclose(base_cfg[key], cand_cfg[key]):
            alt = dict(base_cfg)
            alt[key] = cand_cfg[key]
            r = lookup_row(alt)
            rows.append({
                "Single change (other inputs at baseline)": f"{PARAM_LABELS[key].capitalize()}: {base_cfg[key]:g} → {cand_cfg[key]:g} mm",
                "Δ Capacitance (%)": round((r[C_COL] - base[C_COL]) / base[C_COL] * 100.0, 2),
                "Δ |M| (%)": round((r[M_COL] - base[M_COL]) / base[M_COL] * 100.0, 2),
            })
    return pd.DataFrame(rows)


# -------------------------------- Load data ---------------------------------
try:
    physics, matlab, comparison = load_data()
except Exception as exc:
    st.error(f"**Dashboard data could not be loaded.**\n\n{exc}")
    st.stop()

integrity_issues = validate_data(physics, matlab, comparison)
if integrity_issues:
    st.error("**Dataset integrity check failed.** Fix the source exports before using this dashboard.")
    for issue in integrity_issues:
        st.write(f"- {issue}")
    st.stop()

# Enforce numeric values before calculations and chart rendering.
NUMERIC_COLUMNS = {
    "physics": [
        "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
        "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH",
    ],
    "matlab": [
        "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
        "Capacitance_pF_per_m", "AbsMutualInductance_nH",
    ],
    "comparison": [
        "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
        "MATLAB_Capacitance_pF_per_m", "AI_Capacitance_pF_per_m",
        "MATLAB_AbsMutualInductance_nH", "AI_AbsMutualInductance_nH",
        "Capacitance_Absolute_Difference", "MutualInductance_Absolute_Difference",
    ],
}
for frame_name, frame in (("physics", physics), ("matlab", matlab), ("comparison", comparison)):
    for column in NUMERIC_COLUMNS[frame_name]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            st.error(f"Non-numeric or missing values found in `{frame_name}.{column}`.")
            st.stop()

# Sort consistently for repeatable tables and charts.
physics = physics.sort_values("Config_ID").reset_index(drop=True)
matlab = matlab.sort_values("Config_ID").reset_index(drop=True)
comparison = comparison.sort_values("Config_ID").reset_index(drop=True)

cap_abs = np.abs(
    comparison["MATLAB_Capacitance_pF_per_m"] - comparison["AI_Capacitance_pF_per_m"]
)
mutual_abs = np.abs(
    comparison["MATLAB_AbsMutualInductance_nH"] - comparison["AI_AbsMutualInductance_nH"]
)
cap_pass = int((cap_abs <= CAP_TOL).sum())
mutual_pass = int((mutual_abs <= MUTUAL_TOL).sum())
computational_pass = cap_pass == 216 and mutual_pass == 216


# ------------------------- Supported-domain checking -------------------------
GEOM_COLS = ["PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm"]
GRID_LEVELS = {c: sorted(comparison[c].unique().tolist()) for c in GEOM_COLS}


def classify_geometry(cfg: dict) -> tuple[str, str]:
    """Classify an input geometry against the frozen 216-point design grid."""
    outside = [c for c in GEOM_COLS if not (min(GRID_LEVELS[c]) <= cfg[c] <= max(GRID_LEVELS[c]))]
    if outside:
        return "out-of-domain", f"Outside the design range for: {', '.join(outside)}. Rejected."
    between = [c for c in GEOM_COLS if cfg[c] not in GRID_LEVELS[c]]
    if between:
        return "interpolated", (
            f"Between supported levels for: {', '.join(between)}. The frozen predictor does not support "
            "interpolation; it would need a separately validated pathway. Rejected."
        )
    return "on-grid", "Exact match to one of the 216 configurations. AI–MATLAB agreement was checked."


# --------------------------------- Sidebar ----------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">E</div>
          <div><div class="brand-name">EMC-GUARD</div>
          <div class="brand-sub">ENGINEERING WORKSTATION</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='panel-label' style='margin:0 0 8px 5px'>WORKSPACE NAVIGATION</div>", unsafe_allow_html=True)
    page = st.radio("Engineering modules", NAV_ITEMS, label_visibility="collapsed")
    st.divider()
    st.markdown("<div class='panel-label'>MODEL STATE</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='status-row'><span class='status-name'>MATLAB reference</span><span class='status-state'>FROZEN</span></div>"
        "<div class='status-row'><span class='status-name'>AI surrogate</span><span class='status-state'>FROZEN</span></div>"
        "<div class='status-row'><span class='status-name'>Physical validation</span><span class='status-state status-pending'>PENDING</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.caption("Stage 15C dashboard · frozen Stage 14B-5 exports")
    st.caption("216 supported geometry configurations")
    st.caption("Research prototype · not certified")
    with st.expander("Suggested 5-minute demo path"):
        st.markdown(
            "1. **Command Center**: the workflow and verified status\n"
            "2. **Geometry Explorer**: pick a configuration, show the domain check\n"
            "3. **Design Advisor**: baseline → ranked alternatives\n"
            "4. **Model Agreement**: 216/216 numerical check\n"
            "5. **Validation Status**: limits and the physical-test plan"
        )


# --------------------------------- Top bar ----------------------------------
header_left, header_right = st.columns([5.0, 1.7])
with header_left:
    st.markdown(
        f"<div style='color:#8EA2BA;font-size:.67rem;font-family:DM Mono,monospace;letter-spacing:.14em;text-transform:uppercase'>KONE-ORIENTED ENGINEERING RESEARCH / STAGE 15C</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div style='font-size:1.45rem;font-weight:850;color:#F0F5FC;letter-spacing:-.04em'>{APP_TITLE}</div>", unsafe_allow_html=True)
with header_right:
    state_label = "COMPUTATIONAL CHECK PASS" if computational_pass else "CHECK REQUIRED"
    state_class = "pill-cyan" if computational_pass else "pill-amber"
    st.markdown(
        f"<div style='text-align:right;padding-top:17px'><span class='pill {state_class}'>● {state_label}</span></div>",
        unsafe_allow_html=True,
    )
st.divider()


# ============================================================================
# PAGE 1 · COMMAND CENTER
# ============================================================================
if page == "Command Center":
    add_hero(
        "SYSTEM OVERVIEW / FROZEN COMPUTATIONAL BUILD",
        "A configuration-aware view of elevator cable coupling.",
        "Explore the current four-parameter cable geometry grid, inspect the frozen MATLAB reference, "
        "and compare the AI surrogate against the same 216 supported configurations. The current build "
        "is a computational screening prototype; physical validation remains an open work package.",
        [("216 CONFIGURATIONS", "pill-cyan"), ("MATLAB + AI", ""), ("PHYSICAL TESTING PENDING", "pill-amber")],
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        stat_card("Geometry configurations", "216", "6 × 3 × 3 × 4 grid", "neutral")
    with k2:
        stat_card("MATLAB reference cases", "216", "Stage 13B verified", "ok")
    with k3:
        stat_card(
            "AI–MATLAB comparisons", "216 / 216",
            "within computational tolerances" if computational_pass else "one or more checks outside tolerance",
            "ok" if computational_pass else "pending",
        )
    with k4:
        stat_card("Physical validation", "Pending", "no measurements loaded", "pending")

    section_header("ENGINEERING WORKFLOW", "configure → predict → analyse → optimise → validate")
    st.markdown(
        """
        <div class="flow">
          <div class="flow-step"><div class="n">01 / CONFIGURE</div><div class="t">Geometry Explorer</div><div class="d">Pick a supported cable arrangement; unsupported inputs are rejected.</div></div>
          <div class="flow-step"><div class="n">02 / PREDICT</div><div class="t">MATLAB + AI</div><div class="d">Capacitance and mutual inductance from the frozen reference and surrogate.</div></div>
          <div class="flow-step"><div class="n">03 / ANALYSE</div><div class="t">Pareto & Sensitivity</div><div class="d">Trade-offs, heatmaps and descriptive parameter response.</div></div>
          <div class="flow-step"><div class="n">04 / OPTIMISE</div><div class="t">Design Advisor</div><div class="d">Ranked alternatives under mechanical constraints.</div></div>
          <div class="flow-step pending"><div class="n">05 / VALIDATE</div><div class="t">Physical testing</div><div class="d">Pending: fixture, instrumentation, correlation.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("SYSTEM PIPELINE", "current model-to-dashboard flow")
    p1, p2 = st.columns([1.55, 1])
    with p1:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-label">COMPUTATIONAL WORKFLOW</div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:15px">
                <div style="background:#17283C;border:1px solid #334A64;border-radius:10px;padding:13px">
                  <div style="color:#54D9CB;font-family:DM Mono;font-size:.65rem">01 / INPUT</div>
                  <div style="color:#EDF4FC;font-weight:800;margin-top:6px">Geometry grid</div>
                  <div class="tiny-muted">Separation · pitch · overlap · clearance</div>
                </div>
                <div style="background:#17283C;border:1px solid #334A64;border-radius:10px;padding:13px">
                  <div style="color:#76A9FA;font-family:DM Mono;font-size:.65rem">02 / REFERENCE</div>
                  <div style="color:#EDF4FC;font-weight:800;margin-top:6px">MATLAB model</div>
                  <div class="tiny-muted">Frozen physics-informed response</div>
                </div>
                <div style="background:#17283C;border:1px solid #334A64;border-radius:10px;padding:13px">
                  <div style="color:#B49AFF;font-family:DM Mono;font-size:.65rem">03 / SURROGATE</div>
                  <div style="color:#EDF4FC;font-weight:800;margin-top:6px">AI prediction</div>
                  <div class="tiny-muted">Frozen ExtraTrees models</div>
                </div>
              </div>
              <div style="text-align:center;color:#5A7896;font-size:1.15rem;margin:8px 0">↓</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                <div style="background:rgba(67,214,197,.07);border:1px solid rgba(67,214,197,.28);border-radius:10px;padding:13px">
                  <div style="color:#72E1D6;font-family:DM Mono;font-size:.65rem">04 / VERIFY</div>
                  <div style="color:#EDF4FC;font-weight:800;margin-top:5px">Numerical agreement</div>
                  <div class="tiny-muted">216-row comparison within defined numerical tolerances</div>
                </div>
                <div style="background:rgba(241,182,92,.06);border:1px solid rgba(241,182,92,.25);border-radius:10px;padding:13px">
                  <div style="color:#F1C477;font-family:DM Mono;font-size:.65rem">05 / NEXT PHASE</div>
                  <div style="color:#EDF4FC;font-weight:800;margin-top:5px">Physical validation</div>
                  <div class="tiny-muted">Fixture, low-voltage excitation and instrumented measurements</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-label">BUILD READINESS</div>
              <div class="status-row"><span class="status-name">MATLAB model freeze</span><span class="status-state">COMPLETE</span></div>
              <div class="status-row"><span class="status-name">216-point dataset check</span><span class="status-state">PASS</span></div>
              <div class="status-row"><span class="status-name">AI surrogate integration</span><span class="status-state">PASS</span></div>
              <div class="status-row"><span class="status-name">Dashboard data integration</span><span class="status-state">LOADED</span></div>
              <div class="status-row"><span class="status-name">Physical fixture & testing</span><span class="status-state status-pending">PENDING</span></div>
              <div style="margin-top:15px;padding:11px 12px;background:rgba(241,182,92,.07);border:1px solid rgba(241,182,92,.22);border-radius:9px;color:#D9C49D;font-size:.72rem;line-height:1.55">
                Scope boundary: computational agreement is not independent physical validation, real-cable accuracy, or EMC standards compliance.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("MODEL RESPONSE ENVELOPE", "ranges across the simulated design grid")
    rc1, rc2 = st.columns(2)
    cap_min = float(physics["TotalCapacitance_pF_per_m"].min())
    cap_max = float(physics["TotalCapacitance_pF_per_m"].max())
    mut_min = float(physics["AbsMutualInductance_nH"].min())
    mut_max = float(physics["AbsMutualInductance_nH"].max())
    with rc1:
        add_panel("Total capacitance", f"{cap_min:.3f} – {cap_max:.3f} <span style='font-size:.82rem;color:#98AABD'>pF/m</span>", "Min–max across the 216 simulated configurations")
    with rc2:
        add_panel("Absolute mutual inductance", f"{mut_min:.4f} – {mut_max:.4f} <span style='font-size:.82rem;color:#98AABD'>nH</span>", "Min–max across the 216 simulated configurations")

    section_header("DESIGN-SPACE MAP", "rotate, zoom and hover; every point is one supported configuration")
    land_choice = st.radio(
        "Colour the map by", ["Total capacitance (pF/m)", "Absolute mutual inductance (nH)"],
        horizontal=True, key="land_color",
    )
    land_col = C_COL if land_choice.startswith("Total") else M_COL
    st.plotly_chart(design_landscape(land_col, land_choice), use_container_width=True, config={"displaylogo": False})

    section_header("WHAT THE GRID SHOWS", "descriptive statistics from the 216 configurations")
    for line in grid_insights():
        st.markdown(f"<div class='insight'>{line}</div>", unsafe_allow_html=True)
    st.caption("Level-mean spans are descriptive over this designed grid. They are not causal sensitivity coefficients and not EMI-performance claims.")

    st.markdown("<div class='tiny-muted' style='margin-top:12px'>Model geometry is a simplified four-conductor representation. Cable envelope (35.45 × 5.80 mm) and equivalent conductor radius are current model assumptions, not confirmed manufacturer cable dimensions.</div>", unsafe_allow_html=True)


# ============================================================================
# PAGE 2 · GEOMETRY EXPLORER
# ============================================================================
elif page == "Geometry Explorer":
    add_hero(
        "GEOMETRY LAB / CONFIGURATION EXPLORER",
        "Inspect the conductor arrangement before reading the response.",
        "Choose only supported geometry levels from the frozen design grid. The drawings are generated "
        "from the same conductor-centre equations used by the current model and are schematic—not a verified "
        "cross-section of a KONE cable assembly.",
        [("DISCRETE DESIGN GRID", "pill-cyan"), ("NO EXTRAPOLATION", "pill-amber")],
    )

    sep_values = sorted(comparison["PairSeparation_mm"].unique().tolist())
    pitch_values = sorted(comparison["PairPitch_mm"].unique().tolist())
    overlap_values = sorted(comparison["Overlap_mm"].unique().tolist())
    clearance_values = sorted(comparison["GroundClearance_mm"].unique().tolist())

    # Open with the Stage 12B equal-weight compromise, provided it is present.
    defaults = {"sep": 12.0, "pitch": 2.0, "overlap": 150.0, "clearance": 100.0}

    def default_index(values: list[float], key: str) -> int:
        return values.index(defaults[key]) if defaults[key] in values else 0

    controls = st.container(border=True)
    with controls:
        st.markdown("**GEOMETRY PARAMETERS**")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            separation = st.selectbox("Pair separation · D (mm)", sep_values, index=default_index(sep_values, "sep"))
        with g2:
            pitch = st.selectbox("Pair pitch · p (mm)", pitch_values, index=default_index(pitch_values, "pitch"))
        with g3:
            overlap = st.selectbox("Shared overlap · L (mm)", overlap_values, index=default_index(overlap_values, "overlap"))
        with g4:
            clearance = st.selectbox("Ground clearance (mm)", clearance_values, index=default_index(clearance_values, "clearance"))

    selected = comparison[
        (comparison["PairSeparation_mm"] == separation)
        & (comparison["PairPitch_mm"] == pitch)
        & (comparison["Overlap_mm"] == overlap)
        & (comparison["GroundClearance_mm"] == clearance)
    ]
    if selected.empty:
        st.error("This geometry combination is not present in the frozen 216-point dataset.")
        st.stop()
    row = selected.iloc[0]

    section_header(f"SELECTED CONFIGURATION · {row['Config_ID']}", "all dimensions in mm")
    out1, out2, out3, out4 = st.columns(4)
    out1.metric("MATLAB capacitance", f"{row['MATLAB_Capacitance_pF_per_m']:.4f} pF/m")
    out2.metric("AI capacitance", f"{row['AI_Capacitance_pF_per_m']:.4f} pF/m")
    out3.metric("MATLAB |M|", f"{row['MATLAB_AbsMutualInductance_nH']:.5f} nH")
    out4.metric("AI |M|", f"{row['AI_AbsMutualInductance_nH']:.5f} nH")

    viz_tab, run_tab = st.tabs(["CROSS-SECTION VIEW", "PARALLEL-RUN VIEW"])
    with viz_tab:
        st.plotly_chart(geometry_cross_section(separation, pitch, clearance), use_container_width=True, config={"displaylogo": False})
        st.caption(
            f"Conductor centre height = clearance + half envelope height = {clearance:.1f} + {CABLE_HEIGHT_MM / 2:.2f} = {clearance + CABLE_HEIGHT_MM / 2:.2f} mm. "
            f"Equivalent copper diameter from 0.50 mm² area = {EQUIV_COPPER_DIAMETER_MM:.3f} mm. "
            "The circular markers are schematic and not drawn to scale. No shield is represented."
        )
    with run_tab:
        st.plotly_chart(geometry_3d(separation, pitch, overlap, clearance), use_container_width=True, config={"displaylogo": False})
        st.caption("The longitudinal dimension shows the modelled shared parallel-run length. The outline indicates the assumed cable envelope only; it is not a shield or construction drawing.")

    section_header("AI–MATLAB DIFFERENCE FOR THIS CONFIGURATION")
    d1, d2 = st.columns(2)
    d1.metric("Capacitance absolute difference", f"{row['Capacitance_Absolute_Difference']:.3e} pF/m")
    d2.metric("Mutual inductance absolute difference", f"{row['MutualInductance_Absolute_Difference']:.3e} nH")
    st.info("These values are read from the frozen Stage 14B comparison export. This page does not execute MATLAB or perform a fresh live AI inference.")

    section_header("SUPPORTED DOMAIN AND INPUT CHECK", "predictions exist only for exact grid combinations")
    st.dataframe(
        pd.DataFrame({
            "Input": GEOM_COLS,
            "Min": [min(GRID_LEVELS[c]) for c in GEOM_COLS],
            "Max": [max(GRID_LEVELS[c]) for c in GEOM_COLS],
            "Supported levels (mm)": [", ".join(f"{v:g}" for v in GRID_LEVELS[c]) for c in GEOM_COLS],
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "On-grid: exact match, AI–MATLAB agreement checked. Interpolated: between levels, not supported by the "
        "frozen predictor. Out of domain: outside the original ranges, rejected."
    )
    with st.expander("Test the domain check with arbitrary values"):
        t1, t2, t3, t4 = st.columns(4)
        test_cfg = {
            "PairSeparation_mm": t1.number_input("Separation (mm)", value=float(separation), key="dc_sep"),
            "PairPitch_mm": t2.number_input("Pitch (mm)", value=float(pitch), key="dc_pit"),
            "Overlap_mm": t3.number_input("Overlap (mm)", value=float(overlap), key="dc_ovl"),
            "GroundClearance_mm": t4.number_input("Clearance (mm)", value=float(clearance), key="dc_clr"),
        }
        status, msg = classify_geometry(test_cfg)
        {"on-grid": st.success, "interpolated": st.warning, "out-of-domain": st.error}[status](
            f"{status.upper()}: {msg}"
        )
        if status == "on-grid":
            hit = comparison[(comparison[GEOM_COLS] == pd.Series(test_cfg)).all(axis=1)].iloc[0]
            st.write(
                f"Config {hit['Config_ID']} · MATLAB C = {hit['MATLAB_Capacitance_pF_per_m']:.4f} pF/m, "
                f"|M| = {hit['MATLAB_AbsMutualInductance_nH']:.5f} nH (frozen export, not live inference)."
            )


# ============================================================================
# PAGE · DESIGN ADVISOR
# ============================================================================
elif page == "Design Advisor":
    add_hero(
        "DESIGN ADVISOR / CONSTRAINT-AWARE SCREENING",
        "From a baseline arrangement to a ranked set of better candidates.",
        "Choose a baseline, restrict the levels your mechanical layout allows, and set how much each output matters. "
        "Candidates come only from the supported 216-point grid, so every recommendation has a verified MATLAB reference "
        "and a checked AI agreement. Changes are relative model-output changes, not measured EMI reductions.",
        [("GRID-ONLY CANDIDATES", "pill-cyan"), ("VERIFY IN MATLAB BEFORE USE", "pill-amber")],
    )

    levels = {k: sorted(physics[k].unique().tolist()) for k in PARAM_LABELS}
    adv_defaults = {"PairSeparation_mm": 8.0, "PairPitch_mm": 2.0, "Overlap_mm": 450.0, "GroundClearance_mm": 10.0}

    box = st.container(border=True)
    with box:
        st.markdown("**BASELINE CONFIGURATION**")
        b1, b2, b3, b4 = st.columns(4)
        base_cfg = {}
        for col_ui, key, label in zip(
            (b1, b2, b3, b4), PARAM_LABELS,
            ("Pair separation (mm)", "Pair pitch (mm)", "Overlap (mm)", "Ground clearance (mm)"),
        ):
            base_cfg[key] = col_ui.selectbox(label, levels[key], index=levels[key].index(adv_defaults[key]), key=f"adv_b_{key}")
        st.markdown("**ALLOWED LEVELS · mechanical constraints**")
        a1, a2, a3, a4 = st.columns(4)
        allowed = {}
        for col_ui, key, label in zip(
            (a1, a2, a3, a4), PARAM_LABELS,
            ("Separation", "Pitch", "Overlap", "Clearance"),
        ):
            allowed[key] = col_ui.multiselect(f"{label} levels", levels[key], default=levels[key], key=f"adv_a_{key}")
        cap_priority = st.slider("Priority: mutual inductance  ◄──►  capacitance", 0, 100, 50, 5, key="adv_w",
                                 help="0 = only mutual inductance matters, 100 = only capacitance matters.")

    if any(len(v) == 0 for v in allowed.values()):
        st.warning("Select at least one allowed level for every parameter.")
        st.stop()

    base = lookup_row(base_cfg)
    mask = np.ones(len(physics), dtype=bool)
    for key, vals in allowed.items():
        mask &= physics[key].isin(vals).to_numpy()
    cand = physics[mask & (physics["Config_ID"] != base["Config_ID"]).to_numpy()].copy()
    if cand.empty:
        st.warning("No alternative configurations satisfy these constraints.")
        st.stop()

    cand["Δ Capacitance (%)"] = (cand[C_COL] - base[C_COL]) / base[C_COL] * 100.0
    cand["Δ |M| (%)"] = (cand[M_COL] - base[M_COL]) / base[M_COL] * 100.0
    wc = cap_priority / 100.0
    cand["Weighted change (%)"] = wc * cand["Δ Capacitance (%)"] + (1 - wc) * cand["Δ |M| (%)"]

    def status(r) -> str:
        if r["Δ Capacitance (%)"] < 0 and r["Δ |M| (%)"] < 0:
            return "Improves both"
        if r["Δ Capacitance (%)"] > 0 and r["Δ |M| (%)"] > 0:
            return "Worse on both"
        return "Trade-off"

    cand["Status"] = cand.apply(status, axis=1)
    ranked = cand.sort_values("Weighted change (%)").reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    best = ranked.iloc[0]
    has_improvement = bool(best["Weighted change (%)"] < 0)

    section_header("RECOMMENDATION", f"baseline {base['Config_ID']} · {len(ranked)} feasible candidates")
    r1, r2 = st.columns([1.15, 1])
    with r1:
        if has_improvement:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="panel-label">BEST FEASIBLE CANDIDATE UNDER YOUR WEIGHTS</div>
                  <div class="rec-id">{best['Config_ID']}</div>
                  <div class="status-row"><span class="status-name">Pair separation</span><span class="mono">{base_cfg['PairSeparation_mm']:g} → {best['PairSeparation_mm']:g} mm</span></div>
                  <div class="status-row"><span class="status-name">Pair pitch</span><span class="mono">{base_cfg['PairPitch_mm']:g} → {best['PairPitch_mm']:g} mm</span></div>
                  <div class="status-row"><span class="status-name">Overlap</span><span class="mono">{base_cfg['Overlap_mm']:g} → {best['Overlap_mm']:g} mm</span></div>
                  <div class="status-row"><span class="status-name">Ground clearance</span><span class="mono">{base_cfg['GroundClearance_mm']:g} → {best['GroundClearance_mm']:g} mm</span></div>
                  <div class="status-row"><span class="status-name">Predicted change in capacitance</span><span class="mono">{best['Δ Capacitance (%)']:+.1f}%</span></div>
                  <div class="status-row"><span class="status-name">Predicted change in |M|</span><span class="mono">{best['Δ |M| (%)']:+.1f}%</span></div>
                  <div class="tiny-muted" style="margin-top:10px">Status: <b>{best['Status']}</b> · within allowed levels · supported grid point</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No candidate improves the weighted objective under these constraints and priorities. "
                    "The baseline is already the best feasible choice for this weighting.")
    with r2:
        stat_card("Baseline capacitance", f"{base[C_COL]:.4f} pF/m", f"config {base['Config_ID']}", "neutral")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        stat_card("Baseline |M|", f"{base[M_COL]:.5f} nH", "frozen MATLAB-referenced dataset", "neutral")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        comp_row = comparison[comparison["Config_ID"] == best["Config_ID"]].iloc[0]
        stat_card(
            "AI–MATLAB check for the top candidate",
            f"{comp_row['Capacitance_Absolute_Difference']:.1e} / {comp_row['MutualInductance_Absolute_Difference']:.1e}",
            "absolute differences (pF/m, nH) from the frozen export", "ok",
        )

    section_header("RANKED CANDIDATES")
    show_cols = ["Rank", "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
                 "Δ Capacitance (%)", "Δ |M| (%)", "Weighted change (%)", "Status"]
    table = ranked[show_cols].copy()
    for c in ("Δ Capacitance (%)", "Δ |M| (%)", "Weighted change (%)"):
        table[c] = table[c].round(2)
    st.dataframe(table.head(15), use_container_width=True, hide_index=True)

    top = ranked.head(8)
    bar = go.Figure()
    bar.add_trace(go.Bar(x=top["Config_ID"], y=top["Δ Capacitance (%)"], name="Δ capacitance (%)", marker_color=PALETTE["cyan"]))
    bar.add_trace(go.Bar(x=top["Config_ID"], y=top["Δ |M| (%)"], name="Δ |M| (%)", marker_color=PALETTE["blue"]))
    bar.add_hline(y=0, line=dict(color=PALETTE["amber"], dash="dash", width=1.5))
    bar.update_layout(title="Top 8 candidates · relative change against baseline (negative is lower)",
                      barmode="group", xaxis_title="Configuration", yaxis_title="Change vs baseline (%)")
    st.plotly_chart(plotly_style(bar, 380), use_container_width=True, config={"displaylogo": False})

    if has_improvement:
        section_header("WHY THIS CANDIDATE", "one change at a time, read from the dataset")
        cand_cfg = {k: float(best[k]) for k in PARAM_LABELS}
        sc = single_change_table(base_cfg, cand_cfg)
        if not sc.empty:
            st.dataframe(sc, use_container_width=True, hide_index=True)
        st.caption("Each row applies one parameter change to the baseline with the others held, using exact grid values. "
                   "The combined change is not the sum of the rows because the inputs interact.")

    export = ranked[show_cols].copy()
    export.insert(0, "Baseline_Config_ID", base["Config_ID"])
    export["Recommended"] = (export["Rank"] == 1) & has_improvement
    export["Feasibility"] = "Within allowed levels and supported grid"
    st.download_button(
        "Download advisor results (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="EMC_GUARD_Design_Advisor_Results.csv", mime="text/csv",
    )
    st.warning(
        "Relative changes are differences in the two modelled outputs, capacitance and absolute mutual inductance. "
        "They are not measured EMI reductions. A lower value of either output does not by itself mean lower victim "
        "voltage or compliance. Verify any recommendation in MATLAB and, later, by physical test."
    )


# ============================================================================
# PAGE 3 · MODEL AGREEMENT
# ============================================================================
elif page == "Model Agreement":
    add_hero(
        "MODEL VERIFICATION / STAGE 14B-5",
        "Two computational paths. One frozen configuration grid.",
        "The AI surrogate is compared with the MATLAB reference for each of the original 216 configurations. "
        "The tolerances shown here are numerical cross-tool comparison thresholds only; they are not physical "
        "measurement tolerances or EMC acceptance limits.",
        [("216 / 216 ROWS", "pill-cyan"), ("COMPUTATIONAL ONLY", "pill-amber")],
    )

    a, b, c, d = st.columns(4)
    with a:
        stat_card("Capacitance checks", f"{cap_pass}/216", "within numerical tolerance" if cap_pass == 216 else "check required", "ok" if cap_pass == 216 else "pending")
    with b:
        stat_card("Max |ΔC|", f"{cap_abs.max():.3e}", "pF/m", "neutral")
    with c:
        stat_card("Mutual-inductance checks", f"{mutual_pass}/216", "within numerical tolerance" if mutual_pass == 216 else "check required", "ok" if mutual_pass == 216 else "pending")
    with d:
        stat_card("Max |ΔM|", f"{mutual_abs.max():.3e}", "nH", "neutral")

    if computational_pass:
        st.success("All 216 AI outputs are within the defined numerical comparison tolerances against the exported MATLAB references.")
    else:
        st.warning("At least one comparison exceeds the defined numerical tolerance. Inspect the error table and source exports.")

    t1, t2 = st.tabs(["CAPACITANCE AGREEMENT", "MUTUAL INDUCTANCE AGREEMENT"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison["MATLAB_Capacitance_pF_per_m"],
            y=comparison["AI_Capacitance_pF_per_m"],
            mode="markers", name="216 configurations",
            marker=dict(size=8, color=PALETTE["cyan"], opacity=.78, line=dict(width=.5, color="#D6FFF9")),
            customdata=comparison[["Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm"]],
            hovertemplate="%{customdata[0]}<br>MATLAB: %{x:.10f} pF/m<br>AI: %{y:.10f} pF/m<br>D=%{customdata[1]} · p=%{customdata[2]} · L=%{customdata[3]} · h=%{customdata[4]} mm<extra></extra>",
        ))
        lo = float(min(comparison["MATLAB_Capacitance_pF_per_m"].min(), comparison["AI_Capacitance_pF_per_m"].min()))
        hi = float(max(comparison["MATLAB_Capacitance_pF_per_m"].max(), comparison["AI_Capacitance_pF_per_m"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal agreement", line=dict(color=PALETTE["amber"], dash="dash", width=2)))
        fig.update_layout(title="Capacitance · AI prediction against MATLAB reference", xaxis_title="MATLAB reference (pF/m)", yaxis_title="AI surrogate (pF/m)")
        st.plotly_chart(plotly_style(fig, 440), use_container_width=True, config={"displaylogo": False})
        err_fig = px.histogram(x=cap_abs, nbins=28, labels={"x": "Absolute difference (pF/m)", "count": "Configurations"}, title="Capacitance absolute-difference distribution")
        err_fig.update_traces(marker_color=PALETTE["cyan"], opacity=.85)
        st.plotly_chart(plotly_style(err_fig, 300, legend=False), use_container_width=True, config={"displaylogo": False})
        st.caption(f"Defined comparison tolerance: {CAP_TOL:.1e} pF/m. Maximum observed difference: {cap_abs.max():.3e} pF/m.")

    with t2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison["MATLAB_AbsMutualInductance_nH"],
            y=comparison["AI_AbsMutualInductance_nH"],
            mode="markers", name="216 configurations",
            marker=dict(size=8, color=PALETTE["blue"], opacity=.78, line=dict(width=.5, color="#E5EEFF")),
            customdata=comparison[["Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm"]],
            hovertemplate="%{customdata[0]}<br>MATLAB: %{x:.12f} nH<br>AI: %{y:.12f} nH<br>D=%{customdata[1]} · p=%{customdata[2]} · L=%{customdata[3]} · h=%{customdata[4]} mm<extra></extra>",
        ))
        lo = float(min(comparison["MATLAB_AbsMutualInductance_nH"].min(), comparison["AI_AbsMutualInductance_nH"].min()))
        hi = float(max(comparison["MATLAB_AbsMutualInductance_nH"].max(), comparison["AI_AbsMutualInductance_nH"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal agreement", line=dict(color=PALETTE["amber"], dash="dash", width=2)))
        fig.update_layout(title="Absolute mutual inductance · AI prediction against MATLAB reference", xaxis_title="MATLAB reference (nH)", yaxis_title="AI surrogate (nH)")
        st.plotly_chart(plotly_style(fig, 440), use_container_width=True, config={"displaylogo": False})
        err_fig = px.histogram(x=mutual_abs, nbins=28, labels={"x": "Absolute difference (nH)", "count": "Configurations"}, title="Mutual-inductance absolute-difference distribution")
        err_fig.update_traces(marker_color=PALETTE["blue"], opacity=.85)
        st.plotly_chart(plotly_style(err_fig, 300, legend=False), use_container_width=True, config={"displaylogo": False})
        st.caption(f"Defined comparison tolerance: {MUTUAL_TOL:.1e} nH. Maximum observed difference: {mutual_abs.max():.3e} nH.")

    with st.expander("View row-level comparison results"):
        display = comparison.copy()
        display["Capacitance_check"] = np.where(cap_abs <= CAP_TOL, "PASS", "REVIEW")
        display["Mutual_inductance_check"] = np.where(mutual_abs <= MUTUAL_TOL, "PASS", "REVIEW")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.warning("This verifies numerical reproduction of the MATLAB export over the original design grid. It does not establish accuracy for unseen geometries, a real manufacturer cable, physical measurements, or EMC compliance.")


# ============================================================================
# PAGE 4 · PARETO & SENSITIVITY
# ============================================================================
elif page == "Pareto & Sensitivity":
    add_hero(
        "DESIGN SPACE / MULTI-OBJECTIVE SCREENING",
        "Explore the trade-off between the two model outputs.",
        "The Pareto view treats total capacitance and absolute mutual inductance as two objectives to minimize. "
        "The weighted selector is a mathematical compromise inside the sampled grid—not a proven minimizer of "
        "victim interference, a safety risk, or an EMC compliance metric.",
        [("216 DESIGNS SCREENED", "pill-cyan"), ("MODEL-BASED TRADE-OFF", "pill-amber")],
    )

    pareto = physics[[
        "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm", "GroundClearance_mm",
        "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH",
    ]].copy()
    pareto["Pareto"] = pareto_mask(pareto, "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH")

    weight_col, weight_mut = st.columns([1, 1])
    with weight_col:
        cap_weight = st.slider("Capacitance objective weight", 0, 100, 50, 5)
    with weight_mut:
        mutual_weight = st.slider("Mutual-inductance objective weight", 0, 100, 50, 5)
    candidate = balanced_candidate(pareto, float(cap_weight), float(mutual_weight))
    front_count = int(pareto["Pareto"].sum())

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Non-dominated configurations", front_count)
    pc2.metric("Selected compromise", str(candidate["Config_ID"]))
    pc3.metric("Objective weights", f"{cap_weight}:{mutual_weight}")

    fig = px.scatter(
        pareto,
        x="TotalCapacitance_pF_per_m",
        y="AbsMutualInductance_nH",
        color="Pareto",
        color_discrete_map={True: PALETTE["cyan"], False: "#52647A"},
        symbol="Pareto",
        symbol_map={True: "diamond", False: "circle"},
        hover_data={
            "Config_ID": True,
            "PairSeparation_mm": True,
            "PairPitch_mm": True,
            "Overlap_mm": True,
            "GroundClearance_mm": True,
            "Pareto": True,
        },
        title="Configuration space · both outputs lower is preferred",
        labels={
            "TotalCapacitance_pF_per_m": "Total capacitance (pF/m)",
            "AbsMutualInductance_nH": "Absolute mutual inductance (nH)",
            "Pareto": "Non-dominated",
        },
    )
    fig.add_trace(go.Scatter(
        x=[candidate["TotalCapacitance_pF_per_m"]],
        y=[candidate["AbsMutualInductance_nH"]],
        mode="markers+text", name="Weighted compromise",
        text=[f"{candidate['Config_ID']} · selected"], textposition="top right",
        marker=dict(size=17, color=PALETTE["amber"], symbol="star", line=dict(color="#FFF1D5", width=1.4)),
    ))
    st.plotly_chart(plotly_style(fig, 490), use_container_width=True, config={"displaylogo": False})

    cand1, cand2 = st.columns([1, 1.2])
    with cand1:
        section_header("WEIGHTED COMPROMISE")
        st.markdown(
            f"""
            <div class="panel">
              <div class="panel-label">SELECTED FROM THE 216-POINT GRID</div>
              <div class="panel-value">{candidate['Config_ID']}</div>
              <div class="status-row"><span class="status-name">Pair separation</span><span class="mono">{candidate['PairSeparation_mm']:.0f} mm</span></div>
              <div class="status-row"><span class="status-name">Pair pitch</span><span class="mono">{candidate['PairPitch_mm']:.0f} mm</span></div>
              <div class="status-row"><span class="status-name">Overlap</span><span class="mono">{candidate['Overlap_mm']:.0f} mm</span></div>
              <div class="status-row"><span class="status-name">Ground clearance</span><span class="mono">{candidate['GroundClearance_mm']:.0f} mm</span></div>
              <div class="status-row"><span class="status-name">Capacitance</span><span class="mono">{candidate['TotalCapacitance_pF_per_m']:.4f} pF/m</span></div>
              <div class="status-row"><span class="status-name">Absolute mutual inductance</span><span class="mono">{candidate['AbsMutualInductance_nH']:.5f} nH</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cand2:
        section_header("INTERPRETATION")
        st.markdown(
            """
            <div class="panel">
              <div style="color:#EAF1FA;font-weight:800;margin-bottom:8px">What this selector does</div>
              <div class="tiny-muted" style="font-size:.79rem">
                Each output is min–max normalized over the current 216-row dataset. The selected point minimizes
                a weighted Euclidean distance from the ideal low-capacitance / low-mutual-inductance corner.
                Changing the sliders changes the compromise—not the underlying physics model or dataset.
                <br><br>
                <span style="color:#F1C477;font-weight:800">Engineering caution:</span> lower values of these two model outputs do not automatically imply lower victim voltage,
                lower conducted/radiated emissions, or regulatory compliance. Those require a validated coupling
                circuit, frequency-dependent response and physical test correlation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("DESIGN-SPACE EXPLORATION", "exact grid values, no interpolation")
    heat_tab, par_tab = st.tabs(["RESPONSE HEATMAP", "PARALLEL COORDINATES"])
    with heat_tab:
        h1, h2, h3 = st.columns(3)
        out_choice = h1.selectbox("Output", ["Total capacitance (pF/m)", "Absolute mutual inductance (nH)"], key="hm_out")
        x_key = h2.selectbox("X axis", list(PARAM_LABELS), index=0, key="hm_x", format_func=lambda k: PARAM_LABELS[k].capitalize())
        y_opts = [k for k in PARAM_LABELS if k != x_key]
        y_key = h3.selectbox("Y axis", y_opts, index=y_opts.index("GroundClearance_mm") if "GroundClearance_mm" in y_opts else 0,
                             key="hm_y", format_func=lambda k: PARAM_LABELS[k].capitalize())
        fixed_keys = [k for k in PARAM_LABELS if k not in (x_key, y_key)]
        f1, f2 = st.columns(2)
        fixed_vals = {}
        for col_ui, key in zip((f1, f2), fixed_keys):
            lv = sorted(physics[key].unique().tolist())
            fixed_vals[key] = col_ui.selectbox(f"Hold {PARAM_LABELS[key]} at (mm)", lv, key=f"hm_f_{key}")
        out_col = C_COL if out_choice.startswith("Total") else M_COL
        st.plotly_chart(response_heatmap(out_col, out_choice, x_key, y_key, fixed_vals), use_container_width=True, config={"displaylogo": False})
        st.caption("Each cell is an exact configuration from the frozen dataset with the other two inputs held at the selected levels.")
    with par_tab:
        pc_fig = px.parallel_coordinates(
            physics, dimensions=list(PARAM_LABELS) + [C_COL, M_COL], color=C_COL,
            color_continuous_scale=px.colors.sequential.Tealgrn,
            labels={"PairSeparation_mm": "Separation", "PairPitch_mm": "Pitch", "Overlap_mm": "Overlap",
                    "GroundClearance_mm": "Clearance", C_COL: "Capacitance", M_COL: "|M|"},
        )
        pc_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(family="Manrope, sans-serif", color="#C8D5E5", size=11),
                             height=430, margin=dict(l=40, r=40, t=40, b=20))
        st.plotly_chart(pc_fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Drag along any axis to select a range and see which configurations produce which outputs.")

    section_header("PARAMETER RESPONSE / DESCRIPTIVE SENSITIVITY")
    st.caption("Each curve shows the mean response at each level after averaging across the other sampled variables. Mean spans are descriptive over this grid, not independent causal sensitivity coefficients.")

    params = [
        ("PairSeparation_mm", "Pair separation", "D (mm)"),
        ("PairPitch_mm", "Pair pitch", "p (mm)"),
        ("Overlap_mm", "Parallel overlap", "L (mm)"),
        ("GroundClearance_mm", "Ground clearance", "h (mm)"),
    ]
    sens_rows = []
    chart_cols = st.columns(2)
    for idx, (key, label, axis_label) in enumerate(params):
        grouped = physics.groupby(key, as_index=False).agg(
            Capacitance=("TotalCapacitance_pF_per_m", "mean"),
            MutualInductance=("AbsMutualInductance_nH", "mean"),
        ).sort_values(key)
        sens_rows.append({
            "Parameter": label,
            "Capacitance mean span (pF/m)": grouped["Capacitance"].max() - grouped["Capacitance"].min(),
            "Mutual inductance mean span (nH)": grouped["MutualInductance"].max() - grouped["MutualInductance"].min(),
        })
        with chart_cols[idx % 2]:
            f = go.Figure()
            f.add_trace(go.Scatter(x=grouped[key], y=grouped["Capacitance"], mode="lines+markers", name="Mean capacitance", line=dict(color=PALETTE["cyan"], width=2.5), marker=dict(size=8)))
            f.update_layout(title=f"{label} · mean capacitance", xaxis_title=axis_label, yaxis_title="Mean capacitance (pF/m)")
            st.plotly_chart(plotly_style(f, 300, legend=False), use_container_width=True, config={"displaylogo": False})

    sensitivity_table = pd.DataFrame(sens_rows)
    with st.expander("Compare descriptive mean spans"):
        st.dataframe(sensitivity_table.sort_values("Capacitance mean span (pF/m)", ascending=False), use_container_width=True, hide_index=True)

    # A second paired set for mutual-inductance effects.
    mcols = st.columns(2)
    for idx, (key, label, axis_label) in enumerate(params):
        grouped = physics.groupby(key, as_index=False).agg(
            MutualInductance=("AbsMutualInductance_nH", "mean")
        ).sort_values(key)
        with mcols[idx % 2]:
            f = go.Figure()
            f.add_trace(go.Scatter(x=grouped[key], y=grouped["MutualInductance"], mode="lines+markers", name="Mean |M|", line=dict(color=PALETTE["blue"], width=2.5), marker=dict(size=8)))
            f.update_layout(title=f"{label} · mean mutual inductance", xaxis_title=axis_label, yaxis_title="Mean |M| (nH)")
            st.plotly_chart(plotly_style(f, 300, legend=False), use_container_width=True, config={"displaylogo": False})


# ============================================================================
# PAGE 5 · VALIDATION STATUS
# ============================================================================
elif page == "Validation Status":
    add_hero(
        "VALIDATION GATE / EXPERIMENTAL WORK PACKAGE",
        "Keep computational results separate from measured evidence.",
        "No physical measurements are bundled with this dashboard. The upload area can be used later to inspect "
        "a structured test CSV, but uploaded data is session-only and must be reviewed against the actual instrument "
        "setup, calibration and repeatability before being treated as validation evidence.",
        [("COMPUTATIONAL CHECKS COMPLETE", "pill-cyan"), ("PHYSICAL TESTS NOT YET RECORDED", "pill-amber")],
    )

    stages = pd.DataFrame({
        "Validation work package": [
            "MATLAB reference model and frozen dataset",
            "216-configuration MATLAB export verification",
            "AI surrogate reproduction against MATLAB export",
            "Physical fixture fabrication and geometry measurement",
            "Isolated low-voltage excitation setup",
            "Instrumented measurements and repeated runs",
            "Model-to-measurement correlation",
        ],
        "Status": [
            "Complete — computational",
            "Complete — computational",
            "Complete — computational grid only",
            "Pending",
            "Pending",
            "Pending",
            "Pending",
        ],
        "Evidence type": [
            "MATLAB files and dataset",
            "216-row numerical check",
            "Stage 14B-5 comparison report",
            "Physical fixture record",
            "Test setup and safety record",
            "Raw instrument data",
            "Residuals, uncertainty and repeatability",
        ],
    })
    st.dataframe(stages, use_container_width=True, hide_index=True)

    section_header("KNOWN LIMITATIONS")
    st.warning(
        "Reproduction of MATLAB is verified on the original 216-configuration design grid. "
        "Leave-one-geometry-level-out tests showed weaker prediction on unseen geometry levels, so unseen "
        "levels are rejected rather than predicted. Physical validation and victim-voltage prediction are "
        "future work and need a validated circuit model and additional parameters."
    )
    # Looks in <project>/reports for a leave-one-level-out CSV. Rename/point as needed.
    lolo_matches = sorted(REPORTS_DIR.glob("*leave*out*.csv")) + sorted(REPORTS_DIR.glob("*LOLO*.csv")) if REPORTS_DIR.exists() else []
    if lolo_matches:
        st.caption(f"Leave-one-geometry-level-out results · {lolo_matches[0].name}")
        try:
            st.dataframe(pd.read_csv(lolo_matches[0]), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read {lolo_matches[0].name}: {exc}")
    else:
        st.caption(
            "No leave-one-level-out results file found in the reports folder. Place the CSV there "
            "(file name containing 'leave' and 'out', or 'LOLO') to display it here."
        )

    section_header("FUTURE MEASUREMENT DATA TEMPLATE")
    measurement_columns = [
        "Test_ID", "Config_ID", "PairSeparation_mm", "PairPitch_mm", "Overlap_mm",
        "GroundClearance_mm", "Excitation_Frequency_Hz", "Excitation_Voltage_V",
        "Measured_Capacitance_pF_per_m", "Measured_AbsMutualInductance_nH",
        "Repeat_Number", "Instrument", "Calibration_Reference", "Setup_Notes",
    ]
    template = pd.DataFrame(columns=measurement_columns)
    st.download_button(
        "Download blank experimental-data CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="EMC_GUARD_Experimental_Data_Template.csv",
        mime="text/csv",
    )
    st.caption("Use one row per repeat. Record measured values only after the instrument, calibration state, wiring arrangement and test conditions are documented.")

    section_header("OPTIONAL MEASUREMENT FILE REVIEW")
    uploaded = st.file_uploader("Upload a populated experimental CSV", type=["csv"], help="Expected columns include Config_ID, Measured_Capacitance_pF_per_m and Measured_AbsMutualInductance_nH.")
    if uploaded is None:
        st.info("No physical test file uploaded. The dashboard is currently showing computational evidence only.")
    else:
        try:
            measured = pd.read_csv(uploaded)
            required_measurement = ["Config_ID", "Measured_Capacitance_pF_per_m", "Measured_AbsMutualInductance_nH"]
            missing_measurement = [c for c in required_measurement if c not in measured.columns]
            if missing_measurement:
                st.error(f"Uploaded file is missing required columns: {missing_measurement}")
            elif measured.empty:
                st.warning("The uploaded file has headers but contains no measurement rows.")
            else:
                measured["Config_ID"] = measured["Config_ID"].astype(str)
                measured["Measured_Capacitance_pF_per_m"] = pd.to_numeric(measured["Measured_Capacitance_pF_per_m"], errors="coerce")
                measured["Measured_AbsMutualInductance_nH"] = pd.to_numeric(measured["Measured_AbsMutualInductance_nH"], errors="coerce")
                invalid = measured[required_measurement[1:]].isna().any().any()
                if invalid:
                    st.error("Uploaded measurement columns contain missing or non-numeric values. Correct the file before analysis.")
                else:
                    ref = matlab[["Config_ID", "Capacitance_pF_per_m", "AbsMutualInductance_nH"]]
                    reviewed = measured.merge(ref, on="Config_ID", how="left", validate="many_to_one")
                    unmatched = int(reviewed["Capacitance_pF_per_m"].isna().sum())
                    if unmatched:
                        st.warning(f"{unmatched} row(s) have Config_ID values outside the frozen 216-point dataset. They are excluded from error metrics.")
                    matched = reviewed.dropna(subset=["Capacitance_pF_per_m", "AbsMutualInductance_nH"]).copy()
                    if matched.empty:
                        st.error("No uploaded rows match a supported Config_ID.")
                    else:
                        matched["Capacitance residual (measured − MATLAB)"] = matched["Measured_Capacitance_pF_per_m"] - matched["Capacitance_pF_per_m"]
                        matched["Mutual inductance residual (measured − MATLAB)"] = matched["Measured_AbsMutualInductance_nH"] - matched["AbsMutualInductance_nH"]
                        q1, q2, q3 = st.columns(3)
                        q1.metric("Matched measurement rows", len(matched))
                        q2.metric("Capacitance MAE", f"{matched['Capacitance residual (measured − MATLAB)'].abs().mean():.4g} pF/m")
                        q3.metric("Mutual-inductance MAE", f"{matched['Mutual inductance residual (measured − MATLAB)'].abs().mean():.4g} nH")
                        st.dataframe(matched, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download reviewed measurement comparison",
                            data=matched.to_csv(index=False).encode("utf-8"),
                            file_name="EMC_GUARD_Measurement_Comparison_Review.csv",
                            mime="text/csv",
                        )
                        st.warning("This is a preliminary data comparison only. It does not account for instrument uncertainty, calibration, fixture parasitics, frequency dependence, repeatability, or measurement traceability. Uploads are not saved by this app.")
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")


# ============================================================================
# PAGE 6 · DATA & REPORTS
# ============================================================================
elif page == "Data & Reports":
    add_hero(
        "DATA ROOM / FROZEN EXPORTS",
        "Inspect and export the evidence behind the interface.",
        "Every plotted result in this dashboard is sourced from the frozen Stage 13–14 CSV exports. "
        "The AI–MATLAB comparison is precomputed; this interface does not retrain models, alter the frozen "
        "artifacts, or invoke MATLAB during browsing.",
        [("READ-ONLY DATA VIEW", "pill-cyan"), ("SOURCE EXPORTS PRESERVED", "")],
    )

    data_cards = st.columns(3)
    with data_cards[0]:
        add_panel("Physics dataset", "216 rows", "Geometry inputs and simulated model outputs")
        st.download_button("Download physics CSV", physics.to_csv(index=False).encode("utf-8"), "EMC_GUARD_Physics_Dataset.csv", "text/csv", use_container_width=True)
    with data_cards[1]:
        add_panel("MATLAB reference", "216 rows", "Frozen MATLAB predictions and metadata")
        st.download_button("Download MATLAB CSV", matlab.to_csv(index=False).encode("utf-8"), "EMC_GUARD_MATLAB_Reference.csv", "text/csv", use_container_width=True)
    with data_cards[2]:
        add_panel("AI comparison", "216 rows", "AI surrogate values and numerical differences")
        st.download_button("Download comparison CSV", comparison.to_csv(index=False).encode("utf-8"), "EMC_GUARD_AI_MATLAB_Comparison.csv", "text/csv", use_container_width=True)

    report = {
        "project": APP_TITLE,
        "title": APP_SUBTITLE,
        "dashboard_stage": "15C Premium Engineering Workstation",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_rows": {"physics": len(physics), "matlab_reference": len(matlab), "ai_comparison": len(comparison)},
        "unique_configurations": int(comparison["Config_ID"].nunique()),
        "computational_comparison": {
            "capacitance_tolerance_pF_per_m": CAP_TOL,
            "mutual_inductance_tolerance_nH": MUTUAL_TOL,
            "capacitance_rows_within_tolerance": cap_pass,
            "mutual_inductance_rows_within_tolerance": mutual_pass,
            "maximum_capacitance_absolute_difference_pF_per_m": float(cap_abs.max()),
            "maximum_mutual_inductance_absolute_difference_nH": float(mutual_abs.max()),
            "result": "PASS" if computational_pass else "REVIEW REQUIRED",
        },
        "physical_validation": "PENDING — no physical measurements are bundled with this dashboard",
        "scope_limitations": [
            "The geometry is a simplified four-conductor model, not a verified manufacturer cable cross-section.",
            "The dataset contains the original 216-point design grid only.",
            "The AI comparison is against the exported MATLAB reference, not independent physical measurements.",
            "Leave-one-geometry-level-out tests showed weaker prediction on unseen geometry levels; unseen levels are rejected.",
            "The comparison tolerances are numerical cross-tool thresholds, not EMC acceptance limits.",
            "No EMC standards compliance claim is made.",
        ],
    }
    st.download_button(
        "Download computational validation metadata (JSON)",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name="EMC_GUARD_Dashboard_Validation_Metadata.json",
        mime="application/json",
    )

    section_header("DATASET INSPECTOR")
    dataset = st.selectbox("Select source table", ["Physics dataset", "MATLAB reference", "AI–MATLAB comparison"])
    selected_df = {
        "Physics dataset": physics,
        "MATLAB reference": matlab,
        "AI–MATLAB comparison": comparison,
    }[dataset]
    st.caption(f"{len(selected_df)} rows · {len(selected_df.columns)} columns")
    st.dataframe(selected_df, use_container_width=True, hide_index=True, height=430)

    with st.expander("Model and dataset scope notes"):
        st.markdown(
            """
            **Supported geometry levels**

            - Pair separation: 8, 12, 16, 20, 24, 28 mm
            - Pair pitch: 2, 3, 4 mm
            - Shared overlap: 150, 300, 450 mm
            - Ground clearance: 10, 25, 50, 100 mm

            **Current output scope**

            - Total capacitance (pF/m)
            - Absolute mutual inductance (nH)

            **Not calculated by this build**

            - Frequency-dependent victim voltage or RMS interference
            - Peak-to-peak victim response
            - Real cable/drive-system EMC emissions
            - Product or standards compliance
            - Predictions outside the original 216 configurations
            """
        )


# --------------------------------- Footer -----------------------------------
st.markdown(
    """
    <div class="footer">
      EMC-GUARD · STAGE 15C · RESEARCH PROTOTYPE &nbsp;|&nbsp;
      COMPUTATIONAL PRE-COMPLIANCE SCREENING &nbsp;|&nbsp;
      PHYSICAL VALIDATION PENDING
    </div>
    """,
    unsafe_allow_html=True,
)
