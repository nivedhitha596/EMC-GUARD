"""
EMC-GUARD: Physics-Informed Digital Twin for Elevator EMC Pre-Compliance
=========================================================================
MODULE 6 | Streamlit Engineering Dashboard
File   : Dashboard/app.py

Integrates all five existing modules:
  Module 1 - Dataset & Preprocessing
  Module 2 - AI Prediction Engine     (predict_emc.py)
  Module 3 - Risk Analysis            (risk_analysis.py)
  Module 4 - Explainable AI           (feature_importance.py, prediction_uncertainty.py)
  Module 5 - Cable Routing Optimiser  (optimise_routing.py)

Run:
    streamlit run Dashboard/app.py

DISCLAIMER
----------
All predictions are from models trained on SYNTHETIC DUMMY DATA.
Not for engineering sign-off or certified EMC compliance.
"""

import os
import sys
import json
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATH SETUP — make AI_ML/src importable
# ---------------------------------------------------------------------------
DASH_DIR  = Path(__file__).parent
ROOT_DIR  = DASH_DIR.parent
AI_ML_DIR = ROOT_DIR / "AI_ML"
SRC_DIR   = AI_ML_DIR / "src"
REPORTS   = AI_ML_DIR / "reports"
MODELS    = AI_ML_DIR / "models"
DATASET   = AI_ML_DIR / "dataset"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EMC-GUARD | Digital Twin Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME & CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #07111f;
    color: #dce8f5;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0b1d30;
    border-right: 1px solid #1a3355;
}
/* ── Cards ───────────────────────────────────────────────────── */
.emc-card {
    background: #0f2236;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.emc-card-accent {
    background: linear-gradient(135deg, #0f2236, #0a2a45);
    border: 1px solid #00b4d8;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
/* ── Metric tiles ────────────────────────────────────────────── */
.metric-tile {
    background: #0f2236;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-tile .val {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-tile .lbl {
    font-size: 0.78rem;
    color: #8aa8c4;
    margin-top: 4px;
}
/* ── Risk badges ─────────────────────────────────────────────── */
.risk-lower       { background:#1b3c2a; color:#4caf50; border:1px solid #4caf50; }
.risk-intermediate{ background:#3c2e10; color:#ff9800; border:1px solid #ff9800; }
.risk-higher      { background:#3c1010; color:#f44336; border:1px solid #f44336; }
.risk-unknown     { background:#1e2a38; color:#90a4ae; border:1px solid #546e7a; }
.risk-badge {
    display:inline-block; padding:5px 14px; border-radius:20px;
    font-weight:600; font-size:0.85rem; letter-spacing:0.5px;
}
/* ── Section headers ─────────────────────────────────────────── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00d4ff;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
/* ── Warning / info banners ──────────────────────────────────── */
.banner-warn {
    background:#2e1f00; border-left:4px solid #ff9800;
    padding:10px 14px; border-radius:4px; margin:8px 0;
    font-size:0.85rem; color:#ffd082;
}
.banner-info {
    background:#001f3a; border-left:4px solid #00b4d8;
    padding:10px 14px; border-radius:4px; margin:8px 0;
    font-size:0.85rem; color:#90caf9;
}
.banner-synth {
    background:#1a0a2e; border-left:4px solid #9c27b0;
    padding:10px 14px; border-radius:4px; margin:8px 0;
    font-size:0.85rem; color:#ce93d8;
}
/* ── Buttons ─────────────────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(90deg,#0077b6,#00b4d8);
    color: white; border:none; border-radius:6px;
    padding: 10px 28px; font-weight:600; font-size:0.95rem;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(90deg,#00b4d8,#0077b6);
    transform: translateY(-1px);
}
/* ── Sidebar nav items ───────────────────────────────────────── */
.sidebar-nav-item {
    padding: 10px 14px; border-radius: 6px; cursor: pointer;
    margin-bottom: 4px; font-size: 0.9rem;
    transition: background 0.15s;
}
/* ── Scrollable table ────────────────────────────────────────── */
.scroll-table { overflow-x: auto; }
/* ── Divider ─────────────────────────────────────────────────── */
hr.emc { border-color: #1a3a5c; }
</style>
""", unsafe_allow_html=True)

SYNTH_BANNER = """<div class='banner-synth'>
⚠️ <b>SYNTHETIC DATA:</b> All predictions are based on a model trained on
<b>synthetic dummy data</b>. Results are for pipeline validation and
decision-support only. They do <b>NOT</b> represent validated electromagnetic
physics or certified EMC compliance.
</div>"""

# ---------------------------------------------------------------------------
# PLOTLY DARK TEMPLATE
# ---------------------------------------------------------------------------
EMC_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0f2236",
        plot_bgcolor="#07111f",
        font=dict(color="#dce8f5", family="Segoe UI"),
        xaxis=dict(gridcolor="#1a3a5c", linecolor="#1a3a5c"),
        yaxis=dict(gridcolor="#1a3a5c", linecolor="#1a3a5c"),
        legend=dict(bgcolor="#0f2236", bordercolor="#1a3a5c"),
        colorway=["#00d4ff", "#4caf50", "#ff9800", "#f44336",
                  "#9c27b0", "#00bcd4", "#8bc34a"],
    )
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
ALL_TARGETS = [
    "MutualInductance_nH", "MutualCapacitance_pF",
    "Victim_Voltage_V", "Victim_RMS_V", "Victim_PeakToPeak_V",
]
TARGET_UNITS = {
    "MutualInductance_nH":   "nH",
    "MutualCapacitance_pF":  "pF",
    "Victim_Voltage_V":      "V",
    "Victim_RMS_V":          "V",
    "Victim_PeakToPeak_V":   "V",
}
TARGET_ICONS = {
    "MutualInductance_nH":   "🔄",
    "MutualCapacitance_pF":  "⚡",
    "Victim_Voltage_V":      "📡",
    "Victim_RMS_V":          "〰️",
    "Victim_PeakToPeak_V":   "📶",
}
RISK_COLORS = {
    "LOWER PREDICTED RISK":        "#4caf50",
    "INTERMEDIATE PREDICTED RISK": "#ff9800",
    "HIGHER PREDICTED RISK":       "#f44336",
    "UNDETERMINED RISK":           "#90a4ae",
}
RISK_CSS = {
    "LOWER PREDICTED RISK":        "risk-lower",
    "INTERMEDIATE PREDICTED RISK": "risk-intermediate",
    "HIGHER PREDICTED RISK":       "risk-higher",
    "UNDETERMINED RISK":           "risk-unknown",
}

# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    """Load surrogate model bundle and preprocessor (cached across sessions)."""
    import joblib
    try:
        preprocessor     = joblib.load(MODELS / "emc_preprocessor.pkl")
        surrogate_bundle = joblib.load(MODELS / "emc_surrogate_model.pkl")
        with open(MODELS / "target_model_metadata.json") as f:
            metadata = json.load(f)
        with open(REPORTS / "model_performance.json") as f:
            perf = json.load(f)
        return preprocessor, surrogate_bundle, metadata, perf
    except Exception as e:
        return None, None, None, None


@st.cache_data(show_spinner="Loading dataset …")
def load_dataset():
    try:
        return pd.read_csv(DATASET / "EMC_GUARD_Physics_Dataset.csv")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_report_json(filename: str):
    path = REPORTS / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_report_csv(filename: str):
    path = REPORTS / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def models_available() -> bool:
    return (MODELS / "emc_surrogate_model.pkl").exists()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def risk_badge(category: str) -> str:
    css = RISK_CSS.get(category, "risk-unknown")
    return f"<span class='risk-badge {css}'>{category}</span>"


def _fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def build_config_from_sidebar(prefix="pred") -> dict:
    """Build the 9-feature configuration dict from sidebar sliders."""
    return st.session_state.get(f"_cfg_{prefix}", {})


# ---------------------------------------------------------------------------
# CONFIGURATION INPUT PANEL (shared across pages)
# ---------------------------------------------------------------------------
def render_config_inputs(prefix: str = "pred", compact: bool = False) -> dict:
    """
    Render all 9 input controls.  Returns the configuration dict.
    `prefix` namespaces session_state keys to avoid cross-page collisions.
    """
    if compact:
        c1, c2 = st.columns(2)
    else:
        c1, c2 = st.columns([1, 1])

    with c1:
        sep = st.slider("Separation_mm (mm)",
                        5.0, 200.0, 30.0, 5.0, key=f"{prefix}_sep",
                        help="Cable-to-cable separation distance")
        ovl = st.slider("Overlap_mm (mm)",
                        10.0, 500.0, 200.0, 10.0, key=f"{prefix}_ovl",
                        help="Parallel cable run length")
        cbl = st.slider("Cable_Length_mm (mm)",
                        100.0, 5000.0, 1000.0, 50.0, key=f"{prefix}_cbl",
                        help="Total cable length")
        freq = st.number_input("Frequency_Hz (Hz)",
                               min_value=20000, max_value=30000000,
                               value=500000, step=10000, key=f"{prefix}_freq",
                               help="Signal frequency")
    with c2:
        volt = st.slider("Voltage_V (V)",
                         1.0, 500.0, 48.0, 1.0, key=f"{prefix}_volt")
        rad = st.slider("Conductor_Radius_mm (mm)",
                        0.5, 5.0, 1.0, 0.05, key=f"{prefix}_rad",
                        help="Training min ≈ 0.50 mm. Values at 0.5 mm trigger an extrapolation warning.")
        diel = st.slider("Dielectric_Constant",
                         2.0, 10.0, 4.5, 0.1, key=f"{prefix}_diel")
        shield = st.selectbox("Shield_Type",
                              ["Unshielded", "Shielded"],
                              key=f"{prefix}_shield")
        cable_cfg = st.selectbox("Cable_Configuration",
                                 ["Parallel", "Perpendicular", "Crossed"],
                                 key=f"{prefix}_cfg")

    config = {
        "Separation_mm":       float(sep),
        "Overlap_mm":          float(ovl),
        "Cable_Length_mm":     float(cbl),
        "Frequency_Hz":        float(freq),
        "Voltage_V":           float(volt),
        "Conductor_Radius_mm": float(rad),
        "Dielectric_Constant": float(diel),
        "Shield_Type":         shield,
        "Cable_Configuration": cable_cfg,
    }
    return config


# ---------------------------------------------------------------------------
# PAGE 0 — OVERVIEW
# ---------------------------------------------------------------------------
def page_overview():
    st.markdown("<div class='section-title'>⚡ EMC-GUARD — System Overview</div>",
                unsafe_allow_html=True)
    st.markdown(SYNTH_BANNER, unsafe_allow_html=True)

    # ── Hero description ──────────────────────────────────────────────────
    st.markdown("""<div class='emc-card-accent'>
    <b style='font-size:1.05rem;color:#00d4ff'>EMC-GUARD</b> is a Physics-Informed,
    Configuration-Aware Digital Twin for Predictive EMC Pre-Compliance of Elevator
    Systems. It combines electromagnetic physics simulation data with machine-learning
    surrogate models to help engineers assess and optimise cable configurations
    <em>before</em> physical testing or MATLAB simulation.
    </div>""", unsafe_allow_html=True)

    # ── Dataset metrics ───────────────────────────────────────────────────
    perf = load_report_json("model_performance.json")
    df   = load_dataset()

    cols = st.columns(5)
    tiles = [
        ("Total Records",   perf.get("dataset_size", len(df)),           "#00d4ff"),
        ("Training Records", perf.get("training_records", "—"),          "#4caf50"),
        ("Testing Records",  perf.get("testing_records", "—"),           "#ff9800"),
        ("Prediction Targets", len(perf.get("available_targets", ALL_TARGETS)), "#9c27b0"),
        ("Dataset Type",     perf.get("dataset_type", "SYNTHETIC"),      "#f44336"),
    ]
    for col, (lbl, val, clr) in zip(cols, tiles):
        with col:
            st.markdown(f"""<div class='metric-tile'>
                <div class='val' style='color:{clr}'>{val}</div>
                <div class='lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Module status ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>System Module Status</div>",
                unsafe_allow_html=True)

    modules = [
        ("Module 1", "Dataset Generation & Preparation",
         (DATASET / "EMC_GUARD_Physics_Dataset.csv").exists(), "generate_dummy_data.py"),
        ("Module 2", "AI Surrogate Prediction Engine",
         (MODELS / "emc_surrogate_model.pkl").exists(), "train_model.py / predict_emc.py"),
        ("Module 3", "Configuration Risk Analysis",
         (REPORTS / "risk_thresholds.json").exists(), "risk_analysis.py"),
        ("Module 4", "Explainable AI & Confidence Engine",
         (REPORTS / "feature_importance.csv").exists(), "xai_engine.py"),
        ("Module 5", "Cable-Routing Optimisation Engine",
         (REPORTS / "optimisation_results.csv").exists(), "optimise_routing.py"),
        ("Module 6", "Streamlit Engineering Dashboard",
         True, "Dashboard/app.py"),
    ]

    for mod_id, name, ok, script in modules:
        icon = "✅" if ok else "⚠️"
        clr  = "#4caf50" if ok else "#ff9800"
        status = "ACTIVE" if ok else "FILES MISSING"
        st.markdown(f"""<div class='emc-card' style='padding:12px 18px'>
            <span style='color:{clr};font-size:1.1rem'>{icon}</span>
            <b style='color:#dce8f5;margin-left:8px'>{mod_id}</b>
            <span style='color:#8aa8c4;margin-left:8px'>— {name}</span>
            <span style='float:right;color:{clr};font-size:0.8rem'>{status}</span><br>
            <span style='font-size:0.78rem;color:#546e7a;margin-left:26px'>{script}</span>
        </div>""", unsafe_allow_html=True)

    # ── Model performance summary ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Model Performance Summary</div>",
                unsafe_allow_html=True)

    if perf and "targets" in perf:
        rows = []
        for tgt, tdata in perf["targets"].items():
            best_algo = tdata.get("best_algorithm", "—")
            sel = tdata.get("selected_metrics", {})
            rows.append({
                "Target": tgt,
                "Best Algorithm": best_algo,
                "MAE":  sel.get("MAE",  "—"),
                "RMSE": sel.get("RMSE", "—"),
                "R²":   sel.get("R2",   "—"),
            })
        tbl = pd.DataFrame(rows)
        st.dataframe(
            tbl.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"},
                             na_rep="—"),
            use_container_width=True, hide_index=True
        )

        # R² bar chart
        fig = go.Figure()
        tbl_sorted = tbl.sort_values("R²", ascending=True)
        colors = ["#4caf50" if r >= 0.80 else "#ff9800" if r >= 0.70 else "#f44336"
                  for r in tbl_sorted["R²"].astype(float)]
        fig.add_trace(go.Bar(
            x=tbl_sorted["R²"].astype(float),
            y=tbl_sorted["Target"],
            orientation="h",
            marker_color=colors,
            text=[f"R²={r:.4f}" for r in tbl_sorted["R²"].astype(float)],
            textposition="outside",
        ))
        fig.add_vline(x=0.7, line_dash="dash", line_color="#ff9800",
                      annotation_text="R²=0.70")
        fig.update_layout(
            template=EMC_TEMPLATE,
            title="Model R² by Prediction Target (selected best algorithm)",
            xaxis_title="R² Score", yaxis_title="",
            height=300, margin=dict(l=0, r=60, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 1 — DIGITAL TWIN PREDICTION
# ---------------------------------------------------------------------------
def page_prediction():
    st.markdown("<div class='section-title'>🔬 Digital Twin — EMI Prediction</div>",
                unsafe_allow_html=True)
    st.markdown(SYNTH_BANNER, unsafe_allow_html=True)

    if not models_available():
        st.error("Model files not found. Run train_model.py (Module 2) first.")
        return

    # ── Configuration inputs ──────────────────────────────────────────────
    st.markdown("<div class='emc-card'>", unsafe_allow_html=True)
    st.markdown("**Cable Configuration Parameters**")
    config = render_config_inputs(prefix="pred")
    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("⚡ Predict EMI", key="btn_predict")

    if run:
        with st.spinner("Running Module 2 prediction …"):
            try:
                from predict_emc import predict_interference, ValidationError
                result = predict_interference(config, verbose=False)
                st.session_state["pred_result"] = result
                st.session_state["pred_config"] = copy.deepcopy(config)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

    if "pred_result" not in st.session_state:
        st.info("Set configuration above and click **Predict EMI**.")
        return

    result = st.session_state["pred_result"]
    cfg    = st.session_state["pred_config"]
    preds  = result.get("predictions", {})
    warns  = result.get("warnings", [])
    m_info = result.get("model_info", {})

    # Warnings
    for w in warns:
        st.markdown(f"<div class='banner-warn'>⚠️ {w}</div>",
                    unsafe_allow_html=True)

    # ── Prediction metric cards ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Predicted EMC Outputs</div>",
                unsafe_allow_html=True)

    cols = st.columns(len(preds))
    for col, (tgt, val) in zip(cols, preds.items()):
        unit = TARGET_UNITS.get(tgt, "")
        icon = TARGET_ICONS.get(tgt, "📊")
        r2   = m_info.get(tgt, {}).get("R2") or 0
        r2_c = "#4caf50" if r2 >= 0.80 else "#ff9800" if r2 >= 0.70 else "#f44336"
        with col:
            st.markdown(f"""<div class='metric-tile'>
                <div style='font-size:1.5rem'>{icon}</div>
                <div class='val'>{val:.4f}</div>
                <div class='lbl'>{tgt}<br><span style='color:{unit};'>{unit}</span></div>
                <div style='font-size:0.72rem;color:{r2_c};margin-top:6px'>
                    R²={r2:.4f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Radar chart of predictions (normalised) ───────────────────────────
    if len(preds) >= 3:
        norm_vals = []
        norm_tgts = []
        perf_data = load_report_json("model_performance.json")
        err_df = load_report_csv("prediction_error_analysis.csv")

        for tgt, val in preds.items():
            if not err_df.empty and tgt in err_df["target"].values:
                p90 = err_df[err_df["target"] == tgt]["p90_AE"].values[0]
                low, high = max(0, val - p90), val + p90
            else:
                low, high = 0, val * 1.5 if val > 0 else 1

            norm = val / high if high > 0 else 0
            norm_vals.append(round(min(1, norm), 4))
            norm_tgts.append(tgt.replace("_", " "))

        fig = go.Figure(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=norm_tgts + [norm_tgts[0]],
            fill="toself",
            fillcolor="rgba(0,180,216,0.2)",
            line=dict(color="#00d4ff"),
            name="Predicted (normalised)",
        ))
        fig.update_layout(
            template=EMC_TEMPLATE,
            polar=dict(
                bgcolor="#07111f",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#1a3a5c", color="#8aa8c4"),
                angularaxis=dict(gridcolor="#1a3a5c", color="#8aa8c4"),
            ),
            title="Predicted EMC Profile (normalised to P90 AE bound)",
            height=380, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw prediction table ──────────────────────────────────────────────
    with st.expander("Raw Prediction Details"):
        rows = []
        for tgt, val in preds.items():
            info = m_info.get(tgt, {})
            rows.append({
                "Target": tgt,
                "Unit": TARGET_UNITS.get(tgt, ""),
                "Predicted Value": f"{val:.8f}",
                "Algorithm": info.get("algorithm", "—"),
                "R²": f"{info.get('R2', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 2 — RISK ANALYSIS
# ---------------------------------------------------------------------------
def page_risk_analysis():
    st.markdown("<div class='section-title'>🛡️ Configuration Risk Analysis</div>",
                unsafe_allow_html=True)
    st.markdown(SYNTH_BANNER, unsafe_allow_html=True)
    st.markdown("""<div class='banner-info'>
    Risk categories are <b>dataset-relative</b> (P33/P67 percentile boundaries of the
    training dataset). They are NOT certified EMC safety thresholds.
    </div>""", unsafe_allow_html=True)

    if not models_available():
        st.error("Model files not found. Run train_model.py first.")
        return

    st.markdown("<div class='emc-card'>", unsafe_allow_html=True)
    st.markdown("**Cable Configuration Parameters**")
    config = render_config_inputs(prefix="risk")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🛡️ Analyse Risk", key="btn_risk"):
        with st.spinner("Running Module 3 risk analysis …"):
            try:
                from risk_analysis import analyze_configuration
                result = analyze_configuration(config, config_id="USER_CONFIG")
                st.session_state["risk_result"] = result
                st.session_state["risk_config"] = copy.deepcopy(config)
            except Exception as e:
                st.error(f"Risk analysis failed: {e}")
                return

    if "risk_result" not in st.session_state:
        st.info("Set configuration above and click **Analyse Risk**.")
        return

    result = st.session_state["risk_result"]
    preds  = result.get("predictions", {})
    risk_overall = result.get("overall_risk_category", "UNDETERMINED RISK")
    per_tgt_risk = result.get("per_target_risk", {})
    per_tgt_ind  = result.get("per_target_indicators", {})
    geo_obs      = result.get("geometry_observations", [])
    reliability  = result.get("reliability", {})
    explanation  = result.get("explanation", "")

    # ── Overall risk banner ───────────────────────────────────────────────
    badge = risk_badge(risk_overall)
    clr   = RISK_COLORS.get(risk_overall, "#90a4ae")
    st.markdown(f"""<div class='emc-card-accent' style='border-color:{clr}'>
        <span style='font-size:1.1rem;color:#8aa8c4'>Overall Risk Category: </span>
        {badge}
        <span style='float:right;font-size:0.82rem;color:#8aa8c4'>
            Reliability: {reliability.get("overall_reliability","—")}
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Per-target risk table ─────────────────────────────────────────────
    st.markdown("<div class='section-title'>Per-Target Risk Classification</div>",
                unsafe_allow_html=True)

    risk_cols = st.columns(len(preds) if preds else 1)
    for col, tgt in zip(risk_cols, ALL_TARGETS):
        if tgt not in preds:
            continue
        val  = preds[tgt]
        cat  = per_tgt_risk.get(tgt, "UNDETERMINED RISK")
        ind  = per_tgt_ind.get(tgt)
        clr  = RISK_COLORS.get(cat, "#90a4ae")
        ind_pct = f"{ind*100:.1f}%" if ind is not None else "—"
        with col:
            st.markdown(f"""<div class='metric-tile' style='border-color:{clr}'>
                <div class='val' style='font-size:1.2rem;color:{clr}'>{val:.4f}</div>
                <div class='lbl'>{tgt}<br>{TARGET_UNITS.get(tgt,"")}</div>
                <div style='font-size:0.75rem;margin-top:6px;color:{clr}'>{cat}</div>
                <div style='font-size:0.72rem;color:#546e7a'>Indicator: {ind_pct}</div>
            </div>""", unsafe_allow_html=True)

    # ── Risk indicators gauge chart ───────────────────────────────────────
    valid_inds = {t: v for t, v in per_tgt_ind.items()
                  if v is not None and not np.isnan(v)}
    if valid_inds:
        fig = go.Figure()
        labels = [t.replace("_", "<br>") for t in valid_inds]
        values = list(valid_inds.values())
        bar_colors = ["#4caf50" if v < 0.33 else "#ff9800" if v < 0.67 else "#f44336"
                      for v in values]
        fig.add_trace(go.Bar(
            x=labels, y=values, marker_color=bar_colors,
            text=[f"{v:.2f}" for v in values], textposition="outside",
        ))
        fig.add_hline(y=0.33, line_dash="dot", line_color="#4caf50",
                      annotation_text="P33 boundary")
        fig.add_hline(y=0.67, line_dash="dot", line_color="#f44336",
                      annotation_text="P67 boundary")
        fig.update_layout(
            template=EMC_TEMPLATE,
            title="Normalised Risk Indicators (0 = dataset min, 1 = dataset max)",
            yaxis=dict(range=[0, 1.1], title="Indicator (0–1)"),
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Geometry observations ─────────────────────────────────────────────
    if geo_obs:
        st.markdown("<div class='section-title'>Geometry Observations</div>",
                    unsafe_allow_html=True)
        for obs in geo_obs:
            st.markdown(f"<div class='banner-info'>ℹ️ {obs}</div>",
                        unsafe_allow_html=True)

    # ── Engineering explanation ───────────────────────────────────────────
    with st.expander("Engineering Explanation (Module 3 Output)"):
        for line in explanation.split("\n"):
            if line.strip():
                st.markdown(f"<span style='font-size:0.88rem;color:#b0cfe0'>{line}</span>",
                            unsafe_allow_html=True)

    # ── Reliability warnings ──────────────────────────────────────────────
    rel_warns = reliability.get("warnings", []) + reliability.get("ood_features", [])
    if rel_warns:
        with st.expander("Reliability Warnings"):
            for w in rel_warns:
                st.markdown(f"<div class='banner-warn'>⚠️ {w}</div>",
                            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 3 — EXPLAINABLE AI
# ---------------------------------------------------------------------------
def page_xai():
    st.markdown("<div class='section-title'>🧠 Explainable AI & Prediction Confidence</div>",
                unsafe_allow_html=True)
    st.markdown(SYNTH_BANNER, unsafe_allow_html=True)

    # ── Feature importance (from saved CSV) ───────────────────────────────
    st.markdown("<div class='section-title'>Permutation Feature Importance</div>",
                unsafe_allow_html=True)

    fi_df = load_report_csv("feature_importance.csv")
    if fi_df.empty:
        st.warning("feature_importance.csv not found. Run xai_engine.py (Module 4) first.")
    else:
        targets_in_fi = fi_df["target"].unique().tolist()
        sel_tgt = st.selectbox("Select prediction target:", targets_in_fi, key="fi_tgt")
        sub = fi_df[fi_df["target"] == sel_tgt].sort_values("mean_importance",
                                                              ascending=True)
        fig = go.Figure(go.Bar(
            x=sub["mean_importance"],
            y=sub["feature"],
            orientation="h",
            error_x=dict(type="data", array=sub["std_importance"], visible=True,
                         color="#546e7a"),
            marker_color="#00d4ff",
            text=[f"{v:.4f}" for v in sub["mean_importance"]],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#546e7a")
        fig.update_layout(
            template=EMC_TEMPLATE,
            title=f"Permutation Feature Importance — {sel_tgt}<br>"
                  "<sup>Mean MAE increase when feature is shuffled. "
                  "Predictive dependence, NOT physical causation.</sup>",
            xaxis_title="Mean MAE increase when feature permuted",
            height=360, margin=dict(l=0, r=60, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # All-targets heatmap
        with st.expander("Feature Importance Heatmap (all targets)"):
            pivot = fi_df.pivot_table(index="feature", columns="target",
                                      values="mean_importance", aggfunc="mean")
            fig2 = px.imshow(pivot.round(4), color_continuous_scale="YlOrRd",
                             text_auto=".3f",
                             labels=dict(color="Mean MAE increase"))
            fig2.update_layout(template=EMC_TEMPLATE, height=350,
                               title="Feature Importance Heatmap")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Test-set error statistics ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Test-Set Error Statistics</div>",
                unsafe_allow_html=True)

    err_df = load_report_csv("prediction_error_analysis.csv")
    if err_df.empty:
        st.warning("prediction_error_analysis.csv not found. Run xai_engine.py first.")
    else:
        display_cols = [c for c in ["target", "algorithm", "MAE", "RMSE",
                                     "mean_residual", "median_AE",
                                     "p90_AE", "p95_AE"] if c in err_df.columns]
        st.dataframe(
            err_df[display_cols].style.format(
                {c: "{:.4f}" for c in display_cols if c not in ["target", "algorithm"]}
            ),
            use_container_width=True, hide_index=True,
        )

        # MAE / RMSE comparison chart
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="MAE",  x=err_df["target"], y=err_df["MAE"],
                               marker_color="#00d4ff"))
        fig3.add_trace(go.Bar(name="RMSE", x=err_df["target"], y=err_df["RMSE"],
                               marker_color="#ff9800"))
        fig3.update_layout(template=EMC_TEMPLATE, barmode="group",
                           title="Test-Set MAE and RMSE per Target",
                           yaxis_title="Error (target units)",
                           height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Conformal prediction intervals ────────────────────────────────────
    st.markdown("<div class='section-title'>Conformal Prediction Intervals (90% Nominal)</div>",
                unsafe_allow_html=True)

    calib = load_report_json("uncertainty_calibration.json")
    if not calib:
        st.warning("uncertainty_calibration.json not found. Run xai_engine.py first.")
    else:
        rows = []
        for tgt, cr in calib.items():
            if cr.get("status") == "CALIBRATED":
                emp = cr["coverage_empirical"]
                clr = "#4caf50" if 0.88 <= emp <= 0.98 else "#ff9800"
                rows.append({
                    "Target": tgt,
                    "Half-width (q̂)": cr["q_hat"],
                    "Nominal Coverage": f"{cr['coverage_nominal']:.0%}",
                    "Empirical Coverage": f"{emp:.2%}",
                    "Interval Width": cr["mean_interval_width"],
                    "Calibration Set (n)": cr["n_calib"],
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("""<div class='banner-info'>
            ℹ️ Split conformal prediction. Coverage is <b>marginal</b> (averaged over inputs),
            not conditional. Valid only under exchangeability with training distribution.
            </div>""", unsafe_allow_html=True)

    # ── Model disagreement for current config ─────────────────────────────
    st.markdown("<div class='section-title'>Intra-Ensemble Spread (Model Disagreement)</div>",
                unsafe_allow_html=True)

    st.markdown("<div class='emc-card'>", unsafe_allow_html=True)
    config_dis = render_config_inputs(prefix="dis", compact=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Analyse Prediction Spread", key="btn_spread"):
        if not models_available():
            st.error("Model files not found.")
        else:
            with st.spinner("Computing intra-ensemble spread …"):
                try:
                    from prediction_uncertainty import (
                        calculate_model_disagreement, check_prediction_domain,
                        _load_artefacts
                    )
                    preprocessor, surrogate_bundle, metadata, perf = _load_artefacts()
                    dis = calculate_model_disagreement(
                        config_dis, preprocessor, surrogate_bundle, perf
                    )
                    dom = check_prediction_domain(config_dis)
                    st.session_state["xai_dis"]  = dis
                    st.session_state["xai_dom"]  = dom
                    st.session_state["xai_cfg"]  = copy.deepcopy(config_dis)
                except Exception as e:
                    st.error(f"Spread analysis failed: {e}")

    if "xai_dis" in st.session_state:
        dis = st.session_state["xai_dis"]
        dom = st.session_state["xai_dom"]
        dom_status = dom.get("domain_status", "UNKNOWN")
        dom_clr = {"IN_DOMAIN": "#4caf50",
                   "UNUSUAL_COMBINATION": "#ff9800",
                   "OUT_OF_DOMAIN": "#f44336"}.get(dom_status, "#90a4ae")
        st.markdown(f"""<div class='banner-info' style='border-color:{dom_clr}'>
            Domain Status: <b style='color:{dom_clr}'>{dom_status}</b>
            {"&nbsp;|&nbsp; Mahalanobis: " + str(dom.get("mahal_distance","N/A"))
             if dom.get("mahal_distance") else ""}
        </div>""", unsafe_allow_html=True)

        dis_cols = st.columns(len(dis))
        for col, (tgt, d) in zip(dis_cols, dis.items()):
            mean_p = d.get("mean_prediction", 0)
            std_p  = d.get("std_intra_ensemble", 0)
            frac   = (std_p / mean_p * 100) if mean_p > 0 else 0
            spread_clr = "#f44336" if frac > 20 else "#ff9800" if frac > 10 else "#4caf50"
            with col:
                st.markdown(f"""<div class='metric-tile'>
                    <div class='val' style='font-size:1rem;color:#00d4ff'>
                        {mean_p:.4f}</div>
                    <div class='lbl'>{tgt}</div>
                    <div style='font-size:0.75rem;margin-top:4px;color:{spread_clr}'>
                        Spread: {frac:.1f}%</div>
                    <div style='font-size:0.7rem;color:#546e7a'>
                        std={std_p:.4f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='banner-warn'>
        ⚠️ Intra-ensemble spread is the std of 200 individual tree predictions.
        It is NOT a calibrated confidence interval. High spread indicates
        the ensemble trees disagree for this input — consider MATLAB verification.
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 4 — CABLE ROUTING OPTIMISATION
# ---------------------------------------------------------------------------
def page_optimisation():
    st.markdown("<div class='section-title'>🔧 Cable-Routing Optimisation Engine</div>",
                unsafe_allow_html=True)
    st.markdown(SYNTH_BANNER, unsafe_allow_html=True)
    st.markdown("""<div class='banner-info'>
    Grid-search optimisation. The engine explores Separation × Overlap combinations,
    predicts EMC outputs for each using Module 2, and ranks by lowest predicted
    <b>Victim_Voltage_V</b>. Module 3 risk analysis is applied to every feasible candidate.
    </div>""", unsafe_allow_html=True)

    if not models_available():
        st.error("Model files not found. Run train_model.py first.")
        return

    # ── Baseline + search grid inputs ─────────────────────────────────────
    col_base, col_grid = st.columns([1.2, 1])

    with col_base:
        st.markdown("<div class='emc-card'>", unsafe_allow_html=True)
        st.markdown("**Baseline Configuration**")
        base_config = render_config_inputs(prefix="opt")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_grid:
        st.markdown("<div class='emc-card'>", unsafe_allow_html=True)
        st.markdown("**Grid Search Parameters**")
        s_min  = st.number_input("Separation min (mm)", 5.0,  200.0, 10.0, 5.0, key="opt_smin")
        s_max  = st.number_input("Separation max (mm)", 5.0,  200.0, 80.0, 5.0, key="opt_smax")
        s_step = st.number_input("Separation step (mm)", 1.0, 50.0,  10.0, 1.0, key="opt_sstep")
        o_min  = st.number_input("Overlap min (mm)", 10.0, 500.0, 50.0, 10.0,  key="opt_omin")
        o_max  = st.number_input("Overlap max (mm)", 10.0, 500.0, 450.0, 10.0, key="opt_omax")
        o_step = st.number_input("Overlap step (mm)", 10.0, 100.0, 50.0, 10.0, key="opt_ostep")
        top_n  = st.slider("Top N to display", 3, 10, 5, key="opt_topn")
        n_sep  = max(1, int((s_max - s_min) / s_step) + 1)
        n_ovl  = max(1, int((o_max - o_min) / o_step) + 1)
        n_cand = n_sep * n_ovl
        st.markdown(f"""<div class='banner-info'>
            Grid: {n_sep} sep × {n_ovl} ovl = <b>{n_cand} candidates</b>
            (+ baseline). Est. time: {n_cand * 3 // 2} – {n_cand * 3} sec
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔧 Run Optimisation", key="btn_opt"):
        if s_max <= s_min:
            st.error("Separation max must be greater than min.")
            return
        if o_max <= o_min:
            st.error("Overlap max must be greater than min.")
            return
        with st.spinner(f"Running grid search ({n_cand} candidates + Module 3 risk) …"):
            try:
                from optimise_routing import optimise_cable_routing
                result = optimise_cable_routing(
                    initial_configuration = base_config,
                    sep_min  = float(s_min),  sep_max  = float(s_max),
                    sep_step = float(s_step),
                    ovl_min  = float(o_min),  ovl_max  = float(o_max),
                    ovl_step = float(o_step),
                    top_n    = int(top_n),
                    verbose  = False,
                )
                st.session_state["opt_result"] = result
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                return

    if "opt_result" not in st.session_state:
        st.info("Configure the search grid and click **Run Optimisation**.")
        return

    res = st.session_state["opt_result"]
    pri = res["primary_objective"]
    pred_col = f"Pred_{pri}"
    base_val = res["baseline_value"]
    best_val = res["best_value"]
    rel_impr = res["relative_improvement_pct"]

    # ── Summary banner ────────────────────────────────────────────────────
    impr_clr = "#4caf50" if rel_impr > 0 else "#f44336"
    st.markdown(f"""<div class='emc-card-accent'>
        <b style='color:#00d4ff;font-size:1.1rem'>Optimisation Complete</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Candidates: {res['n_candidates']}
        &nbsp;|&nbsp; Feasible: {res['n_feasible']}
        &nbsp;|&nbsp; Infeasible: {res['n_infeasible']}
        &nbsp;|&nbsp; OOD: {res['n_ood']}
        <br><br>
        <b>Primary objective:</b> {pri}
        &nbsp;&nbsp;
        <b>Baseline:</b> {base_val:.4f}
        &nbsp;&nbsp;
        <b>Best predicted:</b> {best_val:.4f}
        &nbsp;&nbsp;
        <b style='color:{impr_clr}'>Improvement: {rel_impr:+.2f}%</b>
    </div>""", unsafe_allow_html=True)

    # ── Baseline vs recommended cards ─────────────────────────────────────
    col_b, col_r = st.columns(2)
    rec = res["recommended"]
    with col_b:
        st.markdown(f"""<div class='emc-card'>
            <b style='color:#8aa8c4'>Baseline</b><br>
            Separation: <b>{base_config['Separation_mm']:.1f} mm</b> &nbsp;
            Overlap: <b>{base_config['Overlap_mm']:.1f} mm</b><br>
            {pri}: <b style='color:#ff9800'>{base_val:.4f}</b><br>
            Risk: {risk_badge(res['risk_baseline'])}
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown(f"""<div class='emc-card' style='border-color:#4caf50'>
            <b style='color:#4caf50'>Recommended — {rec['Candidate_ID']}</b><br>
            Separation: <b>{rec['Separation_mm']:.1f} mm</b> &nbsp;
            Overlap: <b>{rec['Overlap_mm']:.1f} mm</b><br>
            {pri}: <b style='color:#4caf50'>{best_val:.4f}</b><br>
            Risk: {risk_badge(res['risk_recommended'])}
        </div>""", unsafe_allow_html=True)

    # ── Engineering explanation ───────────────────────────────────────────
    with st.expander("Engineering Explanation"):
        for line in res["explanation"].split("\n"):
            if line.strip():
                st.markdown(f"<span style='font-size:0.88rem;color:#b0cfe0'>{line}</span>",
                            unsafe_allow_html=True)

    # ── Top-N table ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Top Feasible Candidates</div>",
                unsafe_allow_html=True)

    top_df = res["top_n_df"].copy()
    disp_cols = ["Rank", "Candidate_ID", "Separation_mm", "Overlap_mm",
                 pred_col, "Relative_Improvement_pct",
                 "Risk_Category", "Risk_Reliability", "Feasibility_Status"]
    disp_cols = [c for c in disp_cols if c in top_df.columns]
    st.dataframe(
        top_df[disp_cols].style.format({
            pred_col: "{:.4f}",
            "Relative_Improvement_pct": "{:.2f}%",
            "Rank": "{:.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────
    feasible_df = res["feasible_df"].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 vs Separation", "📉 vs Overlap", "🗺️ Grid Heatmap", "🏆 Ranking"
    ])

    with tab1:
        sep_group = (feasible_df.groupby("Separation_mm")[pred_col]
                     .mean().reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sep_group["Separation_mm"], y=sep_group[pred_col],
            mode="lines+markers", name="Mean predicted",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=7),
        ))
        fig.add_hline(y=base_val, line_dash="dash", line_color="#ff9800",
                      annotation_text=f"Baseline {base_val:.2f}")
        fig.update_layout(
            template=EMC_TEMPLATE,
            title=f"Mean Predicted {pri} vs Separation (averaged over all overlaps)",
            xaxis_title="Separation_mm (mm)", yaxis_title=pri,
            height=340, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ovl_group = (feasible_df.groupby("Overlap_mm")[pred_col]
                     .mean().reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ovl_group["Overlap_mm"], y=ovl_group[pred_col],
            mode="lines+markers", name="Mean predicted",
            line=dict(color="#ff9800", width=2),
            marker=dict(size=7, symbol="square"),
        ))
        fig.add_hline(y=base_val, line_dash="dash", line_color="#f44336",
                      annotation_text=f"Baseline {base_val:.2f}")
        fig.update_layout(
            template=EMC_TEMPLATE,
            title=f"Mean Predicted {pri} vs Overlap (averaged over all separations)",
            xaxis_title="Overlap_mm (mm)", yaxis_title=pri,
            height=340, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        try:
            pivot = feasible_df.pivot_table(
                index="Separation_mm", columns="Overlap_mm",
                values=pred_col, aggfunc="mean",
            )
            annotate = pivot.size <= 200
            fig = px.imshow(
                pivot.round(2),
                color_continuous_scale="YlOrRd",
                labels=dict(x="Overlap_mm (mm)", y="Separation_mm (mm)",
                            color=pri),
                text_auto=".2f" if annotate else False,
            )
            fig.update_layout(
                template=EMC_TEMPLATE,
                title=f"Grid Search Heatmap — Predicted {pri}",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")

    with tab4:
        palette = {
            "LOWER PREDICTED RISK":        "#4caf50",
            "INTERMEDIATE PREDICTED RISK": "#ff9800",
            "HIGHER PREDICTED RISK":       "#f44336",
            "UNDETERMINED RISK":           "#90a4ae",
        }
        top_chart = res["top_n_df"].copy()
        top_labels = [
            f"{row['Candidate_ID']}<br>Sep={row['Separation_mm']:.0f}mm "
            f"Ovl={row['Overlap_mm']:.0f}mm"
            for _, row in top_chart.iterrows()
        ]
        bar_colors = [palette.get(r, "#90a4ae")
                      for r in top_chart["Risk_Category"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_labels, y=top_chart[pred_col],
            marker_color=bar_colors, name=pri,
            text=[f"{v:.3f}<br>{i:+.1f}%"
                  for v, i in zip(top_chart[pred_col],
                                  top_chart["Relative_Improvement_pct"])],
            textposition="outside",
        ))
        fig.add_hline(y=base_val, line_dash="dash", line_color="#ff9800",
                      annotation_text=f"Baseline {base_val:.2f}")
        fig.update_layout(
            template=EMC_TEMPLATE,
            title=f"Top-{len(top_chart)} Candidates — {pri}",
            yaxis_title=pri, height=380,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────
    opt_csv = REPORTS / "optimisation_results.csv"
    opt_json = REPORTS / "recommended_configuration.json"
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        if opt_csv.exists():
            st.download_button(
                "⬇️ Download Optimisation CSV",
                data=opt_csv.read_bytes(),
                file_name="optimisation_results.csv",
                mime="text/csv",
            )
    with dl_col2:
        if opt_json.exists():
            st.download_button(
                "⬇️ Download Recommended Config JSON",
                data=opt_json.read_bytes(),
                file_name="recommended_configuration.json",
                mime="application/json",
            )


# ---------------------------------------------------------------------------
# PAGE 5 — MODEL PERFORMANCE
# ---------------------------------------------------------------------------
def page_model_performance():
    st.markdown("<div class='section-title'>📊 Model Performance & Evaluation</div>",
                unsafe_allow_html=True)
    st.markdown(SYNTH_BANNER, unsafe_allow_html=True)

    perf = load_report_json("model_performance.json")
    if not perf:
        st.error("model_performance.json not found. Run train_model.py first.")
        return

    # ── Configuration info ────────────────────────────────────────────────
    cfg = perf.get("training_config", {})
    st.markdown(f"""<div class='emc-card'>
        <b>Evaluation date:</b> {perf.get("evaluation_date","—")} &nbsp;|&nbsp;
        <b>Dataset:</b> {perf.get("dataset_type","—")} &nbsp;|&nbsp;
        <b>Records:</b> {perf.get("dataset_size","—")}
        (Train: {perf.get("training_records","—")},
         Test: {perf.get("testing_records","—")}) &nbsp;|&nbsp;
        <b>Test split:</b> {cfg.get("test_size",0.2)*100:.0f}%
    </div>""", unsafe_allow_html=True)

    # ── Per-target metrics tabs ───────────────────────────────────────────
    targets = list(perf.get("targets", {}).keys())
    tabs = st.tabs(targets)

    for tab, tgt in zip(tabs, targets):
        with tab:
            tdata = perf["targets"][tgt]
            best  = tdata.get("best_algorithm", "—")
            algos = ["Random Forest", "Extra Trees"]
            algo_cols = st.columns(2)
            for col, algo in zip(algo_cols, algos):
                ad = tdata.get(algo, {})
                sel = algo == best
                clr = "#00d4ff" if sel else "#546e7a"
                lbl = " ✦ SELECTED" if sel else ""
                with col:
                    st.markdown(f"""<div class='emc-card' style='border-color:{clr}'>
                        <b style='color:{clr}'>{algo}{lbl}</b><br>
                        MAE:  <b>{ad.get('MAE', '—'):.4f}</b> &nbsp;|&nbsp;
                        RMSE: <b>{ad.get('RMSE','—'):.4f}</b> &nbsp;|&nbsp;
                        R²:   <b>{ad.get('R2',  '—'):.4f}</b>
                    </div>""", unsafe_allow_html=True)

            # RF vs ET comparison bar
            rf = tdata.get("Random Forest", {})
            et = tdata.get("Extra Trees", {})
            metrics = ["MAE", "RMSE", "R2"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Random Forest",
                                 x=metrics,
                                 y=[rf.get(m, 0) for m in metrics],
                                 marker_color="#00d4ff"))
            fig.add_trace(go.Bar(name="Extra Trees",
                                 x=metrics,
                                 y=[et.get(m, 0) for m in metrics],
                                 marker_color="#4caf50"))
            fig.update_layout(
                template=EMC_TEMPLATE, barmode="group",
                title=f"{tgt} — RF vs ET Comparison",
                height=300, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Overall comparison charts ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Algorithm Comparison Across All Targets</div>",
                unsafe_allow_html=True)

    rows_rf, rows_et = [], []
    for tgt, td in perf.get("targets", {}).items():
        rf = td.get("Random Forest", {})
        et = td.get("Extra Trees", {})
        rows_rf.append({"target": tgt, "MAE": rf.get("MAE"), "RMSE": rf.get("RMSE"),
                        "R2": rf.get("R2"), "algorithm": "Random Forest"})
        rows_et.append({"target": tgt, "MAE": et.get("MAE"), "RMSE": et.get("RMSE"),
                        "R2": et.get("R2"), "algorithm": "Extra Trees"})
    comp_df = pd.DataFrame(rows_rf + rows_et)

    if not comp_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(comp_df, x="target", y="R2", color="algorithm",
                         barmode="group", color_discrete_map={
                             "Random Forest": "#00d4ff", "Extra Trees": "#4caf50"},
                         title="R² Score Comparison")
            fig.add_hline(y=0.70, line_dash="dot", line_color="#ff9800",
                          annotation_text="R²=0.70")
            fig.update_layout(template=EMC_TEMPLATE, height=340,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(comp_df, x="target", y="MAE", color="algorithm",
                         barmode="group", color_discrete_map={
                             "Random Forest": "#00d4ff", "Extra Trees": "#4caf50"},
                         title="MAE Comparison")
            fig.update_layout(template=EMC_TEMPLATE, height=340,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Pre-generated evaluation plots ────────────────────────────────────
    st.markdown("<div class='section-title'>Pre-Generated Evaluation Plots</div>",
                unsafe_allow_html=True)
    vis_dir = AI_ML_DIR / "reports" / "visualisations"
    img_files = sorted(vis_dir.glob("*.png"))
    if img_files:
        cols = st.columns(3)
        for i, img in enumerate(img_files[:9]):
            with cols[i % 3]:
                st.image(str(img), caption=img.stem, use_container_width=True)
    else:
        st.info("No pre-generated PNG files found in reports/visualisations/")


# ---------------------------------------------------------------------------
# PAGE 6 — DOWNLOAD REPORTS
# ---------------------------------------------------------------------------
def page_downloads():
    st.markdown("<div class='section-title'>📥 Download Reports</div>",
                unsafe_allow_html=True)

    files = [
        ("Prediction Results CSV",           "prediction_results.csv",         "text/csv"),
        ("Feature Importance CSV",            "feature_importance.csv",          "text/csv"),
        ("Prediction Error Analysis CSV",     "prediction_error_analysis.csv",   "text/csv"),
        ("Risk Analysis Results CSV",         "risk_analysis_results.csv",       "text/csv"),
        ("Optimisation Results CSV",          "optimisation_results.csv",        "text/csv"),
        ("Recommended Configuration JSON",    "recommended_configuration.json",  "application/json"),
        ("Model Performance JSON",            "model_performance.json",          "application/json"),
        ("Risk Thresholds JSON",              "risk_thresholds.json",            "application/json"),
        ("Uncertainty Calibration JSON",      "uncertainty_calibration.json",    "application/json"),
        ("Prediction Reliability Report TXT", "prediction_reliability_report.txt","text/plain"),
        ("Optimisation Summary TXT",          "optimisation_summary.txt",        "text/plain"),
        ("Risk Assessment Report TXT",        "risk_assessment_report.txt",      "text/plain"),
        ("Model Evaluation Report TXT",       "model_evaluation_report.txt",     "text/plain"),
    ]

    cols = st.columns(3)
    for i, (label, filename, mime) in enumerate(files):
        fpath = REPORTS / filename
        with cols[i % 3]:
            if fpath.exists():
                st.download_button(
                    f"⬇️ {label}",
                    data=fpath.read_bytes(),
                    file_name=filename,
                    mime=mime,
                    use_container_width=True,
                )
            else:
                st.markdown(f"""<div class='emc-card' style='opacity:0.5;text-align:center'>
                    ⚠️ {label}<br>
                    <span style='font-size:0.75rem;color:#546e7a'>{filename} not found</span>
                </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("""<div style='text-align:center;padding:14px 0 8px'>
            <span style='font-size:2rem'>⚡</span><br>
            <b style='font-size:1.15rem;color:#00d4ff'>EMC-GUARD</b><br>
            <span style='font-size:0.75rem;color:#546e7a'>Digital Twin Dashboard</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("<hr class='emc'>", unsafe_allow_html=True)

        pages = {
            "🏠  Overview":              "overview",
            "🔬  Digital Twin Prediction":"prediction",
            "🛡️  Risk Analysis":          "risk",
            "🧠  Explainable AI":         "xai",
            "🔧  Route Optimisation":     "optimisation",
            "📊  Model Performance":      "performance",
            "📥  Download Reports":       "downloads",
        }

        if "page" not in st.session_state:
            st.session_state["page"] = "overview"

        for label, key in pages.items():
            selected = st.session_state["page"] == key
            bg = "#1a3a5c" if selected else "transparent"
            clr = "#00d4ff" if selected else "#8aa8c4"
            if st.sidebar.button(label, key=f"nav_{key}",
                                 use_container_width=True):
                st.session_state["page"] = key
                st.rerun()

        st.markdown("<hr class='emc'>", unsafe_allow_html=True)

        # Module status mini-indicators
        checks = [
            ("M2 Models",   (MODELS / "emc_surrogate_model.pkl").exists()),
            ("M3 Thresholds",(REPORTS / "risk_thresholds.json").exists()),
            ("M4 XAI",       (REPORTS / "feature_importance.csv").exists()),
            ("M5 Optimiser", (REPORTS / "optimisation_results.csv").exists()),
        ]
        for name, ok in checks:
            clr = "#4caf50" if ok else "#f44336"
            dot = "●" if ok else "○"
            st.markdown(
                f"<span style='color:{clr};font-size:0.8rem'>{dot} {name}</span><br>",
                unsafe_allow_html=True
            )

        st.markdown("<hr class='emc'>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:0.7rem;color:#2a4a6a;text-align:center'>
            SYNTHETIC DATA PROTOTYPE<br>Not for engineering sign-off
        </div>""", unsafe_allow_html=True)

    return st.session_state["page"]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    page = render_sidebar()

    page_map = {
        "overview":     page_overview,
        "prediction":   page_prediction,
        "risk":         page_risk_analysis,
        "xai":          page_xai,
        "optimisation": page_optimisation,
        "performance":  page_model_performance,
        "downloads":    page_downloads,
    }

    fn = page_map.get(page, page_overview)
    fn()


if __name__ == "__main__":
    main()

