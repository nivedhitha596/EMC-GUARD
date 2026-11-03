"""
supplementary_data.py
---------------------
Safe loading, validation, and engineering analysis helpers for the
three supplementary simulation datasets in KONE EMC-GUARD:

  1. emc_predict_SIMULATED.csv (105 rows x 21 cols): Simulated electrical interference
  2. rg_sweep_data_with_noise.csv (3 rows x 18 cols): Exploratory LTspice MOSFET Rg sweep
  3. rg_sweep_template.csv (3 rows x 20 cols): Incomplete LTspice simulation template

IMPORTANT SCIENTIFIC INTEGRITY RULES:
-------------------------------------
- Supplementary datasets are for VISUALISATION and ENGINEERING ANALYSIS ONLY.
- They are NOT merged with the 216-row MATLAB analytical dataset.
- They are NOT used for model training, validation metrics, or predictions.
- Missing values in templates are preserved as NaN; never imputed as zero or fabricated.
- No regulatory compliance, safety certification, or laboratory validation is claimed.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_AIML_DIR = _SRC_DIR.parent
_REPO_ROOT = _AIML_DIR.parent

SUPPLEMENTARY_DIR = _AIML_DIR / "dataset" / "supplementary"

EMC_SIMULATED_CSV = SUPPLEMENTARY_DIR / "emc_predict_SIMULATED.csv"
RG_SWEEP_CSV      = SUPPLEMENTARY_DIR / "rg_sweep_data_with_noise.csv"
RG_TEMPLATE_CSV   = SUPPLEMENTARY_DIR / "rg_sweep_template.csv"

# ---------------------------------------------------------------------------
# Expected schemas for validation
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS_EMC_SIMULATED = [
    "run_id", "date", "freq_hz", "duty_pct", "supply_v", "load_ohm",
    "spacing_cm", "orientation", "shield_config", "ground_config",
    "repeat_no", "baseline_vpp_mv", "victim_vpp_mv", "victim_vrms_mv",
    "fund_freq_hz", "fund_mag_dbv", "h2_mag_dbv", "h3_mag_dbv",
    "screenshot_file", "notes", "data_source"
]

EXPECTED_COLUMNS_RG_SWEEP = [
    "Run_ID", "Vbus_V", "Pulse_Vhigh_V", "Frequency_kHz", "Duty_percent",
    "L_uH", "R_load_ohm", "MOSFET", "Rg_ohm", "Rise_time_ns", "Vds_peak_V",
    "Overshoot_V", "dv_dt_V_per_ns", "Turn_off_loss_uJ", "Noise_peak_20kHz_dB",
    "Noise_peak_harmonic_dB", "Parasitics", "Data_source"
]

EXPECTED_COLUMNS_RG_TEMPLATE = [
    "run_id", "data_source", "mosfet", "vbus_V", "pwm_freq_kHz",
    "duty_percent", "load_L_uH", "load_R_ohm", "parasitics_added",
    "gate_Rg_ohm", "rise_time_ns", "fall_time_ns", "vds_peak_V",
    "vds_overshoot_V", "dvdt_V_per_ns", "ringing_freq_MHz",
    "switching_loss_uJ", "noise_dBV_at_1MHz", "noise_dBV_at_10MHz",
    "noise_dBV_at_30MHz"
]

TEMPLATE_INPUT_COLUMNS = [
    "run_id", "data_source", "mosfet", "vbus_V", "pwm_freq_kHz",
    "duty_percent", "load_L_uH", "load_R_ohm", "parasitics_added", "gate_Rg_ohm"
]

TEMPLATE_OUTPUT_COLUMNS = [
    "rise_time_ns", "fall_time_ns", "vds_peak_V", "vds_overshoot_V",
    "dvdt_V_per_ns", "ringing_freq_MHz", "switching_loss_uJ",
    "noise_dBV_at_1MHz", "noise_dBV_at_10MHz", "noise_dBV_at_30MHz"
]


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_emc_simulated_dataset(csv_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Safely load and validate emc_predict_SIMULATED.csv.

    Returns
    -------
    (df, info_dict)
    """
    target_path = Path(csv_path) if csv_path else EMC_SIMULATED_CSV
    info: Dict[str, Any] = {
        "dataset_name": "emc_predict_SIMULATED.csv",
        "path": str(target_path),
        "exists": target_path.exists(),
        "status": "UNKNOWN",
        "rows": 0,
        "cols": 0,
        "missing_cols": [],
        "extra_cols": [],
        "null_counts": {},
        "description": "Simulated electrical switching interference dataset across frequencies, spacing, and shielding.",
        "scientific_notice": "Contains simulated electrical interference data, NOT laboratory measurements. For exploratory engineering analysis only.",
    }

    if not target_path.exists():
        info["status"] = "FILE_NOT_FOUND"
        logger.warning("Supplementary dataset not found: %s", target_path)
        return None, info

    try:
        df = pd.read_csv(target_path)
        info["rows"] = len(df)
        info["cols"] = len(df.columns)

        missing = [c for c in EXPECTED_COLUMNS_EMC_SIMULATED if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS_EMC_SIMULATED]
        info["missing_cols"] = missing
        info["extra_cols"] = extra
        info["null_counts"] = df.isnull().sum().to_dict()

        if missing:
            info["status"] = "SCHEMA_WARNING"
            logger.warning("Dataset %s missing expected columns: %s", target_path.name, missing)
        else:
            info["status"] = "VALIDATED"

        return df, info

    except Exception as exc:
        info["status"] = f"ERROR: {exc}"
        logger.error("Failed to load %s: %s", target_path, exc)
        return None, info


def load_rg_sweep_dataset(csv_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Safely load and validate rg_sweep_data_with_noise.csv.

    Returns
    -------
    (df, info_dict)
    """
    target_path = Path(csv_path) if csv_path else RG_SWEEP_CSV
    info: Dict[str, Any] = {
        "dataset_name": "rg_sweep_data_with_noise.csv",
        "path": str(target_path),
        "exists": target_path.exists(),
        "status": "UNKNOWN",
        "rows": 0,
        "cols": 0,
        "missing_cols": [],
        "extra_cols": [],
        "description": "Small exploratory LTspice simulation sweep comparing 3 gate resistance levels (10, 47, 100 ohms).",
        "scientific_notice": "Contains only 3 simulation records. Results illustrate qualitative trade-offs; conclusions are preliminary.",
    }

    if not target_path.exists():
        info["status"] = "FILE_NOT_FOUND"
        logger.warning("Supplementary dataset not found: %s", target_path)
        return None, info

    try:
        df = pd.read_csv(target_path)
        info["rows"] = len(df)
        info["cols"] = len(df.columns)

        missing = [c for c in EXPECTED_COLUMNS_RG_SWEEP if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS_RG_SWEEP]
        info["missing_cols"] = missing
        info["extra_cols"] = extra
        info["null_counts"] = df.isnull().sum().to_dict()

        if missing:
            info["status"] = "SCHEMA_WARNING"
            logger.warning("Dataset %s missing expected columns: %s", target_path.name, missing)
        else:
            info["status"] = "VALIDATED"

        return df, info

    except Exception as exc:
        info["status"] = f"ERROR: {exc}"
        logger.error("Failed to load %s: %s", target_path, exc)
        return None, info


def load_rg_template_dataset(csv_path: Optional[Path] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Safely load and validate rg_sweep_template.csv.
    Explicitly tracks unpopulated simulation outputs without zero-filling.

    Returns
    -------
    (df, info_dict)
    """
    target_path = Path(csv_path) if csv_path else RG_TEMPLATE_CSV
    info: Dict[str, Any] = {
        "dataset_name": "rg_sweep_template.csv",
        "path": str(target_path),
        "exists": target_path.exists(),
        "status": "UNKNOWN",
        "rows": 0,
        "cols": 0,
        "input_columns": TEMPLATE_INPUT_COLUMNS,
        "output_columns": TEMPLATE_OUTPUT_COLUMNS,
        "total_outputs_missing": 0,
        "description": "Incomplete LTspice simulation template awaiting measurement results.",
        "scientific_notice": "Incomplete template. 10 output measurement fields are unpopulated (NaN). Awaiting LTspice runs.",
    }

    if not target_path.exists():
        info["status"] = "FILE_NOT_FOUND"
        logger.warning("Supplementary dataset not found: %s", target_path)
        return None, info

    try:
        df = pd.read_csv(target_path)
        info["rows"] = len(df)
        info["cols"] = len(df.columns)

        # Count missing outputs specifically
        missing_outputs = 0
        for out_col in TEMPLATE_OUTPUT_COLUMNS:
            if out_col in df.columns:
                missing_outputs += int(df[out_col].isnull().sum())
            else:
                missing_outputs += len(df)

        info["total_outputs_missing"] = missing_outputs
        info["total_outputs_expected"] = len(TEMPLATE_OUTPUT_COLUMNS) * len(df)
        info["percent_outputs_missing"] = (
            (missing_outputs / info["total_outputs_expected"]) * 100
            if info["total_outputs_expected"] > 0 else 0
        )
        info["status"] = "AWAITING_SIMULATION_RESULTS"

        return df, info

    except Exception as exc:
        info["status"] = f"ERROR: {exc}"
        logger.error("Failed to load %s: %s", target_path, exc)
        return None, info


def load_all_supplementary_datasets(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience loader that verifies and loads all three supplementary datasets.
    """
    supp_dir = Path(base_dir) if base_dir else SUPPLEMENTARY_DIR
    df_sim, info_sim = load_emc_simulated_dataset(supp_dir / "emc_predict_SIMULATED.csv")
    df_rg, info_rg   = load_rg_sweep_dataset(supp_dir / "rg_sweep_data_with_noise.csv")
    df_tpl, info_tpl = load_rg_template_dataset(supp_dir / "rg_sweep_template.csv")

    return {
        "emc_simulated": {"df": df_sim, "info": info_sim},
        "rg_sweep":      {"df": df_rg,  "info": info_rg},
        "rg_template":   {"df": df_tpl, "info": info_tpl},
        "all_files_exist": bool(info_sim["exists"] and info_rg["exists"] and info_tpl["exists"]),
    }


# ---------------------------------------------------------------------------

def _render_html(html_str: str) -> None:
    """Safely render HTML/CSS bypassing CommonMark 4-space indent code block parsing."""
    import streamlit as st
    clean_html = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


# Streamlit UI Rendering Section
# ---------------------------------------------------------------------------

def render_supplementary_analysis_section() -> None:
    """
    Renders the complete 'Supplementary Simulation Analysis' tab in Streamlit.
    Contains three sub-tabs:
      - Tab 1: Electrical Interference Analysis (105 Runs)
      - Tab 2: MOSFET Switching Sweep Analysis (Rg Sweep)
      - Tab 3: Simulation Template (Unfilled LTspice Template)
    """
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _render_html("""
    <div class="notice">
    <b>Supplementary Simulation Analysis Section</b><br>
    This section provides exploratory engineering visualisations for three supplementary simulation datasets.
    These datasets are auxiliary analytical resources; they are <b>strictly decoupled</b> from the primary
    MATLAB analytical prediction engine and do NOT establish certified regulatory EMC compliance or laboratory validation.
    </div>
    """)

    data_bundle = load_all_supplementary_datasets()

    tab_interf, tab_mosfet, tab_templ = st.tabs([
        "Tab 1 – Electrical Interference Analysis (105 Runs)",
        "Tab 2 – MOSFET Switching Sweep Analysis (Rg Sweep)",
        "Tab 3 – Simulation Template (Awaiting Runs)"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 1: Electrical Interference Analysis
    # ═══════════════════════════════════════════════════════════════════════
    with tab_interf:
        st.subheader("Simulated Electrical Interference & Victim Coupling")
        sim_data = data_bundle["emc_simulated"]
        df_sim = sim_data["df"]
        info_sim = sim_data["info"]

        if df_sim is None or df_sim.empty:
            st.error(f"Cannot load `emc_predict_SIMULATED.csv`: {info_sim.get('status')}")
        else:
            # Scientific notice
            _render_html(f"""
            <div class="warning-box">
            ⚠ <b>Scientific Disclaimer:</b> {info_sim['scientific_notice']}
            Data source: <code>{df_sim['data_source'].iloc[0] if 'data_source' in df_sim.columns else 'SIMULATED'}</code>
            · Dimensions: <b>{info_sim['rows']} rows × {info_sim['cols']} columns</b>.
            </div>
            """)

            # High-level summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Total Simulation Runs</div>
                    <div class="value">{len(df_sim)}</div>
                    <div class="unit">records</div>
                </div>""")
            with m2:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Switching Frequencies</div>
                    <div class="value">{len(df_sim['freq_hz'].unique())}</div>
                    <div class="unit">{sorted(list(df_sim['freq_hz'].unique()))} Hz</div>
                </div>""")
            with m3:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Victim Vpp Range</div>
                    <div class="value">{df_sim['victim_vpp_mv'].min():.1f} – {df_sim['victim_vpp_mv'].max():.1f}</div>
                    <div class="unit">mV (peak-to-peak)</div>
                </div>""")
            with m4:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Victim Vrms Range</div>
                    <div class="value">{df_sim['victim_vrms_mv'].min():.1f} – {df_sim['victim_vrms_mv'].max():.1f}</div>
                    <div class="unit">mV (RMS)</div>
                </div>""")

            _render_html("<div class='section-divider'></div>")

            # Interactive Operating Condition Filters
            _render_html("#### Operating Condition Filters")
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                freq_opts = sorted(list(df_sim["freq_hz"].unique()))
                sel_freq = st.multiselect("Frequency (Hz)", freq_opts, default=freq_opts, key="supp_freq")
            with fc2:
                spacing_opts = sorted(list(df_sim["spacing_cm"].unique()))
                sel_spacing = st.multiselect("Spacing (cm)", spacing_opts, default=spacing_opts, key="supp_spacing")
            with fc3:
                orient_opts = list(df_sim["orientation"].dropna().unique())
                sel_orient = st.multiselect("Orientation", orient_opts, default=orient_opts, key="supp_orient")
            with fc4:
                gnd_opts = list(df_sim["ground_config"].dropna().unique())
                sel_gnd = st.multiselect("Ground Config", gnd_opts, default=gnd_opts, key="supp_gnd")

            # Filter data safely
            filtered_df = df_sim[
                (df_sim["freq_hz"].isin(sel_freq)) &
                (df_sim["spacing_cm"].isin(sel_spacing)) &
                (df_sim["orientation"].isin(sel_orient)) &
                (df_sim["ground_config"].isin(sel_gnd))
            ]

            st.caption(f"Showing **{len(filtered_df)} of {len(df_sim)}** simulation runs matching selected conditions.")

            if not filtered_df.empty:
                # Plots row 1: Spacing decay & Harmonic magnitudes
                pcol1, pcol2 = st.columns(2)

                with pcol1:
                    fig_decay = px.box(
                        filtered_df,
                        x="spacing_cm",
                        y="victim_vpp_mv",
                        color="orientation",
                        title="<b>Victim Vpp Attenuation vs Cable Spacing</b>",
                        labels={"spacing_cm": "Cable Spacing (cm)", "victim_vpp_mv": "Victim Vpp (mV)"},
                        color_discrete_sequence=["#00D4FF", "#F59E0B", "#10B981"],
                    )
                    fig_decay.update_layout(
                        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                        font=dict(color="#CBD5E1"), height=380,
                        yaxis=dict(gridcolor="#334155"),
                        xaxis=dict(gridcolor="#334155"),
                    )
                    st.plotly_chart(fig_decay, use_container_width=True)

                with pcol2:
                    fig_harm = px.scatter(
                        filtered_df,
                        x="victim_vpp_mv",
                        y="fund_mag_dbv",
                        color="freq_hz",
                        size="victim_vrms_mv",
                        hover_data=["shield_config", "ground_config", "spacing_cm"],
                        title="<b>Fundamental Magnitude (dBV) vs Victim Vpp</b>",
                        labels={"victim_vpp_mv": "Victim Vpp (mV)", "fund_mag_dbv": "Fundamental Mag (dBV)"},
                        color_continuous_scale="Plasma",
                    )
                    fig_harm.update_layout(
                        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                        font=dict(color="#CBD5E1"), height=380,
                        yaxis=dict(gridcolor="#334155"),
                        xaxis=dict(gridcolor="#334155"),
                    )
                    st.plotly_chart(fig_harm, use_container_width=True)

                # Plots row 2: Harmonic comparison across shield configurations
                fig_bar = px.bar(
                    filtered_df.groupby(["shield_config", "ground_config"], as_index=False)[["victim_vpp_mv", "victim_vrms_mv"]].mean(),
                    x="shield_config",
                    y="victim_vpp_mv",
                    color="ground_config",
                    barmode="group",
                    title="<b>Mean Victim Vpp by Shield and Grounding Configuration</b>",
                    labels={"shield_config": "Shield Configuration", "victim_vpp_mv": "Mean Victim Vpp (mV)"},
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_bar.update_layout(
                    paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"), height=350,
                    yaxis=dict(gridcolor="#334155"),
                    xaxis=dict(gridcolor="#334155"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Filtered Data Table & Download
                st.markdown("#### Filtered Simulation Records")
                st.dataframe(
                    filtered_df[[
                        "run_id", "freq_hz", "spacing_cm", "orientation",
                        "shield_config", "ground_config", "victim_vpp_mv",
                        "victim_vrms_mv", "fund_mag_dbv", "h2_mag_dbv", "h3_mag_dbv"
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    label="⬇  Download Filtered Simulated Dataset (CSV)",
                    data=filtered_df.to_csv(index=False).encode("utf-8"),
                    file_name="emc_predict_SIMULATED_filtered.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No simulation runs match the selected filter combination.")

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 2: MOSFET Switching Sweep Analysis
    # ═══════════════════════════════════════════════════════════════════════
    with tab_mosfet:
        st.subheader("MOSFET Gate Resistance (Rg) Switching Dynamics Sweep")
        rg_data = data_bundle["rg_sweep"]
        df_rg = rg_data["df"]
        info_rg = rg_data["info"]

        if df_rg is None or df_rg.empty:
            st.error(f"Cannot load `rg_sweep_data_with_noise.csv`: {info_rg.get('status')}")
        else:
            # Scientific notice
            st.markdown(f"""
            <div class="warning-box">
            ⚠ <b>Exploratory Dataset Notice:</b> {info_rg['scientific_notice']}
            Target MOSFET: <b>IRFH5302</b> (N-Channel Power MOSFET) · Bus: <b>12 V</b> · PWM: <b>20 kHz</b>
            · Total populated runs: <b>3 records</b>.
            </div>
            """)

            # Three Gate Resistance Metric Columns
            _render_html("#### Three-Point Gate Resistance Sweep Comparison")
            c1, c2, c3 = st.columns(3)

            for col_widget, (_, r) in zip([c1, c2, c3], df_rg.iterrows()):
                rg_val = int(r["Rg_ohm"])
                with col_widget:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: left; padding: 16px;">
                        <div style="font-size: 1.25rem; font-weight: bold; color: #38BDF8; margin-bottom: 8px;">
                            Run {r['Run_ID']} — Rg = {rg_val} Ω
                        </div>
                        <table style="width: 100%; font-size: 0.88rem; color: #CBD5E1; border-collapse: collapse;">
                            <tr><td style="padding: 3px 0;"><b>Slew Rate (dV/dt):</b></td><td style="text-align: right; color: #F87171;"><b>{r['dv_dt_V_per_ns']:.3f} V/ns</b></td></tr>
                            <tr><td style="padding: 3px 0;"><b>Rise Time (tr):</b></td><td style="text-align: right;">{r['Rise_time_ns']:.1f} ns</td></tr>
                            <tr><td style="padding: 3px 0;"><b>Turn-off Loss:</b></td><td style="text-align: right; color: #FBBF24;"><b>{r['Turn_off_loss_uJ']:.3f} µJ</b></td></tr>
                            <tr><td style="padding: 3px 0;"><b>Voltage Overshoot:</b></td><td style="text-align: right;">{r['Overshoot_V']:.3f} V</td></tr>
                            <tr><td style="padding: 3px 0;"><b>Peak Vds:</b></td><td style="text-align: right;">{r['Vds_peak_V']:.3f} V</td></tr>
                            <tr><td style="padding: 3px 0;"><b>Noise Peak (20kHz):</b></td><td style="text-align: right;">{r['Noise_peak_20kHz_dB']:.1f} dB</td></tr>
                        </table>
                    </div>
                    """)

            _render_html("<div class='section-divider'></div>")

            # Dual-axis Trade-Off Plot: dV/dt (EMI proxy) vs Turn-off Loss (Thermal proxy)
            _render_html("#### Physical Trade-Off: Slew Rate (EMI Risk) vs Switching Dissipation (Thermal)")

            fig_tradeoff = make_subplots(specs=[[{"secondary_y": True}]])

            fig_tradeoff.add_trace(
                go.Scatter(
                    x=df_rg["Rg_ohm"],
                    y=df_rg["dv_dt_V_per_ns"],
                    mode="lines+markers+text",
                    name="Slew Rate dV/dt (V/ns) — EMI Proxy",
                    text=[f"{v:.3f} V/ns" for v in df_rg["dv_dt_V_per_ns"]],
                    textposition="top center",
                    line=dict(color="#EF4444", width=3),
                    marker=dict(size=10, color="#EF4444"),
                ),
                secondary_y=False,
            )

            fig_tradeoff.add_trace(
                go.Scatter(
                    x=df_rg["Rg_ohm"],
                    y=df_rg["Turn_off_loss_uJ"],
                    mode="lines+markers+text",
                    name="Turn-off Loss (µJ) — Thermal Proxy",
                    text=[f"{v:.3f} µJ" for v in df_rg["Turn_off_loss_uJ"]],
                    textposition="bottom center",
                    line=dict(color="#F59E0B", width=3, dash="dot"),
                    marker=dict(size=10, color="#F59E0B"),
                ),
                secondary_y=True,
            )

            fig_tradeoff.update_layout(
                title="<b>Switching Dynamics Trade-Off Across Gate Resistance (Rg)</b>",
                xaxis=dict(title="Gate Resistance Rg (Ω)", tickvals=[10, 47, 100], gridcolor="#334155"),
                paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                font=dict(color="#CBD5E1"), height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_tradeoff.update_yaxes(title_text="<b>dV/dt Slew Rate (V/ns)</b> (Lower = Better EMI)", secondary_y=False, gridcolor="#334155")
            fig_tradeoff.update_yaxes(title_text="<b>Turn-off Loss (µJ)</b> (Lower = Cooler)", secondary_y=True, gridcolor="#334155")

            st.plotly_chart(fig_tradeoff, use_container_width=True)

            # Engineering Interpretation Card
            st.markdown("""
            **Engineering Interpretation & Limitations:**
            - **Low Rg (10 Ω):** Yields rapid turn-on/turn-off ($t_r = 34.2\\text{ ns}$) and minimal turn-off dissipation ($0.172\\,\\mu\\text{J}$). However, the steep voltage slew rate ($dV/dt = 0.281\\text{ V/ns}$) induces substantial high-frequency displacement currents ($i = C \\frac{dv}{dt}$) into adjacent victim elevator cabling.
            - **High Rg (100 Ω):** Drastically curtails slew rate to $0.035\\text{ V/ns}$ (an **87.5% reduction**), significantly softening radiated and conducted emissions. Conversely, turn-off energy dissipation surges to $1.470\\,\\mu\\text{J}$ (**8.5× increase**), increasing thermal dissipation in the power stage.
            - **Moderate Rg (47 Ω):** Provides an intermediate operating compromise ($dV/dt = 0.073\\text{ V/ns}, E_{\\text{off}} = 0.705\\,\\mu\\text{J}$).
            - *Note:* Because these findings are drawn from exactly 3 discrete simulation runs, these observations represent qualitative directional trends rather than universal quantitative models.
            """)

            # Data Table & Export
            st.markdown("#### Complete MOSFET Sweep Dataset")
            st.dataframe(df_rg, use_container_width=True, hide_index=True)

            st.download_button(
                label="⬇  Download MOSFET Sweep Dataset (CSV)",
                data=df_rg.to_csv(index=False).encode("utf-8"),
                file_name="rg_sweep_data_with_noise.csv",
                mime="text/csv",
            )

    # ═══════════════════════════════════════════════════════════════════════
    #  TAB 3: Simulation Template
    # ═══════════════════════════════════════════════════════════════════════
    with tab_templ:
        st.subheader("LTspice Simulation Template (Awaiting Execution)")
        tpl_data = data_bundle["rg_template"]
        df_tpl = tpl_data["df"]
        info_tpl = tpl_data["info"]

        if df_tpl is None or df_tpl.empty:
            st.error(f"Cannot load `rg_sweep_template.csv`: {info_tpl.get('status')}")
        else:
            st.markdown(f"""
            <div class="warning-box">
            ⏳ <b>Status: AWAITING LTSPICE SIMULATION RUNS</b><br>
            {info_tpl['scientific_notice']}
            Missing output measurements are explicitly preserved as <code>NaN</code>.
            They are NOT zero-filled, imputed, or artificially synthesized.
            </div>
            """)

            # Metric overview
            tm1, tm2, tm3, tm4 = st.columns(4)
            with tm1:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Configured Runs</div>
                    <div class="value">{info_tpl['rows']}</div>
                    <div class="unit">runs planned</div>
                </div>""")
            with tm2:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Defined Inputs</div>
                    <div class="value">{len(info_tpl['input_columns'])}</div>
                    <div class="unit">columns ready</div>
                </div>""")
            with tm3:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Pending Outputs</div>
                    <div class="value">{len(info_tpl['output_columns'])}</div>
                    <div class="unit">columns awaiting</div>
                </div>""")
            with tm4:
                _render_html(f"""<div class="metric-card">
                    <div class="label">Missing Measurement Values</div>
                    <div class="value">{info_tpl['total_outputs_missing']} / {info_tpl['total_outputs_expected']}</div>
                    <div class="unit" style="color: #F87171;">{info_tpl['percent_outputs_missing']:.0f}% pending</div>
                </div>""")

            _render_html("<div class='section-divider'></div>")

            # Schema decomposition tables
            sc1, sc2 = st.columns(2)

            with sc1:
                st.markdown("#### Defined Input Parameters (Populated)")
                input_df = df_tpl[info_tpl["input_columns"]]
                st.dataframe(input_df, use_container_width=True, hide_index=True)

            with sc2:
                st.markdown("#### Target Output Measurements (Pending)")
                output_df = df_tpl[["run_id"] + info_tpl["output_columns"]]
                st.dataframe(output_df, use_container_width=True, hide_index=True)

            st.caption("All 10 target measurement columns are presently NaN (Unpopulated), maintaining complete scientific fidelity.")

            st.download_button(
                label="⬇  Download LTspice Sweep Template (CSV)",
                data=df_tpl.to_csv(index=False).encode("utf-8"),
                file_name="rg_sweep_template.csv",
                mime="text/csv",
            )
