"""
visualise.py
------------
Visualisation utilities for the KONE EMC-GUARD AI_ML evaluation suite.

Generates all required plots using Plotly (available in the project venv).
Saves outputs as interactive HTML files and as static PNG where kaleido
is available, otherwise as HTML only.

Figures generated:
  1.  Actual vs Predicted scatter (in-sample, capacitance)
  2.  Actual vs Predicted scatter (in-sample, mutual inductance)
  3.  Residual plot vs predicted (capacitance)
  4.  Residual plot vs predicted (mutual inductance)
  5.  Error distribution histogram (capacitance)
  6.  Error distribution histogram (mutual inductance)
  7.  LOLO validation R² bar chart (capacitance)
  8.  LOLO validation R² bar chart (mutual inductance)
  9.  LOLO RMSE improvement chart (all targets combined)
  10. Feature importance bar chart — MDI (capacitance)
  11. Feature importance bar chart — MDI (mutual inductance)
  12. Feature importance bar chart — PFI (capacitance)
  13. Feature importance bar chart — PFI (mutual inductance)
  14. Sensitivity sweep lines (capacitance)
  15. Sensitivity sweep lines (mutual inductance)
  16. PDP lines (capacitance)
  17. PDP lines (mutual inductance)
  18. XAI global importance chart
  19. Actual vs Predicted by separation level (capacitance)
  20. Actual vs Predicted by separation level (mutual inductance)

Usage (from AI_ML/ directory):
    py src/visualise.py

Usage (from any directory):
    py C:/path/to/kone/AI_ML/src/visualise.py
"""

from __future__ import annotations

import json
import logging
import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    CAP_FEATURES, MUT_FEATURES,
    AIML_REPORTS_DIR,
)

warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)

import plotly.graph_objects as go
import plotly.express as px
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
FIGURES_DIR = AIML_REPORTS_DIR / "figures"

COLOUR_CAP  = "#1565C0"   # dark blue
COLOUR_MUT  = "#B71C1C"   # dark red
COLOUR_LINE = "#2E7D32"   # green
COLOUR_ZERO = "#757575"   # grey
COLOUR_LOLO = "#6A1B9A"   # purple

_PLOTLY_LAYOUT = dict(
    font=dict(family="Arial, sans-serif", size=12),
    paper_bgcolor="white",
    plot_bgcolor="#F8F9FA",
    margin=dict(l=70, r=40, t=70, b=60),
)

CAP_TARGET = "TotalCapacitance_pF_per_m"
MUT_TARGET = "AbsMutualInductance_nH"
CAP_UNIT   = "pF/m"
MUT_UNIT   = "nH"


# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------

def _save_fig(fig: go.Figure, name: str, figures_dir: Path) -> str:
    """Save a Plotly figure as HTML. Attempt PNG if kaleido is available."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info("  Saved HTML: %s", html_path.name)

    # Try PNG (requires kaleido)
    try:
        png_path = figures_dir / f"{name}.png"
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
        logger.info("  Saved PNG:  %s", png_path.name)
        return str(png_path)
    except Exception:
        logger.debug("  kaleido not available — HTML only for %s", name)
        return str(html_path)


# ---------------------------------------------------------------------------
# 1 & 2. Actual vs Predicted scatter
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    unit: str,
    colour: str,
    figures_dir: Path,
    subtitle: str = "In-sample reproduction (ExtraTrees surrogate)",
    extra_info: str = "",
) -> str:
    """Actual vs Predicted scatter plot with identity line."""
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05

    fig = go.Figure()

    # Identity line (perfect prediction)
    fig.add_trace(go.Scatter(
        x=[vmin - margin, vmax + margin],
        y=[vmin - margin, vmax + margin],
        mode="lines",
        name="Perfect prediction",
        line=dict(color=COLOUR_LINE, dash="dash", width=2),
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color=colour, size=6, opacity=0.75,
                    line=dict(width=0.5, color="white")),
        name="Predictions",
        text=[f"Actual: {a:.4g}<br>Predicted: {p:.4g}" for a, p in zip(y_true, y_pred)],
        hoverinfo="text",
    ))

    annotation_text = (
        f"MAE = {mae:.4g} {unit}<br>"
        f"RMSE = {rmse:.4g} {unit}<br>"
        f"R² = {r2:.6f}"
    )
    if extra_info:
        annotation_text += f"<br>{extra_info}"

    fig.add_annotation(
        x=0.98, y=0.05,
        xref="paper", yref="paper",
        text=annotation_text,
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=colour,
        borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"<b>Actual vs Predicted — {label}</b><br>"
                        f"<sup>{subtitle}</sup>", x=0.5),
        xaxis=dict(title=f"Actual [{unit}]", gridcolor="#E0E0E0"),
        yaxis=dict(title=f"Predicted [{unit}]", gridcolor="#E0E0E0"),
        legend=dict(x=0.01, y=0.99),
    )

    fname = f"actual_vs_predicted_{label.lower().replace(' ', '_').replace('/', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 3 & 4. Residual plots
# ---------------------------------------------------------------------------

def plot_residuals(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    label: str,
    unit: str,
    colour: str,
    figures_dir: Path,
) -> str:
    """Residual vs Predicted scatter plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color=colour, size=6, opacity=0.7),
        name="Residual",
        text=[f"Pred: {p:.4g}<br>Residual: {r:.4g}" for p, r in zip(y_pred, residuals)],
        hoverinfo="text",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color=COLOUR_ZERO, dash="dash", width=1.5))

    # ±2σ lines
    sigma = float(np.std(residuals))
    fig.add_hline(y= 2*sigma, line=dict(color="orange", dash="dot", width=1),
                  annotation_text="+2σ", annotation_position="right")
    fig.add_hline(y=-2*sigma, line=dict(color="orange", dash="dot", width=1),
                  annotation_text="−2σ", annotation_position="right")

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"<b>Residuals — {label}</b><br>"
                        f"<sup>Residual = Actual − Predicted (in-sample)</sup>", x=0.5),
        xaxis=dict(title=f"Predicted [{unit}]", gridcolor="#E0E0E0"),
        yaxis=dict(title=f"Residual [{unit}]", gridcolor="#E0E0E0"),
    )

    fname = f"residuals_{label.lower().replace(' ', '_').replace('/', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 5 & 6. Error distribution histograms
# ---------------------------------------------------------------------------

def plot_error_distribution(
    residuals: np.ndarray,
    label: str,
    unit: str,
    colour: str,
    figures_dir: Path,
    validation_type: str = "in-sample",
) -> str:
    """Histogram of prediction errors with normal curve overlay."""
    n_bins = max(10, int(np.sqrt(len(residuals))))

    # Normal fit
    mu    = float(np.mean(residuals))
    sigma = float(np.std(residuals))

    hist_vals, bin_edges = np.histogram(residuals, bins=n_bins, density=True)
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=residuals,
        histnorm="probability density",
        nbinsx=n_bins,
        marker=dict(color=colour, opacity=0.65, line=dict(color="white", width=0.5)),
        name="Error distribution",
    ))

    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm,
        mode="lines",
        line=dict(color=COLOUR_LINE, width=2.5),
        name=f"Normal fit (μ={mu:.3g}, σ={sigma:.3g})",
    ))

    fig.add_vline(x=0, line=dict(color=COLOUR_ZERO, dash="dash", width=1.5))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"<b>Error Distribution — {label}</b><br>"
                        f"<sup>{validation_type} residuals | n={len(residuals)}</sup>", x=0.5),
        xaxis=dict(title=f"Residual [{unit}]", gridcolor="#E0E0E0"),
        yaxis=dict(title="Probability Density", gridcolor="#E0E0E0"),
        legend=dict(x=0.7, y=0.95),
    )

    fname = f"error_distribution_{label.lower().replace(' ', '_').replace('/', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 7 & 8. LOLO validation R² bar charts
# ---------------------------------------------------------------------------

def plot_lolo_r2(
    lolo_results: List[Dict],
    target_name: str,
    unit: str,
    figures_dir: Path,
) -> str:
    """Bar chart of R² per held-out feature level (LOLO)."""
    target_data = [e for e in lolo_results if e["target"] == target_name]
    if not target_data:
        logger.warning("No LOLO data for %s", target_name)
        return ""

    fig = go.Figure()

    colours_palette = px.colors.qualitative.Set2
    for i, entry in enumerate(target_data):
        feat  = entry["held_out_feature"]
        per   = entry["per_level"]
        x_lab = [f"{feat}={m['held_out_level']:.4g}" for m in per]
        y_r2  = [m["R2"] for m in per]

        fig.add_trace(go.Bar(
            name=feat,
            x=x_lab,
            y=y_r2,
            marker_color=colours_palette[i % len(colours_palette)],
            text=[f"{v:.3f}" for v in y_r2],
            textposition="outside",
        ))

    fig.add_hline(y=0, line=dict(color=COLOUR_ZERO, dash="dash", width=1),
                  annotation_text="R²=0 (no better than mean)", annotation_position="right")

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"<b>LOLO Cross-Validation R² — {target_name}</b><br>"
                        f"<sup>Leave-One-Level-Out per input feature [{unit}]</sup>", x=0.5),
        xaxis=dict(title="Held-out configuration", tickangle=-30, gridcolor="#E0E0E0"),
        yaxis=dict(title="R² Score", gridcolor="#E0E0E0"),
        barmode="group",
        legend=dict(title="Feature"),
    )

    fname = f"lolo_r2_{target_name.lower()[:25].replace(' ', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 9. LOLO RMSE improvement chart
# ---------------------------------------------------------------------------

def plot_lolo_rmse_improvement(
    lolo_results: List[Dict],
    figures_dir: Path,
) -> str:
    """Grouped bar chart: RMSE improvement % per target per feature."""
    rows = []
    for entry in lolo_results:
        rows.append({
            "Target":   entry["target"][:30],
            "Feature":  entry["held_out_feature"],
            "Mean_RMSE_Imp_%": entry["mean_rmse_imp_pct"],
            "Mean_RMSE":       entry["mean_RMSE"],
            "Mean_R2":         entry["mean_R2"],
        })
    df = pd.DataFrame(rows)

    colours_palette = [COLOUR_CAP, COLOUR_MUT]
    fig = go.Figure()
    for i, target in enumerate(df["Target"].unique()):
        sub = df[df["Target"] == target]
        fig.add_trace(go.Bar(
            name=target,
            x=sub["Feature"],
            y=sub["Mean_RMSE_Imp_%"],
            marker_color=colours_palette[i % 2],
            text=[f"{v:.1f}%" for v in sub["Mean_RMSE_Imp_%"]],
            textposition="outside",
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="<b>LOLO Validation — Mean RMSE Improvement over Baseline</b><br>"
                        "<sup>How much better is the model vs predicting the training mean?</sup>",
                   x=0.5),
        xaxis=dict(title="Held-out Feature", gridcolor="#E0E0E0"),
        yaxis=dict(title="RMSE Improvement (%)", gridcolor="#E0E0E0"),
        barmode="group",
        legend=dict(title="Target"),
    )

    return _save_fig(fig, "lolo_rmse_improvement", figures_dir)


# ---------------------------------------------------------------------------
# 10-13. Feature importance bar charts
# ---------------------------------------------------------------------------

def plot_feature_importance_bar(
    imp_df: pd.DataFrame,
    value_col: str,
    error_col: Optional[str],
    title: str,
    xlabel: str,
    colour: str,
    figures_dir: Path,
    fname: str,
) -> str:
    """Horizontal bar chart of feature importance."""
    df_sorted = imp_df.sort_values(value_col, ascending=True)

    if error_col and error_col in df_sorted.columns:
        error_x = df_sorted[error_col]
    else:
        error_x = None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["Feature"],
        x=df_sorted[value_col],
        orientation="h",
        marker=dict(
            color=df_sorted[value_col],
            colorscale="Blues",
            line=dict(color=colour, width=1.5),
        ),
        error_x=dict(type="data", array=error_x.tolist(), visible=True) if error_x is not None else None,
        text=[f"{v:.4g}" for v in df_sorted[value_col]],
        textposition="outside",
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis=dict(title=xlabel, gridcolor="#E0E0E0"),
        yaxis=dict(title="Feature", gridcolor="#E0E0E0"),
    )

    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 14 & 15. Sensitivity sweep line plots
# ---------------------------------------------------------------------------

def plot_sensitivity(
    sensitivity: Dict,
    figures_dir: Path,
) -> str:
    """Multi-feature sensitivity sweep line chart."""
    target_name = sensitivity.get("target_name") or sensitivity.get("target", "")
    unit        = sensitivity["unit"]

    n_feats = len(sensitivity["sweeps"])
    fig = make_subplots(
        rows=1, cols=n_feats,
        subplot_titles=list(sensitivity["sweeps"].keys()),
        shared_yaxes=False,
    )

    colours = [COLOUR_CAP, COLOUR_MUT, COLOUR_LOLO]

    for col_idx, (feat, sweep) in enumerate(sensitivity["sweeps"].items(), start=1):
        c = colours[(col_idx - 1) % len(colours)]
        fig.add_trace(
            go.Scatter(
                x=sweep["levels"],
                y=sweep["predictions"],
                mode="lines+markers",
                line=dict(color=c, width=2.5),
                marker=dict(size=8, color=c),
                name=feat,
                showlegend=(col_idx == 1),
                hovertemplate=f"{feat}=%{{x}}<br>Pred={sweep['predictions'][0]:.4g} {unit}<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        # Mark reference level
        ref_idx = sweep["levels"].index(sweep["reference_value"]) \
                  if sweep["reference_value"] in sweep["levels"] else 0
        fig.add_trace(
            go.Scatter(
                x=[sweep["reference_value"]],
                y=[sweep["predictions"][ref_idx]],
                mode="markers",
                marker=dict(size=14, color="gold", symbol="star",
                            line=dict(color=c, width=2)),
                name="Reference level",
                showlegend=(col_idx == 1),
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=400,
        title=dict(
            text=f"<b>Sensitivity Analysis (OAT) — {target_name}</b><br>"
                 f"<sup>One feature varied; others at median grid level. ★ = reference</sup>",
            x=0.5,
        ),
    )
    for col_idx in range(1, n_feats + 1):
        fig.update_xaxes(title_text="Feature Level", row=1, col=col_idx)
        fig.update_yaxes(title_text=f"Predicted [{unit}]", row=1, col=col_idx)

    fname = f"sensitivity_{target_name.lower()[:20].replace(' ', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 16 & 17. PDP line plots
# ---------------------------------------------------------------------------

def plot_pdp(
    ice_pdp: Dict,
    target_name: str,
    unit: str,
    figures_dir: Path,
) -> str:
    """Partial Dependence Plots with confidence bands."""
    n_feats = len(ice_pdp)
    fig = make_subplots(
        rows=1, cols=n_feats,
        subplot_titles=list(ice_pdp.keys()),
    )

    is_cap = "Cap" in target_name or "Capacitance" in target_name
    col_main = COLOUR_CAP if is_cap else COLOUR_MUT

    for col_idx, (feat, data) in enumerate(ice_pdp.items(), start=1):
        x    = data["x_values"]
        mean = np.array(data["pdp_mean"])
        std  = np.array(data["pdp_std"])

        # Confidence band
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=(mean + std).tolist() + (mean - std).tolist()[::-1],
            fill="toself",
            fillcolor=f"rgba({int(col_main[1:3], 16)},{int(col_main[3:5], 16)},{int(col_main[5:7], 16)},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1 std",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        # PDP mean
        fig.add_trace(go.Scatter(
            x=x, y=mean.tolist(),
            mode="lines+markers",
            line=dict(color=col_main, width=2.5),
            marker=dict(size=7, color=col_main),
            name="PDP mean",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=400,
        title=dict(
            text=f"<b>Partial Dependence Plots — {target_name}</b><br>"
                 f"<sup>Average model response as each feature varies (±1 std of ICE)</sup>",
            x=0.5,
        ),
    )
    for col_idx in range(1, n_feats + 1):
        fig.update_xaxes(title_text="Feature Level", row=1, col=col_idx)
        fig.update_yaxes(title_text=f"Predicted [{unit}]", row=1, col=col_idx)

    fname = f"pdp_{target_name.lower()[:20].replace(' ', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# 18. XAI global importance
# ---------------------------------------------------------------------------

def plot_xai_global_importance(
    xai_results: Dict,
    figures_dir: Path,
) -> str:
    """Combined bar chart: global permutation importance (both targets)."""
    fig = go.Figure()

    for tkey in ["capacitance", "mutual_inductance"]:
        if tkey not in xai_results:
            continue
        data = xai_results[tkey]
        gi = data["global_importance"]
        colour = COLOUR_CAP if tkey == "capacitance" else COLOUR_MUT
        fig.add_trace(go.Bar(
            name=data["target_name"][:30],
            x=gi["Feature"],
            y=gi["R2_Drop_Mean"],
            error_y=dict(type="data", array=gi["R2_Drop_Std"].tolist(), visible=True),
            marker_color=colour,
            text=[f"{v:.4f}" for v in gi["R2_Drop_Mean"]],
            textposition="outside",
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(
            text="<b>XAI — Global Permutation Importance (R² drop)</b><br>"
                 "<sup>Mean R² decrease when feature is shuffled — higher = more influential in model</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Feature", gridcolor="#E0E0E0"),
        yaxis=dict(title="Mean R² Drop", gridcolor="#E0E0E0"),
        barmode="group",
        legend=dict(title="Target"),
    )

    return _save_fig(fig, "xai_global_permutation_importance", figures_dir)


# ---------------------------------------------------------------------------
# 19 & 20. Actual vs Predicted coloured by feature level
# ---------------------------------------------------------------------------

def plot_avp_by_feature(
    comparison_df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    colour_feature: str,
    label: str,
    unit: str,
    figures_dir: Path,
) -> str:
    """Actual vs Predicted scatter, coloured by a specific feature level."""
    df = comparison_df.copy()
    df["_actual"] = df[actual_col]
    df["_pred"]   = df[pred_col]

    vmin = min(df["_actual"].min(), df["_pred"].min())
    vmax = max(df["_actual"].max(), df["_pred"].max())
    margin = (vmax - vmin) * 0.05

    levels = sorted(df[colour_feature].unique())
    palette = px.colors.qualitative.Plotly

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[vmin - margin, vmax + margin],
        y=[vmin - margin, vmax + margin],
        mode="lines",
        name="Perfect",
        line=dict(color=COLOUR_LINE, dash="dash", width=1.5),
    ))

    for i, level in enumerate(levels):
        sub = df[df[colour_feature] == level]
        fig.add_trace(go.Scatter(
            x=sub["_actual"], y=sub["_pred"],
            mode="markers",
            name=f"{colour_feature}={level}",
            marker=dict(color=palette[i % len(palette)], size=7, opacity=0.8),
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(
            text=f"<b>Actual vs Predicted — {label}</b><br>"
                 f"<sup>Coloured by {colour_feature}</sup>",
            x=0.5,
        ),
        xaxis=dict(title=f"Actual [{unit}]", gridcolor="#E0E0E0"),
        yaxis=dict(title=f"Predicted [{unit}]", gridcolor="#E0E0E0"),
        legend=dict(title=colour_feature),
    )

    fname = f"avp_by_{colour_feature.lower()[:20]}_{label.lower()[:15].replace('/', '_')}"
    return _save_fig(fig, fname, figures_dir)


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(
    eval_results: Dict,
    importance_results: Dict,
    xai_results: Dict,
    figures_dir: Path = FIGURES_DIR,
) -> Dict:
    """
    Generate all plots from evaluation, importance, and XAI results.

    Parameters
    ----------
    eval_results      : output of evaluate.run_evaluation()
    importance_results: output of feature_importance.run_feature_importance()
    xai_results       : output of explainability.main()
    figures_dir       : directory to save all figures

    Returns
    -------
    dict mapping figure name → saved path
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    logger.info("Generating evaluation plots …")

    # --- 1-2. Actual vs Predicted (in-sample) ---
    for target_key, label, unit, colour, pred_col, actual_col, residual_col in [
        ("capacitance",      "Capacitance",       CAP_UNIT, COLOUR_CAP,
         "Cap_Predicted", CAP_TARGET, "Cap_Residual"),
        ("mutual_inductance", "Mutual Inductance", MUT_UNIT, COLOUR_MUT,
         "Mut_Predicted", MUT_TARGET, "Mut_Residual"),
    ]:
        comp_df = eval_results["in_sample"][target_key]["comparison_df"]
        y_true = comp_df[actual_col].values
        y_pred = comp_df[pred_col].values
        residuals = comp_df[residual_col].values

        # Actual vs Predicted
        p = plot_actual_vs_predicted(
            y_true, y_pred, label, unit, colour, figures_dir,
            subtitle="In-sample (MATLAB dataset — not independent test)"
        )
        saved[f"avp_{target_key}"] = p

        # Residuals
        p = plot_residuals(y_pred, residuals, label, unit, colour, figures_dir)
        saved[f"residuals_{target_key}"] = p

        # Error distribution
        p = plot_error_distribution(
            residuals, label, unit, colour, figures_dir,
            validation_type="In-sample"
        )
        saved[f"error_dist_{target_key}"] = p

        # Coloured by PairSeparation_mm
        p = plot_avp_by_feature(
            comp_df, pred_col, actual_col,
            "PairSeparation_mm", label, unit, figures_dir
        )
        saved[f"avp_by_sep_{target_key}"] = p

    # --- LOLO plots ---
    logger.info("Generating LOLO validation plots …")
    lolo_results = eval_results["lolo"]["lolo_results"]

    for target_name, unit in [(CAP_TARGET, CAP_UNIT), (MUT_TARGET, MUT_UNIT)]:
        p = plot_lolo_r2(lolo_results, target_name, unit, figures_dir)
        key = "capacitance" if "Cap" in target_name else "mutual_inductance"
        saved[f"lolo_r2_{key}"] = p

    p = plot_lolo_rmse_improvement(lolo_results, figures_dir)
    saved["lolo_rmse_improvement"] = p

    # --- LOLO error distribution ---
    for tkey, target_name, unit, colour in [
        ("capacitance",      CAP_TARGET, CAP_UNIT, COLOUR_CAP),
        ("mutual_inductance", MUT_TARGET, MUT_UNIT, COLOUR_MUT),
    ]:
        lolo_res_all = []
        for entry in lolo_results:
            if entry["target"] == target_name:
                for fold in entry["fold_details"]:
                    lolo_res_all.extend(fold["residuals"])
        if lolo_res_all:
            p = plot_error_distribution(
                np.array(lolo_res_all),
                f"{tkey.replace('_', ' ').title()} LOLO",
                unit, colour, figures_dir,
                validation_type="LOLO cross-validation"
            )
            saved[f"error_dist_lolo_{tkey}"] = p

    # --- Feature importance plots ---
    logger.info("Generating feature importance plots …")
    for tkey, colour in [("capacitance", COLOUR_CAP), ("mutual_inductance", COLOUR_MUT)]:
        data = importance_results.get(tkey)
        if data is None:
            continue
        target_label = data["target_name"][:25]

        # MDI
        p = plot_feature_importance_bar(
            data["mdi_df"], "Importance_Pct", None,
            f"MDI Feature Importance — {target_label}",
            "Importance (%)",
            colour, figures_dir,
            f"fi_mdi_{tkey}",
        )
        saved[f"fi_mdi_{tkey}"] = p

        # PFI
        p = plot_feature_importance_bar(
            data["pfi_df"], "Importance_Mean", "Importance_Std",
            f"Permutation Feature Importance — {target_label}",
            "Mean R² Drop (±Std)",
            colour, figures_dir,
            f"fi_pfi_{tkey}",
        )
        saved[f"fi_pfi_{tkey}"] = p

    # --- XAI plots ---
    logger.info("Generating XAI plots …")

    p = plot_xai_global_importance(xai_results, figures_dir)
    saved["xai_global_importance"] = p

    for tkey in ["capacitance", "mutual_inductance"]:
        data = xai_results.get(tkey)
        if data is None:
            continue

        p = plot_sensitivity(data["sensitivity"], figures_dir)
        saved[f"sensitivity_{tkey}"] = p

        p = plot_pdp(data["ice_pdp"], data["target_name"], data["unit"], figures_dir)
        saved[f"pdp_{tkey}"] = p

    logger.info("All plots generated.")
    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full visualisation pipeline."""
    from evaluate import run_evaluation
    from feature_importance import run_feature_importance
    from explainability import main as run_xai

    logger.info("=" * 60)
    logger.info("KONE EMC-GUARD — Visualisation Pipeline")
    logger.info("=" * 60)

    logger.info("Step 1: Running model evaluation …")
    eval_results = run_evaluation(save_results=True)

    logger.info("Step 2: Running feature importance analysis …")
    imp_results = run_feature_importance(save_results=True)

    logger.info("Step 3: Running XAI analysis …")
    xai_results = run_xai(save_results=True)

    logger.info("Step 4: Generating all visualisations …")
    figures = generate_all_plots(eval_results, imp_results, xai_results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index_path = FIGURES_DIR / f"figure_index_{timestamp}.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": timestamp, "figures": figures}, fh, indent=2)
    logger.info("Figure index saved: %s", index_path.name)

    print("\n" + "=" * 60)
    print("Visualisations complete.")
    print(f"Saved to: {FIGURES_DIR}")
    print("=" * 60)
    for name, path in figures.items():
        print(f"  {name}: {Path(path).name}")

    return figures


if __name__ == "__main__":
    main()

