"""
geometry.py
-----------
Digital twin geometry representation for EMC cable configurations.
Generates Plotly 3D figures and 2D cross-section views.

This is a configurable geometric representation of the cable arrangement,
not an electromagnetic field solver.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Cable diameter (representative – not a verified KONE cable cross-section)
# ---------------------------------------------------------------------------
CABLE_DIAMETER_MM = 6.0   # approximate, for visualisation only
CABLE_RADIUS_MM   = CABLE_DIAMETER_MM / 2.0


# ---------------------------------------------------------------------------
# 3D cable geometry
# ---------------------------------------------------------------------------
def _cable_cylinder(
    x0: float, y0: float,
    x1: float, y1: float,
    z_start: float, z_end: float,
    radius: float,
    n_theta: int = 16,
    color: str = "steelblue",
) -> go.Surface:
    """Create a cylindrical Plotly Surface for a cable segment."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_pts = np.array([z_start, z_end])
    THETA, Z = np.meshgrid(theta, z_pts)
    # Cable runs along Z; displaced in X-Y
    X = x0 + radius * np.cos(THETA)
    Y = y0 + radius * np.sin(THETA)
    Z_out = Z

    return go.Surface(
        x=X, y=Y, z=Z_out,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.85,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.3),
        name="",
    )


def build_3d_figure(
    sep_mm: float,
    pitch_mm: float,
    overlap_mm: float,
    ground_mm: float,
) -> go.Figure:
    """
    Build an interactive 3D Plotly figure of the cable configuration.

    Parameters
    ----------
    sep_mm    : PairSeparation_mm  – horizontal distance between cable centres
    pitch_mm  : PairPitch_mm       – vertical stagger (structural pitch parameter)
    overlap_mm: Overlap_mm         – parallel run length
    ground_mm : GroundClearance_mm – height above ground plane

    Returns
    -------
    go.Figure
    """
    r = CABLE_RADIUS_MM

    # Source cable: centred at (0, ground_mm)
    src_x = 0.0
    src_y = ground_mm

    # Victim cable: offset by sep_mm horizontally, pitch_mm vertically
    vic_x = sep_mm
    vic_y = ground_mm + pitch_mm

    # Cable lengths (overlap region is centred; source has extra length)
    src_total = overlap_mm + 40
    vic_total = overlap_mm
    z_src_start = -(src_total / 2)
    z_src_end   =  (src_total / 2)
    z_vic_start = -(vic_total / 2)
    z_vic_end   =  (vic_total / 2)

    traces = []

    # Source cable
    src_surface = _cable_cylinder(
        src_x, src_y, src_x, src_y,
        z_src_start, z_src_end, r,
        color="#2196F3",
    )
    src_surface.name = "Source cable"
    traces.append(src_surface)

    # Victim cable
    vic_surface = _cable_cylinder(
        vic_x, vic_y, vic_x, vic_y,
        z_vic_start, z_vic_end, r,
        color="#FF9800",
    )
    vic_surface.name = "Victim cable"
    traces.append(vic_surface)

    # Ground plane
    x_gnd = np.array([-r - 5, sep_mm + r + 5, sep_mm + r + 5, -r - 5])
    y_gnd = np.array([0.0, 0.0, 0.0, 0.0])
    z_gnd = np.array([z_src_start - 20, z_src_start - 20, z_src_end + 20, z_src_end + 20])
    traces.append(
        go.Mesh3d(
            x=x_gnd, y=y_gnd, z=z_gnd,
            color="#455A64", opacity=0.25,
            name="Ground plane",
        )
    )

    # Overlap bracket markers
    for zv in [z_vic_start, z_vic_end]:
        traces.append(
            go.Scatter3d(
                x=[src_x - r - 2, vic_x + r + 2],
                y=[src_y, vic_y],
                z=[zv, zv],
                mode="lines",
                line=dict(color="#E53935", width=3, dash="dash"),
                name="Overlap boundary",
            )
        )

    # Separation annotation line
    traces.append(
        go.Scatter3d(
            x=[src_x, vic_x],
            y=[src_y, vic_y],
            z=[0.0, 0.0],
            mode="lines+text",
            line=dict(color="#76FF03", width=4),
            text=[f"", f"sep={sep_mm} mm"],
            textfont=dict(size=10, color="#76FF03"),
            name=f"Separation {sep_mm} mm",
        )
    )

    # Dimension annotations
    annotations = [
        dict(
            text=f"<b>Separation</b><br>{sep_mm} mm",
            x=(src_x + vic_x) / 2, y=(src_y + vic_y) / 2 + 2, z=0,
            showarrow=False, font=dict(size=11, color="#76FF03"),
            bgcolor="rgba(0,0,0,0.5)",
        ),
        dict(
            text=f"<b>Overlap</b><br>{overlap_mm} mm",
            x=src_x - r - 8, y=src_y, z=0,
            showarrow=False, font=dict(size=11, color="#FF9800"),
            bgcolor="rgba(0,0,0,0.5)",
        ),
        dict(
            text=f"<b>Ground clearance</b><br>{ground_mm} mm",
            x=src_x - r - 8, y=ground_mm / 2, z=z_src_start,
            showarrow=False, font=dict(size=11, color="#90A4AE"),
            bgcolor="rgba(0,0,0,0.5)",
        ),
        dict(
            text=f"<b>Pitch</b><br>{pitch_mm} mm",
            x=vic_x + r + 4, y=(src_y + vic_y) / 2, z=z_src_end,
            showarrow=False, font=dict(size=11, color="#CE93D8"),
            bgcolor="rgba(0,0,0,0.5)",
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (mm)", showgrid=True, gridcolor="#334155",
                       backgroundcolor="#0F172A", color="#94A3B8"),
            yaxis=dict(title="Y – Height (mm)", showgrid=True, gridcolor="#334155",
                       backgroundcolor="#0F172A", color="#94A3B8"),
            zaxis=dict(title="Z – Cable run (mm)", showgrid=True, gridcolor="#334155",
                       backgroundcolor="#0F172A", color="#94A3B8"),
            bgcolor="#0F172A",
            annotations=annotations,
            camera=dict(eye=dict(x=1.6, y=0.8, z=0.8)),
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#CBD5E1", family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            font=dict(color="#CBD5E1"),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


# ---------------------------------------------------------------------------
# 2D Cross-section view
# ---------------------------------------------------------------------------
def build_cross_section_figure(
    sep_mm: float,
    pitch_mm: float,
    ground_mm: float,
) -> go.Figure:
    """
    Build a 2D cross-section (XY plane) Plotly figure.
    """
    r = CABLE_RADIUS_MM

    src_x, src_y = 0.0, ground_mm
    vic_x, vic_y = sep_mm, ground_mm + pitch_mm

    fig = go.Figure()

    # Ground line
    x_range = [-r - 8, sep_mm + r + 8]
    fig.add_shape(type="line",
                  x0=x_range[0], y0=0, x1=x_range[1], y1=0,
                  line=dict(color="#455A64", width=3))
    fig.add_annotation(x=(x_range[0] + x_range[1]) / 2, y=-2,
                        text="<b>Ground Plane</b>", showarrow=False,
                        font=dict(color="#90A4AE", size=11))

    # Source cable circle
    theta = np.linspace(0, 2 * np.pi, 64)
    fig.add_trace(go.Scatter(
        x=src_x + r * np.cos(theta),
        y=src_y + r * np.sin(theta),
        fill="toself", fillcolor="rgba(33,150,243,0.4)",
        line=dict(color="#2196F3", width=2),
        name="Source cable",
        mode="lines",
    ))
    fig.add_annotation(x=src_x, y=src_y, text="<b>SRC</b>",
                        showarrow=False, font=dict(color="white", size=10))

    # Victim cable circle
    fig.add_trace(go.Scatter(
        x=vic_x + r * np.cos(theta),
        y=vic_y + r * np.sin(theta),
        fill="toself", fillcolor="rgba(255,152,0,0.4)",
        line=dict(color="#FF9800", width=2),
        name="Victim cable",
        mode="lines",
    ))
    fig.add_annotation(x=vic_x, y=vic_y, text="<b>VIC</b>",
                        showarrow=False, font=dict(color="white", size=10))

    # Separation arrow
    fig.add_annotation(
        x=vic_x, y=(src_y + vic_y) / 2, ax=src_x, ay=(src_y + vic_y) / 2,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"sep={sep_mm} mm",
        showarrow=True, arrowhead=2, arrowcolor="#76FF03",
        font=dict(color="#76FF03", size=10),
    )

    # Ground clearance arrow
    fig.add_annotation(
        x=src_x - r - 4, y=src_y, ax=src_x - r - 4, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"GC={ground_mm} mm",
        showarrow=True, arrowhead=2, arrowcolor="#90A4AE",
        font=dict(color="#90A4AE", size=10),
    )

    fig.update_layout(
        xaxis=dict(title="X (mm)", showgrid=True, gridcolor="#334155", color="#94A3B8",
                   range=[x_range[0] - 2, x_range[1] + 2]),
        yaxis=dict(title="Y – Height (mm)", showgrid=True, gridcolor="#334155", color="#94A3B8",
                   range=[-6, max(src_y, vic_y) + r + 8],
                   scaleanchor="x", scaleratio=1),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#CBD5E1", family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(bgcolor="#1E293B", bordercolor="#334155", font=dict(color="#CBD5E1")),
        height=380,
        title=dict(
            text=f"Cable Cross-Section  |  Sep={sep_mm} mm  |  Pitch={pitch_mm} mm  |  GC={ground_mm} mm",
            font=dict(color="#E2E8F0", size=13),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

