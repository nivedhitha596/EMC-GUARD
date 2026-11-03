"""
report_generator.py
-------------------
Generates engineering PDF reports using ReportLab.

Reports include:
  - Project and configuration metadata
  - Frozen model predictions (exact values from MATLAB)
  - Model identifiers and domain validity
  - Clearly stated model limitations
  - Comparison table (if provided)
  - Date, time and report version
"""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


REPORT_VERSION = "1.0"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── Colour palette (matches dashboard) ──────────────────────────────────────
_NAVY   = colors.HexColor("#0F172A")
_BLUE   = colors.HexColor("#2563EB")
_LIGHT  = colors.HexColor("#CBD5E1")
_HEADER = colors.HexColor("#1E3A5F")
_ROW1   = colors.HexColor("#1E293B")
_ROW2   = colors.HexColor("#0F172A")
_WHITE  = colors.white
_WARN   = colors.HexColor("#F59E0B")
_GREEN  = colors.HexColor("#22C55E")


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"],
            textColor=_WHITE, fontSize=18, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            textColor=_LIGHT, fontSize=11, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            textColor=_BLUE, fontSize=13, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            textColor=_LIGHT, fontSize=10, spaceAfter=4,
        ),
        "warning": ParagraphStyle(
            "Warning", parent=base["Normal"],
            textColor=_WARN, fontSize=9, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["Normal"],
            textColor=colors.HexColor("#64748B"), fontSize=8,
        ),
    }


def _table(data, col_widths=None):
    """Create a styled table."""
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  _HEADER),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  _WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_ROW1, _ROW2]),
        ("TEXTCOLOR",   (0, 1), (-1, -1), _LIGHT),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#334155")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def generate_report(
    config_id: str,
    sep: float,
    pitch: float,
    overlap: float,
    ground: float,
    cap_value: float,
    mut_value: float,
    cap_model: str,
    mut_model: str,
    engine: str,
    recommendation: Optional[str] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    filename: Optional[str] = None,
) -> bytes:
    """
    Generate an engineering PDF report and return the bytes.

    Parameters
    ----------
    config_id   : Configuration identifier (e.g. "EMC-001")
    sep, pitch, overlap, ground : geometry inputs
    cap_value   : predicted total capacitance [pF/m]
    mut_value   : predicted absolute mutual inductance [nH]
    cap_model   : model name used for capacitance
    mut_model   : model name used for inductance
    engine      : prediction engine identifier
    recommendation : EMC-GUARD recommendation text (optional)
    comparison_df  : DataFrame of compared configurations (optional)
    filename    : output filename (optional; if None returns bytes only)

    Returns
    -------
    bytes : raw PDF bytes
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is not installed. "
            "Run: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="EMC-GUARD Engineering Report",
    )

    S = _styles()
    story = []
    W = A4[0] - 40 * mm  # usable width

    now = datetime.now(timezone.utc)

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("EMC-PREDICT + EMC-GUARD", S["title"]))
    story.append(Paragraph(
        "Physics-Informed Configuration-Aware EMC Digital Twin for Elevator Cable Systems",
        S["subtitle"],
    ))
    story.append(HRFlowable(width=W, color=_BLUE, thickness=1.5))
    story.append(Spacer(1, 6))

    # ── Report metadata ───────────────────────────────────────────────────
    story.append(Paragraph("Report Metadata", S["h2"]))
    meta_data = [
        ["Field", "Value"],
        ["Project",         "Physics-Informed Configuration-Aware Digital Twin"],
        ["Report version",  REPORT_VERSION],
        ["Generated",       now.strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Configuration ID", config_id],
        ["Prediction engine", engine],
    ]
    story.append(_table(meta_data, [60 * mm, W - 60 * mm]))
    story.append(Spacer(1, 10))

    # ── Input configuration ───────────────────────────────────────────────
    story.append(Paragraph("Input Geometry Configuration", S["h2"]))
    geom_data = [
        ["Parameter",             "Value",    "Unit",  "Supported levels"],
        ["Cable pair separation",  f"{sep}",   "mm",   "8, 12, 16, 20, 24, 28"],
        ["Cable pitch",            f"{pitch}", "mm",   "2, 3, 4"],
        ["Parallel overlap length", f"{overlap}", "mm", "150, 300, 450"],
        ["Ground clearance",       f"{ground}", "mm",  "10, 25, 50, 100"],
    ]
    story.append(_table(geom_data, [65 * mm, 30 * mm, 20 * mm, W - 115 * mm]))
    story.append(Spacer(1, 10))

    # ── Predictions ───────────────────────────────────────────────────────
    story.append(Paragraph("EMC-PREDICT Outputs", S["h2"]))
    pred_data = [
        ["Parameter",                   "Predicted Value", "Unit", "Model"],
        ["Total capacitance",           f"{cap_value:.6g}", "pF/m",  cap_model],
        ["Absolute mutual inductance",  f"{mut_value:.6g}", "nH",    mut_model],
    ]
    story.append(_table(pred_data, [70 * mm, 35 * mm, 20 * mm, W - 125 * mm]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Domain validity: Original 216-point MATLAB design grid. "
        "No extrapolation is permitted.",
        S["body"],
    ))
    story.append(Spacer(1, 10))

    # ── Configuration comparison ──────────────────────────────────────────
    if comparison_df is not None and len(comparison_df) > 0:
        story.append(Paragraph("Configuration Comparison", S["h2"]))
        disp_cols = [
            "Config_ID", "PairSeparation_mm", "PairPitch_mm",
            "Overlap_mm", "GroundClearance_mm",
            "TotalCapacitance_pF_per_m", "AbsMutualInductance_nH",
        ]
        available = [c for c in disp_cols if c in comparison_df.columns]
        sub = comparison_df[available].head(12)

        tbl_data = [available]
        for _, row in sub.iterrows():
            tbl_data.append([
                str(row[c]) if isinstance(row[c], str)
                else f"{row[c]:.4g}"
                for c in available
            ])
        col_w = W / len(available)
        story.append(_table(tbl_data, [col_w] * len(available)))
        story.append(Spacer(1, 10))

    # ── Recommendation ────────────────────────────────────────────────────
    if recommendation:
        story.append(Paragraph("EMC-GUARD Recommendation", S["h2"]))
        story.append(Paragraph(recommendation, S["body"]))
        story.append(Spacer(1, 10))

    # ── Limitations ───────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, color=colors.HexColor("#334155"), thickness=0.5))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Important Model Limitations", S["h2"]))
    limitations = [
        "Predictions reproduce a deterministic MATLAB-generated dataset (216 configurations).",
        "The frozen models are polynomial (Stage 12A) and ExtraTrees (Stage 14A) surrogates "
        "fitted to the analytical dataset.",
        "This platform does NOT predict victim voltage, conducted emission or radiated emission.",
        "No EMC compliance, safety certification or installation approval is established.",
        "Independent experimental validation has not yet been performed.",
        "Do not extrapolate predictions outside the 216-point supported design grid.",
        "Terminology: 'lower predicted coupling' refers to relative model output comparison only.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"• {lim}", S["warning"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Generated by EMC-PREDICT + EMC-GUARD v1.0  |  "
        f"{now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        S["small"],
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if filename:
        out_path = REPORT_DIR / filename
        out_path.write_bytes(pdf_bytes)

    return pdf_bytes

