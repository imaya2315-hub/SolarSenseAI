"""
report.py – Generate a downloadable PDF prediction report using ReportLab.
"""

import os
import io
import time
from datetime import datetime

from reportlab.lib.pagesizes  import A4
from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units      import cm
from reportlab.lib             import colors
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums      import TA_CENTER, TA_LEFT

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "plots")
os.makedirs(REPORT_DIR, exist_ok=True)

# Brand colours
SOLAR_ORANGE = colors.HexColor("#F97316")
SOLAR_DARK   = colors.HexColor("#0F172A")
SOLAR_LIGHT  = colors.HexColor("#F8FAFC")
SLATE        = colors.HexColor("#64748B")


def generate_report(result: dict, inputs: dict,
                    importance_img: str | None = None,
                    batch_img: str | None = None) -> str:
    """
    Build a PDF report and return its filename (relative to static/plots/).

    Parameters
    ----------
    result        : dict from predictor.predict_single
    inputs        : dict with Temperature, Wind Speed, GHI, Previous Active Power
    importance_img: abs path to the feature-importance PNG
    batch_img     : abs path to the batch timeline PNG (optional)
    """
    fname    = f"report_{int(time.time()*1000)}.pdf"
    abs_path = os.path.join(REPORT_DIR, fname)

    doc  = SimpleDocTemplate(
        abs_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )
    styles  = getSampleStyleSheet()
    story   = []

    # ── Title block ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        textColor=SOLAR_DARK, fontSize=22, leading=28,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "sub", parent=styles["Normal"],
        textColor=SLATE, fontSize=10, leading=14,
    )

    story.append(Paragraph("SolarSense AI", title_style))
    story.append(Paragraph("Solar Power Prediction Report", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=SOLAR_ORANGE))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M:%S')}",
        ParagraphStyle("date", parent=styles["Normal"], textColor=SLATE, fontSize=9),
    ))
    story.append(Spacer(1, 0.6*cm))

    # ── Key result ───────────────────────────────────────────────────────────
    result_style = ParagraphStyle(
        "result", parent=styles["Normal"],
        textColor=SOLAR_ORANGE, fontSize=30, leading=36,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    story.append(Paragraph(
        f"Predicted Output: {result['predicted_power']:.2f} kWh",
        result_style,
    ))
    story.append(Paragraph(
        f"Confidence: {result['confidence']:.0f}%",
        ParagraphStyle("conf", parent=styles["Normal"], textColor=SLATE,
                       fontSize=12, alignment=TA_CENTER),
    ))
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.5*cm))

    # ── Input parameters table ────────────────────────────────────────────────
    story.append(Paragraph(
        "Input Parameters",
        ParagraphStyle("h2", parent=styles["Heading2"],
                       textColor=SOLAR_DARK, fontSize=13),
    ))
    story.append(Spacer(1, 0.3*cm))

    input_data = [
        ["Parameter",            "Value",               "Unit"],
        ["Temperature",          f"{inputs.get('temperature', 'N/A'):.1f}", "°C"],
        ["Wind Speed",           f"{inputs.get('wind_speed',  'N/A'):.1f}", "m/s"],
        ["GHI",                  f"{inputs.get('ghi',         'N/A'):.1f}", "W/m²"],
        ["Previous Active Power",f"{inputs.get('prev_power',  'N/A'):.1f}", "kW"],
    ]
    t = Table(input_data, colWidths=[8*cm, 5*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), SOLAR_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SOLAR_LIGHT, colors.white]),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))

    # ── Feature importance chart ──────────────────────────────────────────────
    if importance_img and os.path.exists(importance_img):
        story.append(Paragraph(
            "Feature Importance",
            ParagraphStyle("h2", parent=styles["Heading2"],
                           textColor=SOLAR_DARK, fontSize=13),
        ))
        story.append(Spacer(1, 0.3*cm))
        story.append(RLImage(importance_img, width=14*cm, height=7*cm))
        story.append(Spacer(1, 0.5*cm))

    # ── Batch chart (if provided) ─────────────────────────────────────────────
    if batch_img and os.path.exists(batch_img):
        story.append(Paragraph(
            "Batch Prediction Timeline",
            ParagraphStyle("h2", parent=styles["Heading2"],
                           textColor=SOLAR_DARK, fontSize=13),
        ))
        story.append(Spacer(1, 0.3*cm))
        story.append(RLImage(batch_img, width=14*cm, height=7*cm))
        story.append(Spacer(1, 0.5*cm))

    # ── Model info ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Model Information",
        ParagraphStyle("h2", parent=styles["Heading2"],
                       textColor=SOLAR_DARK, fontSize=13),
    ))
    model_text = (
        "This prediction was generated by a Hybrid CNN-LSTM neural network trained on "
        "historical solar-farm data. The CNN layers extract local feature patterns "
        "from meteorological inputs while the LSTM layers capture temporal dependencies "
        "in power generation. SHAP-style perturbation analysis was used to compute "
        "feature importances."
    )
    story.append(Paragraph(
        model_text,
        ParagraphStyle("body", parent=styles["Normal"], fontSize=9,
                       textColor=SLATE, leading=14),
    ))
    story.append(Spacer(1, 0.6*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Paragraph(
        "SolarSense AI  •  Explainable Solar Power Prediction  •  Confidential",
        ParagraphStyle("footer", parent=styles["Normal"],
                       textColor=SLATE, fontSize=8, alignment=TA_CENTER),
    ))

    doc.build(story)
    return fname
