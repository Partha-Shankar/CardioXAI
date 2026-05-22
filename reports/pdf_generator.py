"""
reports/pdf_generator.py
========================
Clinical-style PDF report using ReportLab.

Sections:
  1. Header (logo + metadata)
  2. Patient input summary table
  3. Risk assessment badge
  4. Model agreement panel
  5. Top SHAP risk factors
  6. Clinical recommendations
  7. Disclaimer
  8. Footer
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.colors import HexColor, white, black

# ---------------------------------------------------------------------------
# Design tokens (Apple-inspired clinical palette)
# ---------------------------------------------------------------------------
C_PRIMARY   = HexColor("#0071E3")   # Apple blue
C_BG        = HexColor("#F5F5F7")   # Apple off-white
C_BORDER    = HexColor("#E5E5EA")
C_TEXT      = HexColor("#1D1D1F")
C_SECONDARY = HexColor("#6E6E73")
C_GREEN     = HexColor("#34C759")
C_AMBER     = HexColor("#FF9F0A")
C_RED       = HexColor("#FF3B30")
C_PURPLE    = HexColor("#AF52DE")

RISK_COLORS = {
    "Low":      C_GREEN,
    "Moderate": C_AMBER,
    "High":     C_RED,
    "Critical": C_PURPLE,
}

RECOMMENDATIONS = {
    "Low": (
        "Continue healthy lifestyle. Annual cardiovascular screening recommended. "
        "Maintain regular physical activity (≥150 min/week moderate intensity), "
        "balanced diet low in saturated fats, and routine health check-ups."
    ),
    "Moderate": (
        "Consult cardiologist within 3 months. Monitor blood pressure and cholesterol. "
        "Consider lifestyle modifications including dietary changes, increased physical "
        "activity, and stress management. Follow-up lipid panel recommended."
    ),
    "High": (
        "Urgent cardiologist referral recommended. Consider stress test and lipid panel. "
        "Medication review advised. Immediate lifestyle interventions required. "
        "Avoid strenuous activity until cleared by a physician."
    ),
    "Critical": (
        "Immediate medical evaluation required. High risk of acute coronary event. "
        "Do not delay seeking emergency care if experiencing chest pain, shortness of "
        "breath, or radiating arm pain. Call emergency services immediately if symptomatic."
    ),
}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()

    def make(name, **kwargs) -> ParagraphStyle:
        font_name = kwargs.pop("fontName", "Helvetica")
        return ParagraphStyle(name=name, fontName=font_name, **kwargs)

    return {
        "title":       make("title",       fontSize=22, textColor=C_TEXT,      spaceAfter=2,  alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "subtitle":    make("subtitle",    fontSize=10, textColor=C_SECONDARY,  spaceAfter=6,  alignment=TA_CENTER),
        "section":     make("section",     fontSize=13, textColor=C_TEXT,       spaceAfter=4,  fontName="Helvetica-Bold"),
        "body":        make("body",        fontSize=9,  textColor=C_TEXT,       spaceAfter=3,  leading=14),
        "small":       make("small",       fontSize=8,  textColor=C_SECONDARY,  spaceAfter=2),
        "risk_label":  make("risk_label",  fontSize=28, textColor=white,        alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "risk_prob":   make("risk_prob",   fontSize=14, textColor=white,        alignment=TA_CENTER),
        "disclaimer":  make("disclaimer",  fontSize=8,  textColor=C_SECONDARY,  alignment=TA_CENTER, leading=12),
        "footer":      make("footer",      fontSize=7,  textColor=C_SECONDARY,  alignment=TA_CENTER),
        "right":       make("right",       fontSize=9,  textColor=C_SECONDARY,  alignment=TA_RIGHT),
    }


def _hr(width=1, color=C_BORDER) -> HRFlowable:
    return HRFlowable(width="100%", thickness=width, color=color, spaceAfter=4, spaceBefore=4)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_pdf(
    patient_id: str,
    input_values: dict,          # {feature_label: value}
    ensemble_result: dict,       # output of CardioEnsemble.predict_proba()
    shap_data: dict,             # output of SHAPExplainer.get_shap_values()
    feature_labels: dict | None = None,
) -> bytes:
    """
    Generate a clinical risk assessment PDF report.

    Returns
    -------
    bytes  — raw PDF bytes suitable for st.download_button
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"CardioXAI Report — {patient_id}",
        author="CardioXAI v1.0",
    )

    S   = _styles()
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    story = []

    # -----------------------------------------------------------------------
    # 1. Header
    # -----------------------------------------------------------------------
    story.append(Paragraph("CardioXAI", S["title"]))
    story.append(Paragraph("Clinical Heart Disease Risk Assessment Report", S["subtitle"]))
    story.append(_hr(2, C_PRIMARY))
    story.append(Spacer(1, 3 * mm))

    meta_data = [
        ["Patient ID:", patient_id, "Report Date:", now],
        ["System:", "CardioXAI v1.0", "Dataset:", "UCI Cleveland Heart Disease"],
    ]
    meta_table = Table(meta_data, colWidths=[35 * mm, 65 * mm, 35 * mm, 45 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",   (0, 0), (-1, -1), C_SECONDARY),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), C_TEXT),
        ("TEXTCOLOR",   (2, 0), (2, -1), C_TEXT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # 2. Patient Input Summary
    # -----------------------------------------------------------------------
    story.append(Paragraph("Patient Input Summary", S["section"]))
    story.append(_hr())

    feat_labels = feature_labels or {}
    rows   = [["Feature", "Value", "Feature", "Value"]]
    items  = list(input_values.items())
    for i in range(0, len(items), 2):
        left_k, left_v  = items[i]
        right_k, right_v = items[i + 1] if i + 1 < len(items) else ("", "")
        rows.append([
            feat_labels.get(left_k,  left_k),
            str(left_v),
            feat_labels.get(right_k, right_k),
            str(right_v),
        ])

    input_table = Table(rows, colWidths=[50 * mm, 40 * mm, 50 * mm, 40 * mm])
    input_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, HexColor("#F5F5F7")]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), C_TEXT),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 1), (2, -1), "Helvetica-Bold"),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # 3. Risk Assessment
    # -----------------------------------------------------------------------
    story.append(Paragraph("Risk Assessment", S["section"]))
    story.append(_hr())

    prob       = ensemble_result["ensemble_prob"]
    risk_level = ensemble_result["risk_level"]
    risk_color = RISK_COLORS.get(risk_level, C_AMBER)
    conf       = ensemble_result["confidence"]

    risk_pct   = f"{prob * 100:.1f}%"

    risk_data  = [
        [
            Paragraph(risk_level, S["risk_label"]),
            Paragraph(f"Probability: {risk_pct}", S["risk_prob"]),
        ]
    ]
    risk_table = Table(risk_data, colWidths=[80 * mm, 100 * mm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), risk_color),
        ("ROUNDEDCORNERS", [8]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"Model confidence (inter-model agreement): {conf * 100:.1f}%",
        S["small"],
    ))
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # 4. Model Agreement
    # -----------------------------------------------------------------------
    story.append(Paragraph("Model Agreement", S["section"]))
    story.append(_hr())

    model_rows = [["Model", "Probability", "Risk Weight"]]
    for name, mp in ensemble_result["model_probs"].items():
        model_rows.append([name, f"{mp * 100:.1f}%", f"{mp:.4f}"])

    model_table = Table(model_rows, colWidths=[80 * mm, 60 * mm, 40 * mm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, HexColor("#F5F5F7")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # 5. Top Risk Factors (SHAP)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Key Risk Factors (SHAP Analysis)", S["section"]))
    story.append(_hr())

    shap_rows = [["Rank", "Feature", "SHAP Value", "Direction"]]
    for rank, (feat, val, direction) in enumerate(shap_data["top_features"], 1):
        label     = feat_labels.get(feat, feat)
        dir_text  = "Increases Risk ↑" if direction == "↑" else "Decreases Risk ↓"
        shap_rows.append([str(rank), label, f"{val:+.4f}", dir_text])

    shap_table = Table(shap_rows, colWidths=[15 * mm, 70 * mm, 40 * mm, 55 * mm])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, HexColor("#F5F5F7")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        # Colour direction column
        *[
            ("TEXTCOLOR", (3, i + 1), (3, i + 1),
             C_RED if "↑" in shap_rows[i + 1][3] else C_GREEN)
            for i in range(len(shap_data["top_features"]))
        ],
    ]))
    story.append(shap_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # 6. Clinical Recommendations
    # -----------------------------------------------------------------------
    story.append(Paragraph("Clinical Recommendations", S["section"]))
    story.append(_hr())

    rec_text = RECOMMENDATIONS.get(risk_level, RECOMMENDATIONS["Moderate"])
    story.append(Paragraph(f"<b>Risk Level: {risk_level}</b>", S["body"]))
    story.append(Paragraph(rec_text, S["body"]))
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # 7. Disclaimer
    # -----------------------------------------------------------------------
    story.append(_hr(0.5))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI system for educational and "
        "research purposes only. It is NOT a substitute for professional clinical diagnosis, "
        "advice, or treatment. Always seek the guidance of a qualified healthcare provider "
        "with questions you may have regarding a medical condition.",
        S["disclaimer"],
    ))
    story.append(Spacer(1, 3 * mm))

    # -----------------------------------------------------------------------
    # 8. Footer
    # -----------------------------------------------------------------------
    story.append(_hr(0.5))
    story.append(Paragraph(
        "CardioXAI v1.0  ·  IEEE Paper: 'CardioXAI: A Multi-Model Ensemble Framework "
        "with Explainability for Early Heart Disease Risk Stratification'  ·  "
        f"Generated: {now}",
        S["footer"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
