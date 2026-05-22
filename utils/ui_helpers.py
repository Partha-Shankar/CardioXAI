"""
utils/ui_helpers.py
===================
Apple-inspired CSS injection and reusable UI components for CardioXAI.
"""

from __future__ import annotations

import streamlit as st

APPLE_FONT = "'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"

GLOBAL_CSS = """
<style>
/* ── Reset & Base ── */
.stApp { background: #F2F2F7 !important; }
.block-container { 
    max-width: 1280px !important; 
    padding: 2rem 2rem 4rem !important; 
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 0.5px solid #D1D1D6 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ── Sidebar Radio (nav) ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #1C1C1E !important;
    transition: background 0.15s ease !important;
    cursor: pointer !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #F2F2F7 !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: #EBF2FF !important;
    color: #007AFF !important;
}

/* ── Sliders ── */
.stSlider > div > div > div > div { background: #007AFF !important; }
.stSlider > div > div > div { background: #E5E5EA !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    border: 1px solid #D1D1D6 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: #007AFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: -0.2px !important;
    width: 100% !important;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 2px 8px rgba(0,122,255,0.3) !important;
}
.stButton > button:hover {
    background: #0056CC !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,122,255,0.4) !important;
}
.stButton > button:active {
    transform: scale(0.98) translateY(0) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: #34C759 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(52,199,89,0.3) !important;
}

/* ── Expanders — fix icon bug ── */
[data-testid="stExpander"] summary { list-style: none !important; }
[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: 16px !important; }

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C7C7CC; border-radius: 3px; }

/* ── Custom Component Styles ── */
.cxai-card {
    background: #FFFFFF;
    border: 1px solid #D1D1D6;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.cxai-metric-mini {
    background: #FFFFFF;
    border: 1px solid #D1D1D6;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.cxai-metric-mini .val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1C1C1E;
    line-height: 1.1;
}
.cxai-metric-mini .lbl {
    font-size: 0.78rem;
    color: #636366;
    margin-top: 4px;
    font-weight: 500;
}
.cxai-metric-mini .sublbl {
    font-size: 0.7rem;
    color: #AEAEB2;
}

.cxai-shap-legend {
    display: flex;
    gap: 20px;
    margin-bottom: 8px;
}
.cxai-shap-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #636366;
}
.cxai-shap-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.cxai-abstract {
    background: #FFFFFF;
    border: 1px solid #D1D1D6;
    border-left: 4px solid #007AFF;
    border-radius: 16px;
    padding: 24px 28px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #1C1C1E;
}
.cxai-citation {
    background: #FBFBFD;
    border: 1px solid #D1D1D6;
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #636366;
}
.cxai-tag {
    display: inline-block;
    background: #EBF2FF;
    color: #007AFF;
    border: 1px solid #D1D1D6;
    border-radius: 980px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap" rel="stylesheet">
"""

def inject_css() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<div style="font-size:0.85rem; color:#636366; margin-bottom:16px;">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div style="margin-bottom:12px">'
        f'  <div style="font-size:1.1rem; font-weight:600; color:#1C1C1E; margin-bottom:4px;">{title}</div>'
        f'  {sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

def section_card(title: str, icon: str = ""):
    st.markdown(f"""
<div style="padding-top:16px; padding-bottom:8px; border-top: 1px solid #D1D1D6; margin-top:16px; margin-bottom: 8px;">
<div style="font-size:1.05rem; font-weight:600; color:#1C1C1E;">
            {icon} {title}
</div>
</div>
    """, unsafe_allow_html=True)

def render_hero(title: str = "CardioXAI", subtitle: str = "") -> None:
    st.markdown(f"""
<div class="cxai-card" style="text-align:center; padding: 32px; background: linear-gradient(135deg, #FFFFFF 0%, #F0F4FF 100%);">
<div style="font-size:0.9rem; font-weight:600; color:#007AFF; margin-bottom:8px; letter-spacing:0.5px;">
            CardioXAI · Clinical Risk Assessment
</div>
<h1 style="font-size:2.4rem; font-weight:700; color:#1C1C1E; letter-spacing:-0.5px; margin-bottom:8px; margin-top:0;">
            Heart Disease Risk Prediction
</h1>
<p style="font-size:1.05rem; color:#636366; margin:0;">
            4-model ensemble · SHAP explainability · Clinical stratification
</p>
</div>
    """, unsafe_allow_html=True)

def render_risk_result(prob: float, risk_level: str, color: str, confidence: float):
    pct = int(prob * 100)
    conf_pct = int(confidence * 100)
    
    st.markdown(f"""
<div class="cxai-card" style="text-align:center;">
<div style="font-size:1.1rem; font-weight:600; color:#1C1C1E; margin-bottom:20px;">
            Risk Assessment Result
</div>
<div style="display:flex; justify-content:center; align-items:center; flex-direction:column; gap:8px;">
<div style="width:140px; height:140px; border-radius:50%; display:flex; align-items:center; justify-content:center; border:8px solid {color}; background:#FBFBFD;">
<span style="font-size:2.5rem; font-weight:700; color:#1C1C1E;">{pct}%</span>
</div>
<div style="font-size:1.2rem; font-weight:600; color:{color}; margin-top:8px;">
                {risk_level} Risk
</div>
<div style="font-size:0.9rem; color:#636366;">
                {pct}% probability of heart disease
</div>
<div style="font-size:0.85rem; font-weight:500; color:#636366; margin-top:8px; padding: 4px 12px; background:#F2F2F7; border-radius: 8px;">
                Confidence: ±{100 - conf_pct}%
</div>
</div>
</div>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, sublabel: str = "", color: str = "#007AFF"):
    st.markdown(f"""
<div class="cxai-metric-mini">
<div class="val" style="color:{color};">{value}</div>
<div class="lbl">{label}</div>
<div class="sublbl">{f'{sublabel}' if sublabel else ''}</div>
</div>
    """, unsafe_allow_html=True)

def render_model_bars(model_probs: dict):
    model_colors = {
        "Random Forest": "#34C759",
        "XGBoost": "#FF9F0A", 
        "LightGBM": "#007AFF",
        "TabPFN": "#AF52DE"
    }
    bars_html = ""
    for name, prob in model_probs.items():
        color = model_colors.get(name, "#007AFF")
        pct = int(prob * 100)
        display_name = name + " 🤗" if name == "TabPFN" else name
        bars_html += f"""
<div style="display:flex; align-items:center; margin-bottom:12px; gap: 12px;">
<div style="width:120px; font-size:0.9rem; font-weight:500; color:#1C1C1E; text-align:right;">
                {display_name}
</div>
<div style="flex:1; background:#E5E5EA; border-radius:980px; height:10px; overflow:hidden;">
<div style="width:{pct}%; background:{color}; height:100%; border-radius:980px;"></div>
</div>
<div style="width:40px; font-size:0.9rem; font-weight:600; color:#636366;">
                {pct}%
</div>
</div>
        """
    
    st.markdown(f"""
<div class="cxai-card">
<div style="font-size:1.1rem; font-weight:600; color:#1C1C1E; margin-bottom:16px; text-align:center;">
            Model Agreement
</div>
        {bars_html}
</div>
    """, unsafe_allow_html=True)

def render_shap_legend() -> None:
    st.markdown(
        """
<div class="cxai-shap-legend">
<div class="cxai-shap-legend-item">
<div class="cxai-shap-dot" style="background:#FF3B30;"></div>
                Increases disease risk
</div>
<div class="cxai-shap-legend-item">
<div class="cxai-shap-dot" style="background:#34C759;"></div>
                Decreases disease risk
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

def render_paper_abstract(abstract: str, keywords: list[str] | None = None) -> None:
    kw_html = ""
    if keywords:
        tags = "".join(f'<span class="cxai-tag">{kw}</span>' for kw in keywords)
        kw_html = f'<div style="margin-top:14px;">{tags}</div>'

    st.markdown(
        f'<div class="cxai-abstract">{abstract}{kw_html}</div>',
        unsafe_allow_html=True,
    )

def render_citation(text: str) -> None:
    st.markdown(
        f'<div class="cxai-citation">{text}</div>',
        unsafe_allow_html=True,
    )

def sidebar_footer() -> None:
    st.sidebar.markdown(
        """
<div style="position:fixed; bottom:20px; left:0; width:260px;
                    text-align:center; font-size:0.72rem; color:#AEAEB2;
                    border-top:1px solid #E5E5EA; padding-top:10px;">
            CardioXAI<br>Developed by Partha Shankar
</div>
        """,
        unsafe_allow_html=True,
    )
