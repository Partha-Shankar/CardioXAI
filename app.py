"""
app.py  —  CardioXAI: Clinical Heart Disease Risk Assessment
============================================================
Streamlit entry point.
"""

from __future__ import annotations

import json
import pathlib
import sys
import uuid
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="CardioXAI — Clinical Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui_helpers import (
    inject_css,
    render_hero,
    render_risk_result,
    render_metric_card,
    render_model_bars,
    render_shap_legend,
    render_paper_abstract,
    render_citation,
    section_header,
    section_card,
    sidebar_footer,
    APPLE_FONT,
)
from data.loader import FEATURE_LABELS, FEATURE_NAMES, load_raw_dataframe
from models.train import models_exist, train_all_models, SAVED_DIR
from models.ensemble import CardioEnsemble

inject_css()

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="SF Pro Display, -apple-system, sans-serif", color="#1C1C1E"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#F2F2F7", linecolor="#D1D1D6"),
    yaxis=dict(gridcolor="#F2F2F7", linecolor="#D1D1D6"),
)

IEEE_ABSTRACT = """
<b>Abstract</b> — Cardiovascular disease remains the leading cause of global mortality,
necessitating early and accurate diagnostic tools. In this paper, we present
<b>CardioXAI</b>, an explainable multi-model ensemble system for heart disease risk
stratification. Our approach combines Random Forest, XGBoost, LightGBM, and TabPFN
classifiers under a weighted soft-voting scheme, where weights are derived from individual model
cross-validated AUC-ROC performance. To address clinical interpretability — a critical barrier to AI adoption
in healthcare — we integrate SHAP (SHapley Additive exPlanations) to generate
patient-level feature attribution. CardioXAI stratifies patients into four risk tiers
(Low, Moderate, High, Critical). Our key
contributions are: (1) a confidence-weighted ensemble surpassing individual model
baselines, (2) patient-level SHAP explanations enabling clinician trust, (3) integration of TabPFN for tabular meta-learning, and (4) a
four-tier risk stratification schema aligned with clinical practice. Experimental results
demonstrate the feasibility of deploying explainable AI in early cardiovascular risk
assessment.
"""

KEYWORDS = [
    "Heart Disease Prediction", "Ensemble Learning", "XAI",
    "SHAP", "Clinical Decision Support", "TabPFN",
    "XGBoost", "LightGBM",
]

def _init_state() -> None:
    defaults = {
        "prediction_done": False,
        "ensemble_result": None,
        "shap_data":       None,
        "input_df":        None,
        "input_values":    {},
        "patient_id":      f"PT-{uuid.uuid4().hex[:8].upper()}",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

@st.cache_resource(show_spinner=False)
def _load_ensemble() -> CardioEnsemble:
    if not models_exist():
        train_all_models()
    ens = CardioEnsemble()
    ens.load()
    return ens

def _ensure_models_ready() -> bool:
    if not models_exist():
        st.info("🔄 First run: Training ensemble models on the heart disease dataset…")
        with st.spinner("Training RF, XGBoost, LightGBM, and TabPFN — this takes ~30 seconds…"):
            train_all_models()
        st.success("✅ Models trained successfully!")
        st.rerun()
        return False
    return True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_clinical_recommendations(risk_level: str):
    recs = {
        "Low": ("#34C759", "✅ Continue healthy lifestyle. Routine annual cardiovascular screening."),
        "Moderate": ("#FF9F0A", "⚠️ Consult cardiologist within 3 months. Monitor BP and lipids closely."),
        "High": ("#FF3B30", "🔴 Urgent referral recommended. Schedule stress test and lipid panel."),
        "Critical": ("#AF52DE", "🚨 Immediate medical evaluation required. High risk of acute coronary event.")
    }
    color, text = recs.get(risk_level, ("#1C1C1E", "No recommendations available."))
    st.markdown(f"""
    <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 16px; border-radius: 8px; margin-top: 16px; margin-bottom: 16px;">
        <div style="color: {color}; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px;">Clinical Recommendation</div>
        <div style="color: #1C1C1E; font-size: 0.95rem;">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def render_radar_comparison(patient_dict: dict):
    df_raw = load_raw_dataframe()
    features_to_plot = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    
    mean_disease = df_raw[df_raw['target']==1][features_to_plot].mean()
    mean_healthy = df_raw[df_raw['target']==0][features_to_plot].mean()
    max_vals = df_raw[features_to_plot].max()
    
    pat_norm = [patient_dict[f] / max_vals[f] for f in features_to_plot]
    dis_norm = [mean_disease[f] / max_vals[f] for f in features_to_plot]
    hea_norm = [mean_healthy[f] / max_vals[f] for f in features_to_plot]
    
    labels = [FEATURE_LABELS.get(f, f) for f in features_to_plot]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pat_norm + [pat_norm[0]], theta=labels + [labels[0]],
        fill='toself', name='This Patient', line_color='#007AFF'
    ))
    fig.add_trace(go.Scatterpolar(
        r=dis_norm + [dis_norm[0]], theta=labels + [labels[0]],
        name='Avg Disease', line_color='#FF3B30'
    ))
    fig.add_trace(go.Scatterpolar(
        r=hea_norm + [hea_norm[0]], theta=labels + [labels[0]],
        name='Avg Healthy', line_color='#34C759'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

def render_simulator(ensemble, input_dict):
    st.markdown("### Clinical Intervention Simulator")
    st.markdown("<p style='color:#636366;'>What if we change one factor?</p>", unsafe_allow_html=True)
    
    sim_feat = st.selectbox("Select Feature to Modify", ["trestbps", "chol", "thalach", "oldpeak", "age"])
    current_val = input_dict[sim_feat]
    
    bounds = {
        "age": (20, 90),
        "trestbps": (80, 220),
        "chol": (100, 600),
        "thalach": (60, 220),
        "oldpeak": (0.0, 8.0)
    }
    min_val, max_val = bounds[sim_feat]
    
    if isinstance(current_val, float):
        new_val = st.slider(f"New {FEATURE_LABELS.get(sim_feat)}", float(min_val), float(max_val), float(current_val), step=0.1, key="sim_slider_float")
    else:
        new_val = st.slider(f"New {FEATURE_LABELS.get(sim_feat)}", int(min_val), int(max_val), int(current_val), key="sim_slider_int")
        
    if st.button("Recalculate Risk", key="sim_btn"):
        sim_input = input_dict.copy()
        sim_input[sim_feat] = new_val
        feat_order = ensemble.get_feature_names()
        ordered = {k: sim_input[k] for k in feat_order if k in sim_input}
        sim_df = pd.DataFrame([ordered])
        res = ensemble.predict_proba(sim_df)
        
        st.markdown(f"""
        <div style="background:#F2F2F7; padding: 12px; border-radius: 8px; margin-top: 12px;">
            <div style="color: #636366; font-size: 0.9rem;">Simulated Risk</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1C1C1E;">
                {res['ensemble_prob']*100:.1f}% — <span style="color: {res['risk_color']}">{res['risk_level']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _roc_curves(ensemble: CardioEnsemble) -> go.Figure:
    from sklearn.metrics import roc_curve, auc
    from data.loader import load_data
    X_tr, X_te, y_tr, y_te, feats, scaler = load_data()
    COLORS = {"Random Forest": "#34C759", "XGBoost": "#FF9F0A", "LightGBM": "#007AFF", "TabPFN": "#AF52DE"}
    fig = go.Figure()
    for name, model in ensemble.models.items():
        proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc:.3f})",
            line=dict(color=COLORS.get(name, "#636366"), width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#D1D1D6", dash="dot", width=1.5),
        name="Random Classifier",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="ROC Curves — All Models", font=dict(size=15, color="#1C1C1E")),
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E5EA", borderwidth=1),
        height=400,
    )
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")
    return fig

def _confusion_matrix_fig(ensemble: CardioEnsemble) -> go.Figure:
    from sklearn.metrics import confusion_matrix
    from data.loader import load_data
    X_tr, X_te, y_tr, y_te, feats, scaler = load_data()
    metrics = ensemble.get_metrics()
    best_name = max(metrics, key=lambda n: metrics[n].get("auc_roc", 0))
    model = ensemble.models[best_name]
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    labels = ["No Disease", "Disease"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=cm, texttemplate="%{text}",
        textfont=dict(size=20, color="white"),
        colorscale=[[0, "#EBF2FF"], [1, "#007AFF"]], showscale=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Confusion Matrix — {best_name}", font=dict(size=15, color="#1C1C1E")),
        height=380,
    )
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", autorange="reversed")
    return fig

def _feature_importance_fig(ensemble: CardioEnsemble) -> go.Figure:
    from explainability.shap_explainer import SHAPExplainer
    from data.loader import load_data
    X_tr, X_te, y_tr, y_te, feats, scaler = load_data()
    rf_model = ensemble.models["Random Forest"]
    explainer = SHAPExplainer(rf_model, feats)
    sample = X_te.iloc[:60]
    mean_shap = explainer.compute_mean_shap(sample)
    fig = explainer.plot_feature_importance(mean_shap, FEATURE_LABELS)
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_risk_assessment(ensemble: CardioEnsemble) -> None:
    render_hero("CardioXAI", "Explainable AI-powered cardiovascular risk stratification")
    col_inputs, col_results = st.columns([1.1, 0.9], gap="large")

    with col_inputs:
        with st.container():
            st.markdown('<div class="cxai-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:1.4rem; font-weight:600; color:#1C1C1E; margin-bottom:12px;">Patient Parameters</div>', unsafe_allow_html=True)
            
            section_card("Section A — Demographics", "👤")
            age = st.slider("Age (years)", 20, 90, 55, key="age")
            sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="sex")

            section_card("Section B — Clinical Measurements", "🩺")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: {0: "0 — Typical Angina", 1: "1 — Atypical Angina", 2: "2 — Non-Anginal Pain", 3: "3 — Asymptomatic"}[x], key="cp")
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 220, 130, key="trestbps")
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240, key="chol")
            fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dl", value=False, key="fbs")

            section_card("Section C — ECG & Exercise", "📈")
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: {0: "0 — Normal", 1: "1 — ST-T Wave Abnormality", 2: "2 — Left Ventricular Hypertrophy"}[x], key="restecg")
            thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150, key="thalach")
            exang = st.checkbox("Exercise-Induced Angina", value=False, key="exang")
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 8.0, 1.0, step=0.1, key="oldpeak")
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: {0: "0 — Upsloping", 1: "1 — Flat", 2: "2 — Downsloping"}[x], key="slope")

            section_card("Section D — Advanced Imaging", "🔬")
            ca = st.slider("Number of Major Vessels (Fluoroscopy, 0–3)", 0, 3, 0, key="ca")
            thal = st.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: {1: "1 — Normal", 2: "2 — Fixed Defect", 3: "3 — Reversible Defect"}[x], key="thal")

            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🫀 Analyze Risk →", type="primary", key="analyze_btn")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        if run_btn:
            input_dict = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": int(fbs), "restecg": restecg,
                "thalach": thalach, "exang": int(exang), "oldpeak": oldpeak,
                "slope": slope, "ca": ca, "thal": thal,
            }
            feat_order = ensemble.get_feature_names()
            ordered = {k: input_dict[k] for k in feat_order if k in input_dict}
            X_input = pd.DataFrame([ordered])

            with st.spinner("Running ensemble & SHAP analysis…"):
                result = ensemble.predict_proba(X_input)
                from explainability.shap_explainer import SHAPExplainer
                rf_model = ensemble.models["Random Forest"]
                explainer = SHAPExplainer(rf_model, feat_order)
                shap_data = explainer.get_shap_values(X_input)

            st.session_state.prediction_done = True
            st.session_state.ensemble_result = result
            st.session_state.shap_data = shap_data
            st.session_state.input_df = X_input
            st.session_state.input_values = input_dict

        if st.session_state.prediction_done:
            result = st.session_state.ensemble_result
            prob = result["ensemble_prob"]
            color = result["risk_color"]
            level = result["risk_level"]

            render_risk_result(prob, level, color, result["confidence"])
            render_clinical_recommendations(level)
            render_model_bars(result["model_probs"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Patient Profile Comparison")
            render_radar_comparison(st.session_state.input_values)

            # PDF Download
            st.markdown("<br>", unsafe_allow_html=True)
            from reports.pdf_generator import generate_pdf
            pdf_bytes = generate_pdf(
                patient_id=st.session_state.patient_id,
                input_values=st.session_state.input_values,
                ensemble_result=result,
                shap_data=st.session_state.shap_data,
                feature_labels=FEATURE_LABELS,
            )
            st.download_button(
                label="📄 Download Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"CardioXAI_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )
        else:
            st.markdown("""
            <div class="cxai-card" style="text-align:center; padding:64px 24px;">
                <div style="font-size:4rem; margin-bottom:12px; animation: pulse 2s infinite;">🫀</div>
                <div style="font-size:1.1rem; font-weight:600; color:#1C1C1E; margin-bottom:8px;">Ready for Analysis</div>
                <div style="font-size:0.9rem; color:#636366;">Complete the form and click<br><strong>Analyze Risk →</strong> to get results</div>
                <style>@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }</style>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.prediction_done and st.session_state.shap_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="cxai-card">', unsafe_allow_html=True)
        section_header("🧠 Key Risk Factors Driving This Prediction", "SHAP (SHapley Additive exPlanations) — patient-level feature attribution")
        render_shap_legend()

        from explainability.shap_explainer import SHAPExplainer
        rf_model = ensemble.models["Random Forest"]
        feat_order = ensemble.get_feature_names()
        explainer = SHAPExplainer(rf_model, feat_order)
        fig = explainer.plot_waterfall(st.session_state.shap_data, FEATURE_LABELS)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
        render_simulator(ensemble, st.session_state.input_values)
        st.markdown('</div>', unsafe_allow_html=True)


def page_model_performance(ensemble: CardioEnsemble) -> None:
    st.markdown('<div style="font-size:1.8rem; font-weight:700; color:#1C1C1E; margin-bottom:24px;">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)
    metrics = ensemble.get_metrics()
    if not metrics:
        st.warning("No metrics found. Please retrain models.")
        return

    best_acc = max(m.get("accuracy", 0) for m in metrics.values())
    best_auc = max(m.get("auc_roc", 0) for m in metrics.values())
    avg_f1 = np.mean([m.get("f1", 0) for m in metrics.values()])
    
    best_auc_model = max(metrics, key=lambda n: metrics[n].get("auc_roc", 0))

    cols = st.columns(4, gap="small")
    with cols[0]: render_metric_card("Best Accuracy", f"{best_acc:.1%}", "Across models")
    with cols[1]: render_metric_card("Best CV AUC-ROC", f"{best_auc:.3f}", best_auc_model)
    with cols[2]: render_metric_card("Avg F1 Score", f"{avg_f1:.3f}", "Across models")
    with cols[3]: render_metric_card("Dataset Size", "303", "UCI Cleveland")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Detailed Model Metrics")
    
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name + " 🤗" if name == "TabPFN" else name,
            "Accuracy": m.get("accuracy", 0),
            "AUC-ROC": m.get("auc_roc", 0),
            "F1": m.get("f1", 0),
        })
    metrics_df = pd.DataFrame(rows)
    
    st.dataframe(
        metrics_df,
        column_config={
            "Model": st.column_config.TextColumn(width="medium"),
            "Accuracy": st.column_config.ProgressColumn(format="%.1%", min_value=0, max_value=1),
            "AUC-ROC": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "F1": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("### ROC Curves")
        with st.spinner("Computing ROC curves…"):
            roc_fig = _roc_curves(ensemble)
        st.plotly_chart(roc_fig, use_container_width=True)
    with col2:
        st.markdown(f"### Confusion Matrix ({best_auc_model})")
        with st.spinner("Building confusion matrix…"):
            cm_fig = _confusion_matrix_fig(ensemble)
        st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔑 Global Feature Importance (Random Forest)")
    with st.spinner("Computing SHAP feature importance…"):
        fi_fig = _feature_importance_fig(ensemble)
    st.plotly_chart(fi_fig, use_container_width=True)


def page_dataset_explorer() -> None:
    st.markdown('<div style="font-size:1.8rem; font-weight:700; color:#1C1C1E; margin-bottom:24px;">🔬 Dataset Explorer</div>', unsafe_allow_html=True)
    with st.spinner("Loading dataset…"):
        df = load_raw_dataframe()

    st.markdown(
        """
        <div class="cxai-card" style="display:flex; align-items:center; gap:16px; padding:16px 24px;">
            <div style="font-size:2rem;">🏥</div>
            <div>
                <div style="font-weight:600; color:#1C1C1E; font-size:0.95rem;">UCI Heart Disease Dataset</div>
                <div style="font-size:0.82rem; color:#636366; margin-top:2px;">
                    Sourced via HuggingFace Datasets (<code>imodels/heart_disease</code>) · Cleveland Clinic Foundation
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: render_metric_card("Total Patients", str(len(df)))
    with col2: render_metric_card("Features", "13")
    with col3: render_metric_card("Disease Cases", str(df["target"].sum()), color="#FF3B30")
    with col4: render_metric_card("Healthy Cases", str((df["target"] == 0).sum()), color="#34C759")

    st.markdown("<br>", unsafe_allow_html=True)
    col_pie, col_age = st.columns(2, gap="medium")

    with col_pie:
        st.markdown("### Target Distribution")
        counts = df["target"].value_counts().reset_index()
        counts.columns = ["target", "count"]
        counts["label"] = counts["target"].map({0: "No Disease", 1: "Heart Disease"})
        pie_fig = go.Figure(go.Pie(
            labels=counts["label"], values=counts["count"], hole=0.5,
            marker=dict(colors=["#34C759", "#FF3B30"], line=dict(color="white", width=2)),
            textinfo="label+percent"
        ))
        pie_fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, annotations=[dict(text=f"<b>{len(df)}</b>", x=0.5, y=0.5, font_size=16, showarrow=False)])
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_age:
        st.markdown("### Age Distribution by Outcome")
        age_fig = go.Figure()
        for target_val, label, color in [(0, "No Disease", "#34C759"), (1, "Disease", "#FF3B30")]:
            age_fig.add_trace(go.Histogram(
                x=df[df["target"] == target_val]["age"], name=label, marker_color=color, opacity=0.75, nbinsx=20
            ))
        age_fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=320, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.05))
        st.plotly_chart(age_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Correlation Heatmap")
    numeric_feats = [f for f in FEATURE_NAMES if f in df.columns]
    corr_df = df[numeric_feats].corr()
    heat_fig = go.Figure(go.Heatmap(
        z=corr_df.values, x=[FEATURE_LABELS.get(c, c) for c in corr_df.columns], y=[FEATURE_LABELS.get(c, c) for c in corr_df.index],
        colorscale=[[0, "#FF3B30"], [0.5, "#FFFFFF"], [1, "#007AFF"]], zmid=0, zmin=-1, zmax=1
    ))
    heat_fig.update_layout(**PLOTLY_LAYOUT, height=480)
    st.plotly_chart(heat_fig, use_container_width=True)


def page_about() -> None:
    st.markdown('<div style="font-size:1.8rem; font-weight:700; color:#1C1C1E; margin-bottom:24px;">📄 About CardioXAI & IEEE Paper</div>', unsafe_allow_html=True)
    st.markdown("### Abstract")
    render_paper_abstract(IEEE_ABSTRACT, KEYWORDS)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="cxai-card">
            <div style="font-size: 1.1rem; font-weight: 600; color: #1C1C1E; margin-bottom: 8px;">🤗 HuggingFace Model Integration</div>
            <div style="color: #636366; line-height: 1.6;">
                TabPFN (Prior-Labs/TabPFN) — A transformer-based Prior-Fitted Network for 
                tabular classification. Unlike tree-based methods, TabPFN performs in-context meta-learning 
                trained on millions of synthetic datasets, achieving strong performance without hyperparameter 
                tuning. Accessed via HuggingFace Hub and integrated as the 4th ensemble member.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Methodology Overview")
    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.markdown("""
            <div class="cxai-card">
            <div style="font-weight:600; margin-bottom:12px;">🏗️ System Architecture</div>
            <pre style="background:#F2F2F7; border-radius:10px; padding:14px; font-size:0.75rem; color:#1C1C1E; overflow-x:auto;">
Data (UCI Heart Disease)
         ↓
  Preprocessing
         ↓
┌───────────────────────┐
│ Ensemble Predictors   │
│ ├─ Random Forest      │
│ ├─ XGBoost            │
│ ├─ LightGBM           │
│ └─ TabPFN 🤗          │
└───────────────────────┘
         ↓
  Weighted Ensemble (AUC)
         ↓
  Risk Level & SHAP
            </pre>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
            <div class="cxai-card">
            <div style="font-weight:600; margin-bottom:12px;">✨ Key Contributions</div>
            <ul style="font-size:0.9rem; color:#1C1C1E; line-height:1.8; padding-left:18px;">
                <li><b>Confidence-weighted ensemble</b> outperforming individual learners</li>
                <li><b>Patient-level SHAP explanations</b> for clinical trust</li>
                <li><b>Four-tier risk stratification</b> aligned with practice guidelines</li>
                <li><b>TabPFN transformer integration</b> for robust zero-shot meta-learning</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### IEEE Citation (BibTeX)")
    st.code("""
@article{cardioxai2025,
  title={CardioXAI: A Multi-Model Ensemble Framework with Explainability 
         for Early Heart Disease Risk Stratification},
  author={[Your Name]},
  journal={IEEE Access / IEEE ICHI 2025},
  year={2025},
  note={Submitted}
}
    """, language="bibtex")

def page_batch_prediction(ensemble: CardioEnsemble) -> None:
    st.markdown('<div style="font-size:1.8rem; font-weight:700; color:#1C1C1E; margin-bottom:24px;">📂 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#636366;'>Run the ensemble model over a batch of patients via CSV upload.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV containing patient features", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Running ensemble for all rows..."):
                results = []
                features = [f for f in FEATURE_NAMES if f in df.columns]
                
                if len(features) < len(FEATURE_NAMES):
                    st.warning(f"Missing some features. Found {len(features)}/{len(FEATURE_NAMES)}.")
                    
                input_df = df[features]
                
                for i in range(len(input_df)):
                    row = pd.DataFrame([input_df.iloc[i]])
                    # ensure columns match expected by setting the subset
                    missing = set(FEATURE_NAMES) - set(row.columns)
                    for m in missing:
                        row[m] = 0 # Dummy fill for missing if any (shouldn't happen in clean data)
                    
                    ordered = row[ensemble.get_feature_names()]
                    res = ensemble.predict_proba(ordered)
                    results.append({
                        "Patient_Index": i,
                        "Ensemble_Prob": res["ensemble_prob"],
                        "Risk_Level": res["risk_level"],
                        "Confidence": res["confidence"]
                    })
                
                res_df = pd.DataFrame(results)
                
                def style_risk(val):
                    color = "#34C759" if val == "Low" else "#FF9F0A" if val == "Moderate" else "#FF3B30" if val == "High" else "#AF52DE"
                    return f'background-color: {color}; color: white;'

                styled_df = res_df.style.map(style_risk, subset=['Risk_Level'])
                st.dataframe(styled_df, use_container_width=True)
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=csv,
                    file_name='cardioxai_batch_predictions.csv',
                    mime='text/csv',
                )

# ---------------------------------------------------------------------------
# Sidebar + main router
# ---------------------------------------------------------------------------

def main() -> None:
    if not _ensure_models_ready():
        return

    ensemble = _load_ensemble()

    st.sidebar.markdown(
        """
        <div style="padding:12px 0 20px; border-bottom:1px solid #D1D1D6; margin-bottom:20px;">
            <div style="font-size:1.3rem; font-weight:700; color:#1C1C1E; letter-spacing:-0.5px;">
                🫀 CardioXAI
            </div>
            <div style="font-size:0.78rem; color:#636366; margin-top:4px;">
                Clinical Risk Assessment System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🫀 Risk Assessment",
            "📊 Model Performance",
            "🔬 Dataset Explorer",
            "📂 Batch Prediction",
            "📄 About & IEEE Paper",
        ],
        label_visibility="collapsed",
    )

    metrics = ensemble.get_metrics()
    if metrics:
        best_auc = max(m.get("auc_roc", 0) for m in metrics.values())
        st.sidebar.markdown(
            f"""
            <div style="margin:16px 0 8px; padding:10px 14px; background:#F0FFF4;
                        border:1px solid #BBF7D0; border-radius:12px;">
                <div style="font-size:0.72rem; color:#166534; font-weight:600;">
                    ✅ Models Loaded
                </div>
                <div style="font-size:0.82rem; color:#1C1C1E; margin-top:2px;">
                    Best CV AUC-ROC: <b>{best_auc:.3f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sidebar_footer()

    # Isolated routing blocks to prevent page bleed
    if page == "🫀 Risk Assessment":
        page_risk_assessment(ensemble)
    elif page == "📊 Model Performance":
        page_model_performance(ensemble)
    elif page == "🔬 Dataset Explorer":
        page_dataset_explorer()
    elif page == "📂 Batch Prediction":
        page_batch_prediction(ensemble)
    elif page == "📄 About & IEEE Paper":
        page_about()

if __name__ == "__main__":
    main()
