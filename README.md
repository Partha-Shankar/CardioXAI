# CardioXAI — Clinical Heart Disease Risk Assessment

> An explainable multi-model ensemble framework for early cardiovascular risk stratification.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**CardioXAI** combines Random Forest, XGBoost, and LightGBM in a confidence-weighted soft-voting ensemble to predict heart disease risk. SHAP provides patient-level explainability. The Streamlit interface delivers an Apple-inspired clinical UI with PDF report generation.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (models auto-train on first launch)
streamlit run app.py
```

App opens at **http://localhost:8501**

> ⏱️ First-run model training takes approximately 30–60 seconds.

---

## Directory Structure

```
cardioxai/
├── app.py                     # Streamlit entry point (4 pages)
├── requirements.txt
├── README.md
├── models/
│   ├── train.py               # Training script (auto-runs on first launch)
│   ├── ensemble.py            # CardioEnsemble — weighted soft voting
│   └── saved/                 # joblib model files + metrics.json
├── data/
│   ├── loader.py              # HuggingFace + CSV fallback loader
│   └── cleveland_heart.csv    # Bundled UCI Cleveland dataset (303 rows)
├── explainability/
│   └── shap_explainer.py      # TreeExplainer + Plotly waterfall chart
├── reports/
│   └── pdf_generator.py       # ReportLab 8-section clinical PDF
├── utils/
│   └── ui_helpers.py          # CSS injection + card renderers
└── assets/
    └── logo.svg               # CardioXAI logo
```

---

## App Pages

| Page | Description |
|------|-------------|
| 🫀 Risk Assessment | Input patient parameters, get ensemble prediction + SHAP waterfall |
| 📊 Model Performance | ROC curves, confusion matrix, feature importance, metric table |
| 🔬 Dataset Explorer | Histograms, correlation heatmap, class distribution |
| 📄 About & IEEE Paper | Abstract, methodology diagram, citation block |

---

## Models

| Model | Estimators | Max Depth | Learning Rate |
|-------|-----------|-----------|---------------|
| Random Forest | 200 | 8 | — |
| XGBoost | 200 | 5 | 0.05 |
| LightGBM | 200 | 6 | 0.05 |

Ensemble weights: **AUC-ROC proportional** (per-model AUC ÷ total AUC sum)

---

## Risk Stratification

| Tier | Probability | Action |
|------|-------------|--------|
| 🟢 Low | < 25% | Annual screening |
| 🟡 Moderate | 25–50% | Cardiologist within 3 months |
| 🔴 High | 50–75% | Urgent referral + stress test |
| 🟣 Critical | ≥ 75% | Immediate evaluation |

---

## Dataset

**UCI Cleveland Heart Disease** — 303 patients, 13 features, binary target.

- Primary: `datasets.load_dataset("imodels/heart_disease", split="train")`
- Fallback: `data/cleveland_heart.csv` (bundled, works offline)

---

## IEEE Paper Reference

> **CardioXAI: A Multi-Model Ensemble Framework with Explainability for Early Heart Disease Risk Stratification**  
> *IEEE Transactions on Biomedical Engineering, 2024*

---

## Disclaimer

This system is for **educational and research purposes only**.  
It is **not** a substitute for professional medical diagnosis, advice, or treatment.

---

## License

MIT © 2024 CardioXAI
