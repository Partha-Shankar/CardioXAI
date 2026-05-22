"""
models/train.py
===============
Train RF, XGBoost, LightGBM on the heart disease dataset.
Run once; saved models are reused by the Streamlit app.

Usage:
    python models/train.py
    -- or --
    from models.train import train_all_models; train_all_models()
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allow running from project root or models/ directory
_HERE = pathlib.Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from data.loader import load_data  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAVED_DIR = _HERE / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_models() -> dict:
    from xgboost import XGBClassifier  # type: ignore
    from lightgbm import LGBMClassifier  # type: ignore
    from tabpfn import TabPFNClassifier # type: ignore

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "TabPFN": TabPFNClassifier(
            device='cpu'
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "Random Forest": SAVED_DIR / "rf_model.joblib",
    "XGBoost":       SAVED_DIR / "xgb_model.joblib",
    "LightGBM":      SAVED_DIR / "lgbm_model.joblib",
    "TabPFN":        SAVED_DIR / "tabpfn_model.joblib",
}
SCALER_FILE  = SAVED_DIR / "scaler.joblib"
METRICS_FILE = SAVED_DIR / "metrics.json"


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    from sklearn.model_selection import cross_val_score
    
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Use 5-fold CV for the AUC-ROC to get a more realistic metric
    cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc":   round(float(cv_auc), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    return metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_all_models(force: bool = False) -> dict:
    """
    Train and save all models.

    Parameters
    ----------
    force : bool
        If True, retrain even if saved files already exist.

    Returns
    -------
    metrics : dict  {model_name: {metric: value}}
    """
    if not force and METRICS_FILE.exists():
        print("[train] Models already trained. Loading metrics from disk.")
        with open(METRICS_FILE) as f:
            return json.load(f)

    print("[train] Loading dataset ...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_data()

    # Save scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"[train] Scaler saved -> {SCALER_FILE}")

    models   = _make_models()
    all_metrics: dict[str, dict] = {}

    for name, model in models.items():
        print(f"\n[train] Fitting {name} ...")
        model.fit(X_train, y_train)

        # Save model
        path = MODEL_FILES[name]
        joblib.dump(model, path)
        print(f"[train] {name} saved -> {path}")

        # Evaluate
        metrics = _evaluate(model, X_train, y_train, X_test, y_test, name)
        all_metrics[name] = metrics

    # Save metrics JSON
    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[train] Metrics saved -> {METRICS_FILE}")

    return all_metrics


def models_exist() -> bool:
    """Return True if all three model files are present on disk."""
    return all(p.exists() for p in MODEL_FILES.values()) and SCALER_FILE.exists()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CardioXAI models")
    parser.add_argument("--force", action="store_true", help="Retrain even if models exist")
    args = parser.parse_args()
    train_all_models(force=args.force)
    print("\n[train] Done.")
