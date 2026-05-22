"""
models/ensemble.py
==================
Weighted soft-voting ensemble over RF, XGBoost, and LightGBM.
Weights are derived from per-model AUC-ROC stored in metrics.json.
"""

from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import pandas as pd

_HERE    = pathlib.Path(__file__).parent
SAVED    = _HERE / "saved"

MODEL_FILES = {
    "Random Forest": SAVED / "rf_model.joblib",
    "XGBoost":       SAVED / "xgb_model.joblib",
    "LightGBM":      SAVED / "lgbm_model.joblib",
    "TabPFN":        SAVED / "tabpfn_model.joblib",
}
SCALER_FILE  = SAVED / "scaler.joblib"
METRICS_FILE = SAVED / "metrics.json"

# Risk tier thresholds
RISK_TIERS = [
    (0.25, "Low",      "#34C759"),   # Apple green
    (0.50, "Moderate", "#FF9F0A"),   # Apple amber
    (0.75, "High",     "#FF3B30"),   # Apple red
    (1.01, "Critical", "#AF52DE"),   # Apple purple
]


class CardioEnsemble:
    """
    Weighted soft-voting ensemble.

    Attributes
    ----------
    models  : dict[str, classifier]
    weights : dict[str, float]      — normalised AUC-ROC weights
    scaler  : StandardScaler
    """

    def __init__(self) -> None:
        self.models:  dict = {}
        self.weights: dict = {}
        self.scaler   = None
        self._loaded  = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "CardioEnsemble":
        """Load models, scaler, and weights from disk."""
        if self._loaded:
            return self

        # Models
        for name, path in MODEL_FILES.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Run `python models/train.py` first."
                )
            self.models[name] = joblib.load(path)

        # Scaler
        self.scaler = joblib.load(SCALER_FILE)

        # Weights from AUC-ROC
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                metrics = json.load(f)
            raw_weights = {
                name: metrics.get(name, {}).get("auc_roc", 1.0)
                for name in self.models
            }
        else:
            raw_weights = {name: 1.0 for name in self.models}

        total = sum(raw_weights.values())
        self.weights = {k: v / total for k, v in raw_weights.items()}

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _scale_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to input data."""
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )

    def predict_proba(self, X_raw: pd.DataFrame) -> dict:
        """
        Run the weighted ensemble prediction.

        Parameters
        ----------
        X_raw : pd.DataFrame  — unscaled feature row(s)

        Returns
        -------
        dict with keys:
            ensemble_prob : float   — final probability [0, 1]
            risk_level    : str     — Low / Moderate / High / Critical
            risk_color    : str     — hex colour for that tier
            model_probs   : dict    — {model_name: prob}
            confidence    : float   — inter-model agreement [0, 1]
        """
        if not self._loaded:
            self.load()

        X_scaled = self._scale_input(X_raw)

        model_probs: dict[str, float] = {}
        weighted_sum = 0.0

        for name, model in self.models.items():
            prob = float(model.predict_proba(X_scaled)[:, 1][0])
            model_probs[name] = round(prob, 4)
            weighted_sum += prob * self.weights[name]

        ensemble_prob = round(min(max(weighted_sum, 0.0), 1.0), 4)

        # Confidence = 1 - std_dev (higher agreement → higher confidence)
        probs_arr   = np.array(list(model_probs.values()))
        confidence  = round(float(1.0 - probs_arr.std()), 4)
        confidence  = max(0.0, min(1.0, confidence))

        risk_level, risk_color = self.risk_stratify(ensemble_prob)

        return {
            "ensemble_prob": ensemble_prob,
            "risk_level":    risk_level,
            "risk_color":    risk_color,
            "model_probs":   model_probs,
            "confidence":    confidence,
        }

    # ------------------------------------------------------------------
    # Risk stratification
    # ------------------------------------------------------------------

    @staticmethod
    def risk_stratify(prob: float) -> tuple[str, str]:
        """
        Map a probability to a clinical risk tier.

        < 0.25   → Low,      #34C759 (green)
        0.25–0.5 → Moderate, #FF9F0A (amber)
        0.50–0.75→ High,     #FF3B30 (red)
        ≥ 0.75   → Critical, #AF52DE (purple)
        """
        for threshold, level, color in RISK_TIERS:
            if prob < threshold:
                return level, color
        return "Critical", "#AF52DE"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return per-model metrics loaded from metrics.json."""
        if not METRICS_FILE.exists():
            return {}
        with open(METRICS_FILE) as f:
            return json.load(f)

    def get_feature_names(self) -> list[str]:
        """Return the feature names from the scaler."""
        if not self._loaded:
            self.load()
        return list(self.scaler.feature_names_in_)

    def models_loaded(self) -> bool:
        return self._loaded
