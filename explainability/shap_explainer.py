"""
explainability/shap_explainer.py
================================
SHAP-based explainability wrapper for CardioXAI.

Uses TreeExplainer on the Random Forest model and returns
Plotly-compatible visualisations (no matplotlib).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_HERE = pathlib.Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

APPLE_FONT = "'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"

# Colour palette — Apple status colours
COLOR_RISK_INCREASE = "#FF3B30"   # Apple red
COLOR_RISK_DECREASE = "#34C759"   # Apple green
COLOR_NEUTRAL       = "#636366"   # Apple secondary label


class SHAPExplainer:
    """
    Thin SHAP wrapper around the Random Forest model.

    Parameters
    ----------
    model      : fitted RandomForestClassifier
    feature_names : list[str]
    """

    def __init__(self, model, feature_names: list[str]) -> None:
        self.model         = model
        self.feature_names = feature_names
        self._explainer    = None

    def _get_explainer(self):
        if self._explainer is None:
            import shap  # type: ignore
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    # ------------------------------------------------------------------
    # Core SHAP computation
    # ------------------------------------------------------------------

    def get_shap_values(self, X_input: pd.DataFrame) -> dict:
        """
        Compute SHAP values for a single input row.

        Parameters
        ----------
        X_input : pd.DataFrame  — scaled feature row(s)

        Returns
        -------
        dict with keys:
            shap_values    : np.ndarray   — shape (n_features,)
            base_value     : float
            feature_names  : list[str]
            feature_values : list[float]  — original input values
            top_features   : list of (feature_name, shap_val, direction)
                             Top 8 sorted by |shap_val|, direction ∈ {"↑","↓"}
        """
        import shap  # type: ignore

        explainer = self._get_explainer()
        sv        = explainer(X_input)

        # For binary classification: pick class-1 SHAP values
        if len(sv.values.shape) == 3:
            # shape: (n_samples, n_features, n_classes)
            shap_vals = sv.values[0, :, 1]
            base_val  = float(sv.base_values[0][1])
        else:
            shap_vals = sv.values[0]
            base_val  = float(sv.base_values[0])

        feat_vals = X_input.iloc[0].tolist()

        # Top-8 features by absolute SHAP value
        indices    = np.argsort(np.abs(shap_vals))[::-1][:8]
        top_feats  = [
            (
                self.feature_names[i],
                float(shap_vals[i]),
                "↑" if shap_vals[i] > 0 else "↓",
            )
            for i in indices
        ]

        return {
            "shap_values":    shap_vals,
            "base_value":     base_val,
            "feature_names":  self.feature_names,
            "feature_values": feat_vals,
            "top_features":   top_feats,
        }

    # ------------------------------------------------------------------
    # Plotly waterfall chart
    # ------------------------------------------------------------------

    def plot_waterfall(self, shap_data: dict, feature_labels: dict | None = None) -> go.Figure:
        """
        Horizontal bar chart of the top SHAP contributions (waterfall style).

        Parameters
        ----------
        shap_data      : output of get_shap_values()
        feature_labels : optional human-readable label mapping

        Returns
        -------
        plotly.graph_objects.Figure
        """
        top = shap_data["top_features"]
        labels = feature_labels or {}

        names  = [labels.get(f, f) for f, _, _ in top]
        values = [v for _, v, _ in top]
        colors = [COLOR_RISK_INCREASE if v > 0 else COLOR_RISK_DECREASE for v in values]

        # Reverse so largest bar is at the top
        names  = names[::-1]
        values = values[::-1]
        colors = colors[::-1]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=names,
                x=values,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
                textfont=dict(family=APPLE_FONT, size=11, color="#1D1D1F"),
            )
        )

        fig.add_vline(x=0, line_width=1, line_color="#C7C7CC")

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family=APPLE_FONT, size=12, color="#1D1D1F"),
            margin=dict(l=20, r=60, t=40, b=20),
            xaxis=dict(
                title="SHAP Value (impact on model output)",
                title_font=dict(size=11, color="#6E6E73"),
                tickfont=dict(size=10),
                gridcolor="#F5F5F7",
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=11),
                gridcolor="rgba(0,0,0,0)",
            ),
            height=max(300, 50 * len(top) + 60),
            showlegend=False,
            bargap=0.35,
        )

        return fig

    # ------------------------------------------------------------------
    # Mean absolute SHAP for feature importance (across test set)
    # ------------------------------------------------------------------

    def compute_mean_shap(self, X_set: pd.DataFrame) -> dict[str, float]:
        """
        Compute mean |SHAP| across a dataset.

        Returns {feature_name: mean_abs_shap}
        """
        import shap  # type: ignore

        explainer = self._get_explainer()
        sv        = explainer(X_set)

        if len(sv.values.shape) == 3:
            shap_matrix = sv.values[:, :, 1]
        else:
            shap_matrix = sv.values

        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        return dict(zip(self.feature_names, mean_abs.tolist()))

    def plot_feature_importance(
        self,
        mean_shap: dict[str, float],
        feature_labels: dict | None = None,
    ) -> go.Figure:
        """Bar chart of mean |SHAP| — global feature importance."""
        labels   = feature_labels or {}
        sorted_items = sorted(mean_shap.items(), key=lambda x: x[1])

        names    = [labels.get(k, k) for k, _ in sorted_items]
        values   = [v for _, v in sorted_items]
        n_feats  = len(names)

        # Colour gradient: more important → deeper blue
        max_v   = max(values) if values else 1
        colours = [
            f"rgba(0, 113, 227, {0.3 + 0.7 * v / max_v:.2f})"
            for v in values
        ]

        fig = go.Figure(
            go.Bar(
                y=names,
                x=values,
                orientation="h",
                marker=dict(color=colours, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family=APPLE_FONT, size=12, color="#1D1D1F"),
            margin=dict(l=20, r=40, t=40, b=20),
            xaxis=dict(
                title="Mean |SHAP Value|",
                title_font=dict(size=11, color="#6E6E73"),
                gridcolor="#F5F5F7",
                zeroline=False,
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=max(350, 40 * n_feats + 60),
            showlegend=False,
            bargap=0.3,
        )
        return fig
