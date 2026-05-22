"""
data/loader.py
==============
Loads the UCI Cleveland Heart Disease dataset.

Priority:
  1. HuggingFace Datasets Hub  — imodels/heart_disease
  2. Bundled CSV fallback       — data/cleveland_heart.csv

Returns
-------
X_train, X_test, y_train, y_test, feature_names, scaler
"""

from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=M, 0=F)",
    "cp":       "Chest Pain Type (0–3)",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar >120",
    "restecg":  "Resting ECG (0–2)",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope (0–2)",
    "ca":       "Major Vessels (0–3)",
    "thal":     "Thalassemia (1–3)",
}

# ---------------------------------------------------------------------------
# CSV fallback path (relative to project root or this file's parent)
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent
_CSV_PATH = _HERE / "cleveland_heart.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_from_huggingface() -> pd.DataFrame:
    """Attempt to pull the dataset from HuggingFace Hub."""
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("imodels/heart_disease", split="train")
    df = ds.to_pandas()
    return df


def _load_from_csv() -> pd.DataFrame:
    """Load the bundled Cleveland CSV file."""
    col_names = FEATURE_NAMES + ["target"]
    df = pd.read_csv(
        _CSV_PATH,
        names=col_names,
        header=0,          # first row is our header
        na_values=["?"],
    )
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, fill missing values, and binarise the target."""
    # Replace '?' already handled via na_values; fill numeric NaN with median
    for col in FEATURE_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Binarise target: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    # Keep only expected columns (HF dataset may have extra)
    cols = FEATURE_NAMES + ["target"]
    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    return df


def load_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    list[str],
    StandardScaler,
]:
    """
    Load, clean, split, and scale the heart disease dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # --- 1. Load ---
    df: pd.DataFrame | None = None
    source = "unknown"

    try:
        df = _load_from_huggingface()
        source = "HuggingFace"
        print(f"[loader] Dataset loaded from HuggingFace ({len(df)} rows)")
    except Exception as hf_err:
        print(f"[loader] HuggingFace unavailable ({hf_err}). Using bundled CSV.")

    if df is None:
        df = _load_from_csv()
        source = "CSV"
        print(f"[loader] Dataset loaded from bundled CSV ({len(df)} rows)")

    # --- 2. Clean ---
    df = _clean(df)
    print(f"[loader] After cleaning: {df.shape} | source={source}")

    # --- 3. Split ---
    feature_names = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- 4. Scale ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def load_raw_dataframe() -> pd.DataFrame:
    """Return the full cleaned DataFrame (for Explorer page)."""
    try:
        df = _load_from_huggingface()
    except Exception:
        df = _load_from_csv()
    return _clean(df)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, sc = load_data()
    print("X_train shape:", X_tr.shape)
    print("Features:", feats)
    print("Target distribution:\n", pd.concat([y_tr, y_te]).value_counts())
