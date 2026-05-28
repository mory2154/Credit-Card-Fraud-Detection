"""
Fonctions utilitaires pour l'app Streamlit.
"""
import numpy as np
import pandas as pd
import joblib
import json
import shap
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"


@st_cache_resource
def load_artifacts():
    """Charge tous les artefacts (mis en cache par Streamlit)."""
    import streamlit as st
    model        = joblib.load(MODELS_DIR / "pipeline_final.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    with open(MODELS_DIR / "feature_config.json")  as f: feat_cfg  = json.load(f)
    with open(MODELS_DIR / "final_model_config.json") as f: final_cfg = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, preprocessor, feat_cfg, final_cfg, explainer


def st_cache_resource(func):
    """Wrapper — remplacé par @st.cache_resource dans main.py."""
    return func


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les features dérivées (identique Phase 4)."""
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = (df['Time'] // 3600) % 24
    df['hour_sin']           = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']           = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_night']           = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    bins                     = [-np.inf, 1, 10, 100, np.inf]
    df['amount_bin']         = pd.cut(df['Amount'], bins=bins,
                                      labels=[0, 1, 2, 3]).astype(int)
    df['high_amount']        = (df['Amount'] > 9.82).astype(int)
    df['night_x_log_amount'] = df['is_night'] * np.log1p(df['Amount'])
    return df


def predict_transaction(row_df, model, preprocessor, feat_cfg):
    """
    Prédit le score de fraude pour une transaction (DataFrame 1 ligne).
    Retourne (score, features_array).
    """
    row_df = add_engineered_features(row_df)
    FEAT_COLS = feat_cfg['all_feat_cols']
    X = preprocessor.transform(row_df[FEAT_COLS])
    score = float(model.predict_proba(X)[0, 1])
    return score, X


def get_shap_local(X_row, explainer, feat_names):
    """
    Calcule les valeurs SHAP pour une seule transaction.
    Retourne un DataFrame trié par |SHAP| décroissant.
    """
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()
    df = pd.DataFrame({
        'feature': feat_names,
        'shap'   : sv,
        'abs'    : np.abs(sv),
        'value'  : X_row.flatten(),
    }).sort_values('abs', ascending=False)
    return df


def score_to_risk(score, threshold):
    """Convertit un score en niveau de risque."""
    if score >= threshold:
        if score >= 0.8:
            return "🔴 RISQUE TRÈS ÉLEVÉ", "error"
        return "🟠 RISQUE ÉLEVÉ", "warning"
    elif score >= threshold * 0.6:
        return "🟡 RISQUE MODÉRÉ", "warning"
    return "🟢 RISQUE FAIBLE", "success"
