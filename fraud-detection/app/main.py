"""
Application Streamlit — Détection de fraude par carte bancaire
Phase 12 du projet Credit Card Fraud Detection (ULB / Worldline)

Usage : streamlit run app/main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
import joblib, json, shap, warnings

from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              classification_report)
from src.config      import COST_FALSE_POSITIVE, TOTAL_FRAUD_AMOUNT
from src.cost_metric import compute_cost, optimal_threshold

warnings.filterwarnings('ignore')

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
MODELS   = ROOT / "models"
DATA     = ROOT / "data"

# ── Chargement des artefacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load(MODELS / "pipeline_final.joblib")
    preprocessor = joblib.load(MODELS / "preprocessor.joblib")
    with open(MODELS / "feature_config.json")     as f: feat_cfg  = json.load(f)
    with open(MODELS / "final_model_config.json") as f: final_cfg = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, preprocessor, feat_cfg, final_cfg, explainer

@st.cache_data
def load_test_data():
    return pd.read_parquet(DATA / "test.parquet")

# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = (df['Time'] // 3600) % 24
    df['hour_sin']           = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']           = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_night']           = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    bins                     = [-np.inf, 1, 10, 100, np.inf]
    df['amount_bin']         = pd.cut(df['Amount'], bins=bins,
                                      labels=[0,1,2,3]).astype(int)
    df['high_amount']        = (df['Amount'] > 9.82).astype(int)
    df['night_x_log_amount'] = df['is_night'] * np.log1p(df['Amount'])
    return df

def predict(row_df, model, preprocessor, feat_cols):
    row_df = add_features(row_df)
    X = preprocessor.transform(row_df[feat_cols])
    score = float(model.predict_proba(X)[0, 1])
    return score, X

def score_label(score, thr):
    if score >= thr:
        return ("🔴 FRAUDE PROBABLE",  "error")   if score >= 0.8 \
          else ("🟠 RISQUE ÉLEVÉ",     "warning")
    elif score >= thr * 0.5:
        return ("🟡 RISQUE MODÉRÉ",    "warning")
    return     ("🟢 TRANSACTION SAINE", "success")

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement ────────────────────────────────────────────────────────────────
model, preprocessor, feat_cfg, final_cfg, explainer = load_artifacts()
test = load_test_data()
test = add_features(test)

FEAT_COLS  = feat_cfg['all_feat_cols']
THRESHOLD  = final_cfg['threshold']
V_COLS     = [f'V{i}' for i in range(1, 29)]
FEAT_NAMES = (V_COLS + ['Amount_log1p']
              + feat_cfg.get('cyclic_cols',     [])
              + feat_cfg.get('binary_cols',      [])
              + feat_cfg.get('ordinal_cols',     [])
              + feat_cfg.get('interaction_cols', []))
INTERP     = ['Amount_log1p', 'hour_sin', 'hour_cos',
              'is_night', 'high_amount', 'amount_bin',
              'night_x_log_amount']

# Scores pré-calculés sur tout le test
@st.cache_data
def get_test_scores():
    X = preprocessor.transform(test[FEAT_COLS])
    return model.predict_proba(X)[:, 1]

scores_all = get_test_scores()
test['score'] = scores_all
test['pred']  = (scores_all >= THRESHOLD).astype(int)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Detector")
    st.caption("LGB Optuna tuned · AUPRC 0.755")
    st.divider()
    st.metric("Seuil opérationnel", f"{THRESHOLD:.3f}")
    st.metric("Coût optimal (test)", "2 307 €")
    st.metric("Fraudes dans le test", f"{int(test['Class'].sum())} / {len(test):,}")
    st.divider()
    st.caption("Projet : ULB Credit Card Fraud Detection  \n"
               "Dataset : 284 807 transactions, 2 jours, sept. 2013")

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋 Analyse d'une transaction",
    "✍️ Saisie manuelle",
    "⚙️ Optimisation du seuil",
])

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Analyse d'une transaction du test
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Analyse d'une transaction")
    st.caption("Sélectionnez une transaction du test set ou tirez-en une aléatoirement.")

    col_sel, col_rand = st.columns([3, 1])
    with col_rand:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🎲 Transaction aléatoire"):
            st.session_state['tx_idx'] = int(np.random.choice(len(test)))
    with col_sel:
        tx_idx = st.number_input(
            "Index de transaction (0 – " + str(len(test)-1) + ")",
            min_value=0, max_value=len(test)-1,
            value=st.session_state.get('tx_idx', 0),
            step=1
        )

    row   = test.iloc[tx_idx]
    score = float(row['score'])
    label, alert_type = score_label(score, THRESHOLD)

    # ── Fiche transaction ─────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score de fraude",  f"{score:.4f}")
    c2.metric("Montant",          f"{row['Amount']:.2f} €")
    c3.metric("Heure",            f"{int(row['hour'])}h")
    c4.metric("Label réel",       "🔴 FRAUDE" if int(row['Class']) == 1 else "🟢 LÉGIT")

    getattr(st, alert_type)(f"**Décision @ seuil {THRESHOLD}** : {label}")

    # ── SHAP local ────────────────────────────────────────────────────────────
    st.subheader("Drivers de la décision (SHAP)")
    X_row  = preprocessor.transform(test.iloc[[tx_idx]][FEAT_COLS])
    sv     = explainer.shap_values(X_row)
    sv     = sv[1].flatten() if isinstance(sv, list) else sv.flatten()
    top_n  = 10
    top_idx    = np.argsort(np.abs(sv))[::-1][:top_n]
    top_names  = [FEAT_NAMES[i] for i in top_idx]
    top_vals   = [sv[i]          for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ['tomato' if v > 0 else 'steelblue' for v in top_vals]
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [f"{'★' if n in INTERP else '·'} {n}" for n in top_names[::-1]],
        fontsize=9
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Valeur SHAP  (→ fraude / ← légit)")
    ax.set_title("Top 10 drivers  ·  ★ = feature interprétable")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top 3 en texte
    st.markdown("**Top 3 drivers :**")
    for i in range(3):
        n, v = top_names[i], top_vals[i]
        direction = "↑ vers fraude" if v > 0 else "↓ vers légit"
        interp = " *(interprétable)*" if n in INTERP else " *(PCA anonyme)*"
        st.markdown(f"- **{n}**{interp} — SHAP = {v:+.3f} {direction}")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Saisie manuelle
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Saisie manuelle d'une transaction")
    st.caption("Renseignez les caractéristiques d'une transaction pour scorer en temps réel.")

    with st.form("manual_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Montant (€)", min_value=0.01,
                                     max_value=30000.0, value=50.0, step=0.01)
            hour   = st.slider("Heure de la transaction", 0, 23, 14)
        with col_b:
            st.markdown("**Composantes PCA (V1–V28)**")
            st.caption("Valeurs typiques entre −5 et +5. "
                       "Laissez à 0 si inconnues.")
            v_vals = {}
            cols3 = st.columns(4)
            for i, vi in enumerate([f'V{j}' for j in range(1, 29)]):
                with cols3[i % 4]:
                    v_vals[vi] = st.number_input(vi, value=0.0,
                                                  format="%.2f",
                                                  key=f"v_{vi}",
                                                  label_visibility="visible")
        submitted = st.form_submit_button("🔍 Analyser la transaction")

    if submitted:
        # Construire la ligne
        row_dict = {'Time': hour * 3600, 'Amount': amount}
        row_dict.update(v_vals)
        row_df = pd.DataFrame([row_dict])
        score_m, X_m = predict(row_df, model, preprocessor, FEAT_COLS)
        label_m, alert_m = score_label(score_m, THRESHOLD)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Score de fraude", f"{score_m:.4f}")
        m2.metric("Seuil", f"{THRESHOLD:.3f}")
        m3.metric("Décision", "FRAUDE" if score_m >= THRESHOLD else "LÉGIT")
        getattr(st, alert_m)(label_m)

        # SHAP local
        sv_m = explainer.shap_values(X_m)
        sv_m = sv_m[1].flatten() if isinstance(sv_m, list) else sv_m.flatten()
        top_idx_m  = np.argsort(np.abs(sv_m))[::-1][:8]
        top_names_m = [FEAT_NAMES[i] for i in top_idx_m]
        top_vals_m  = [sv_m[i]        for i in top_idx_m]

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.barh(range(8), top_vals_m[::-1],
                 color=['tomato' if v > 0 else 'steelblue' for v in top_vals_m[::-1]],
                 edgecolor='white')
        ax2.set_yticks(range(8))
        ax2.set_yticklabels(top_names_m[::-1], fontsize=9)
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_xlabel("SHAP")
        ax2.set_title("Drivers de la décision")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.info("ℹ️ Si V1–V28 sont laissés à 0, le score est peu fiable "
                "(89% du signal provient des PCA).")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — Optimisation du seuil
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Optimisation du seuil de décision")
    st.caption("Ajustez le seuil et observez l'impact en temps réel sur le coût,  "
               "le rappel et la précision.")

    thr_slider = st.slider(
        "Seuil de décision",
        min_value=0.01, max_value=0.99,
        value=THRESHOLD, step=0.01,
        help="Valeur optimale calculée en Phase 9 : 0.281"
    )

    # Métriques au seuil choisi
    preds_thr = (scores_all >= thr_slider).astype(int)
    y_test    = test['Class'].values
    amounts   = test['Amount'].values
    cost      = compute_cost(y_test, preds_thr, amounts, COST_FALSE_POSITIVE)
    rep       = classification_report(y_test, preds_thr, output_dict=True)
    f_m       = rep.get('1', {})

    # Affichage métriques
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Coût total",   f"{cost['cout_total']:,.0f} €",
              delta=f"{cost['cout_total'] - 2307:+,.0f} € vs optimal")
    c2.metric("Rappel",       f"{f_m.get('recall',0)*100:.1f} %")
    c3.metric("Précision",    f"{f_m.get('precision',0)*100:.1f} %")
    c4.metric("TP / FN",      f"{cost['n_tp']} / {cost['n_fn']}")
    c5.metric("FP (alertes)", f"{cost['n_fp']}")

    # Courbe coût vs seuil
    thresholds  = np.linspace(0.01, 0.99, 200)
    costs_curve = []
    recalls_c   = []
    precs_c     = []
    for t in thresholds:
        p = (scores_all >= t).astype(int)
        c = compute_cost(y_test, p, amounts, COST_FALSE_POSITIVE)
        costs_curve.append(c['cout_total'])
        rep_t = classification_report(y_test, p, output_dict=True, zero_division=0)
        recalls_c.append(rep_t.get('1', {}).get('recall',    0))
        precs_c.append(  rep_t.get('1', {}).get('precision', 0))

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4))

    # Coût vs seuil
    axes3[0].plot(thresholds, costs_curve, color='tomato', linewidth=2)
    axes3[0].axvline(thr_slider, color='black', linestyle='--', linewidth=1.5,
                     label=f'Seuil actuel = {thr_slider:.2f}')
    axes3[0].axvline(THRESHOLD, color='gray', linestyle=':', linewidth=1,
                     label=f'Seuil optimal = {THRESHOLD:.3f}')
    axes3[0].scatter([thr_slider], [cost['cout_total']],
                     color='black', s=80, zorder=5)
    axes3[0].set_xlabel('Seuil')
    axes3[0].set_ylabel('Coût total (€)')
    axes3[0].set_title('Coût total vs seuil')
    axes3[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    axes3[0].legend(fontsize=9)

    # Rappel & Précision vs seuil
    axes3[1].plot(thresholds, [r*100 for r in recalls_c],
                  color='steelblue', linewidth=2, label='Rappel')
    axes3[1].plot(thresholds, [p*100 for p in precs_c],
                  color='tomato',    linewidth=2, label='Précision')
    axes3[1].axvline(thr_slider, color='black', linestyle='--', linewidth=1.5)
    axes3[1].axvline(THRESHOLD,  color='gray',  linestyle=':',  linewidth=1)
    axes3[1].set_xlabel('Seuil')
    axes3[1].set_ylabel('%')
    axes3[1].set_title('Rappel & Précision vs seuil')
    axes3[1].legend(fontsize=9)

    plt.suptitle('Impact du seuil sur les métriques opérationnelles',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Message contextuel
    if thr_slider < 0.15:
        st.warning("⚠️ Seuil très bas : beaucoup de FP — les analystes seront saturés d'alertes.")
    elif thr_slider > 0.8:
        st.warning("⚠️ Seuil très haut : beaucoup de FN — la majorité des fraudes passent.")
    else:
        st.success(f"✅ Seuil {thr_slider:.2f} — coût estimé : {cost['cout_total']:,.0f} €  "
                   f"| Rappel : {f_m.get('recall',0)*100:.1f} %  "
                   f"| Précision : {f_m.get('precision',0)*100:.1f} %")

    st.divider()
    st.markdown(
        "**Lecture** : le seuil optimal (0.281) minimise le coût total sur le test set.  \n"
        "Abaisser le seuil augmente le rappel (moins de fraudes manquées) mais aussi les FP  \n"
        "(plus d'alertes à traiter manuellement). L'équipe Risque choisit son point de fonctionnement."
    )
