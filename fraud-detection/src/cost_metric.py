"""
Fonctions de calcul du coût métier.
Réutilisées en Phase 7 (pondération), Phase 9 (évaluation), Phase 12 (app).
"""
import numpy as np
from src.config import COST_FALSE_POSITIVE, TOTAL_FRAUD_AMOUNT


def compute_cost(y_true, y_pred, amounts, cost_fp=COST_FALSE_POSITIVE):
    """
    Calcule le coût total d'une politique de décision binaire.

    Paramètres
    ----------
    y_true   : array-like — labels réels (0 = légit, 1 = fraude)
    y_pred   : array-like — prédictions binaires (0/1)
    amounts  : array-like — montants des transactions (€)
    cost_fp  : float — coût fixe par faux positif (€), défaut = config.COST_FALSE_POSITIVE

    Retourne
    --------
    dict :
        cout_total       — coût total (FN + FP)
        cout_fn          — coût des fraudes manquées (= somme des Amount FN)
        cout_fp          — coût des blocages injustifiés (= n_FP × cost_fp)
        montant_recupere — montant fraudé correctement bloqué (= somme Amount TP)
        pct_recupere     — % du montant total fraudé récupéré
        n_fn, n_fp, n_tp — comptages
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    amounts = np.asarray(amounts)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tp_mask = (y_true == 1) & (y_pred == 1)

    cost_fn_total = amounts[fn_mask].sum()
    cost_fp_total = fp_mask.sum() * cost_fp
    recovered     = amounts[tp_mask].sum()

    return {
        "cout_total"       : cost_fn_total + cost_fp_total,
        "cout_fn"          : cost_fn_total,
        "cout_fp"          : cost_fp_total,
        "montant_recupere" : recovered,
        "pct_recupere"     : recovered / TOTAL_FRAUD_AMOUNT * 100,
        "n_fn"             : int(fn_mask.sum()),
        "n_fp"             : int(fp_mask.sum()),
        "n_tp"             : int(tp_mask.sum()),
    }


def cost_at_threshold(y_true, scores, amounts, threshold, cost_fp=COST_FALSE_POSITIVE):
    """
    Calcule le coût pour un seuil donné sur les scores de probabilité.

    Paramètres
    ----------
    scores    : array-like — probabilités de fraude (entre 0 et 1)
    threshold : float — seuil de décision (0–1)
    """
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    return compute_cost(y_true, y_pred, amounts, cost_fp)


def optimal_threshold(y_true, scores, amounts, cost_fp=COST_FALSE_POSITIVE,
                      n_steps=200):
    """
    Trouve le seuil qui minimise le coût total sur la plage [0.01, 0.99].

    Retourne
    --------
    best_threshold : float
    best_cost      : dict (résultat de compute_cost au seuil optimal)
    costs_df       : DataFrame avec colonnes [threshold, cout_total, ...]
    """
    import pandas as pd

    thresholds = np.linspace(0.01, 0.99, n_steps)
    records = []
    for thr in thresholds:
        c = cost_at_threshold(y_true, scores, amounts, thr, cost_fp)
        c["threshold"] = thr
        records.append(c)

    costs_df      = pd.DataFrame(records)
    best_idx      = costs_df["cout_total"].idxmin()
    best_threshold = costs_df.loc[best_idx, "threshold"]
    best_cost      = costs_df.loc[best_idx].to_dict()

    return best_threshold, best_cost, costs_df
