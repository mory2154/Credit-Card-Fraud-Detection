"""
Configuration globale du projet.
Toutes les constantes modifiables sont centralisées ici.
"""
import os

# ── Reproductibilité ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
MODEL_PATH    = os.path.join(MODELS_DIR, "pipeline_final.joblib")

# ── Dataset Kaggle ────────────────────────────────────────────────────────────
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
KAGGLE_FILE    = "creditcard.csv"

# ── Split temporel (Phase 3) ──────────────────────────────────────────────────
# Time max = 172 792 s (48h). Seuil = 138 234 s (38,4 h)
TEST_TIME_THRESHOLD = 0.80   # fraction de Time max pour la coupure train/test

# ── Chiffres EDA corrigés (Phase 1 + 2 + 3) ──────────────────────────────────
N_TOTAL            = 283_726
N_FRAUDS           = 473
N_LEGIT            = 283_253
TOTAL_FRAUD_AMOUNT = 58_591.39    # €

# Chiffres post-split (Phase 3)
N_TRAIN            = 210_292
N_TEST             =  73_434
N_FRAUDS_TRAIN     =     376
N_FRAUDS_TEST      =      97
N_LEGIT_TRAIN      = 209_916
FRAUD_RATE_TRAIN   = 0.00179      # 0,179 %
FRAUD_RATE_TEST    = 0.00132      # 0,132 %

# ── Structure de coût (Phase 2) ───────────────────────────────────────────────
COST_FALSE_POSITIVE = 5.0                         # € par FP
SCALE_POS_WEIGHT    = N_LEGIT_TRAIN / N_FRAUDS_TRAIN  # ≈ 558 pour LightGBM

# ── Métrique principale ───────────────────────────────────────────────────────
MAIN_METRIC = "average_precision"  # AUPRC
