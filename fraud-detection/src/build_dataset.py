"""
Prépare le dataset analytique de manière reproductible.
Produit : data/train.parquet, data/test.parquet, models/preprocessor.joblib

Usage : python src/build_dataset.py
        ou via : make data
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.config import (
    RAW_DATA_PATH, DATA_DIR, MODELS_DIR,
    RANDOM_SEED, TEST_TIME_THRESHOLD
)

np.random.seed(RANDOM_SEED)

# ── Colonnes ──────────────────────────────────────────────────────────────────
V_COLS     = [f'V{i}' for i in range(1, 29)]
FEAT_COLS  = V_COLS + ['Amount', 'hour']
TARGET_COL = 'Class'


def load_and_clean(path: str) -> pd.DataFrame:
    """Charge le CSV brut et supprime les doublons."""
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Doublons supprimés : {n_before - len(df)}  "
          f"({n_before:,} → {len(df):,})")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées de Time."""
    df = df.copy()
    df['hour'] = (df['Time'] // 3600) % 24
    return df


def temporal_split(df: pd.DataFrame, threshold: float = TEST_TIME_THRESHOLD):
    """
    Split temporel : train sur les `threshold` premiers % de Time,
    test sur le reste.
    """
    time_max   = df['Time'].max()
    time_split = time_max * threshold

    train = df[df['Time'] <= time_split].copy()
    test  = df[df['Time'] >  time_split].copy()

    print(f"\nSplit temporel (seuil = {time_split:,.0f} s — "
          f"{time_split/3600:.1f} h)")
    for name, d in [("Train", train), ("Test", test)]:
        n_f = d['Class'].sum()
        print(f"  {name:<6} : {len(d):>8,} lignes | "
              f"{n_f:>4} fraudes ({n_f/len(d)*100:.3f} %)")

    return train, test


def build_preprocessor() -> ColumnTransformer:
    """Construit le pipeline de prétraitement (non fitté)."""
    log1p_scaler = Pipeline([
        ('log1p',  FunctionTransformer(np.log1p, validate=False)),
        ('scaler', RobustScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ('v_cols', 'passthrough',  V_COLS),
            ('amount', log1p_scaler,   ['Amount']),
            ('hour',   RobustScaler(), ['hour']),
        ],
        remainder='drop'
    )


def main():
    DATA_PATH   = Path(DATA_DIR)
    MODELS_PATH = Path(MODELS_DIR)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Chargement & nettoyage
    print("=== Chargement ===")
    df = load_and_clean(RAW_DATA_PATH)

    # 2. Feature engineering
    df = add_features(df)

    # 3. Split temporel
    print("\n=== Split ===")
    train, test = temporal_split(df)

    # 4. Prétraitement (fit sur train uniquement)
    print("\n=== Prétraitement ===")
    preprocessor = build_preprocessor()
    preprocessor.fit(train[FEAT_COLS])
    print("Pipeline fitté sur le train uniquement.")

    # 5. Sauvegarde
    print("\n=== Sauvegarde ===")
    train.to_parquet(DATA_PATH / 'train.parquet', index=False)
    test.to_parquet(DATA_PATH  / 'test.parquet',  index=False)
    joblib.dump(preprocessor, MODELS_PATH / 'preprocessor.joblib')

    print(f"  data/train.parquet         ({len(train):,} lignes)")
    print(f"  data/test.parquet          ({len(test):,} lignes)")
    print(f"  models/preprocessor.joblib")
    print("\nPhase 3 terminée.")


if __name__ == "__main__":
    main()
