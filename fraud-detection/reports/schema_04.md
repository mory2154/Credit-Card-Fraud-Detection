# Schéma du dataset analytique

Mis à jour par `notebooks/04_feature_engineering.ipynb`.

## Features finales — 35 colonnes

| Groupe | Feature | Transformation | Cohen's d | Décision |
|---|---|---|---|---|
| V* | V1–V28 | Passthrough | V17=9.12, V14=7.69 | INCLURE |
| Montant | `Amount` | `log1p` + `RobustScaler` | 0.141 (brut) | INCLURE (base dérivées) |
| Cyclique | `hour_sin` | Passthrough | 0.313 | INCLURE |
| Cyclique | `hour_cos` | Passthrough | 0.246 | INCLURE (encodage cyclique) |
| Binaire | `is_night` | Passthrough | **0.608** | INCLURE |
| Binaire | `high_amount` | Passthrough | 0.358 | INCLURE |
| Ordinal | `amount_bin` | `RobustScaler` | 0.361 | INCLURE |
| Interaction | `night_x_log_amount` | `RobustScaler` | 0.304 | INCLURE |

## Split temporel

| Split | Lignes | Fraudes | % fraudes |
|---|---|---|---|
| Train | 210 292 | 376 | 0.179 % |
| Test | 73 434 | 97 | 0.132 % |

## Règles anti-fuite
- 1 081 doublons supprimés avant le split
- `preprocessor.fit()` sur train uniquement
- SMOTE uniquement sur le train (Phase 5)
- Overlap train/test = 0
