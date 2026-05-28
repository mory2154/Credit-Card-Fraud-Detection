# 🛡️ Détection de Fraude par Carte Bancaire

> **Modèle de Machine Learning** pour la détection de transactions frauduleuses, entraîné sur le dataset *Credit Card Fraud Detection* de l'ULB / Worldline.

---

## 📊 Données & Métriques Clés

| Élément | Valeur / Description |
| :--- | :--- |
| **Transactions** | 284 807 (48h, Europe, sept. 2013) |
| **Fraudes** | 473 après nettoyage — **0,167 %** |
| **Modèle final** | LightGBM (Optuna tuned) |
| **AUPRC (test)** | **0,755** |
| **Seuil opérationnel** | 0,281 |
| **Coût optimal** | **2 307 €** vs 58 591 € sans modèle |

---

## 🎯 Résultats Clés

- ✅ Le modèle détecte **75 fraudes sur 97** dans le test set avec une **précision de 38 %**.
- 👉 **1 alerte sur 2,6** correspond à une vraie fraude.
- 🎯 En inspectant seulement **0,51 % des transactions**, on capture **80 % des fraudes en valeur**.

> ⚠️ **Limite principale** : 89 % du signal prédictif repose sur des composantes PCA anonymisées (V1–V28). Sans elles, l'AUPRC tombe à **0,002**.  
> 📋 *Ce point est à documenter impérativement pour la conformité RGPD avant tout déploiement en production.*

---

## 🧱 Structure du Projet

```text
fraud-detection/
├── data/                        # Données brutes (non versionnées — voir .gitignore)
│   ├── train.parquet            # Split temporel train (0–38,4h)
│   └── test.parquet             # Split temporel test (38,4–48h)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploration & profiling
│   ├── 02_cost_structure.ipynb  # Matrice de coût FN/FP
│   ├── 03_data_preparation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_resampling.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_advanced_modeling.ipynb
│   ├── 08_hybrid_unsupervised_supervised.ipynb
│   ├── 09_final_evaluation.ipynb
│   ├── 10_interpretability.ipynb
│   ├── 11_robustness_audit.ipynb
│   └── 12_monitoring.ipynb
├── src/
│   ├── config.py                # Constantes globales (seed, chemins, coûts, seuils)
│   ├── download_data.py         # Téléchargement via l'API Kaggle
│   ├── build_dataset.py         # Pipeline de préparation reproductible
│   └── cost_metric.py           # Fonctions de coût métier (compute_cost, optimal_threshold)
├── models/
│   ├── pipeline_final.joblib    # Modèle final sérialisé
│   ├── preprocessor.joblib      # Pipeline de prétraitement (fitté sur train)
│   ├── feature_config.json      # Liste des 35 features finales
│   └── final_model_config.json  # Seuil, AUPRC, coût du modèle retenu
├── app/
│   └── main.py                  # Application Streamlit (scoring + SHAP + seuil)
├── reports/
│   ├── schema.md                # Schéma documenté de toutes les colonnes
│   ├── findings_escalade.csv    # Findings à escalader (Risque / Legal)
│   └── monitoring_triggers.csv  # Règles de déclenchement du retraining
├── tests/
├── Makefile
├── requirements.txt
└── .env.example
