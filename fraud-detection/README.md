Détection de fraude par carte bancaire
Modèle de machine learning pour la détection de transactions frauduleuses, entraîné sur le dataset Credit Card Fraud Detection de l'ULB / Worldline.
	
Transactions	284 807 (48h, Europe, sept. 2013)
Fraudes	473 après nettoyage — 0,167 %
Modèle final	LightGBM Optuna tuned
AUPRC (test)	0,755
Coût optimal	2 307 € vs 58 591 € sans modèle
Seuil opérationnel	0,281
---
Résultats clés
Le modèle détecte 75 fraudes sur 97 dans le test set avec une précision de 38 % — chaque alerte sur 2,6 est une vraie fraude. En inspectant seulement 0,51 % des transactions, on capture 80 % des fraudes en valeur.
Limite principale : 89 % du signal prédictif repose sur des composantes PCA anonymisées (V1–V28). Sans elles, l'AUPRC tombe à 0,002. Ce point est à documenter pour la conformité RGPD avant tout déploiement en production.
---
Structure du projet
```
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
```
---
Installation
1. Cloner le dépôt
```bash
git clone <url-du-repo>
cd fraud-detection
```
2. Créer et activer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```
3. Installer les dépendances
```bash
pip install -r requirements.txt
```
4. Placer le dataset
Télécharger `creditcard.csv` depuis Kaggle (~150 Mo) et le déposer dans `data/`. Le fichier est exclu du dépôt git (`.gitignore`).
> **Alternative via l'API Kaggle :** copier vos credentials dans `~/.kaggle/kaggle.json` puis lancer `make data`.
---
Utilisation
```bash
make data    # Télécharge creditcard.csv via l'API Kaggle
make train   # Prépare les données et entraîne le modèle final
make app     # Lance l'app Streamlit → http://localhost:8501
make test    # Lance les tests unitaires
```
L'application Streamlit expose trois onglets :
Analyse d'une transaction — score, niveau de risque, top drivers SHAP
Saisie manuelle — scoring en temps réel d'une transaction fictive
Optimisation du seuil — curseur interactif coût / rappel / précision
---
Reproduire les résultats
```bash
# Depuis le CSV brut jusqu'au modèle final en une séquence
make data
python src/build_dataset.py     # split + prétraitement → data/*.parquet + models/preprocessor.joblib
jupyter nbconvert --to notebook --execute notebooks/07_advanced_modeling.ipynb
```
Tous les `random_state` sont fixés à 42 dans `src/config.py`.
---
Honesty
Ce qui est	Détail
Implémenté	Pipeline complet de l'EDA au monitoring, app Streamlit, rapport PDF/Word
Approximé	Coût FP = 5 € fixe (hypothèse conservatrice, à valider avec l'équipe Risque)
Limite connue	Optuna tuné sur 75 fraudes (val set) — risque de surapprentissage au val
Future work	Retraining automatisé, calibration Platt pour LR, données post-2013
Usage de LLM	Assistance à la rédaction du rapport final et des conclusions des notebooks
Les clés API ne sont jamais committées — voir `.gitignore` et `.env.example`.
---
Références
Dal Pozzolo et al. — Calibrating Probability with Undersampling for Unbalanced Classification, IEEE CIDM 2015
Carcillo et al. — Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection, Information Sciences 2019
Le Borgne & Bontempi — Reproducible Machine Learning for Credit Card Fraud Detection, Practical Handbook (fraud-detection-handbook.github.io)
