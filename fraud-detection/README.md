Détection de fraude par carte bancaire
Projet de machine learning visant à détecter des transactions frauduleuses à partir du dataset Credit Card Fraud Detection (ULB / Worldline).
---
📊 Données
Élément	Valeur
Transactions	284 807 (48h, Europe, septembre 2013)
Fraudes	473 après nettoyage — 0,167 %
Source	ULB / Worldline
---
🤖 Modèle
Composant	Description
Modèle final	LightGBM (optimisé avec Optuna)
AUPRC (test)	0,755
Seuil opérationnel	0,281
Coût optimisé	2 307 € vs 58 591 € sans modèle
---
🎯 Résultats clés
Le modèle détecte 75 fraudes sur 97 dans le jeu de test, avec une précision de 38 %.
👉 En pratique, 1 alerte sur 2,6 correspond à une vraie fraude.
En ne contrôlant que 0,51 % des transactions, on capture 80 % des fraudes en valeur.
---
⚠️ Limites importantes
89 % du signal prédictif provient de composantes PCA anonymisées (V1–V28)
Sans ces variables, l’AUPRC chute à 0,002
Ce point doit être explicitement documenté pour la conformité RGPD avant mise en production
---
🧱 Structure du projet
fraud-detection/
├── data/
│   ├── train.parquet
│   └── test.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cost_structure.ipynb
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
│   ├── config.py
│   ├── download_data.py
│   ├── build_dataset.py
│   └── cost_metric.py
├── models/
│   ├── pipeline_final.joblib
│   ├── preprocessor.joblib
│   ├── feature_config.json
│   └── final_model_config.json
├── app/
│   └── main.py
├── reports/
│   ├── schema.md
│   ├── findings_escalade.csv
│   └── monitoring_triggers.csv
├── tests/
├── Makefile
├── requirements.txt
└── .env.example
---
🚀 Installation
git clone <url-du-repo>
cd fraud-detection
python -m venv venv
source venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
---
▶️ Utilisation
make data
make train
make app
make test
---
🧾 Transparence (Honesty)
Pipeline complet de bout en bout
Coût FP approximé à 5 €
Limite : dépendance forte aux features PCA
Future work : calibration + retraining + données post-2013
---
📚 Références
Dal Pozzolo et al. (2015)
Carcillo et al. (2019)
Le Borgne & Bontempi
