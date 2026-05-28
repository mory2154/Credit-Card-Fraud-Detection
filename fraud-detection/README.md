# Détection de fraude par carte bancaire

Modèle de machine learning pour la détection de transactions frauduleuses,
entraîné sur le dataset [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
de l'ULB / Worldline (284 807 transactions, 492 fraudes — 0,172 %).

---

## Structure du projet

```
fraud-detection/
├── data/               # Données brutes (non versionnées)
├── notebooks/          # Exploration et analyses (EDA, etc.)
├── src/
│   ├── config.py       # Constantes globales (seed, chemins, coûts)
│   ├── download_data.py
│   ├── build_features.py   # (Phase 4)
│   ├── train.py            # (Phase 6-7)
│   └── evaluate.py         # (Phase 9)
├── models/             # Pipelines entraînés (.joblib)
├── app/
│   └── main.py         # Application Streamlit (Phase 12)
├── reports/            # Graphiques et rapport final
├── tests/              # Tests unitaires
├── Makefile
├── requirements.txt
└── .env.example
```

---

## Installation

**1. Cloner le dépôt**
```bash
git clone <url-du-repo>
cd fraud-detection
```

**2. Créer et activer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**4. Configurer les credentials Kaggle**
```bash
cp .env.example .env
# Renseigner KAGGLE_USERNAME et KAGGLE_KEY dans .env
# (Récupérables sur https://www.kaggle.com/settings → API)
```

---

## Utilisation

```bash
make data    # Télécharge creditcard.csv (~150 Mo) dans data/
make train   # Entraîne le modèle final
make app     # Lance l'app Streamlit sur http://localhost:8501
make test    # Lance les tests unitaires
```

---

## Honesty

Ce qui est implémenté, approximé ou laissé en future work est documenté
dans `reports/honesty.md`.  
Les clés API ne sont jamais committées (voir `.gitignore` + `.env.example`).

---

## Références

- Dal Pozzolo et al. — *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE CIDM 2015
- Carcillo et al. — *Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection*, Information Sciences 2019
- Le Borgne & Bontempi — *Reproducible Machine Learning for Credit Card Fraud Detection*, Practical Handbook
