.PHONY: help data train app test clean

# Affiche l'aide par défaut
help:
	@echo "Commandes disponibles :"
	@echo "  make data    — Télécharge creditcard.csv depuis Kaggle"
	@echo "  make train   — Entraîne le modèle final"
	@echo "  make app     — Lance l'application Streamlit"
	@echo "  make test    — Lance les tests unitaires"
	@echo "  make clean   — Supprime les artefacts temporaires"

# Télécharge le dataset depuis Kaggle
data:
	python src/download_data.py

# Entraîne le modèle (implémenté en Phase 6-7)
train:
	python src/train.py

# Lance l'app Streamlit (implémentée en Phase 12)
app:
	streamlit run app/main.py

# Tests unitaires
test:
	pytest tests/ -v

# Nettoyage
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
