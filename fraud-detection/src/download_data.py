"""
Télécharge creditcard.csv depuis Kaggle dans data/.
Nécessite : KAGGLE_USERNAME et KAGGLE_KEY dans .env (ou variables d'env système).

Usage : python src/download_data.py
        ou     make data
"""
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Chemins relatifs au script
ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
CSV_PATH  = DATA_DIR / "creditcard.csv"


def download_data() -> None:
    """Télécharge et dézippe creditcard.csv depuis Kaggle."""
    # Import ici pour ne pas crasher si kaggle n'est pas installé
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Le package 'kaggle' n'est pas installé.\n"
            "Lancez : pip install kaggle"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Authentification Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print(f"Téléchargement de mlg-ulb/creditcardfraud → {DATA_DIR} ...")
    api.dataset_download_files(
        "mlg-ulb/creditcardfraud",
        path=str(DATA_DIR),
        unzip=False,
        quiet=False,
    )

    # Dézip du fichier téléchargé
    zip_path = DATA_DIR / "creditcardfraud.zip"
    if zip_path.exists():
        print(f"Décompression de {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
        print("Archive supprimée.")

    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / 1_048_576
        print(f"✓ {CSV_PATH} ({size_mb:.1f} Mo)")
    else:
        raise FileNotFoundError(
            f"Le fichier {CSV_PATH} est introuvable après le téléchargement."
        )


if __name__ == "__main__":
    if CSV_PATH.exists():
        print(f"'{CSV_PATH.name}' existe déjà — pas de téléchargement.")
    else:
        download_data()
