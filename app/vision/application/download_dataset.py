import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Carpeta destino del dataset
DATA_DIR = os.path.join("app", "vision", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Inicializar Kaggle API
api = KaggleApi()
api.authenticate()

# Nombre del dataset (ejemplo: rayhaan/python-chest-xray-pneumonia)
DATASET = "paultimothymooney/chest-xray-pneumonia"

print(f"⬇️ Descargando dataset {DATASET}...")
api.dataset_download_files(DATASET, path=DATA_DIR, unzip=True)
print(f"✅ Dataset descargado y extraído en: {DATA_DIR}")
