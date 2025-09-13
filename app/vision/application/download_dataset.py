#  script de descarga del dataset desde Kaggle.
import os
# import zipfile  
#  Es un módulo estándar de Python para trabajar con archivos .zip    (crear, leer, extraer, comprimir).
#  En este caso no lo usamos porque la función dataset_download_files ya descomprime por sí misma.

from kaggle.api.kaggle_api_extended import KaggleApi
# Importamos KaggleApi, la clase que nos permite autenticar 
#    y descargar datasets directamente desde Kaggle usando su API oficial.

DATA_DIR = os.path.join("app", "vision", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Inicializar Kaggle API
api = KaggleApi() #  Creamos un objeto de la clase KaggleApi.
api.authenticate()
# Autenticamos usando la configuración de Kaggle (necesita el archivo kaggle.json en la carpeta correcta).
#    Esto es como "iniciar sesión" en Kaggle desde Python.

DATASET = "paultimothymooney/chest-xray-pneumonia" #  Aquí indicamos el dataset que queremos descargar, en formato "usuario/nombre_dataset".

print(f"⬇️ Descargando dataset {DATASET}...")
api.dataset_download_files(DATASET, path=DATA_DIR, unzip=True)
#  Descarga el dataset al directorio DATA_DIR.
#  unzip=True significa que lo descomprime automáticamente al llegar.
#     Si pusieras unzip=False, sí necesitarías "import zipfile" para extraerlo manualmente.

print(f"✅ Dataset descargado y extraído en: {DATA_DIR}")