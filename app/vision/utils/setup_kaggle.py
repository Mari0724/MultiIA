import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

if not username or not key:
    raise ValueError("⚠️ Faltan KAGGLE_USERNAME o KAGGLE_KEY en el .env")

# Ruta a ~/.kaggle/kaggle.json
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_path = kaggle_dir / "kaggle.json"

# Crear el archivo kaggle.json
creds = {
    "username": username,
    "key": key
}

with open(kaggle_path, "w") as f:
    json.dump(creds, f)

# Dar permisos correctos (Unix/Linux/Mac)
try:
    os.chmod(kaggle_path, 0o600)
except Exception:
    pass  # En Windows no aplica

print(f"✅ kaggle.json generado en {kaggle_path}")
