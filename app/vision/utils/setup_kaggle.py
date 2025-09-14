import os
import json
from pathlib import Path
from dotenv import load_dotenv

# 🔹 1. Cargar las variables desde un archivo .env
# (Este archivo debe tener: KAGGLE_USERNAME y KAGGLE_KEY)
load_dotenv()

# 🔹 2. Leer las credenciales desde las variables de entorno
username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

# 🔹 3. Validar que existan las credenciales
if not username or not key:
    raise ValueError("⚠️ Faltan KAGGLE_USERNAME o KAGGLE_KEY en el .env")

# 🔹 4. Definir la ruta donde Kaggle espera encontrar sus credenciales:
# En la carpeta del usuario, dentro de ".kaggle/kaggle.json"
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe
kaggle_path = kaggle_dir / "kaggle.json"

# 🔹 5. Crear el contenido del kaggle.json con los datos leídos del .env
creds = {
    "username": username,
    "key": key
}

# 🔹 6. Guardar ese contenido como archivo kaggle.json
with open(kaggle_path, "w") as f:
    json.dump(creds, f)

# 🔹 7. Ajustar permisos (solo Unix/Linux/Mac),
# Kaggle exige que el archivo no sea "público"
try:
    os.chmod(kaggle_path, 0o600)
except Exception:
    # En Windows no aplica, así que lo ignoramos
    pass

print(f"✅ kaggle.json generado en {kaggle_path}")
