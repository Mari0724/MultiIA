import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 📂 Cargar variables del archivo .env
load_dotenv()

# ⚙️ Leer credenciales desde .env
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def get_engine():
    """
    Crea y retorna un motor de conexión SQLAlchemy a PostgreSQL
    usando las variables definidas en el archivo .env.
    """
    if not all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("❌ Faltan variables de entorno para la conexión a la base de datos.")
    
    url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"🔌 Conectando a PostgreSQL → {DB_HOST}:{DB_PORT}/{DB_NAME}")
    return create_engine(url)
