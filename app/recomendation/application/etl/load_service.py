import pandas as pd
import os
from sqlalchemy import text
from app.recommendation.infrastructure.db_connection import get_engine

def load_data():
    """
    Carga los datos limpios (compras_clean.csv) a PostgreSQL.
    - Crea la tabla si no existe.
    - Inserta los datos validados.
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ruta_clean = os.path.join(base_dir, "infrastructure", "data", "clean", "compras_clean.csv")

    # 🧠 Leer el CSV limpio
    df = pd.read_csv(ruta_clean, encoding="utf-8-sig", sep=";")
    print(f"✅ Archivo limpio leído correctamente ({len(df)} registros).")

    # 🔌 Conexión a PostgreSQL
    engine = get_engine()
    table_name = "compras"

    # 📊 Crear tabla si no existe (tipos compatibles con PostgreSQL)
    create_table_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id_compra INT PRIMARY KEY,
            cliente VARCHAR(100),
            producto VARCHAR(100),
            categoria VARCHAR(100),
            precio FLOAT,
            fecha_compra DATE
        );
    """)

    with engine.connect() as conn:
        conn.execute(create_table_sql)
        conn.commit()
        print(f"🧱 Tabla '{table_name}' verificada o creada.")

    # 🚀 Cargar datos (reemplazar o agregar según tu preferencia)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print(f"📦 Datos cargados correctamente en la tabla '{table_name}'.")

    return f"Se cargaron {len(df)} registros en la tabla '{table_name}'."
