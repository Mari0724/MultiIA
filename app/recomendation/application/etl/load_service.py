import pandas as pd
import os
from sqlalchemy import text
from app.recomendation.infrastructure.db_connection import get_engine

def load_data():
    """
    Fase L (Load) del proceso ETL.
    Carga los datos limpios (compras_clean.csv) a PostgreSQL y guarda una copia procesada.
    
    - Verifica o crea la tabla 'compras'.
    - Inserta los datos validados.
    - Genera un respaldo local en /data/processed/compras_processed.csv.
    """

    # 📂 1️⃣ Definir rutas base
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ruta_clean = os.path.join(base_dir, "infrastructure", "data", "clean", "compras_clean.csv")
    ruta_processed = os.path.join(base_dir, "infrastructure", "data", "processed")
    os.makedirs(ruta_processed, exist_ok=True)
    ruta_salida = os.path.join(ruta_processed, "compras_processed.csv")

    # 📥 2️⃣ Leer el CSV limpio
    try:
        df = pd.read_csv(ruta_clean, encoding="utf-8-sig", sep=";")
        print(f"✅ Archivo limpio leído correctamente ({len(df)} registros).")
    except Exception as e:
        raise RuntimeError(f"❌ Error al leer el archivo limpio: {e}")

    # 🔌 3️⃣ Conectar a PostgreSQL
    try:
        engine = get_engine()
        print("🔗 Conexión a PostgreSQL establecida correctamente.")
    except Exception as e:
        raise ConnectionError(f"❌ No se pudo conectar a la base de datos: {e}")

    table_name = "compras"

    # 🧱 4️⃣ Crear tabla si no existe
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

    try:
        with engine.connect() as conn:
            conn.execute(create_table_sql)
            conn.commit()
        print(f"🧩 Tabla '{table_name}' verificada o creada exitosamente.")
    except Exception as e:
        raise RuntimeError(f"❌ Error al crear/verificar la tabla: {e}")

    # 🚀 5️⃣ Cargar los datos al motor
    try:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"📦 {len(df)} registros cargados correctamente en la tabla '{table_name}'.")
    except Exception as e:
        raise RuntimeError(f"❌ Error al insertar los datos: {e}")

    # 💾 6️⃣ Guardar copia procesada local
    try:
        df.to_csv(ruta_salida, index=False, encoding="utf-8-sig", sep=";")
        print(f"💽 Copia procesada guardada en: {ruta_salida}")
    except Exception as e:
        raise RuntimeError(f"❌ Error al guardar el archivo procesado: {e}")

    print("✅ Fase L (Load) completada con éxito.")
    return f"Se cargaron {len(df)} registros en PostgreSQL y se guardó copia local."
