import pandas as pd
import os

def transform_data():
    """
    Limpia y transforma los datos del archivo compras_raw.csv:
    - Detecta el delimitador correcto (coma o punto y coma)
    - Normaliza nombres de columnas
    - Convierte tipos de datos
    - Elimina duplicados y valores nulos
    - Guarda el resultado limpio en /data/clean/compras_clean.csv
    """

    # 🧭 Rutas de entrada y salida
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ruta_raw = os.path.join(base_dir, "infrastructure", "data", "raw", "compras_raw.csv")
    ruta_clean = os.path.join(base_dir, "infrastructure", "data", "clean")
    os.makedirs(ruta_clean, exist_ok=True)
    ruta_salida = os.path.join(ruta_clean, "compras_clean.csv")

    # 🔍 Intentamos leer el CSV con distintos separadores
    try:
        df = pd.read_csv(ruta_raw, encoding="utf-8-sig", sep=",")
        if len(df.columns) == 1:  # si todo quedó en una sola columna...
            df = pd.read_csv(ruta_raw, encoding="utf-8-sig", sep=";")
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return None

    print("📊 Datos cargados correctamente. Columnas detectadas:")
    print(df.columns)

    # 🧼 Limpieza básica
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Eliminamos filas vacías o duplicadas
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    # Convertimos tipos de datos
    if "precio" in df.columns:
        df["precio"] = pd.to_numeric(df["precio"], errors="coerce")

    if "fecha_compra" in df.columns:
        df["fecha_compra"] = pd.to_datetime(df["fecha_compra"], errors="coerce")

    # Quitamos filas sin fecha o sin cliente
    df = df.dropna(subset=["cliente", "fecha_compra"], how="any")

    # Guardamos archivo limpio
    df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print(f"✅ Archivo limpio guardado en: {ruta_salida}")
    print(f"📈 Registros finales: {len(df)}")
    return ruta_salida
