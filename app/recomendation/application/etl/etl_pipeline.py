import os
from app.recomendation.application.etl.extract_service import generar_dataset
from app.recomendation.application.etl.transform_service import transform_data
from app.recomendation.application.etl.load_service import load_data

def run_etl_pipeline():
    """
    Ejecuta el pipeline completo ETL (Extract → Transform → Load).
    Cada fase se ejecuta de forma secuencial y segura.
    """

    print("\n INICIANDO PIPELINE ETL COMPLETO...\n")

    try:
        print(" Extrayendo datos...")
        extract_result = generar_dataset()
        print(f"✅ Extracción completada: {extract_result}")
    except Exception as e:
        print(f"❌ Error en la fase Extract: {e}")
        return "Error en Extract."

    try:
        print("\n Transformando datos...")
        transform_result = transform_data()
        print(f"✅ Transformación completada: {transform_result}")
    except Exception as e:
        print(f"❌ Error en la fase Transform: {e}")
        return "Error en Transform."

    try:
        print("\n📦 3️⃣ Cargando datos...")
        load_result = load_data()
        print(f"✅ Carga completada: {load_result}")
    except Exception as e:
        print(f"❌ Error en la fase Load: {e}")
        return "Error en Load."

    print("\n🎉 PIPELINE ETL EJECUTADO EXITOSAMENTE.")
    return "Pipeline ETL completado con éxito."
