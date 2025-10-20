from fastapi import APIRouter
from app.recomendation.application.etl.extract_service import generar_dataset
from app.recomendation.application.etl.transform_service import transform_data
from app.recomendation.application.etl.load_service import load_data
from app.recomendation.application.etl.etl_pipeline import run_etl_pipeline


router = APIRouter(prefix="/recomendation", tags=["Recomendation"])

@router.post("/generate-data")
def generar_datos(cantidad: int = 3000):
    """Genera datos sintéticos de compras"""
    ruta = generar_dataset(cantidad)
    return {"message": f"Dataset generado con {cantidad} registros", "archivo": ruta}

@router.post("/transform-data")
def transformar_datos():
    ruta = transform_data()
    return {"message": "Transformación completada", "archivo": ruta}

@router.post("/load-data")
def cargar_datos():
    """
    Carga los datos limpios (compras_clean.csv) en PostgreSQL.
    """
    resultado = load_data()
    return {"message": resultado}


@router.post("/run-etl")
def ejecutar_pipeline():
    """
    Ejecuta todo el pipeline ETL (Extract → Transform → Load).
    """
    resultado = run_etl_pipeline()
    return {"message": resultado}