from fastapi import APIRouter
from app.recomendation.application.etl.extract_service import generar_dataset
from app.recomendation.application.etl.transform_service import transform_data



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
