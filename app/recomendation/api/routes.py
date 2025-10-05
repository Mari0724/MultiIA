from fastapi import APIRouter
from app.recomendation.application.etl.extract_service import generar_dataset

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

@router.post("/generate-data")
def generar_datos(cantidad: int = 3000):
    """Genera datos sintéticos de compras"""
    ruta = generar_dataset(cantidad)
    return {"message": f"Dataset generado con {cantidad} registros", "archivo": ruta}
