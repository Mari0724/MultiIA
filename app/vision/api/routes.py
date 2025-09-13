from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path

# Importamos servicios y entrenamiento
from app.vision.application.vision_service import VisionService
from app.vision.application.pneumonia_service import PneumoniaService
from app.vision.training.train_pneumonia import train_pneumonia_model

# ======================
# 🚏 Configuración del Router
# ======================
router = APIRouter(
    prefix="/vision",
    tags=["Vision"],
    responses={404: {"description": "No encontrado"}}
)

# 📂 Directorios base
VISION_DIR = Path(__file__).resolve().parents[1]    # Raíz de la carpeta vision/
UPLOAD_DIR = VISION_DIR / "uploads"                 # Carpeta donde se suben imágenes temporales
PROCESSED_DIR = UPLOAD_DIR / "processed"            # Carpeta de imágenes procesadas
PLOTS_DIR = VISION_DIR / "infrastructure" / "plots" # Carpeta para guardar gráficas de métricas

# mkdir(..., exist_ok=True) — crea las carpetas si no existen; no falla si ya existen.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# 🚀 Lazy Loading
# ======================
# En lugar de instanciar VisionService() y PneumoniaService() al iniciar el servidor,
# los cargamos "a demanda" la primera vez que se usen.
# Esto evita que la API tarde en arrancar si los modelos son pesados.

_vision_service = None
_pneumonia_service = None

def get_vision_service():
    global _vision_service
    if _vision_service is None:
        print("⏳ Cargando VisionService por primera vez...")
        _vision_service = VisionService()
    return _vision_service

def get_pneumonia_service():
    global _pneumonia_service
    if _pneumonia_service is None:
        print("⏳ Cargando PneumoniaService por primera vez...")
        _pneumonia_service = PneumoniaService()
    return _pneumonia_service


# 🔍 DETECCIÓN GENERAL (YOLO)
@router.post(
    "/detect",
    summary="🔍 Detección de objetos en imagen",
    description="Sube una imagen para detectar y clasificar objetos en ella."
)
async def detect_objects(file: UploadFile = File(..., description="Imagen a analizar")):
    """
    Sube una imagen y recibe las detecciones de objetos encontradas.
    Retorna un JSON con las clases detectadas y sus probabilidades.
    """
    # Validación básica: si no hay nombre de archivo → responder con 400 Bad Request.
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo inválido")

    temp_path = UPLOAD_DIR / f"temp_{file.filename}"
    try:
        # Guardamos la imagen subida temporalmente en disco
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Procesamos con YOLO usando lazy loading
        try:
            result = get_vision_service().detect_objects(str(temp_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return result
    finally:
        # Siempre borramos la imagen temporal, haya éxito o error
        if temp_path.exists():
            os.remove(temp_path)


# 🩻 ANÁLISIS DE RAYOS X (NEUMONÍA)
@router.post(
    "/analyze-xray",
    summary="🩻 Análisis de radiografía (Neumonía)",
    description="Sube una radiografía de tórax y obtiene predicción de neumonía."
)
async def analyze_xray(file: UploadFile = File(..., description="Radiografía a analizar")):
    """
    Recibe una imagen de rayos X y devuelve si hay indicios de neumonía
    según el modelo CNN entrenado.
    """
    return await get_pneumonia_service().analyze_xray(file, file.filename)


# 📈 MÉTRICAS NEUMONÍA
@router.get(
    "/training-metrics",
    summary="📈 Ver métricas de entrenamiento neumonía",
    description="Devuelve la gráfica con las métricas del modelo de neumonía."
)
async def get_pneumonia_metrics():
    """
    Devuelve la gráfica `pneumonia_training.png` con las curvas
    de pérdida y exactitud del entrenamiento más reciente.
    """
    plot_path = PLOTS_DIR / "pneumonia_training.png" # Busca pneumonia_training.png en PLOTS_DIR.
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="No hay métricas entrenadas aún")
    return FileResponse(str(plot_path))


# 🔧 ENTRENAR MODELO NEUMONÍA
@router.post(
    "/train",
    summary="🔧 Reentrenar modelo de neumonía",
    description="Permite reentrenar el modelo CNN de neumonía especificando "
                "número de épocas y tasa de aprendizaje."
)
async def train_pneumonia(
    epochs: int = Query(5, description="Número de épocas de entrenamiento"),
    lr: float = Query(0.001, description="Tasa de aprendizaje")
):
    """
    Reentrena el modelo CNN de neumonía con los parámetros indicados.
    Genera nueva gráfica de métricas y actualiza el modelo guardado.
    """
    train_pneumonia_model(epochs=epochs, lr=lr)

    # ⚠️ Importante:
    # train_pneumonia_model es síncrono y pesado (entrenamiento).
    # Ejecutarlo desde un endpoint bloquea el event loop del servidor mientras se entrena
    # → la API dejará de responder hasta que termine.
    
    return {"message": f"Modelo de neumonía reentrenado por {epochs} épocas"}
