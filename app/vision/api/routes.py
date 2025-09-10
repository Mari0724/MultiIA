from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path
from app.vision.application.vision_service import VisionService
from app.vision.application.pneumonia_service import PneumoniaService
from app.vision.training.train_pneumonia import train_pneumonia_model
from app.vision.training.train_organ import train_organ_model

router = APIRouter(
    prefix="/vision",
    tags=["Vision"],
    responses={404: {"description": "No encontrado"}}
)

vision_service = VisionService()
pneumonia_service = PneumoniaService()

# 📂 Directorios base
VISION_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = VISION_DIR / "uploads"
PROCESSED_DIR = UPLOAD_DIR / "processed"
PLOTS_DIR = VISION_DIR / "infrastructure" / "plots"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# 🔍 DETECCIÓN GENERAL
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
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo inválido")

    temp_path = UPLOAD_DIR / f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            result = vision_service.detect_objects(str(temp_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return result
    finally:
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
    según el modelo entrenado.
    """
    return pneumonia_service.analyze_xray(file, file.filename)


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
    plot_path = PLOTS_DIR / "pneumonia_training.png"
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
    return {"message": f"Modelo de neumonía reentrenado por {epochs} épocas"}


# 🧠 ENTRENAR MODELO ÓRGANOS
@router.post(
    "/train-organ",
    summary="🧠 Reentrenar modelo de órganos/tórax",
    description="Entrena o reentrena el clasificador general para detectar si es tórax u otro órgano."
)
async def train_organ(
    epochs: int = Query(5, description="Número de épocas de entrenamiento"),
    lr: float = Query(0.001, description="Tasa de aprendizaje")
):
    """
    Entrena el modelo OrganClassifier, genera su gráfica y guarda el modelo.
    """
    train_organ_model(epochs=epochs, lr=lr)
    return {"message": f"Modelo de órgano reentrenado por {epochs} épocas"}


# 📊 MÉTRICAS ÓRGANOS
@router.get(
    "/training-metrics-organ",
    summary="📊 Ver métricas de entrenamiento de órgano",
    description="Devuelve la gráfica con las pérdidas del clasificador de órgano."
)
async def get_organ_metrics():
    """
    Devuelve la gráfica `organ_training.png` con las curvas de pérdida
    del entrenamiento del clasificador de órgano.
    """
    plot_path = PLOTS_DIR / "organ_training.png"
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="No hay métricas de órgano aún")
    return FileResponse(str(plot_path))
