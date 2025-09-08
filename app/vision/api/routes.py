from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path
from app.vision.application.vision_service import VisionService
from app.vision.application.pneumonia_service import PneumoniaService
from app.vision.training.train_pneumonia import train_pneumonia_model

router = APIRouter(prefix="/vision", tags=["Vision"])

vision_service = VisionService()
pneumonia_service = PneumoniaService()

# 📂 app/vision como base
VISION_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = VISION_DIR / "uploads"
PROCESSED_DIR = UPLOAD_DIR / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
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
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass


@router.post("/analyze-xray")
async def analyze_xray(file: UploadFile = File(...)):
    return pneumonia_service.analyze_xray(file, file.filename)


@router.get("/training-metrics")
async def get_pneumonia_metrics():
    plot_path = "vision/infrastructure/plots/pneumonia_training.png"
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="No hay métricas entrenadas aún")
    return FileResponse(plot_path)


@router.post("/train")
async def train_pneumonia(epochs: int = 5, lr: float = 0.001):
    train_pneumonia_model(epochs=epochs, lr=lr)
    return {"message": f"Modelo reentrenado por {epochs} épocas"}