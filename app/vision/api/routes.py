from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from pathlib import Path
from app.vision.application.vision_service import VisionService

router = APIRouter(prefix="/vision", tags=["Vision"])
service = VisionService()

# 📂 app/vision como base
VISION_DIR = Path(__file__).resolve().parents[1]  # .../app/vision
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

        # Detectar objetos (el servicio guardará la imagen procesada en app/vision/uploads/processed)
        result = service.detect_objects(str(temp_path))
        return result

    finally:
        # Limpieza segura
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass
