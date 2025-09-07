from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from pathlib import Path
from app.vision.application.vision_service import VisionService
from vision.application.pneumonia_service import PneumoniaService

router = APIRouter(prefix="/vision", tags=["Vision"])
service = VisionService()
service = PneumoniaService()

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

        try:
            result = service.detect_objects(str(temp_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return result

    finally:
        # Limpieza segura
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass

@router.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    result = service.analyze_xray(file, file.filename)
    return result
