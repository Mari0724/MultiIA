from pathlib import Path
import torch

from app.vision.domain.pneumonia_model import SimpleCNN
from app.vision.domain.organ_model import OrganClassifier
from app.vision.utils.preprocess import preprocess_image
from app.vision.utils.draw import draw_xray_annotation

# ⚠️ Ajusta este import si tu repository está en otro lugar:
#   si lo tienes en app/vision/infraestructure/pneumonia/repository.py:
from app.vision.infrastructure.pneumonia_repository import PneumoniaRepository
#   (si lo tienes como app/vision/infraestructure/pneumonia_repository.py,
#    cambia el import a: from app.vision.infraestructure.pneumonia_repository import PneumoniaRepository)

class PneumoniaService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── carpetas base ───────────────────────────────────────────
        self.BASE_DIR = Path(__file__).resolve().parents[1]  # app/vision
        self.MODELS_DIR = self.BASE_DIR / "infraestructure" / "model"
        self.UPLOADS_DIR = self.BASE_DIR / "uploads"
        self.XRAY_PROC_DIR = self.UPLOADS_DIR / "xray_proc"
        self.XRAY_PROC_DIR.mkdir(parents=True, exist_ok=True)

        # ── rutas de modelos ───────────────────────────────────────
        pneumonia_path = self.MODELS_DIR / "pneumonia_cnn.pth"
        organ_path = self.MODELS_DIR / "organ_cnn.pth"

        # ── cargar modelos ─────────────────────────────────────────
        self.pneumonia_model = SimpleCNN().to(self.device)
        try:
            self.pneumonia_model.load_state_dict(
                torch.load(pneumonia_path, map_location=self.device)
            )
            self.pneumonia_model.eval()
            self.pneumonia_model_loaded = True
        except FileNotFoundError:
            self.pneumonia_model_loaded = False

        self.organ_model = OrganClassifier().to(self.device)
        try:
            self.organ_model.load_state_dict(
                torch.load(organ_path, map_location=self.device)
            )
            self.organ_model.eval()
            self.organ_model_loaded = True
        except FileNotFoundError:
            self.organ_model_loaded = False

        self.repo = PneumoniaRepository()  # guarda en uploads/raw y uploads/xray_proc

    def analyze_xray(self, file, filename: str):
        # 1) Guardar original en uploads/raw
        file_path = self.repo.save_raw(file, filename)

        # 2) Preprocesar a tensor
        img_tensor = preprocess_image(file_path).to(self.device)

        # 3) Validación de órgano (radiografía de tórax)
        if not self.organ_model_loaded:
            return {"error": "Modelo de órgano no entrenado"}

        with torch.no_grad():
            is_chest = self.organ_model(img_tensor).item() > 0.5

        if not is_chest:
            annotated = draw_xray_annotation(
                img_path=file_path, is_chest=False, prediction="Imagen inválida", confidence=0.0
            )
            proc_path = self.repo.save_processed(annotated, f"invalid_{filename}")
            return {
                "file_path": file_path,
                "processed_path": proc_path,
                "prediction": "Imagen inválida: no es radiografía de tórax",
                "confidence": None
            }

        # 4) Predicción de neumonía
        if not self.pneumonia_model_loaded:
            return {
                "file_path": file_path,
                "prediction": "Modelo de neumonía no entrenado",
                "confidence": None
            }

        with torch.no_grad():
            prob = self.pneumonia_model(img_tensor).item()

        prediction = "Pneumonia" if prob > 0.5 else "Normal"

        # 5) Anotar y guardar en uploads/xray_proc
        annotated = draw_xray_annotation(
            img_path=file_path,
            is_chest=True,
            prediction=prediction,
            confidence=prob
        )
        processed_path = self.repo.save_processed(annotated, f"proc_{filename}")

        return {
            "file_path": file_path,
            "processed_path": processed_path,
            "prediction": prediction,
            "confidence": prob
        }
