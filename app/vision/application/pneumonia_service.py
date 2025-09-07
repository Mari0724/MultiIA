import torch
from vision.domain.pneumonia_model import SimpleCNN
from vision.domain.organ_model import OrganClassifier
from vision.infrastructure.pneumonia_repository import PneumoniaRepository
from vision.utils.preprocess import preprocess_image
from vision.utils.draw import draw_xray_annotation


class PneumoniaService:
    def __init__(self,
        pneumonia_path="vision/models/pneumonia_cnn.pth",
        organ_path="vision/models/organ_cnn.pth"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelo de neumonía
        self.pneumonia_model = SimpleCNN().to(self.device)
        try:
            self.pneumonia_model.load_state_dict(torch.load(pneumonia_path, map_location=self.device))
            self.pneumonia_model.eval()
            self.pneumonia_model_loaded = True
        except FileNotFoundError:
            self.pneumonia_model_loaded = False

        # Modelo de validación de órgano
        self.organ_model = OrganClassifier().to(self.device)
        try:
            self.organ_model.load_state_dict(torch.load(organ_path, map_location=self.device))
            self.organ_model.eval()
            self.organ_model_loaded = True
        except FileNotFoundError:
            self.organ_model_loaded = False

        self.repo = PneumoniaRepository()

    def analyze_xray(self, file, filename: str):
        # 1. Guardar imagen original
        file_path = self.repo.save_raw(file, filename)

        # 2. Preprocesar imagen
        img_tensor = preprocess_image(file_path).to(self.device)

        # 3. Validar que sí sea tórax
        if not self.organ_model_loaded:
            return {"error": "Modelo de órgano no entrenado"}

        with torch.no_grad():
            is_chest = self.organ_model(img_tensor).item() > 0.5

        if not is_chest:
            return {
                "file_path": file_path,
                "prediction": "Imagen inválida: no es radiografía de tórax",
                "confidence": None
            }

        # 4. Predecir neumonía
        if not self.pneumonia_model_loaded:
            return {"file_path": file_path, "prediction": "Modelo de neumonía no entrenado", "confidence": None}

        with torch.no_grad():
            prob = self.pneumonia_model(img_tensor).item()

        prediction = "Pneumonia" if prob > 0.5 else "Normal"

        # 5. Dibujar anotación
        annotated = draw_xray_annotation(
            file_path=file_path,
            is_chest=is_chest,
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
