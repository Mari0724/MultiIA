from ultralytics import YOLO
from typing import List, Dict
from app.vision.domain.vision_interface import DetectorInterface
import os

class YoloDetector(DetectorInterface):
    def __init__(self, model_name: str = "app/vision/infrastructure/model/yolov8n.pt"):
        print("🔍 Cargando modelo YOLO...")
        self.model = YOLO(model_name)

    def detect(self, image_path: str) -> List[Dict]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ No se encontró la imagen: {image_path}")

        try:
            results = self.model(image_path)
        except Exception as e:
            raise RuntimeError(f"❌ Error al procesar la imagen con YOLO: {str(e)}")

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]
                conf = float(box.conf[0])
                detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "status": "seguro" if conf >= 0.5 else "dudoso",
                    "bbox": box.xyxy.tolist()[0]
                })
        return detections
