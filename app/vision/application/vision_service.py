from pathlib import Path            # Manejo de rutas de forma más amigable (objetos Path)
import cv2                          # OpenCV: usado para leer, escribir y dibujar sobre imágenes
from app.vision.infrastructure.vision_yolo import YoloDetector  # Detector basado en YOLO

class VisionService:
    def __init__(self):
        # Inicializa el detector YOLO
        self.detector = YoloDetector()

        # 🟥 Zona restringida (x1, y1, x2, y2) – ajusta a tu gusto
        self.restricted_area = (50, 50, 300, 300)

        # 📂 app/vision/uploads/processed
        self.VISION_DIR = Path(__file__).resolve().parents[1]              # .../app/vision
        self.UPLOAD_DIR = self.VISION_DIR / "uploads"
        self.PROCESSED_DIR = self.UPLOAD_DIR / "processed"
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Método principal: detección de objetos
    # ─────────────────────────────────────────────
    def detect_objects(self, image_path: str):
        detections = self.detector.detect(image_path)

        #  Alertas por intersección (mejor que "contenida 100%")
        alerts = []
        for det in detections:
            # Extrae coordenadas de cada detección (caja delimitadora)
            x1, y1, x2, y2 = map(int, det["bbox"])
            # Verifica si intersecta con la zona restringida
            if self._intersects_restricted_area(x1, y1, x2, y2):
                alerts.append(f"⚠️ {det['label']} dentro/encima de zona restringida")

        #  Dibuja zona + cajas y guarda imagen procesada dentro de app/vision
        processed_path = self._draw_on_image(image_path, detections)

        #  Resumen
        summary = {
            "total_objects": len(detections),
            "by_label": self._count_by_label(detections),
            "processed_image": str(processed_path)  # ruta del archivo en app/vision/uploads/processed
        }

        return {
            "summary": summary,
            "detections": detections,
            "alerts": alerts
        }

    # ─────────────────────────────────────────────
    # Método: chequea si una caja intersecta con la zona restringida
    # ─────────────────────────────────────────────
    def _intersects_restricted_area(self, x1, y1, x2, y2):
        # Coordenadas de la zona restringida
        rx1, ry1, rx2, ry2 = self.restricted_area

        # Calcula la intersección de rectángulos
        inter_x1 = max(x1, rx1)
        inter_y1 = max(y1, ry1)
        inter_x2 = min(x2, rx2)
        inter_y2 = min(y2, ry2)

        # Retorna True si hay área de intersección (> 0)
        return (inter_x2 - inter_x1) > 0 and (inter_y2 - inter_y1) > 0



    # ─────────────────────────────────────────────
    # Método: contar cuántos objetos hay de cada clase
    # ─────────────────────────────────────────────
    def _count_by_label(self, detections):
        counts = {}
        for det in detections:
            counts[det["label"]] = counts.get(det["label"], 0) + 1
        return counts


   
        # ─────────────────────────────────────────────
    # Método: dibuja cajas, textos y zona restringida en la imagen
    # ─────────────────────────────────────────────
    def _draw_on_image(self, image_path, detections):
        # Leer imagen desde disco
        img = cv2.imread(image_path)
        if img is None:
            # Manejo robusto en caso de error de lectura (evita el "need at least one array to stack")
            raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

        #  Zona restringida (rojo)
        rx1, ry1, rx2, ry2 = map(int, self.restricted_area)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        cv2.putText(img, "Zona Restringida", 
                    (rx1, max(15, ry1 - 8)),      # posición del texto
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #  Detecciones
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])   # coordenadas
            label = det.get("label", "obj")          # clase
            conf = det.get("confidence", 0.0)        # confianza
            tag = f"{label} {conf:.2f}"              # texto: clase + confianza

            # caja
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Determina posición del texto encima o debajo
            ty = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(img, tag, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #  Guardar dentro de app/vision/uploads/processed
        out_name = f"result_{Path(image_path).name}"
        out_path = self.PROCESSED_DIR / out_name
        cv2.imwrite(str(out_path), img)
        return out_path
