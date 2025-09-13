from ultralytics import YOLO
from typing import List, Dict
from app.vision.domain.vision_interface import DetectorInterface
import os

class YoloDetector(DetectorInterface):
    def __init__(self, model_name: str = "app/vision/infrastructure/model/yolov8n.pt"):
        print("🔍 Cargando modelo YOLO...")
        # Crea una instancia del modelo YOLO y carga los pesos del archivo especificado.
        # Esto prepara el modelo para la detección de objetos
        self.model = YOLO(model_name)

    def detect(self, image_path: str) -> List[Dict]:
        
        # Verifica si el archivo de imagen existe en la ruta proporcionada.
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ No se encontró la imagen: {image_path}")

        try:
            # Llama al modelo para que procese la imagen.
            # Este es el paso principal donde se ejecuta la detección de objetos.
            results = self.model(image_path)
        # Si ocurre una excepción, la captura y la maneja.
        except Exception as e:
            raise RuntimeError(f"❌ Error al procesar la imagen con YOLO: {str(e)}")

            # Inicializa una lista vacía para almacenar los resultados de las detecciones.
        detections = []
        # Itera sobre los resultados de la detección. 'results' es una lista,
        # incluso si solo se procesó una imagen.
        for r in results:
            # Itera sobre cada cuadro delimitador ('box') que el modelo encontró.
            # Cada 'box' contiene la clase, confianza y coordenadas del objeto.
            for box in r.boxes:
                # Obtiene el ID de la clase detectada (ej. 0 para 'persona', 1 para 'coche').
                cls = int(box.cls[0])
                # Usa el ID para obtener el nombre de la etiqueta (ej. "persona", "coche").
                label = r.names[cls]
                # Obtiene el nivel de confianza de la detección, un valor entre 0 y 1.
                conf = float(box.conf[0])
                # Añade un diccionario a la lista 'detections' con toda la información.
                detections.append({
                    "label": label, # El nombre del objeto.
                    "confidence": float(conf), # Nivel de confianza.
                    # Asigna un estado basado en la confianza: 'seguro' si es >= 0.5,
                    # de lo contrario, 'dudoso'.
                    "status": "seguro" if conf >= 0.5 else "dudoso",
                    # Obtiene las coordenadas del cuadro delimitador y las convierte en una lista.
                    "bbox": box.xyxy.tolist()[0]
                })
        # Devuelve la lista completa de objetos detectados.
        return detections