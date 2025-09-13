from pathlib import Path
import torch
import cv2
import numpy as np
from ultralytics import YOLO

from app.vision.domain.pneumonia_model import SimpleCNN
from app.vision.utils.preprocess import preprocess_image
from app.vision.utils.draw import draw_xray_annotation
from app.vision.infrastructure.pneumonia_repository import PneumoniaRepository


class PneumoniaService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── raíz del proyecto
        ROOT_DIR = Path(__file__).resolve().parents[3]  # multiia/
        self.BASE_DIR = ROOT_DIR / "app" / "vision"

        # ── carpetas base
        self.MODELS_DIR = self.BASE_DIR / "infrastructure" / "model"
        self.UPLOADS_DIR = self.BASE_DIR / "uploads"
        self.XRAY_PROC_DIR = self.UPLOADS_DIR / "xray_proc"
        self.XRAY_PROC_DIR.mkdir(parents=True, exist_ok=True)

        print("DEBUG Base dir:", self.BASE_DIR)
        print("DEBUG Uploads dir:", self.UPLOADS_DIR)
        print("DEBUG Models dir:", self.MODELS_DIR)

        # ── rutas de modelos
        pneumonia_path = self.MODELS_DIR / "pneumonia_cnn.pth"

        # ── cargar modelo de neumonía
        self.pneumonia_model = SimpleCNN().to(self.device)
        try:
            self.pneumonia_model.load_state_dict(
                torch.load(pneumonia_path, map_location=self.device)
            )
            self.pneumonia_model.eval()
            self.pneumonia_model_loaded = True
            print("✅ Modelo de neumonía cargado.")
        except Exception as e:
            self.pneumonia_model_loaded = False
            print("⚠️ Modelo de neumonía NO encontrado o error cargándolo:", e)

        # ── cargar YOLO (opcional)
        try:
            yolo_path = self.MODELS_DIR / "yolov8n.pt"  # si está aquí
            if yolo_path.exists():
                self.yolo_model = YOLO(str(yolo_path))
                self.yolo_loaded = True
                print("✅ YOLO cargado desde:", yolo_path)
            else:
                self.yolo_model = None
                self.yolo_loaded = False
                print("ℹ️ YOLO no encontrado en", yolo_path)
        except Exception as e:
            print("⚠️ Error cargando YOLO:", e)
            self.yolo_loaded = False
            self.yolo_model = None

        # repo (maneja uploads/raw y uploads/xray_proc)
        self.repo = PneumoniaRepository(self.UPLOADS_DIR)

        # parámetros ajustables (modifícalos si quieres)
        self.grayscale_tolerance = 6      # tolerancia en diferencia de canales (0 = estrictamente iguales)
        self.yolo_conf_threshold = 0.45   # confianza mínima para considerar una detección "fuerte"
        self.yolo_allowed = {"person"}    # si todas las detecciones fuertes son de estas clases, permitir imagen

    async def analyze_xray(self, file, filename: str):
        """
        1) Guarda la imagen subida (async)
        2) Verifica que sea grayscale (o RGB que sea efectivamente gris)
        3) (Opcional) Pasa por YOLO para descartar imágenes con objetos típicos (gato, auto, etc.)
        4) Preprocesa y pasa el tensor al modelo de neumonía
        5) Devuelve paths y predicción
        """
        # 1) Guardar original en uploads/raw
        file_path = await self.repo.save_raw(file, filename)

        # 2) Leer imagen con OpenCV
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {
                "file_path": str(file_path),
                "prediction": "Error leyendo imagen",
                "confidence": None
            }

        # 2.a) Detectar si es escala de grises (permitimos pequeños desvíos)
        is_grayscale = False
        if len(img.shape) == 2:
            # ya es monocanal
            is_grayscale = True
        elif len(img.shape) == 3 and img.shape[2] == 3:
            b, g, r = cv2.split(img)
            # usar diferencia máxima entre canales (tolerancia)
            max_bg = int(np.max(np.abs(b.astype(np.int16) - g.astype(np.int16))))
            max_gr = int(np.max(np.abs(g.astype(np.int16) - r.astype(np.int16))))
            if max_bg <= self.grayscale_tolerance and max_gr <= self.grayscale_tolerance:
                is_grayscale = True

        if not is_grayscale:
            annotated = draw_xray_annotation(
                img_path=file_path,
                is_chest=False,
                prediction="Imagen inválida (no está en escala de grises)",
                confidence=0.0
            )
            proc_path = self.repo.save_processed(annotated, f"invalid_{filename}")
            return {
                "file_path": str(file_path),
                "processed_path": proc_path,
                "prediction": "Imagen inválida (no está en escala de grises)",
                "confidence": None
            }

        # 3) Filtro YOLO: si YOLO está cargado, verificar detecciones "fuertes"
        if self.yolo_loaded and self.yolo_model is not None:
            try:
                results = self.yolo_model(str(file_path))  # devuelve lista de Results
                if results and len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        # intentar obtener clases y confs; si falla, consideramos "detecciones" como motivo de rechazo
                        try:
                            cls_tensor = boxes.cls  # tensor
                            conf_tensor = boxes.conf
                            cls_ids = cls_tensor.cpu().numpy().astype(int).tolist()
                            confs = conf_tensor.cpu().numpy().tolist()
                            labels = [r.names[int(c)] for c in cls_ids]
                            # detecciones "fuertes"
                            strong = [(lab, conf) for lab, conf in zip(labels, confs) if conf >= self.yolo_conf_threshold]
                            if len(strong) > 0:
                                # si TODAS las detecciones fuertes están en la lista de permitidas -> permitir,
                                # si alguna NO está -> RECHAZAR (no es radiografía)
                                if not all(lab in self.yolo_allowed for lab, _ in strong):
                                    annotated = draw_xray_annotation(
                                        img_path=file_path,
                                        is_chest=False,
                                        prediction="Imagen inválida (objetos detectados por YOLO)",
                                        confidence=0.0
                                    )
                                    proc_path = self.repo.save_processed(annotated, f"invalid_{filename}")
                                    return {
                                        "file_path": str(file_path),
                                        "processed_path": proc_path,
                                        "prediction": "Imagen inválida (objetos detectados por YOLO)",
                                        "confidence": None
                                    }
                        except Exception as e:
                            # fallback: si no pudimos leer clases/confidencias, y hay cajas -> RECHAZAR
                            print("⚠️ Error extrayendo clases de YOLO o cajas presentes:", e)
                            annotated = draw_xray_annotation(
                                img_path=file_path,
                                is_chest=False,
                                prediction="Imagen inválida (detección no válida)",
                                confidence=0.0
                            )
                            proc_path = self.repo.save_processed(annotated, f"invalid_{filename}")
                            return {
                                "file_path": str(file_path),
                                "processed_path": proc_path,
                                "prediction": "Imagen inválida (detección no válida)",
                                "confidence": None
                            }
            except Exception as e:
                # si YOLO falla por cualquier motivo, NO bloqueamos el flujo: solo lo logueamos
                print("⚠️ YOLO inference error (se continúa sin usar la salida):", e)

        # 4) Preprocesar a tensor para modelo de neumonía
        img_tensor = preprocess_image(str(file_path)).to(self.device)  # devuelve tensor [1,1,H,W]

        # 5) Predicción de neumonía
        if not self.pneumonia_model_loaded:
            return {
                "file_path": str(file_path),
                "prediction": "Modelo de neumonía no entrenado",
                "confidence": None
            }

        with torch.no_grad():
            out = self.pneumonia_model(img_tensor)
            prob = float(torch.sigmoid(out).cpu().numpy().item())  # prob en [0,1]

        prediction = "Pneumonia" if prob > 0.5 else "Normal"

        # 6) Anotar y guardar procesada
        annotated = draw_xray_annotation(
            img_path=file_path,
            is_chest=True,
            prediction=prediction,
            confidence=prob
        )
        processed_path = self.repo.save_processed(annotated, f"proc_{filename}")

        return {
            "file_path": str(file_path),
            "processed_path": processed_path,
            "prediction": prediction,
            "confidence": prob
        }
