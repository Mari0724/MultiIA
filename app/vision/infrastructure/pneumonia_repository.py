from pathlib import Path
import shutil
import cv2

class PneumoniaRepository:
    def __init__(self, base_dir: Path = None):
        """
        Maneja rutas de imágenes en app/vision/uploads/raw y app/vision/uploads/xray_proc
        sin importar desde dónde se ejecute el servidor.
        """
        if base_dir is None:
            # Sube 2 niveles hasta la raíz del proyecto (multiia/)
            root_dir = Path(__file__).resolve().parent.parent.parent  
            base_dir = root_dir / "app" / "vision" / "uploads"

        self.base_dir = base_dir
        self.raw_dir = base_dir / "raw"
        self.proc_dir = base_dir / "xray_proc"

        # Crear carpetas si no existen
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, file, filename: str) -> str:
        file_path = self.raw_dir / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        return str(file_path)

    def save_processed(self, image, filename: str) -> str:
        file_path = self.proc_dir / filename
        cv2.imwrite(str(file_path), image)
        return str(file_path)
