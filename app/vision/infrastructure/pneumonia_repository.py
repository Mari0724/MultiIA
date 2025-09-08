import os
import shutil

class PneumoniaRepository:
    def __init__(self, base_dir="vision/uploads"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "xray")
        self.proc_dir = os.path.join(base_dir, "xray_proc")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.proc_dir, exist_ok=True)

    def save_raw(self, file, filename: str) -> str:
        """Guarda la radiografía original."""
        file_path = os.path.join(self.raw_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        return file_path

    def save_processed(self, image, filename: str) -> str:
        """Guarda la radiografía procesada con marcas."""
        file_path = os.path.join(self.proc_dir, filename)
        import cv2
        cv2.imwrite(file_path, image)
        return file_path
