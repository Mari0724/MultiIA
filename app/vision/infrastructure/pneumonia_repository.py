import os
import shutil

import os
import shutil
import cv2


class PneumoniaRepository:
    def __init__(self, base_dir="vision/uploads"):
        # Directorios base
        self.base_dir = base_dir
        self.raw_dir = os.path.join(self.base_dir, "raw")        # Radiografías originales
        self.proc_dir = os.path.join(self.base_dir, "xray_proc") # Radiografías procesadas

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.proc_dir, exist_ok=True)

    def save_raw(self, file, filename: str) -> str:
        """
        Guarda la radiografía original en vision/uploads/raw.
        """
        file_path = os.path.join(self.raw_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        return file_path

    def save_processed(self, image, filename: str) -> str:
        """
        Guarda la radiografía procesada con anotaciones en vision/uploads/xray_proc.
        """
        file_path = os.path.join(self.proc_dir, filename)
        cv2.imwrite(file_path, image)
        return file_path
