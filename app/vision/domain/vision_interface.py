from typing import List, Dict
from abc import ABC, abstractmethod

class DetectorInterface(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> List[Dict]:
        """
        Detecta objetos en una imagen y devuelve una lista con:
        - label (nombre del objeto)
        - confidence (probabilidad)
        - bbox (coordenadas de la caja)
        """
        pass
