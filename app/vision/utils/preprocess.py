import cv2
import torch
import numpy as np
from torchvision import transforms

def preprocess_image(file_path: str, target_size=(224, 224), for_batch: bool = True):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Escalar a [0,1]

    # Normalización estándar
    mean = 0.5
    std = 0.5
    img = (img - mean) / std  # Escalar a [-1,1]

    # Expandir canal (grayscale → 1 canal)
    img = np.expand_dims(img, axis=0)  # [1,H,W]

    if for_batch:
        img = np.expand_dims(img, axis=0)  # [1,1,H,W]

    return torch.tensor(img, dtype=torch.float32)
