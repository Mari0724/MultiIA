import cv2
import torch
import numpy as np

def preprocess_image(file_path: str, target_size=(224, 224)):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Escalar a [0,1]

    # Normalización estándar
    mean = 0.5
    std = 0.5
    img = (img - mean) / std  # Escalar a [-1,1]

    # Expandir dimensiones para que PyTorch lo entienda: [batch, channel, H, W]
    img = np.expand_dims(img, axis=0)  # canal
    img = np.expand_dims(img, axis=0)  # batch

    return torch.tensor(img, dtype=torch.float32)
