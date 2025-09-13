import cv2
import torch
import numpy as np
from torchvision import transforms

def preprocess_image(file_path: str, target_size=(224, 224), for_batch: bool = True):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = (img - 0.5) / 0.5  # Normalizar a [-1,1]

    img = np.expand_dims(img, axis=0)  # [1,H,W] canal único

    tensor = torch.tensor(img, dtype=torch.float32)
    if for_batch:
        tensor = tensor.unsqueeze(0)  # [1,1,H,W]
    return tensor
