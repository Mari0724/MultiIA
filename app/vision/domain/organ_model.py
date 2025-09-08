import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganClassifier(nn.Module):
    """
    Modelo simple para clasificar si una imagen es tórax o no.
    Salida: probabilidad [0,1] de que sea una radiografía de tórax.
    """
    def __init__(self):
        super(OrganClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # depende del tamaño de entrada
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Supone entrada normalizada a 1 canal (escala de grises)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # probabilidad entre 0 y 1
        return x
