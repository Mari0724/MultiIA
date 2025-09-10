import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganClassifier(nn.Module):
    """
    Clasificador simple: entrada en escala de grises (1 canal).
    Devuelve probabilidad de que sea una radiografía de tórax.
    Totalmente flexible al tamaño de entrada.
    """
    def __init__(self):
        super(OrganClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # AdaptiveAvgPool2d hace que la salida sea SIEMPRE 8x8
        self.gap = nn.AdaptiveAvgPool2d((8, 8))
        
        # Ahora el feature_dim siempre es 32*8*8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)  # salida fija
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
