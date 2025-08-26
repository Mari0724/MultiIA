import torch  
from torch import nn  # nn = “neural networks”, como la caja de juguetes del gato para aprender

# Definimos nuestra clase de modelo de regresión lineal
class LinearRegressor(nn.Module):
    """
    Modelo de **regresión lineal simple**.
    
    Fórmula matemática:
        y = Wx + b
    
    - `W` = peso que el modelo aprende (pendiente de la recta).
    - `b` = sesgo o intercepto.
    """

    def __init__(self):
        """
        Constructor del modelo:
        - Define una capa lineal con 1 entrada y 1 salida.
        - Significa que, dado un valor `x`, el modelo calculará:
              y = W * x + b
        """
        super().__init__()  # Llamamos al constructor de nn.Module (gato se prepara para aprender)

        # Creamos una capa lineal con 1 entrada y 1 salida
        # Esto significa que el modelo aprenderá un peso y un sesgo: y = Wx + b
        self.linear = nn.Linear(1, 1)  # La regla de juego del gato: multiplicar y sumar para llegar a la comida

    def forward(self, x):
        """
        Definimos la pasada hacia adelante (forward pass) del modelo.
        Toma la entrada x y devuelve la predicción y_pred = Wx + b.
        """
        return self.linear(x)  # El gato toma x, aplica su regla lineal y devuelve la salida

class LogisticRegressor(nn.Module):
    """
    Modelo de **regresión logística** para clasificación binaria.
    
    Fórmula matemática:
        y_pred = sigmoid(Wx + b)
    
    - `Wx + b`: combinación lineal de las entradas.
    - `sigmoid`: función que convierte el resultado en una probabilidad [0,1].
    """
    def __init__(self):
        """
        Constructor del modelo:
        - Define una capa lineal con 2 entradas y 1 salida.
        - Ejemplo de uso: (velocidad, energía) -> probabilidad de atrapar al ratón.
        """
        super().__init__()
        self.linear = nn.Linear(2, 1)   # 2 entradas -> 1 salida (clasificación binaria)

    def forward(self, x):
        """
        Pasada hacia adelante:
        - Calcula z = Wx + b con la capa lineal.
        - Aplica la función sigmoide para convertir z en probabilidad.
        - Devuelve un valor entre 0 y 1.
        """
        return torch.sigmoid(self.linear(x))  # probabilidad entre 0 y 1