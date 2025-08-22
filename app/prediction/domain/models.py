from torch import nn  # nn = “neural networks”, como la caja de juguetes del gato para aprender


# Definimos nuestra clase de modelo de regresión lineal
class LinearRegressor(nn.Module):
    """
    Modelo de regresión lineal simple:
    y = Wx + b
    """

    def __init__(self):
        """
        Inicializador del modelo.
        Aquí definimos las capas que usará nuestro modelo.
        """
        super().__init__()  # Llamamos al constructor de nn.Module (gato se prepara para aprender)

        # Creamos una capa lineal con 1 entrada y 1 salida
        # Esto significa que el modelo aprenderá un peso y un sesgo: y = Wx + b
        self.linear = nn.Linear(1, 1)  # La regla de juego del gato: multiplicar y sumar para llegar a la comida

    def forward(self, x):
        """
        Definimos la pasada hacia adelante (forward pass) del modelo.
        Toma la entrada x y devuelve la predicción y_pred.
        """
        return self.linear(x)  # El gato toma x, aplica su regla lineal y devuelve la salida
