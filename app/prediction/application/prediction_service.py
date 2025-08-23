from app.prediction.infrastructure.model_storage import save_model, load_model
from app.prediction.domain.models import LinearRegressor, LogisticRegressor
import torch
from torch import nn

# Ruta donde se guardará el modelo entrenado
MODEL_PATH = "app/prediction/infrastructure/models/linear_regression.pth"
MODEL_PATH_LOGISTIC = "app/prediction/infrastructure/models/logistic_regression.pth"

def train_linear_model():
    """
    Función para entrenar un modelo de regresión lineal simple.
    Genera datos ficticios, entrena el modelo y lo guarda.
    """
    
    # Creamos 200 datos de entrada x aleatorios de dimensión (200,1)
    x = torch.randn(200, 1)
    # Creamos y = 3*x + 2 + ruido aleatorio (0.5) para simular datos reales
    y = 3 * x + 2 + 0.5 * torch.randn(200, 1)

    # Creamos el modelo de regresión lineal (1 input, 1 output)
    model = LinearRegressor()
    # Definimos el optimizador SGD con tasa de aprendizaje 0.05
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    # Función de pérdida: error cuadrático medio
    loss_fn = nn.MSELoss() # 🐱 le decimos al gato cuán lejos está de la comida perfecta


    # Entrenamos el modelo durante 300 iteraciones
    for _ in range(300):
        pred = model(x)              # 🔹 predicción del modelo (el gato intenta atrapar la pelota)
        loss = loss_fn(pred, y)      # 🔹 calculamos qué tan mal lo hizo
        optim.zero_grad()            # 🔹 limpiamos gradientes previos (gato se sacude las patas)
        loss.backward()              # 🔹 calculamos gradientes (gato aprende)
        optim.step()                 # 🔹 actualizamos pesos (gato ajusta su salto)

    # Guardamos el modelo entrenado en la ruta definida
    save_model(model, MODEL_PATH)
    # Retornamos un mensaje y la última pérdida
    return {"message": "Modelo entrenado", "loss": loss.item()}

def predict_linear(x: float):
    """
    Función para hacer predicciones usando el modelo entrenado.
    Recibe un valor x y devuelve y_pred.
    """
    
    # Cargamos el modelo entrenado desde disco
    model = load_model(LinearRegressor, MODEL_PATH)  # 🐱 sacamos al gato de su casita
    if model is None:
        raise FileNotFoundError()  # 🚨 si no existe el modelo, lanzamos error
    
    # No necesitamos gradientes para predicción
    with torch.no_grad():
        # Convertimos x a tensor y hacemos la predicción
        y_pred = model(torch.tensor([[x]])).item()  # 🐱 gato mira dónde caerá la pelota
        return {"x": x, "y_pred": y_pred}  # 🔹 devolvemos la predicción




# -------- LOGÍSTICA (nuevo) -------- #
def train_logistic_model():
    """
    Entrena un modelo de clasificación binaria simple
    """
    # Datos ficticios: 200 puntos con 2 características (x1, x2)
    x = torch.randn(200, 2)
    # Etiquetas binarias: 0 o 1 (con probabilidad 0.5)
    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)

    model = LogisticRegressor()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy

    for _ in range(300):
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    save_model(model, MODEL_PATH_LOGISTIC)
    return {"message": "Modelo logístico entrenado", "loss": loss.item()}

def predict_logistic(x1: float, x2: float):
    model = load_model(LogisticRegressor, MODEL_PATH_LOGISTIC)
    if model is None:
        raise FileNotFoundError()
    with torch.no_grad():
        prob = model(torch.tensor([[x1, x2]])).item()
        clase = 1 if prob >= 0.5 else 0
        return {"x1": x1, "x2": x2, "probabilidad": prob, "clase": clase}