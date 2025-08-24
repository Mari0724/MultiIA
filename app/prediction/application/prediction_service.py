from app.prediction.infrastructure.model_storage import save_model, load_model
from app.prediction.domain.models import LinearRegressor, LogisticRegressor
import torch
from torch import nn
import math

# -------------------------
# 🔹 Función de validación profesional
# -------------------------
def safe_float(value: float) -> float:
    """Convierte a float válido para JSON. 
    Si el valor es NaN o infinito, lanza un ValueError para detener el flujo."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Valor no válido detectado: {value}")
    return float(value)


# -------------------------
# 🔹 Rutas donde se guardarán los modelos
# -------------------------
LINEAR_MODEL_PATH = "app/prediction/infrastructure/models/linear_regression.pth"
LOGISTIC_MODEL_PATH = "app/prediction/infrastructure/models/logistic_regression.pth"


# -------------------------
# 🔹 Regresión Lineal (peso ~ tamaño)
# -------------------------
def train_linear_model():
    # Entradas: largo del gato (20–60 cm)
    x_raw = torch.linspace(20, 60, 200).unsqueeze(1)

    # Normalizar para entrenamiento (media 40, std 10 aprox)
    x = (x_raw - x_raw.mean()) / x_raw.std()

    # Relación realista peso~largo (ejemplo)
    # Relación más realista entre largo (cm) y peso (kg)
    y = 0.18 * x_raw - 2 + 0.5 * torch.randn(200, 1)

    model = LinearRegressor()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    last_loss = None
    for _ in range(1000):
        pred = model(x)
        loss = loss_fn(pred, y)

        if torch.isnan(loss):
            raise ValueError("El entrenamiento produjo NaN en la pérdida")

        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = loss.item()

    save_model(model, LINEAR_MODEL_PATH)
    return {
        "message": "Modelo lineal entrenado (peso ~ tamaño)",
        "loss": safe_float(last_loss)
    }


def predict_linear(size: float):
    model = load_model(LinearRegressor, LINEAR_MODEL_PATH)
    if model is None:
        raise FileNotFoundError()

    # Normalizar con misma lógica que entrenamiento
    x_raw = torch.tensor([[size]], dtype=torch.float32)
    x_norm = (x_raw - 40) / 10  # media ~40 cm, std ~10 cm

    with torch.no_grad():
        peso_pred = model(x_norm).item()
        return {
            "tamaño_cm": safe_float(size),
            "peso_pred_kg": safe_float(max(peso_pred, 0.5))  # nunca <0.5 kg
        }



# -------------------------
# 🔹 Regresión Logística (atrapa ratón o no)
# -------------------------
def train_logistic_model():
    # Datos: velocidad (0-10 m/s) y energía (0-1)
    velocidad = torch.rand(500, 1) * 10
    energia = torch.rand(500, 1)

    # Etiqueta: atrapa ratón (1) si velocidad*energía > 3
    y = ((velocidad * energia) > 3).float()
    x = torch.cat([velocidad, energia], dim=1)  # entradas (500,2)

    model = LogisticRegressor()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    last_loss = None
    for _ in range(500):
        pred = model(x)
        loss = loss_fn(pred, y)

        if torch.isnan(loss):
            raise ValueError("El entrenamiento produjo NaN en la pérdida")

        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = loss.item()

    save_model(model, LOGISTIC_MODEL_PATH)
    return {
        "message": "Modelo logístico entrenado (atrapa ratón)",
        "loss": safe_float(last_loss)
    }


def predict_logistic(x1: float, x2: float):
    model = load_model(LogisticRegressor, LOGISTIC_MODEL_PATH)
    if model is None:
        raise FileNotFoundError()

    with torch.no_grad():
        prob = model(torch.tensor([[x1, x2]])).item()
        clase = 1 if prob >= 0.5 else 0
        return {
            "velocidad": safe_float(x1),
            "energia": safe_float(x2),
            "probabilidad": safe_float(prob),
            "clase": clase
        }
