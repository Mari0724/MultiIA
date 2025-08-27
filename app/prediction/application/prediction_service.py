import torch
from torch import nn
# 👉 Librería PyTorch para trabajar con tensores y redes neuronales.

import math
import os
# 👉 Math = validaciones matemáticas (NaN, infinito).
# 👉 OS = crear carpetas donde se guardan modelos y gráficas.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
# 👉 Métricas de evaluación para el modelo logístico.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 👉 Librería para graficar pérdida, matrices de confusión, curva ROC.
# 👉 "Agg" = modo sin interfaz gráfica (evita errores en servidor).

from app.prediction.domain.models import LinearRegressor, LogisticRegressor
from app.prediction.infrastructure.model_storage import save_model, load_model


# ===================== FUNCIÓN DE VALIDACIÓN =====================

def safe_float(value: float) -> float:
    """
    Convierte un número a float seguro para serialización JSON.
    - Si el valor es NaN (Not a Number) o infinito → lanza un error.
    - Garantiza que las respuestas de la API sean válidas en JSON.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Valor no válido detectado: {value}")
    return float(value)

# ===================== RUTAS DE MODELOS Y PLOTS =====================

LINEAR_MODEL_PATH = "app/prediction/infrastructure/models/linear_regression.pth"
LOGISTIC_MODEL_PATH = "app/prediction/infrastructure/models/logistic_regression.pth"
PLOT_DIR = "app/prediction/infrastructure/plots"

os.makedirs(PLOT_DIR, exist_ok=True)
# 👉 Se asegura de que la carpeta para guardar gráficas exista.


# ===================== MODELO LINEAL =====================

def train_linear_model(save_plot: bool = True):
    """
    Entrena un modelo de regresión lineal para predecir el **peso del gato**
    a partir de su **tamaño**.
    """
    # --- 1. Datos simulados (20–60 cm, con ruido aleatorio) ---
    x_raw = torch.linspace(20, 60, 200).unsqueeze(1)
    x = (x_raw - x_raw.mean()) / x_raw.std()
    y = (
        0.18 * x_raw - 2
        + 0.5 * torch.randn(200, 1)  # ruido normal
        + 0.02 * (x_raw**1.5) / 100  # curva ligera
    )

    # --- 2. Modelo y entrenamiento ---
    model = LinearRegressor()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in range(1000):
        pred = model(x)             # predicciones
        loss = loss_fn(pred, y)     # error MSE

        """
        Qué valida: Que la función de pérdida no explote y devuelva NaN.
        Por qué: a veces el gradiente se descontrola y el modelo “revienta”.
        Protege: el proceso de entrenamiento.
        """
        if torch.isnan(loss):
            raise ValueError("El entrenamiento produjo NaN en la pérdida")

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    # --- 3. Guardar modelo entrenado ---
    if save_plot: 
        save_model(model, LINEAR_MODEL_PATH)

    # --- 4. Guardar gráfica de pérdidas ---
    linear_plot_path = os.path.join(PLOT_DIR, "linear_loss.png") if save_plot else None
    if save_plot:  # 👈 solo guarda si lo pedimos
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Evolución de la pérdida (Lineal)")
        linear_plot_path = os.path.join(PLOT_DIR, "linear_loss.png")
        plt.savefig(linear_plot_path)
        plt.close()

    # --- 5. Retorno con métricas ---
    return {
        "message": "Modelo lineal entrenado (peso ~ tamaño)",
        "loss": safe_float(losses[-1]),
        "mse": safe_float(losses[-1]),
        "rmse": safe_float(math.sqrt(losses[-1])),
        "plot_path": linear_plot_path
    }


def predict_linear(size: float):
    """
    Predice el **peso de un gato** usando el modelo de regresión lineal entrenado.
    """

    # --- Validación de entrada ---
    if size < 15 or size > 125:
        raise ValueError("El tamaño debe estar entre 15 y 125 cm.")

    # --- Cargar modelo ---
    model = load_model(LinearRegressor, LINEAR_MODEL_PATH)
    if model is None:
        raise FileNotFoundError("Modelo lineal no entrenado aún.")

    # --- Normalizar entrada y predecir ---
    x_norm = (torch.tensor([[size]], dtype=torch.float32) - 40) / 10

    with torch.no_grad():
        peso_pred = model(x_norm).item()
        return {
            "tamaño_cm": safe_float(size),
            "peso_pred_kg": safe_float(max(peso_pred, 0.5))  # mínimo 0.5 kg
        }

# ===================== MODELO LOGÍSTICO =====================
def train_logistic_model(save_plot: bool = True):
    """
    Entrena un modelo de regresión logística para predecir si un gato atrapa un ratón.
    """
    
    # --- 1. Datos simulados ---
    velocidad = torch.rand(500, 1) * 20
    energia = torch.rand(500, 1)
    # Condición base: velocidad * energia
    base = velocidad * energia

    # Le agregamos ruido y una condición "difusa"
    y = ((base + 0.5 * torch.randn(500, 1)) > 3).float()

    x = torch.cat([velocidad, energia], dim=1)

    # --- 2. Modelo y entrenamiento ---
    model = LogisticRegressor()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()


    losses = []
    for _ in range(500):
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    # --- 3. Evaluación ---
    with torch.no_grad():
        y_pred_prob = model(x).numpy()
        y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # --- 4. Métricas ---
    acc = accuracy_score(y.numpy(), y_pred_class)
    prec = precision_score(y.numpy(), y_pred_class)
    rec = recall_score(y.numpy(), y_pred_class)
    f1 = f1_score(y.numpy(), y_pred_class)

       # --- Gráficas y guardado opcionales ---
    cm_plot_path = None
    roc_plot_path = None

    if save_plot:
        # --- 5. Matriz de confusión ---
        cm = confusion_matrix(y.numpy(), y_pred_class)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión (Logístico)")
        cm_plot_path = os.path.join(PLOT_DIR, "logistic_confusion.png")
        plt.savefig(cm_plot_path)
        plt.close()

        # --- 6. Curva ROC ---
        fpr, tpr, _ = roc_curve(y.numpy(), y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC (AUC={roc_auc:.2f})")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC (Logístico)")
        plt.legend(loc="lower right")
        roc_plot_path = os.path.join(PLOT_DIR, "logistic_roc.png")
        plt.savefig(roc_plot_path)
        plt.close()

        # --- 7. Guardar modelo ---
        save_model(model, LOGISTIC_MODEL_PATH)

    # --- 8. Retorno ---
    return {
        "message": "Modelo logístico entrenado (atrapa ratón)",
        "loss": safe_float(losses[-1]),
        "accuracy": safe_float(acc),
        "precision": safe_float(prec),
        "recall": safe_float(rec),
        "f1_score": safe_float(f1),
        "plots": {
            "confusion_matrix": cm_plot_path,
            "roc_curve": roc_plot_path
        }
    }

def predict_logistic(x1: float, x2: float):
    """
    Predice si un gato atrapará un ratón usando el modelo de regresión logística.
    """

    # --- Validación de entrada ---
    if x1 < 0 or x1 > 20:
        raise ValueError("La velocidad debe estar entre 0 y 20 m/s.")
    if x2 < 0 or x2 > 1:
        raise ValueError("La energía debe estar entre 0 y 1.")

    # --- Cargar modelo ---
    """
    Qué valida: Que el modelo ya esté entrenado y guardado.
    Por qué: si intentas predecir sin entrenar primero → lanza error claro.
    """
    model = load_model(LogisticRegressor, LOGISTIC_MODEL_PATH)
    if model is None:
        raise FileNotFoundError("Modelo logístico no entrenado aún.")

    # --- Predicción ---
    with torch.no_grad():
        prob = model(torch.tensor([[x1, x2]])).item()
        clase = 1 if prob >= 0.5 else 0
        return {
            "velocidad": safe_float(x1),
            "energia": safe_float(x2),
            "probabilidad": safe_float(prob),
            "clase": clase # 0=no atrapa, 1=atrapa
        } 
