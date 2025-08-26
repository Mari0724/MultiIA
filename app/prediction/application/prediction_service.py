from app.prediction.infrastructure.model_storage import save_model, load_model
from app.prediction.domain.models import LinearRegressor, LogisticRegressor
import torch
from torch import nn
import math
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib
matplotlib.use("Agg")   # 👈 backend sin GUI, evita Tkinter
import matplotlib.pyplot as plt

# Función de validación profesional

def safe_float(value: float) -> float:
    """
    Convierte un número a float seguro para serialización JSON.
    
    - Si el valor es NaN (Not a Number) o infinito → lanza un error.
    - Garantiza que las respuestas de la API sean válidas en JSON.
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Valor no válido detectado: {value}")
    return float(value)

# Rutas donde se guardarán los modelos

LINEAR_MODEL_PATH = "app/prediction/infrastructure/models/linear_regression.pth"
LOGISTIC_MODEL_PATH = "app/prediction/infrastructure/models/logistic_regression.pth"
PLOT_DIR = "app/prediction/infrastructure/plots"

# Crear la carpeta plots si no existe
os.makedirs(PLOT_DIR, exist_ok=True)

# Regresión Lineal (peso ~ tamaño)

def train_linear_model():
    """
    Entrena un modelo de regresión lineal para predecir el **peso del gato**
    a partir de su **tamaño**.

    Proceso:
    1. Genera datos simulados (tamaño en cm entre 20–60).
    2. Normaliza los datos para mejorar el entrenamiento.
    3. Define modelo, optimizador (SGD) y función de pérdida (MSE).
    4. Realiza 1000 iteraciones de entrenamiento.
    5. Guarda el modelo entrenado y genera una gráfica de la pérdida.
    
    Retorna:
    - Última pérdida (MSE y RMSE).
    - Ruta de la gráfica de evolución de la pérdida.
    """
    # Datos simulados: tamaño entre 20–60 cm
    x_raw = torch.linspace(20, 60, 200).unsqueeze(1)
    x = (x_raw - x_raw.mean()) / x_raw.std()
    y = 0.18 * x_raw - 2 + 0.5 * torch.randn(200, 1)

    # Definición de modelo, optimizador y función de pérdida
    model = LinearRegressor()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in range(1000):
        pred = model(x)
        loss = loss_fn(pred, y)

        if torch.isnan(loss):
            raise ValueError("El entrenamiento produjo NaN en la pérdida")

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    # Guardar modelo entrenado
    save_model(model, LINEAR_MODEL_PATH)

    # Guardar gráfica de pérdida
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Evolución de la pérdida (Lineal)")
    linear_plot_path = os.path.join(PLOT_DIR, "linear_loss.png")
    plt.savefig(linear_plot_path)
    plt.close()

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

    - Entrada: tamaño del gato en cm (15–125).
    - Normaliza la entrada y obtiene la predicción.
    - Devuelve el peso estimado (mínimo 0.5 kg por realismo).
    """
    if size < 15 or size > 125:
        raise ValueError("El tamaño debe estar entre 15 y 125 cm.")

    model = load_model(LinearRegressor, LINEAR_MODEL_PATH)
    if model is None:
        raise FileNotFoundError("Modelo lineal no entrenado aún.")

    # Normalización aproximada
    x_norm = (torch.tensor([[size]], dtype=torch.float32) - 40) / 10

    with torch.no_grad():
        peso_pred = model(x_norm).item()
        return {
            "tamaño_cm": safe_float(size),
            "peso_pred_kg": safe_float(max(peso_pred, 0.5))  # mínimo 0.5 kg
        }

# Regresión Logística (atrapa ratón o no)

def train_logistic_model():
    """
    Entrena un modelo de regresión logística para predecir si un gato atrapa un ratón.

    Proceso:
    1. Genera datos simulados (velocidad y energía).
    2. Define etiquetas (1 si velocidad*energía > 3, sino 0).
    3. Entrena el modelo con BCE (Binary Cross Entropy).
    4. Evalúa métricas: accuracy, precision, recall, F1.
    5. Genera gráficas de:
        - Matriz de confusión.
        - Curva ROC.
    6. Guarda modelo y gráficas.
    
    Retorna:
    - Última pérdida.
    - Métricas de evaluación.
    - Rutas de las gráficas.
    """
    # Datos simulados
    velocidad = torch.rand(500, 1) * 10
    energia = torch.rand(500, 1)
    y = ((velocidad * energia) > 3).float()
    x = torch.cat([velocidad, energia], dim=1)

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

    # Evaluación
    with torch.no_grad():
        y_pred_prob = model(x).numpy()
        y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y.numpy(), y_pred_class)
    prec = precision_score(y.numpy(), y_pred_class)
    rec = recall_score(y.numpy(), y_pred_class)
    f1 = f1_score(y.numpy(), y_pred_class)

    # Matriz de confusión
    cm = confusion_matrix(y.numpy(), y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión (Logístico)")
    cm_plot_path = os.path.join(PLOT_DIR, "logistic_confusion.png")
    plt.savefig(cm_plot_path)
    plt.close()

    # Curva ROC
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

    save_model(model, LOGISTIC_MODEL_PATH)
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

    - Entradas:
        - x1: velocidad del gato en m/s (0–20).
        - x2: nivel de energía (0–1).
    - Validaciones: error si está fuera de rango.
    - Salida:
        - velocidad y energía ingresadas.
        - probabilidad (0–1).
        - clase (0 = no atrapa, 1 = atrapa).
    """
    if x1 < 0 or x1 > 20:
        raise ValueError("La velocidad debe estar entre 0 y 20 m/s.")
    if x2 < 0 or x2 > 1:
        raise ValueError("La energía debe estar entre 0 y 1.")

    model = load_model(LogisticRegressor, LOGISTIC_MODEL_PATH)
    if model is None:
        raise FileNotFoundError("Modelo logístico no entrenado aún.")

    with torch.no_grad():
        prob = model(torch.tensor([[x1, x2]])).item()
        clase = 1 if prob >= 0.5 else 0
        return {
            "velocidad": safe_float(x1),
            "energia": safe_float(x2),
            "probabilidad": safe_float(prob),
            "clase": clase
        }
