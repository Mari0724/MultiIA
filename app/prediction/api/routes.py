from fastapi import APIRouter, HTTPException, Query
from app.prediction.application.prediction_service import (
    train_linear_model, predict_linear,
    train_logistic_model, predict_logistic
)

router = APIRouter(prefix="/prediction", tags=["Predicción"])  # 👈 agrupamos rutas bajo /prediction


# ------------------------------
# 🌱 Endpoint raíz
# ------------------------------
@router.get("/", summary="Estado del API", response_description="Mensaje de bienvenida")
def root():
    """
    Devuelve un mensaje indicando que la API de predicciones está activa.
    """
    return {"message": "Prediction API lista 🚀"}


# ------------------------------
# 📈 Regresión Lineal
# ------------------------------
@router.get(
    "/linear/train",
    summary="Entrenar modelo lineal",
    response_description="Métricas del entrenamiento (MSE, pérdida)"
)
def train_linear():
    """
    Entrena un modelo de **regresión lineal** para predecir el **peso del gato (kg)** a partir de su tamaño.

    - **Datos simulados**: gatos de 20–60 cm
    - **Salida**: pérdida final y métricas MSE
    """
    return train_linear_model()


@router.get(
    "/linear/predict",
    summary="Predecir peso con modelo lineal",
    response_description="Peso estimado (kg) del gato"
)
def predict_linear_endpoint(
    x: float = Query(..., description="Tamaño del gato en cm (15–125)", example=42.0)
):
    """
    Predice el **peso estimado** de un gato (en kg) a partir de su **tamaño en cm**.

    - **Input**: tamaño del gato en cm
    - **Rango válido**: 15–125
    - **Output**: JSON con métricas y peso estimado
    """
    try:
        return predict_linear(x)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo no entrenado")


# ------------------------------
# 🐭 Regresión Logística
# ------------------------------
@router.get(
    "/logistic/train",
    summary="Entrenar modelo logístico",
    response_description="Métricas del entrenamiento (accuracy, F1, etc.)"
)
def train_logistic():
    """
    Entrena un modelo de **regresión logística** que predice si un gato atrapa un ratón.

    - **Features**:
        - velocidad (0–20 m/s)
        - energía (0–1)
    - **Output**: métricas como accuracy, precision, recall, f1_score
    """
    return train_logistic_model()


@router.get(
    "/logistic/predict",
    summary="Predecir captura de ratón",
    response_description="Probabilidad y clase de predicción"
)
def predict_logistic_endpoint(
    x1: float = Query(..., description="Velocidad del gato en m/s (0–20)", example=12.5),
    x2: float = Query(..., description="Nivel de energía del gato (0–1)", example=0.7)
):
    """
    Predice si un gato **atrapa un ratón** o no.

    - **Input**:
        - velocidad (m/s) entre 0 y 20
        - energía entre 0 y 1
    - **Output**:
        - probabilidad entre [0,1]
        - clase: 0 = no atrapa, 1 = atrapa
    """
    try:
        return predict_logistic(x1, x2)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo no entrenado")
