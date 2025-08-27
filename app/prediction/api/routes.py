from fastapi import APIRouter, HTTPException, Query
from app.prediction.application.prediction_service import (
    train_linear_model, predict_linear,
    train_logistic_model, predict_logistic
)

# Definición del router de FastAPI

# Se define un router para agrupar las rutas relacionadas con predicciones.
# Todas las rutas estarán bajo el prefijo "/prediction".
# Además, se agrupan en la etiqueta "Predicción" para la documentación automática de Swagger.

router = APIRouter(prefix="/prediction", tags=["Predicción"])  # 👈 agrupamos rutas bajo /prediction

# Endpoint raíz

@router.get("/", summary="Estado del API", response_description="Mensaje de bienvenida")
def root():
    """
    Endpoint raíz de la API de predicciones.
    
    Retorna un mensaje indicando que el servicio está activo.
    Útil para verificar que el servidor corre correctamente.
    """
    return {"message": "Prediction API lista 🚀"}

# Regresión Lineal

@router.get(
    "/linear/train",
    summary="Entrenar modelo lineal",
    response_description="Métricas del entrenamiento (MSE, pérdida)"
)
def train_linear():
    """
    Entrena un modelo de **regresión lineal**.
    
    - **Objetivo**: Predecir el **peso de un gato (kg)** a partir de su tamaño.
    - **Datos simulados**: tamaños de gatos entre 20–60 cm.
    - **Salida**: métricas del entrenamiento, como error cuadrático medio (MSE) y pérdida final.
    """
    return train_linear_model(save_plot=True)


@router.get(
    "/linear/predict",
    summary="Predecir peso con modelo lineal",
    response_description="Peso estimado (kg) del gato"
)
def predict_linear_endpoint(
    x: float = Query(..., description="Tamaño del gato en cm (15–125)", example=42.0)
):
    """
    Usa el modelo entrenado de regresión lineal para predecir el **peso de un gato**.

    - **Parámetro de entrada**:
        - `x`: tamaño del gato en cm (valores entre 15 y 125).
    - **Salida**:
        - JSON con métricas y el peso estimado en kg.
    """
    try:
        return predict_linear(x)
    except FileNotFoundError:
        # Si no existe un modelo entrenado previamente, retorna error 404
        raise HTTPException(status_code=404, detail="Modelo no entrenado")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Regresión Logística

@router.get(
    "/logistic/train",
    summary="Entrenar modelo logístico",
    response_description="Métricas del entrenamiento (accuracy, F1, etc.)"
)
def train_logistic():
    """
    Entrena un modelo de **regresión logística**.
    
    - **Objetivo**: Predecir si un gato atrapa un ratón.
    - **Características usadas (features)**:
        - velocidad del gato (0–20 m/s)
        - nivel de energía (0–1)
    - **Salida**: métricas del modelo como accuracy, precision, recall y F1-score.
    """
    return train_logistic_model(save_plot=True)

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
    Usa el modelo entrenado de regresión logística para predecir si un gato atrapará un ratón.

    - **Parámetros de entrada**:
        - `x1`: velocidad en m/s (0–20).
        - `x2`: nivel de energía (0–1).
    - **Salida**:
        - `probabilidad`: valor entre [0,1] indicando la confianza del modelo.
        - `clase`: 
            - 0 = no atrapa al ratón
            - 1 = atrapa al ratón
    """
    try:
        return predict_logistic(x1, x2)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo no entrenado")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
