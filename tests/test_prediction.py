import pytest
from fastapi.testclient import TestClient
from main import app
from app.prediction.application.prediction_service import (
    train_linear_model,
    train_logistic_model
)

client = TestClient(app)

# ==================== TEST REGRESIÓN LINEAL ====================

def test_train_linear():
    """
    Entrena el modelo lineal SIN guardar gráficas.
    Verifica que la pérdida sea un valor válido.
    """
    data = train_linear_model(save_plot=False)  # 👈 evita sobrescribir imágenes
    assert "loss" in data
    assert data["loss"] >= 0

def test_predict_linear():
    """
    Usa el endpoint de predicción para validar el modelo lineal.
    """
    # aseguramos que el modelo esté entrenado
    train_linear_model(save_plot=False)

    # predicción válida
    response = client.get("/prediction/linear/predict", params={"x": 40})
    assert response.status_code == 200
    data = response.json()
    assert "peso_pred_kg" in data
    assert "tamaño_cm" in data

    # predicción inválida (fuera de rango)
    response = client.get("/prediction/linear/predict", params={"x": 200})
    assert response.status_code in (400, 422)


# ==================== TEST REGRESIÓN LOGÍSTICA ====================

def test_train_logistic():
    """
    Entrena el modelo logístico SIN guardar gráficas.
    Verifica métricas de pérdida y exactitud.
    """
    data = train_logistic_model(save_plot=False)  # 👈 evita sobrescribir imágenes
    assert "loss" in data
    assert "accuracy" in data
    assert 0 <= data["accuracy"] <= 1


def test_predict_logistic():
    """
    Usa el endpoint de predicción para validar el modelo logístico.
    """
    # aseguramos que el modelo esté entrenado
    train_logistic_model(save_plot=False)

    # predicción válida
    response = client.get("/prediction/logistic/predict", params={"x1": 5, "x2": 0.5})
    assert response.status_code == 200
    data = response.json()
    assert "probabilidad" in data
    assert 0 <= data["probabilidad"] <= 1

    # predicción inválida (energía fuera de rango)
    response = client.get("/prediction/logistic/predict", params={"x1": 5, "x2": 2})
    assert response.status_code in (400, 422)
