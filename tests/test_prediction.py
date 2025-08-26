import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# TEST LINEAR
def test_train_linear():
    response = client.get("/prediction/linear/train")
    assert response.status_code == 200
    data = response.json()
    assert "loss" in data
    assert data["loss"] >= 0  # el loss no debería ser negativo

def test_predict_linear():
    # primero entrenamos el modelo
    client.get("/prediction/linear/train")

    # predicción válida
    response = client.get("/prediction/linear/predict", params={"x": 40})
    assert response.status_code == 200
    data = response.json()
    assert "peso_pred_kg" in data
    assert "tamaño_cm" in data

    # predicción inválida
    response = client.get("/prediction/linear/predict", params={"x": 200})
    assert response.status_code in (400, 422)

# TEST LOGISTIC
def test_train_logistic():
    response = client.get("/prediction/logistic/train")
    assert response.status_code == 200
    data = response.json()
    assert "loss" in data
    assert "accuracy" in data
    assert 0 <= data["accuracy"] <= 1  # accuracy siempre entre 0 y 1

def test_predict_logistic():
    # primero entrenamos
    client.get("/prediction/logistic/train")

    # predicción válida
    response = client.get("/prediction/logistic/predict", params={"x1": 5, "x2": 0.5})
    assert response.status_code == 200
    data = response.json()
    assert "probabilidad" in data
    assert 0 <= data["probabilidad"] <= 1

    # predicción inválida (energía fuera de rango)
    response = client.get("/prediction/logistic/predict", params={"x1": 5, "x2": 2})
    assert response.status_code in (400, 422)
