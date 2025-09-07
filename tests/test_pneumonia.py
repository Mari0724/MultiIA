import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_pneumonia_prediction_no_model():
    """
    Verifica que el endpoint funcione aunque el modelo no esté entrenado.
    """
    # Usamos un archivo ficticio (puede ser cualquier binario)
    file_content = b"fake-image-content"

    response = client.post(
        "/vision/pneumonia/analyze",
        files={"file": ("test.png", file_content, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()

    # Validamos la estructura de la respuesta
    assert "file_path" in data
    assert "prediction" in data
    assert "confidence" in data
