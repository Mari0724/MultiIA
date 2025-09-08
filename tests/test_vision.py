# tests/test_vision.py
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)


def test_detect_objects_success():
    """✅ Caso feliz: el servicio procesa bien la imagen"""
    with patch("app.vision.application.vision_service.VisionService.detect_objects") as mock_service:
        mock_service.return_value = {"status": "ok", "objects": ["cat", "dog"]}

        fake_image = io.BytesIO(b"fake image content")
        response = client.post(
            "/vision/detect",
            files={"file": ("test.jpg", fake_image, "image/jpeg")}
        )

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "objects": ["cat", "dog"]}


def test_detect_objects_failure():
    """❌ Caso fallido: el servicio lanza error → ahora debe dar 400"""
    with patch("app.vision.application.vision_service.VisionService.detect_objects") as mock_service:
        mock_service.side_effect = Exception("Error en detección")

        fake_image = io.BytesIO(b"fake image content")
        response = client.post(
            "/vision/detect",
            files={"file": ("test.jpg", fake_image, "image/jpeg")}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Error en detección"


def test_detect_objects_no_file():
    """🚫 Caso borde: no se manda archivo"""
    response = client.post("/vision/detect", files={})
    assert response.status_code == 422  # FastAPI valida antes de entrar al endpoint
