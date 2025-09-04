import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ==================== TEST RESUMEN ====================

def test_resumen_texto_valido():
    """
    Verifica que el endpoint /nlp/resumen genere un resumen correcto.
    """
    texto = {
        "texto": "La inteligencia artificial está revolucionando la industria. "
                 "Gracias a sus avances, ahora podemos automatizar procesos, "
                 "mejorar diagnósticos médicos y optimizar cadenas de suministro."
    }

    response = client.post("/nlp/resumen", json=texto)
    assert response.status_code == 200

    data = response.json()
    # Validamos que tenga las claves esperadas
    assert "texto_original" in data
    assert "resumen" in data
    assert "palabras_original" in data
    assert "palabras_resumen" in data
    assert "reduccion" in data

    # El resumen debe ser más corto que el texto original
    assert data["palabras_resumen"] <= data["palabras_original"]


def test_resumen_texto_vacio():
    """
    Verifica que el endpoint falle cuando no se envía texto.
    """
    response = client.post("/nlp/resumen", json={"texto": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Debes enviar un texto para resumir"
